#!/usr/bin/env python3
"""
Evaluation runner: unpack contestant submission, start platform, run eval.

Usage:
    python bin/run_eval.py \
        --submission /path/to/team_alpha.tar.gz \
        --team-name team_alpha \
        --team-token tok_alpha \
        --duration 60 \
        --output /tmp/result.json
"""

import argparse
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import zipfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("run_eval")

UBISERVICE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(UBISERVICE_DIR, "src")
CONFIG_DIR = os.path.join(UBISERVICE_DIR, "config")
BIN_DIR = os.path.join(UBISERVICE_DIR, "bin")

# Add bin/ to path for start_platform import
if BIN_DIR not in sys.path:
    sys.path.insert(0, BIN_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _extract_submission(archive_path, dest_dir):
    """Extract .tar.gz or .zip to dest_dir."""
    os.makedirs(dest_dir, exist_ok=True)
    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            tf.extractall(dest_dir)
    elif zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_path}")

    # If extracted into a single subdirectory, flatten it
    entries = os.listdir(dest_dir)
    if len(entries) == 1 and os.path.isdir(os.path.join(dest_dir, entries[0])):
        sub = os.path.join(dest_dir, entries[0])
        for item in os.listdir(sub):
            shutil.move(os.path.join(sub, item), dest_dir)
        os.rmdir(sub)


def _generate_contest_config(output_path, platform_url, contestant_port, duration_s, model_path):
    """Generate contest.json for the contestant."""
    # Read SLA and SamplingParam from defination_base.json
    defination_path = os.path.join(CONFIG_DIR, "defination_base.json")
    with open(defination_path) as f:
        defination = json.load(f)

    sla_levels = {}
    for name, spec in defination.get("SLA", {}).items():
        sla_levels[name] = {"latency_max": spec.get("ttft_avg", 10.0)}

    config = {
        "platform_url": platform_url,
        "model_name": "Qwen3-32B",
        "model_path": model_path,
        "contestant_port": contestant_port,
        "duration_s": duration_s,
        "sla_levels": sla_levels,
        "sampling_params": defination.get("SamplingParam", {}),
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    return config


def main():
    parser = argparse.ArgumentParser(description="Run evaluation for a contestant submission")
    parser.add_argument("--submission", required=True, help="Path to .tar.gz or .zip submission")
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--team-token", required=True)
    parser.add_argument("--duration", type=int, default=60, help="Eval duration in seconds (from contestant start)")
    parser.add_argument("--output", default="/tmp/eval_result.json", help="Output JSON path")
    parser.add_argument("--model-path", default="/mnt/model/Qwen3-32B")
    parser.add_argument("--contestant-port", type=int, default=9000)
    parser.add_argument("--platform-port", type=int, default=8003)
    parser.add_argument("--setup-timeout", type=int, default=300)
    parser.add_argument("--work-dir", default=None, help="Working directory (default: /tmp/eval_{team})")
    args = parser.parse_args()

    work_dir = args.work_dir or f"/tmp/eval_{args.team_name}"
    submission_dir = os.path.join(work_dir, "submission")
    contest_config_path = os.path.join(work_dir, "contest.json")
    platform_url = f"http://127.0.0.1:{args.platform_port}"

    result = {
        "team_name": args.team_name,
        "team_token": args.team_token,
        "duration_s": args.duration,
        "score": 0.0,
        "tasks_completed": 0,
        "tasks_accepted": 0,
        "avg_correctness": 0.0,
        "avg_latency_ms": 0.0,
        "credit": 1.0,
        "setup_ok": False,
        "startup_ok": False,
        "error": None,
    }

    contestant_proc = None
    platform_started = False

    try:
        # ── Phase 1: Prepare ──────────────────────────────────────
        logger.info(f"[Phase 1] Extracting {args.submission} → {submission_dir}")
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        _extract_submission(args.submission, submission_dir)

        run_sh = os.path.join(submission_dir, "run.sh")
        if not os.path.exists(run_sh):
            result["error"] = "run.sh not found in submission"
            logger.error(result["error"])
            return

        # setup.sh (optional)
        setup_sh = os.path.join(submission_dir, "setup.sh")
        if os.path.exists(setup_sh):
            logger.info(f"[Phase 1] Running setup.sh (timeout={args.setup_timeout}s)...")
            try:
                subprocess.run(
                    ["bash", "setup.sh"],
                    cwd=submission_dir,
                    timeout=args.setup_timeout,
                    check=True,
                )
                logger.info("[Phase 1] setup.sh completed.")
            except subprocess.TimeoutExpired:
                result["error"] = f"setup.sh timed out after {args.setup_timeout}s"
                logger.error(result["error"])
                return
            except subprocess.CalledProcessError as e:
                result["error"] = f"setup.sh failed with exit code {e.returncode}"
                logger.error(result["error"])
                return
        result["setup_ok"] = True

        # ── Phase 2: Start platform ──────────────────────────────
        logger.info("[Phase 2] Starting ubiservice platform...")

        # Kill stale Redis
        import redis as redis_sync
        try:
            _r = redis_sync.from_url("redis://localhost:6379")
            _r.ping()
            _r.shutdown(nosave=True)
            time.sleep(1)
        except Exception:
            pass

        sys.path.insert(0, SRC_DIR)
        from start_platform import start_platform, stop_platform
        start_platform()
        platform_started = True

        # Wait for tasks
        r = redis_sync.from_url("redis://localhost:6379")
        for i in range(60):
            try:
                if r.llen("task_queue") >= 1:
                    logger.info(f"[Phase 2] Tasks ready after {i}s")
                    break
            except Exception:
                pass
            time.sleep(1)
        else:
            result["error"] = "No tasks appeared after 60s"
            logger.error(result["error"])
            return

        # ── Phase 3: Generate config + start contestant ──────────
        logger.info("[Phase 3] Generating contest.json and starting contestant...")
        _generate_contest_config(
            contest_config_path, platform_url,
            args.contestant_port, args.duration, args.model_path,
        )

        env = os.environ.copy()
        env["CONTESTANT_PORT"] = str(args.contestant_port)
        env["MODEL_PATH"] = args.model_path
        env["CONFIG_PATH"] = contest_config_path
        env["PLATFORM_URL"] = platform_url
        env["TEAM_NAME"] = args.team_name
        env["TEAM_TOKEN"] = args.team_token

        contestant_proc = subprocess.Popen(
            ["bash", "run.sh"],
            cwd=submission_dir,
            env=env,
            preexec_fn=os.setsid,
            stdout=open(os.path.join(work_dir, "contestant_stdout.log"), "w"),
            stderr=open(os.path.join(work_dir, "contestant_stderr.log"), "w"),
        )
        logger.info(f"[Phase 3] Contestant started (PID={contestant_proc.pid})")
        result["startup_ok"] = True

        # ── Phase 4: Run evaluation ──────────────────────────────
        # Duration starts NOW — from the moment contestant is launched
        logger.info(f"[Phase 4] Evaluation running for {args.duration}s...")
        import urllib.request
        deadline = time.time() + args.duration
        while time.time() < deadline:
            sleep_time = min(30, deadline - time.time())
            if sleep_time <= 0:
                break
            time.sleep(sleep_time)

            # Progress check
            try:
                elapsed = args.duration - (deadline - time.time())
                req = urllib.request.Request(f"{platform_url}/leaderboard")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    lb = json.loads(resp.read())
                entries = lb.get("leaderboard", [])
                for e in entries:
                    if e.get("name") == args.team_name:
                        logger.info(
                            f"[Progress] {elapsed:.0f}s: score={e['score']:.2f} "
                            f"completed={e.get('tasks_completed', '?')} "
                            f"latency={e.get('avg_latency_ms', '?')}ms"
                        )
                        break
                else:
                    logger.info(f"[Progress] {elapsed:.0f}s: team not on leaderboard yet")
            except Exception:
                pass

        # ── Phase 5: Collect results ─────────────────────────────
        logger.info("[Phase 5] Collecting results...")

        # Stop contestant
        if contestant_proc and contestant_proc.poll() is None:
            try:
                os.killpg(os.getpgid(contestant_proc.pid), signal.SIGTERM)
                contestant_proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(contestant_proc.pid), signal.SIGKILL)
                    contestant_proc.wait(timeout=5)
                except Exception:
                    pass
            logger.info("[Phase 5] Contestant stopped")

        # Fetch leaderboard
        try:
            req = urllib.request.Request(f"{platform_url}/leaderboard")
            with urllib.request.urlopen(req, timeout=5) as resp:
                lb = json.loads(resp.read())
            for e in lb.get("leaderboard", []):
                if e.get("name") == args.team_name:
                    result["score"] = e.get("score", 0.0)
                    result["tasks_completed"] = e.get("tasks_completed", 0)
                    result["tasks_accepted"] = e.get("tasks_accepted", 0)
                    result["avg_correctness"] = e.get("avg_correctness", 0.0)
                    result["avg_latency_ms"] = e.get("avg_latency_ms", 0.0)
                    result["credit"] = e.get("credit", 1.0)
                    break
        except Exception as ex:
            logger.warning(f"[Phase 5] Failed to fetch leaderboard: {ex}")

    except Exception as ex:
        result["error"] = str(ex)
        logger.error(f"Eval error: {ex}", exc_info=True)

    finally:
        # Stop contestant if still running
        if contestant_proc and contestant_proc.poll() is None:
            try:
                os.killpg(os.getpgid(contestant_proc.pid), signal.SIGKILL)
            except Exception:
                pass

        # Stop platform
        if platform_started:
            try:
                stop_platform()
            except Exception:
                pass

        # Write result
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Result written to {args.output}")

        print("\n" + "=" * 60)
        print(f"  EVAL RESULT: {args.team_name}")
        print("=" * 60)
        print(f"  Score:            {result['score']:.4f}")
        print(f"  Tasks completed:  {result['tasks_completed']}")
        print(f"  Tasks accepted:   {result['tasks_accepted']}")
        print(f"  Avg correctness:  {result['avg_correctness']:.4f}")
        print(f"  Avg latency:      {result['avg_latency_ms']:.1f}ms")
        print(f"  Setup OK:         {result['setup_ok']}")
        print(f"  Startup OK:       {result['startup_ok']}")
        if result["error"]:
            print(f"  Error:            {result['error']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
