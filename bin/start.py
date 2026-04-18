#!/usr/bin/env python3
"""
ubiservice one-click launcher.

Starts Redis, background task generation, and the Matcher API server.
Ctrl+C to shut down everything gracefully.
"""

import os
import sys
import json
import time
import signal
import subprocess
import threading

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
sys.path.insert(0, SRC_DIR)

from defination import ConfigRegistry, Task
from task_builder import TaskBuilder
from fake_generator import generate_messages

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REDIS_PORT = 6379
MATCHER_PORT = 8003
CONFIG_PATH = os.path.join(CONFIG_DIR, "defination_base.json")

# Sub-processes to clean up
_redis_proc = None
_matcher_proc = None
_stop_event = threading.Event()


# ---------------------------------------------------------------------------
# Redis management
# ---------------------------------------------------------------------------
def kill_stale_redis():
    """Kill any existing Redis on our port."""
    try:
        result = subprocess.run(
            ["redis-cli", "-p", str(REDIS_PORT), "shutdown", "nosave"],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass
    time.sleep(0.5)


def start_redis():
    global _redis_proc
    kill_stale_redis()
    _redis_proc = subprocess.Popen(
        [
            "redis-server",
            "--port", str(REDIS_PORT),
            "--save", "",
            "--maxmemory", "4gb",
            "--maxmemory-policy", "allkeys-lru",
            "--loglevel", "warning",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for Redis to be ready
    for _ in range(20):
        try:
            result = subprocess.run(
                ["redis-cli", "-p", str(REDIS_PORT), "ping"],
                capture_output=True, timeout=2,
            )
            if b"PONG" in result.stdout:
                print(f"[start] Redis started on port {REDIS_PORT}")
                return
        except Exception:
            pass
        time.sleep(0.3)
    print("[start] WARNING: Redis may not be ready")


# ---------------------------------------------------------------------------
# Task generation thread
# ---------------------------------------------------------------------------
def task_generator_loop():
    """
    Background thread: continuously generate fake messages,
    build tasks, and push to Redis.
    Rate: ~10 tasks/second.
    """
    import redis as sync_redis

    rdb = sync_redis.Redis(host="localhost", port=REDIS_PORT, decode_responses=False)

    registry = ConfigRegistry(CONFIG_PATH)
    builder = TaskBuilder(rank=0, registry=registry, max_wait_time=0.3)
    gen = generate_messages(registry, msg_id_start=0)

    task_count = 0
    print("[start] Task generator started")

    while not _stop_event.is_set():
        # Generate enough messages to fill all buckets (4+ per round)
        num_buckets = len(builder.buckets)
        for _ in range(num_buckets + 1):
            msg = next(gen)
            builder.put(msg)

        # Try to build a task
        task = builder.maybe_build()
        if task is not None:
            task_id = task.overview.task_id
            task_data = task.model_dump()

            pipe = rdb.pipeline()
            pipe.hset(
                f"task:{task_id}",
                mapping={
                    "overview": json.dumps(task.overview.model_dump()),
                    "full": json.dumps(task_data),
                    "max_winners": task.overview.max_winners,
                },
            )
            pipe.rpush("task_queue", task_id)
            pipe.execute()

            task_count += 1
            if task_count % 50 == 0:
                print(f"[start] Generated {task_count} tasks so far")

        time.sleep(0.1)

    rdb.close()
    print(f"[start] Task generator stopped ({task_count} tasks total)")


# ---------------------------------------------------------------------------
# Matcher management
# ---------------------------------------------------------------------------
def start_matcher():
    global _matcher_proc
    env = os.environ.copy()
    env["REDIS_URL"] = f"redis://localhost:{REDIS_PORT}"

    _matcher_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "matcher:app",
            "--host", "0.0.0.0",
            "--port", str(MATCHER_PORT),
            "--log-level", "info",
        ],
        cwd=SRC_DIR,
        env=env,
    )
    print(f"[start] Matcher starting on port {MATCHER_PORT}")


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
def cleanup(*args):
    print("\n[start] Shutting down...")
    _stop_event.set()

    if _matcher_proc and _matcher_proc.poll() is None:
        _matcher_proc.terminate()
        try:
            _matcher_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _matcher_proc.kill()
        print("[start] Matcher stopped")

    if _redis_proc and _redis_proc.poll() is None:
        try:
            subprocess.run(
                ["redis-cli", "-p", str(REDIS_PORT), "shutdown", "nosave"],
                capture_output=True, timeout=5,
            )
        except Exception:
            _redis_proc.kill()
        print("[start] Redis stopped")

    print("[start] Bye!")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("=" * 60)
    print("  ubiservice mock server")
    print("=" * 60)

    # 1. Start Redis
    start_redis()

    # 2. Start background task generator
    gen_thread = threading.Thread(target=task_generator_loop, daemon=True)
    gen_thread.start()

    # Wait a moment for initial tasks to be generated
    time.sleep(1.0)

    # 3. Start Matcher API
    start_matcher()

    # Wait for matcher to boot
    time.sleep(1.5)

    print()
    print("=" * 60)
    print("  Mock server ready!")
    print(f"  API:         http://127.0.0.1:{MATCHER_PORT}")
    print(f"  Register:    POST http://127.0.0.1:{MATCHER_PORT}/register")
    print(f"  Query:       POST http://127.0.0.1:{MATCHER_PORT}/query")
    print(f"  Ask:         POST http://127.0.0.1:{MATCHER_PORT}/ask")
    print(f"  Submit:      POST http://127.0.0.1:{MATCHER_PORT}/submit")
    print(f"  Leaderboard: GET  http://127.0.0.1:{MATCHER_PORT}/leaderboard")
    print("=" * 60)
    print("  Press Ctrl+C to stop")
    print()

    # Keep main thread alive
    try:
        while True:
            # Check child processes
            if _matcher_proc and _matcher_proc.poll() is not None:
                print("[start] WARNING: Matcher process exited unexpectedly")
                cleanup()
            if _redis_proc and _redis_proc.poll() is not None:
                print("[start] WARNING: Redis process exited unexpectedly")
                cleanup()
            time.sleep(2)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
