"""
Platform start/stop utilities — shared by start.py and run_eval.py.
"""

import json
import os
import subprocess
import sys
import threading
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
CONFIG_DIR = os.path.join(BASE_DIR, "config")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

REDIS_PORT = 6379
MATCHER_PORT = 8003
CONFIG_PATH = os.path.join(CONFIG_DIR, "defination_base.json")

_redis_proc = None
_matcher_proc = None
_stop_event = threading.Event()


def start_platform():
    """Start Redis + task generator thread + matcher. Non-blocking."""
    global _redis_proc, _matcher_proc

    # Redis
    try:
        subprocess.run(
            ["redis-cli", "-p", str(REDIS_PORT), "shutdown", "nosave"],
            capture_output=True, timeout=5,
        )
        time.sleep(0.5)
    except Exception:
        pass

    _redis_proc = subprocess.Popen(
        ["redis-server", "--port", str(REDIS_PORT),
         "--save", "", "--maxmemory", "4gb",
         "--maxmemory-policy", "allkeys-lru", "--loglevel", "warning"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(20):
        try:
            result = subprocess.run(
                ["redis-cli", "-p", str(REDIS_PORT), "ping"],
                capture_output=True, timeout=2,
            )
            if b"PONG" in result.stdout:
                break
        except Exception:
            pass
        time.sleep(0.3)

    # Task generator thread
    _stop_event.clear()
    t = threading.Thread(target=_task_generator_loop, daemon=True)
    t.start()
    time.sleep(1.0)

    # Matcher
    env = os.environ.copy()
    env["REDIS_URL"] = f"redis://localhost:{REDIS_PORT}"
    _matcher_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "matcher:app",
         "--host", "0.0.0.0", "--port", str(MATCHER_PORT),
         "--log-level", "warning"],
        cwd=SRC_DIR, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2.0)


def stop_platform():
    """Gracefully stop matcher + Redis."""
    global _redis_proc, _matcher_proc
    _stop_event.set()

    if _matcher_proc and _matcher_proc.poll() is None:
        _matcher_proc.terminate()
        try:
            _matcher_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _matcher_proc.kill()

    if _redis_proc and _redis_proc.poll() is None:
        try:
            subprocess.run(
                ["redis-cli", "-p", str(REDIS_PORT), "shutdown", "nosave"],
                capture_output=True, timeout=5,
            )
        except Exception:
            _redis_proc.kill()


def _task_generator_loop():
    import redis as sync_redis
    from defination import ConfigRegistry
    from task_builder import TaskBuilder
    from fake_generator import generate_messages

    rdb = sync_redis.Redis(host="localhost", port=REDIS_PORT, decode_responses=False)
    registry = ConfigRegistry(CONFIG_PATH)
    builder = TaskBuilder(rank=0, registry=registry, max_wait_time=0.3)
    gen = generate_messages(registry, msg_id_start=0)

    while not _stop_event.is_set():
        for _ in range(len(builder.buckets) + 1):
            builder.put(next(gen))
        task = builder.maybe_build()
        if task is not None:
            task_id = task.overview.task_id
            pipe = rdb.pipeline()
            pipe.hset(f"task:{task_id}", mapping={
                "overview": json.dumps(task.overview.model_dump()),
                "full": json.dumps(task.model_dump()),
                "max_winners": task.overview.max_winners,
            })
            pipe.rpush("task_queue", task_id)
            pipe.execute()
        time.sleep(0.1)
    rdb.close()
