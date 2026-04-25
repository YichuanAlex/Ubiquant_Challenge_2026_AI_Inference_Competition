"""
Fake contestant client for testing the evaluation pipeline.

Reads configuration from environment variables and contest.json,
then runs a continuous query → ask → fake-inference → submit loop.

This is a minimal example — real contestants would replace the
fake inference with actual model calls.
"""

import json
import os
import time

import requests

# ── Configuration from environment ──────────────────────────────────
PLATFORM_URL = os.environ.get("PLATFORM_URL", "http://127.0.0.1:8003")
TOKEN = os.environ.get("TEAM_TOKEN", "test_token")
TEAM_NAME = os.environ.get("TEAM_NAME", "test_team")

# Read contest config if available
config_path = os.environ.get("CONFIG_PATH", "")
if config_path and os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
    PLATFORM_URL = config.get("platform_url", PLATFORM_URL)
    print(f"[client] Loaded config from {config_path}")
    print(f"[client] Platform: {PLATFORM_URL}, Duration: {config.get('duration_s')}s")

print(f"[client] Team: {TEAM_NAME}, Token: {TOKEN}")

# ── Register ────────────────────────────────────────────────────────
requests.post(f"{PLATFORM_URL}/register", json={"name": TEAM_NAME, "token": TOKEN})
print(f"[client] Registered")

# ── Main loop ───────────────────────────────────────────────────────
completed = 0
while True:
    try:
        # Query
        r = requests.post(f"{PLATFORM_URL}/query", json={"token": TOKEN}, timeout=5)
        if r.status_code != 200:
            time.sleep(0.3)
            continue

        ov = r.json()
        tid = ov.get("task_id")
        if not tid:
            time.sleep(0.3)
            continue

        # Ask
        r = requests.post(f"{PLATFORM_URL}/ask", json={
            "token": TOKEN, "task_id": tid, "sla": ov["target_sla"],
        }, timeout=5)
        ask = r.json()
        if ask.get("status") != "accepted":
            continue

        # Fake inference: fill dummy responses
        # Real contestants would call their inference service here
        task = ask["task"]
        for msg in task["messages"]:
            rt = msg.get("eval_request_type", "")
            if rt == "generate_until":
                msg["response"] = "fake response"
            elif rt == "loglikelihood":
                msg["accuracy"] = -1.0
            elif rt == "loglikelihood_rolling":
                msg["accuracy"] = -10.0
            else:
                msg["response"] = ""

        # Submit
        requests.post(f"{PLATFORM_URL}/submit", json={
            "user": {"name": TEAM_NAME, "token": TOKEN},
            "msg": task,
        }, timeout=10)
        completed += 1

        if completed % 50 == 0:
            print(f"[client] Completed {completed} tasks")

    except KeyboardInterrupt:
        break
    except Exception as e:
        time.sleep(0.5)

print(f"[client] Done. Completed {completed} tasks total.")
