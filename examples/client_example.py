#!/usr/bin/env python3
"""
ubiservice client example.

Demonstrates the full competition flow:
  1. Register a team
  2. Query for available tasks
  3. Ask (bid) for a task
  4. Fill in responses for each message
  5. Submit the completed task
  6. Check leaderboard

Usage:
    python bin/start.py          # start mock server first
    python examples/client_example.py  # run this example
"""

import requests
import json
import time
import sys

MATCHER_URL = "http://127.0.0.1:8003"


def register(name: str, token: str):
    """
    Step 1: Register your team.
    Must be called before any other operation.
    """
    resp = requests.post(
        f"{MATCHER_URL}/register",
        json={"name": name, "token": token},
    )
    resp.raise_for_status()
    print(f"[register] {resp.json()}")
    return resp.json()


def query_task(token: str):
    """
    Step 2: Query for an available task.
    Returns a TaskOverview (task_id, target_sla, len_distribution, etc.)
    You do NOT get the full task data yet — just metadata for bidding.
    """
    resp = requests.post(
        f"{MATCHER_URL}/query",
        json={"token": token},
    )
    if resp.status_code == 404:
        print("[query] No tasks available, waiting...")
        return None
    resp.raise_for_status()
    overview = resp.json()
    print(f"[query] Got task {overview['task_id']} "
          f"(SLA={overview['target_sla']}, reward={overview['target_reward']})")
    return overview


def ask_task(token: str, task_id: int, sla: str):
    """
    Step 3: Ask (bid) for the task at a given SLA level.
    In preliminary mode, sla must match target_sla exactly.
    Returns the full task data if accepted.
    """
    resp = requests.post(
        f"{MATCHER_URL}/ask",
        json={"token": token, "task_id": task_id, "sla": sla},
    )
    resp.raise_for_status()
    result = resp.json()
    status = result["status"]
    print(f"[ask] task {task_id}: status={status}")
    return result


def submit_task(token: str, name: str, task_data: dict):
    """
    Step 4: Submit the completed task with your responses.

    For each message in the task:
      - If it has a 'prompt' (generate_until type): fill in 'response' field
      - If it needs loglikelihood: fill in 'accuracy' field

    The submit endpoint expects JSON body with 'user' and 'msg' fields.
    """
    resp = requests.post(
        f"{MATCHER_URL}/submit",
        json={
            "user": {"name": name, "token": token},
            "msg": task_data,
        },
    )
    resp.raise_for_status()
    print(f"[submit] {resp.json()}")
    return resp.json()


def get_leaderboard():
    """
    Step 5: Check the leaderboard to see your score.
    """
    resp = requests.get(f"{MATCHER_URL}/leaderboard")
    resp.raise_for_status()
    return resp.json()


def fill_fake_responses(task: dict) -> dict:
    """
    Fill in fake responses for all messages in the task.

    In a real competition, this is where your LLM inference happens.
    Check eval_request_type to determine what to do:

      - generate_until: call your model with eval_gen_kwargs params,
        generate text until hitting a stop token. Fill 'response' field.
      - loglikelihood: compute log-probability of eval_continuation
        given the prompt. Fill 'accuracy' field.
      - loglikelihood_rolling: compute total log-probability of the
        entire prompt text. Fill 'accuracy' field.
    """
    for msg in task["messages"]:
        rt = msg.get("eval_request_type", "")

        if rt == "generate_until":
            gen_kwargs = msg.get("eval_gen_kwargs", {})
            # Your model would use: temperature, top_p, max_gen_toks, until, etc.
            print(f"    [generate_until] gen_kwargs={gen_kwargs}")
            msg["response"] = f"Fake response to: {msg['prompt'][:40]}..."

        elif rt == "loglikelihood":
            continuation = msg.get("eval_continuation", "")
            # Your model would compute: log P(continuation | prompt)
            print(f"    [loglikelihood] continuation='{continuation}'")
            msg["accuracy"] = -0.5  # fake log-prob

        elif rt == "loglikelihood_rolling":
            # Your model would compute: total log P(prompt)
            print(f"    [loglikelihood_rolling] prompt_len={len(msg.get('prompt', ''))}")
            msg["accuracy"] = -10.0  # fake log-prob

        else:
            msg["response"] = ""

    return task


def main():
    team_name = "team_test"
    token = "test_token"
    num_rounds = 5

    print("=" * 60)
    print("  ubiservice client example")
    print("=" * 60)

    # Step 1: Register
    register(team_name, token)
    print()

    completed = 0

    for i in range(num_rounds):
        print(f"--- Round {i + 1}/{num_rounds} ---")

        # Step 2: Query for a task
        overview = query_task(token)
        if overview is None:
            time.sleep(1)
            continue

        task_id = overview["task_id"]
        target_sla = overview["target_sla"]

        # Step 3: Ask (bid) — must match target SLA in preliminary mode
        result = ask_task(token, task_id, sla=target_sla)
        if result["status"] != "accepted":
            print(f"  Bid not accepted ({result['status']}), trying next...")
            continue

        # Step 4: Fill in responses (this is where your model would run)
        task = result["task"]
        print(f"  Task has {len(task['messages'])} messages")
        task = fill_fake_responses(task)

        # Step 5: Submit
        submit_task(token, team_name, task)
        completed += 1
        print()

    # Step 6: Check leaderboard
    print("=" * 60)
    lb = get_leaderboard()
    print("Leaderboard:")
    for entry in lb["leaderboard"]:
        print(f"  {entry['name']:20s}  score={entry['score']}")
    print()
    print(f"Completed {completed}/{num_rounds} tasks")
    print("Done!")


if __name__ == "__main__":
    main()
