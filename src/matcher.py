"""
Simplified Matcher for ubiservice mock server.
Based on ubioracle's src/service/matcher.py — eval/ZMQ code removed,
submit gives a fixed 1.0 score per completed task.
"""

import os
import time
import json
import logging
import asyncio
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from typing import Optional

from defination import User, ParamAsk, ParamQuery, TaskOverview, Task

# -----------------------
# Configurations
# -----------------------
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
MAX_QUERY_PER_USER = 1024     # limit outstanding queried tasks
BID_WINDOW = 10               # seconds
MAX_INFLIGHT_PER_USER = 64
RATE_LIMIT_QUERY = 32         # max /query per second per user
RATE_LIMIT_ASK = 32           # max /ask per second per user

# Competition mode: "preliminary" or "final"
COMPETITION_MODE = os.environ.get("COMPETITION_MODE", "preliminary")

logger = logging.getLogger(__name__)

# SLA ranking (higher is better)
SLA_ORDER = {
    "Bronze": 1,
    "Silver": 2,
    "Gold": 3,
    "Platinum": 4,
    "Diamond": 5,
    "Stellar": 6,
    "Glorious": 7,
    "Supreme": 8,
}

# SLA -> ttft_avg (for deadline computation)
SLA_TTFT = {
    "Bronze": 10.0,
    "Silver": 8.0,
    "Gold": 6.0,
    "Platinum": 4.0,
    "Diamond": 2.0,
    "Stellar": 1.5,
    "Glorious": 0.8,
    "Supreme": 0.5,
}

# Create ASGI app instance
app = FastAPI(title="ubiservice mock matcher")
r = redis.from_url(REDIS_URL)

# Service start time (for round_elapsed_s in leaderboard)
_service_start_time = time.time()

# ── Per-user rate limiter (token bucket in memory, separate for query/ask) ──
_rate_buckets: dict = {}  # key = "token:endpoint" → [remaining, last_refill_time]

async def _check_rate_limit(token: str, endpoint: str, limit: int):
    """
    Token-bucket rate limiter. Separate buckets per endpoint.
    Raises 429 if bucket is empty.
    """
    now = time.time()
    key = f"{token}:{endpoint}"
    if key not in _rate_buckets:
        _rate_buckets[key] = [limit, now]

    bucket = _rate_buckets[key]
    elapsed = now - bucket[1]
    bucket[0] = min(limit, bucket[0] + elapsed * limit)
    bucket[1] = now

    if bucket[0] < 1.0:
        raise HTTPException(429, "Rate limit exceeded, slow down")
    bucket[0] -= 1.0


async def is_task_open(task_id: int) -> bool:
    data = await r.hgetall(f"task:{task_id}")
    if not data:
        return False

    max_winners = int(
        data[b"max_winners"] if b"max_winners" in data else data.get("max_winners", 5)
    )
    winners = await r.scard(f"task:{task_id}:winners")
    if winners >= max_winners:
        return False

    # BID_WINDOW counts from first ask
    first_bid_at = float(
        data.get(b"first_bid_at", data.get("first_bid_at", 0)) or 0
    )
    if first_bid_at > 0 and (time.time() - first_bid_at > BID_WINDOW):
        return False

    return True


async def record_first_bid(task_id: int):
    key = f"task:{task_id}"
    data = await r.hgetall(key)
    if b"first_bid_at" not in data or data[b"first_bid_at"] == b"":
        await r.hset(key, "first_bid_at", time.time())


# ---------------------------------------------------------------
# /register
# ---------------------------------------------------------------
@app.post("/register")
async def on_register(user: User):
    await r.set(f"user:{user.token}:name", user.name)
    await r.setnx(f"user:{user.token}:score", "0.0")
    await r.setnx(f"user:{user.token}:credit", "1.0")
    return {"status": "ok"}


# ---------------------------------------------------------------
# /query — find an open task for the user
# ---------------------------------------------------------------
@app.post("/query", response_model=TaskOverview)
async def query(msg: ParamQuery):
    await _check_rate_limit(msg.token, "query", RATE_LIMIT_QUERY)
    queried_key = f"user:{msg.token}:queried"

    # Prune queried set: remove tasks that are fully claimed or expired.
    queried_members = await r.smembers(queried_key)
    if queried_members:
        stale = []
        pipe = r.pipeline()
        for tid_raw in queried_members:
            tid = int(tid_raw)
            pipe.scard(f"task:{tid}:winners")
            pipe.hgetall(f"task:{tid}")
        prune_results = await pipe.execute()
        for idx, tid_raw in enumerate(queried_members):
            winners = prune_results[2 * idx]
            data = prune_results[2 * idx + 1]
            if not data:
                stale.append(tid_raw)   # task expired/deleted
            elif winners >= 1:          # max_winners=1 typical
                stale.append(tid_raw)   # task fully claimed
        if stale:
            await r.srem(queried_key, *stale)

    cnt = await r.scard(queried_key)
    if cnt >= MAX_QUERY_PER_USER + MAX_INFLIGHT_PER_USER:
        raise HTTPException(400, "Too many tasks")

    # Sliding window scan: fetch 200 tasks at a time from the head
    _window = 200

    queue_len = await r.llen("task_queue")
    if queue_len == 0:
        raise HTTPException(404, "No tasks")

    selected = None
    now = time.time()
    _start = 0

    while _start < queue_len:
        task_ids = await r.lrange("task_queue", _start, _start + _window - 1)
        task_ids = [int(t) for t in task_ids]
        if not task_ids:
            break

        pipe = r.pipeline()
        for tid in task_ids:
            pipe.sismember(queried_key, tid)
        for tid in task_ids:
            pipe.hgetall(f"task:{tid}")
            pipe.scard(f"task:{tid}:winners")
        try:
            results = await pipe.execute()
        except asyncio.TimeoutError:
            raise HTTPException(500, "Redis timeout")

        n = len(task_ids)
        queried_flags = results[:n]
        rest = results[n:]

        for i, tid in enumerate(task_ids):
            if queried_flags[i]:
                continue
            data = rest[2 * i]
            winners = rest[2 * i + 1]
            if data is None or not data:
                continue

            data = {
                (k.decode() if isinstance(k, bytes) else k):
                (v.decode() if isinstance(v, bytes) else v)
                for k, v in data.items()
            }
            max_winners = int(data["max_winners"])
            if winners >= max_winners:
                await r.srem(queried_key, tid)
                continue
            first_bid_at = float(data.get("first_bid_at") or 0)
            if first_bid_at > 0 and (now - first_bid_at > BID_WINDOW):
                continue

            selected = tid
            break

        if selected is not None:
            break
        _start += _window

    if selected is None:
        raise HTTPException(404, "No suitable task")

    pipe = r.pipeline()
    pipe.sadd(queried_key, selected)
    pipe.hget(f"task:{selected}", "overview")
    try:
        _, data = await pipe.execute()
    except asyncio.TimeoutError:
        raise HTTPException(500, "Redis timeout")

    if data is None:
        # Task was expired/deleted between selection and fetch — retry
        raise HTTPException(404, "Task expired, retry")

    return TaskOverview(**json.loads(data))


# ---------------------------------------------------------------
# /ask — bid for a task
# ---------------------------------------------------------------
@app.post("/ask")
async def ask(msg: ParamAsk):
    await _check_rate_limit(msg.token, "ask", RATE_LIMIT_ASK)
    token = msg.token
    task_id = msg.task_id
    sla = msg.sla

    queried_key = f"user:{token}:queried"
    try:
        is_queried = await r.sismember(queried_key, task_id)
    except asyncio.TimeoutError:
        raise HTTPException(500, "Redis timeout")
    if not is_queried:
        raise HTTPException(400, "Not queried")

    inflight_key = f"user:{token}:inflight"
    inflight_cnt = await r.scard(inflight_key)
    if inflight_cnt >= MAX_INFLIGHT_PER_USER:
        raise HTTPException(400, "too_many_inflight")

    key = f"task:{task_id}"
    pipe = r.pipeline()
    pipe.hgetall(key)
    pipe.scard(f"{key}:winners")
    pipe.exists(f"{key}:bids")
    try:
        results = await pipe.execute()
    except asyncio.TimeoutError:
        raise HTTPException(500, "Redis timeout")
    data_raw, winners_count, _ = results
    if not data_raw:
        return {"status": "closed"}

    # decode bytes -> str
    data = {
        k.decode() if isinstance(k, bytes) else k:
        v.decode() if isinstance(v, bytes) else v
        for k, v in data_raw.items()
    }
    max_winners = int(data["max_winners"])
    first_bid_at = float(data.get("first_bid_at") or 0)
    now = time.time()

    if first_bid_at > 0 and winners_count >= max_winners:
        return {"status": "closed"}

    # -- SLA validation -------------------------------------------
    overview_raw = data.get("overview")
    overview = json.loads(overview_raw) if overview_raw else {}
    target_sla = overview.get("target_sla", "Bronze")

    bid_sla_rank = SLA_ORDER.get(sla, 0)
    target_sla_rank = SLA_ORDER.get(target_sla, 0)

    if COMPETITION_MODE == "preliminary":
        # Preliminary: must bid exactly the target SLA
        if bid_sla_rank != target_sla_rank:
            return {"status": "rejected", "reason": "SLA must match target"}
    else:
        # Final: must bid >= target SLA
        if bid_sla_rank < target_sla_rank:
            return {"status": "rejected", "reason": "SLA below minimum"}

    # -- Credit check (final mode) --------------------------------
    credit = 1.0
    if COMPETITION_MODE == "final":
        credit_raw = await r.get(f"user:{token}:credit")
        credit = float(credit_raw) if credit_raw else 1.0
        credit_ts = await r.get(f"user:{token}:credit_ts")
        if credit_ts and (now - float(credit_ts) > 600):
            credit = 1.0
            await r.set(f"user:{token}:credit", "1.0")

    # -- Bid and rank ---------------------------------------------
    if first_bid_at == 0:
        try:
            await r.hsetnx(key, "first_bid_at", now)
        except asyncio.TimeoutError:
            pass

    # Fairness boost: losers accumulate boost until they win
    boost_raw = await r.get(f"user:{token}:boost")
    boost = float(boost_raw) if boost_raw else 0.0

    score = (bid_sla_rank + boost)
    if COMPETITION_MODE == "final":
        score = (bid_sla_rank + boost) * credit

    pipe = r.pipeline()
    pipe.zadd(f"{key}:bids", {token: score})
    pipe.zrevrank(f"{key}:bids", token)
    try:
        _, rank = await pipe.execute()
    except asyncio.TimeoutError:
        raise HTTPException(500, "Redis timeout")

    if rank is None or rank >= max_winners:
        # Only boost if lost to a same-score competitor (tie-breaking fairness)
        winner_score_raw = await r.zrevrange(f"{key}:bids", 0, 0, withscores=True)
        if winner_score_raw and abs(winner_score_raw[0][1] - score) < 0.01:
            await r.incrbyfloat(f"user:{token}:boost", 0.5)
        await r.srem(f"user:{token}:queried", task_id)
        return {"status": "rejected"}

    # -- Atomic winner slot acquisition (Lua) ---------------------
    LUA_TRY_ADD_WINNER = """
    local winners_key = KEYS[1]
    local max_w = tonumber(ARGV[1])
    local token = ARGV[2]
    local current = redis.call('SCARD', winners_key)
    if current >= max_w then
        return 0
    end
    redis.call('SADD', winners_key, token)
    return 1
    """
    added = await r.eval(
        LUA_TRY_ADD_WINNER, 1, f"{key}:winners", max_winners, token
    )
    if not added:
        # Lost Lua race — always a same-score tie (we passed rank check above)
        await r.incrbyfloat(f"user:{token}:boost", 0.5)
        await r.srem(f"user:{token}:queried", task_id)
        return {"status": "closed"}

    # Winner: reset boost to 0
    await r.set(f"user:{token}:boost", "0")

    await r.sadd(inflight_key, task_id)

    # Track stats for leaderboard
    await r.incr(f"user:{token}:tasks_accepted")
    await r.set(f"user:{token}:ask_ts:{task_id}", str(now), ex=700)

    # -- Return full task -----------------------------------------
    try:
        raw = await asyncio.wait_for(r.hget(key, "full"), timeout=1)
    except asyncio.TimeoutError:
        raise HTTPException(500, "Redis timeout")

    return {"status": "accepted", "task": json.loads(raw)}


# ---------------------------------------------------------------
# /submit — simplified: +1.0 score per completed task
# ---------------------------------------------------------------
@app.post("/submit")
async def on_submit(user: User, msg: Task):
    """
    Handle user's submission of a task result.
    Simplified for mock server: just award 1.0 score per task.
    """
    token = user.token
    task_id = msg.overview.task_id
    task_key = f"task:{task_id}"

    # Check if the user is allowed to submit (winner)
    winners_set = f"{task_key}:winners"
    is_winner = await r.sismember(winners_set, token)
    if not is_winner:
        raise HTTPException(400, "User is not allowed to submit for this task")

    # Track stats: tasks_completed and latency
    await r.incr(f"user:{token}:tasks_completed")
    ask_ts_raw = await r.get(f"user:{token}:ask_ts:{task_id}")
    if ask_ts_raw:
        latency_ms = (time.time() - float(ask_ts_raw)) * 1000.0
        await r.incrbyfloat(f"user:{token}:latency_sum_ms", latency_ms)
        await r.incr(f"user:{token}:latency_count")
        await r.delete(f"user:{token}:ask_ts:{task_id}")

    # Store the response
    response_key = f"{task_key}:responses"
    await r.hset(response_key, token, json.dumps(msg.model_dump()))

    # Remove from inflight/queried
    inflight_key = f"user:{token}:inflight"
    queried_key = f"user:{token}:queried"
    await r.srem(inflight_key, task_id)
    await r.srem(queried_key, task_id)

    # Award fixed score of 1.0 per completed task
    await r.incrbyfloat(f"user:{token}:score", 1.0)

    # Track correctness (mock: always 1.0 since we give full score)
    await r.incrbyfloat(f"user:{token}:correctness_sum", 1.0)
    await r.incr(f"user:{token}:correctness_count")

    return {"status": "ok"}


# ---------------------------------------------------------------
# /leaderboard
# ---------------------------------------------------------------
@app.get("/leaderboard")
async def leaderboard(top_n: int = 100):
    """Return leaderboard sorted by score with detailed stats."""
    now = time.time()
    cursor = 0
    entries = []
    while True:
        cursor, keys = await r.scan(cursor, match="user:*:score", count=200)
        for key in keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            parts = key_str.split(":")
            if len(parts) >= 3:
                token = parts[1]

                pipe = r.pipeline()
                pipe.get(f"user:{token}:score")
                pipe.get(f"user:{token}:name")
                pipe.get(f"user:{token}:tasks_completed")
                pipe.get(f"user:{token}:tasks_accepted")
                pipe.get(f"user:{token}:correctness_sum")
                pipe.get(f"user:{token}:correctness_count")
                pipe.get(f"user:{token}:latency_sum_ms")
                pipe.get(f"user:{token}:latency_count")
                pipe.get(f"user:{token}:credit")
                vals = await pipe.execute()

                score = float(vals[0]) if vals[0] else 0.0
                name = vals[1].decode() if isinstance(vals[1], bytes) and vals[1] else token
                tasks_completed = int(vals[2]) if vals[2] else 0
                tasks_accepted = int(vals[3]) if vals[3] else 0
                corr_sum = float(vals[4]) if vals[4] else 0.0
                corr_count = int(vals[5]) if vals[5] else 0
                lat_sum = float(vals[6]) if vals[6] else 0.0
                lat_count = int(vals[7]) if vals[7] else 0
                credit = float(vals[8]) if vals[8] else 1.0

                entries.append({
                    "name": name,
                    "score": round(score, 4),
                    "tasks_completed": tasks_completed,
                    "tasks_accepted": tasks_accepted,
                    "avg_correctness": round(corr_sum / corr_count, 4) if corr_count > 0 else 0.0,
                    "avg_latency_ms": round(lat_sum / lat_count, 1) if lat_count > 0 else 0.0,
                    "credit": round(credit, 3),
                })
        if cursor == 0:
            break

    entries.sort(key=lambda x: x["score"], reverse=True)
    for i, e in enumerate(entries):
        e["rank"] = i + 1

    return {
        "timestamp": int(now),
        "round_elapsed_s": round(now - _service_start_time, 1),
        "leaderboard": entries[:top_n],
    }
