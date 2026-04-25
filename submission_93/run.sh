#!/bin/bash
# poppycock — competition entrypoint.
# Stability-first: per-task and per-message hard timeouts, bounded buffering.

set -u  # do not 'set -e' — we want the python process to handle its own retries

cd "$(dirname "$0")"

echo "[run] poppycock client starting"
echo "[run] PLATFORM_URL = ${PLATFORM_URL:-http://127.0.0.1:8003}"
echo "[run] MODEL_PATH   = ${MODEL_PATH:-<unset>}"
echo "[run] CONFIG_PATH  = ${CONFIG_PATH:-<unset>}"

# Activate the venv created by setup.sh, if it exists. If not we fall through
# to the system python — the contest machine may already have everything.
if [ -d "/tmp/contestant_env" ] && [ -f "/tmp/contestant_env/bin/activate" ]; then
    # shellcheck disable=SC1091
    source /tmp/contestant_env/bin/activate
    echo "[run] venv activated: $(which python)"
fi

# Conservative defaults (override via env). The competition is a single-shot
# offline eval, so we prioritise stability over peak throughput.
export MAX_CONCURRENT_TASKS=${MAX_CONCURRENT_TASKS:-4}
export MAX_BUFFERED_TASKS=${MAX_BUFFERED_TASKS:-8}
export TASK_TIMEOUT_S=${TASK_TIMEOUT_S:-600}
export MSG_TIMEOUT_S=${MSG_TIMEOUT_S:-60}
export ABORT_TIMEOUT_S=${ABORT_TIMEOUT_S:-3}
export POLL_INTERVAL=${POLL_INTERVAL:-0.15}
export IDLE_SLEEP=${IDLE_SLEEP:-0.40}
export HTTP_TIMEOUT=${HTTP_TIMEOUT:-15}
export HEARTBEAT_INTERVAL_S=${HEARTBEAT_INTERVAL_S:-30}

echo "[run] MAX_CONCURRENT_TASKS=$MAX_CONCURRENT_TASKS  MAX_BUFFERED_TASKS=$MAX_BUFFERED_TASKS"
echo "[run] TASK_TIMEOUT_S=$TASK_TIMEOUT_S  MSG_TIMEOUT_S=$MSG_TIMEOUT_S"

# Pick a python — prefer the venv one, then python3, then python.
PY="python"
if command -v python3 >/dev/null 2>&1; then
    PY="python3"
fi
if [ -x "/tmp/contestant_env/bin/python" ]; then
    PY="/tmp/contestant_env/bin/python"
fi

exec "$PY" client.py
