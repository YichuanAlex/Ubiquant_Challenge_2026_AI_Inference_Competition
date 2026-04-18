#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "[run] Starting poppycock competition client v2..."
echo "[run] Platform URL: ${PLATFORM_URL:-http://127.0.0.1:8003}"
echo "[run] Model path: ${MODEL_PATH:-Not set}"
echo "[run] Max concurrent tasks: ${MAX_CONCURRENT_TASKS:-32}"

# Activate virtual environment if it exists
if [ -d "/tmp/contestant_env" ]; then
    source /tmp/contestant_env/bin/activate
fi

# Set default max concurrent tasks if not set
export MAX_CONCURRENT_TASKS=${MAX_CONCURRENT_TASKS:-32}

# Run the client
python client.py
