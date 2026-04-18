#!/bin/bash
set -e

echo "[setup] Installing dependencies..."

# Create virtual environment if not exists
if [ ! -d "/tmp/contestant_env" ]; then
    python3.12 -m venv /tmp/contestant_env
fi

source /tmp/contestant_env/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install httpx>=0.27.0
pip install pydantic>=2.0

# Install vLLM only if MODEL_PATH is set (GPU environment)
if [ -n "$MODEL_PATH" ] && [ -d "$MODEL_PATH" ]; then
    echo "[setup] Installing vLLM for GPU inference..."
    pip install vllm>=0.8.5,<0.10
    pip install transformers>=4.40.0
else
    echo "[setup] MODEL_PATH not set, skipping vLLM installation"
fi

echo "[setup] Dependencies installed successfully"
