#!/bin/bash
# poppycock — environment bootstrap.
# Robust to: missing python3.12, no internet, vLLM already installed.

set -u

echo "[setup] start"

# ---------------------------------------------------------------------------
# Pick a Python interpreter (>= 3.10). Prefer 3.12 if available.
# ---------------------------------------------------------------------------
PY=""
for cand in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        ver=$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "0.0")
        major=${ver%%.*}
        minor=${ver##*.}
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PY="$cand"
            echo "[setup] selected interpreter: $cand ($ver)"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "[setup] FATAL: no python >= 3.10 found"
    exit 1
fi

# ---------------------------------------------------------------------------
# Virtual environment (best-effort; if venv module is missing, fall through).
# ---------------------------------------------------------------------------
VENV_DIR="/tmp/contestant_env"
if [ ! -d "$VENV_DIR" ]; then
    if "$PY" -m venv "$VENV_DIR" 2>/dev/null; then
        echo "[setup] venv created at $VENV_DIR"
    else
        echo "[setup] WARN: venv creation failed; will install into system python"
    fi
fi

if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PIP="$VENV_DIR/bin/pip"
else
    PIP="$PY -m pip"
fi

# ---------------------------------------------------------------------------
# Install core dependencies. We try once; if pip fails (e.g. offline),
# we still exit 0 so run.sh has a chance to use whatever is already on the
# machine. The platform's eval will surface the real failure if any.
# ---------------------------------------------------------------------------
$PIP install --upgrade pip 2>/dev/null || echo "[setup] WARN: pip self-upgrade failed"
$PIP install -r requirements.txt || echo "[setup] WARN: requirements.txt install had errors"

# vLLM is only needed when MODEL_PATH points at a real directory.
# We try to install it but never fail setup if it cannot be installed —
# client.py falls back to a mock inference path so the process still runs.
if [ -n "${MODEL_PATH:-}" ] && [ -d "${MODEL_PATH}" ]; then
    echo "[setup] MODEL_PATH=$MODEL_PATH detected, installing vLLM"
    if ! $PIP show vllm >/dev/null 2>&1; then
        $PIP install "vllm>=0.7.0,<0.10" || echo "[setup] WARN: vllm install failed"
    else
        echo "[setup] vllm already installed: $($PIP show vllm | grep -i ^version)"
    fi
    if ! $PIP show transformers >/dev/null 2>&1; then
        $PIP install "transformers>=4.40.0" || echo "[setup] WARN: transformers install failed"
    fi
else
    echo "[setup] MODEL_PATH not set or invalid; skipping vLLM"
fi

echo "[setup] done"
exit 0
