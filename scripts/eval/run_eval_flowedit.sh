#!/bin/bash
# ============================================================================
# One-click FlowEdit 3D Guidance Evaluation
# ============================================================================
# Prerequisites:
#   - TRELLIS2 renders already exist at /local-ssd/eval_flowedit/ (or S3)
#   - Run setup_trellis2_env.sh + prepare_trellis2_renders.py first if needed
#
# Usage:
#   bash scripts/eval/run_eval_flowedit.sh
# ============================================================================
set -euo pipefail

PROJECT_DIR="/data/work/vllm-omni"
VENV_OMNI="/tmp/uv-venv-omni"
RENDERS_DIR="/local-ssd/eval_flowedit"
PORT=8092
SERVER="http://localhost:${PORT}"

cd "${PROJECT_DIR}"

# ============================================================================
# [1/4] Setup vllm-omni environment
# ============================================================================
echo "=== [1/4] Setting up vllm-omni environment ==="
if ! "${VENV_OMNI}/bin/python" -c "import vllm" 2>/dev/null; then
    . scripts/setup_koala.sh
else
    echo "  Already set up"
    # Ensure env vars are set
    export HF_HOME="/local-ssd/hf_cache"
    export HF_HUB_DISABLE_XET=1
    export SETUPTOOLS_SCM_PRETEND_VERSION=0.22.0
    export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

    # LD_LIBRARY_PATH for CUDA 13
    CU13_LIB="${VENV_OMNI}/lib/python3.12/site-packages/nvidia/cu13/lib"
    if [ -d "${CU13_LIB}" ]; then
        export LD_LIBRARY_PATH="${CU13_LIB}:${LD_LIBRARY_PATH:-}"
    fi
fi

# ============================================================================
# [2/4] Start FlowEdit server
# ============================================================================
echo "=== [2/4] Starting FlowEdit server ==="
if curl -s "${SERVER}/health" >/dev/null 2>&1; then
    echo "  Server already running"
else
    echo "  Launching server on port ${PORT}..."
    "${VENV_OMNI}/bin/python" -m vllm_omni.entrypoints.cli.main serve \
        Qwen/Qwen-Image-Edit-2511 \
        --omni --port ${PORT} --host 0.0.0.0 \
        --model-class-name QwenImageFlowEditPipeline \
        > /tmp/flowedit_server.log 2>&1 &
    SERVER_PID=$!
    echo "  PID: ${SERVER_PID}"

    # Wait for server ready
    echo "  Waiting for server to be ready..."
    for i in $(seq 1 60); do
        if curl -s "${SERVER}/health" >/dev/null 2>&1; then
            echo "  Server ready (${i}s)"
            break
        fi
        if ! kill -0 ${SERVER_PID} 2>/dev/null; then
            echo "  ERROR: Server died. Check /tmp/flowedit_server.log"
            tail -20 /tmp/flowedit_server.log
            exit 1
        fi
        sleep 5
    done

    if ! curl -s "${SERVER}/health" >/dev/null 2>&1; then
        echo "  ERROR: Server not ready after 300s"
        exit 1
    fi
fi

# ============================================================================
# [3/4] Run evaluation
# ============================================================================
echo "=== [3/4] Running FlowEdit evaluation ==="
"${VENV_OMNI}/bin/python" scripts/eval/eval_flowedit_guidance.py \
    --renders "${RENDERS_DIR}" \
    --server "${SERVER}"

# ============================================================================
# [4/4] Cleanup
# ============================================================================
echo "=== [4/4] Done ==="
if [ -n "${SERVER_PID:-}" ]; then
    echo "  Stopping server (PID: ${SERVER_PID})..."
    kill ${SERVER_PID} 2>/dev/null || true
    wait ${SERVER_PID} 2>/dev/null || true
fi

echo ""
echo "Results: ${RENDERS_DIR}/results/"
echo "  - results.json (summary + per-sample metrics)"
echo "  - results.csv (tabular data)"
echo "  - grid.png (visual comparison)"
