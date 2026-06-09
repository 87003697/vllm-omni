#!/bin/bash
# ============================================================================
# KOALA 环境恢复脚本 — vLLM-Omni serving
# ============================================================================
# 用法：
#   . scripts/setup_koala.sh [--model Qwen/Qwen-Image-Edit-2511]
#
# 在 debug pod 中 source 执行，初始化环境后手动启动 server。
#
# 注意：
# - Koala 默认镜像不含 vllm，脚本会自动安装 vllm + 匹配的 torch
# - vllm 0.22.0 PyPI wheel 链接 CUDA 13，需设 LD_LIBRARY_PATH 指向 pip 内置 cu13 lib
# - 不能用 `uv run`（会触发 uv sync 覆盖手动装的 torch），直接用 python -m
# ============================================================================
set -euo pipefail

# --- 参数解析 ---
MODEL="${MODEL:-Qwen/Qwen-Image-Edit-2511}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            if [[ $# -lt 2 ]]; then echo "ERROR: --model requires a value"; exit 1; fi
            MODEL="$2"; shift 2 ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

# --- 环境变量 ---
export HF_HOME="/local-ssd/hf_cache"
export UV_PROJECT_ENVIRONMENT="/tmp/uv-venv-omni"
export HF_HUB_DISABLE_XET=1
export SETUPTOOLS_SCM_PRETEND_VERSION=0.22.0

# --- 路径配置 ---
MODEL_SHORT=$(echo "${MODEL}" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]')
S3_PREFIX_URI="s3://arcwm-code-us-west-2/${KOALA_USER:-$USER}"
HF_CACHE_TAR_URI="${S3_PREFIX_URI}/tools/hf_cache_${MODEL_SHORT}.tar"
PROJECT_DIR="/data/work/vllm-omni"
VENV="/tmp/uv-venv-omni"

# ============================================================================
# 函数定义
# ============================================================================

setup_hf_cache() {
    echo "  HF model cache (${MODEL})..."
    if [ -d "${HF_HOME}/hub" ]; then
        echo "    Already present, skipping"
        return
    fi
    if s5cmd ls "${HF_CACHE_TAR_URI}" &>/dev/null; then
        s5cmd cat "${HF_CACHE_TAR_URI}" | tar xf - -C /local-ssd
        echo "    Restored from ${HF_CACHE_TAR_URI}"
    else
        echo "    No tar at ${HF_CACHE_TAR_URI}, will download on first use"
    fi
}

setup_python_deps() {
    echo "  Python dependencies (vllm-omni + vllm)..."

    # Step 1: Install vllm-omni and its deps (vllm itself not declared as dep)
    uv sync --all-extras

    # Step 2: Install vllm (base engine) — PyPI wheel is built against CUDA 13
    uv pip install --python "${VENV}/bin/python" "vllm==0.22.0"

    # Step 3: Verify torch CUDA works
    if ! "${VENV}/bin/python" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "    WARNING: torch CUDA not available after vllm install"
    fi

    echo "    Done."
}

setup_ld_library_path() {
    # vllm 0.22.0 wheel links against libcudart.so.13 (CUDA 13).
    # Koala pods have CUDA 12.8 system-wide, but pip installs nvidia-cu13 package
    # with the needed .so files.
    local cu13_lib="${VENV}/lib/python3.12/site-packages/nvidia/cu13/lib"
    if [ -d "${cu13_lib}" ]; then
        export LD_LIBRARY_PATH="${cu13_lib}:${LD_LIBRARY_PATH:-}"
        echo "    LD_LIBRARY_PATH includes ${cu13_lib}"
    else
        echo "    WARNING: nvidia/cu13/lib not found, vllm may fail to load"
    fi
}

# ============================================================================
# 主流程
# ============================================================================
cd "${PROJECT_DIR}"

echo "=== [1/3] Python dependencies ==="
setup_python_deps

echo "=== [2/3] CUDA library path ==="
setup_ld_library_path

echo "=== [3/3] HF model cache ==="
setup_hf_cache

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Start server (standard Qwen-Image-Edit):"
echo "  ${VENV}/bin/python -m vllm_omni.entrypoints.cli.main serve ${MODEL} --omni --port 8092 --host 0.0.0.0"
echo ""
echo "Start server (FlowEdit pipeline):"
echo "  ${VENV}/bin/python -m vllm_omni.entrypoints.cli.main serve ${MODEL} --omni --port 8092 --host 0.0.0.0 --model-class-name QwenImageFlowEditPipeline"
echo ""
echo "Local port forwarding (Mac):"
echo "  ssh -L 8092:localhost:8092 <pod-name>"
echo ""
echo "Test standard edit:"
echo "  bash examples/online_serving/image_to_image/run_curl_image_edit.sh input.png \"your prompt\""
echo ""
echo "Test FlowEdit (2 images: source + condition):"
echo "  bash examples/online_serving/image_to_image/run_curl_flowedit.sh source.png condition.png \"your prompt\""
