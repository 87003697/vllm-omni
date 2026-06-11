#!/bin/bash
# ============================================================================
# TRELLIS2 渲染环境恢复脚本（精简版）
# ============================================================================
# 用法：. scripts/eval/setup_trellis2_env.sh
#
# 从 S3 tar 恢复 TRELLIS2 推理+渲染所需的全部环境。
# 复用 Flow-Factory 已打好的 tar 资产，无需重新编译。
#
# 恢复后可执行：
#   /tmp/uv-venv/bin/python scripts/eval/prepare_trellis2_renders.py --n-samples 5
# ============================================================================
set -euo pipefail

# --- 路径配置 ---
USER="${KOALA_USER:-ericzyma}"
S3_DATA="s3://arcwm-code-us-west-2/${USER}/data/flow_grpo"
PROJECT_DIR="/data/work/vllm-omni"
VENV="/tmp/uv-venv"
WEIGHTS_LOCAL="/local-ssd/pretrained_weights"
DATASET_LOCAL="/local-ssd/alphaimages_v2_formatted"

cd "${PROJECT_DIR}"

# --- 环境变量 ---
export PATH="${VENV}/bin:${PATH}"
export UV_PROJECT_ENVIRONMENT="${VENV}"
export HF_HOME="/local-ssd/hf_cache"
export HF_HUB_DISABLE_XET=1
export TORCH_CUDA_ARCH_LIST="9.0"
export ATTN_BACKEND=flash_attn
export PYTHONPATH="${PROJECT_DIR}/third_party/TRELLIS.2:${PYTHONPATH:-}"

# ============================================================================
# [1/6] Python venv + PyTorch 2.6.0+cu124
# ============================================================================
echo "=== [1/6] Python venv + PyTorch ==="
if [ ! -d "${VENV}" ]; then
    uv venv --python 3.12 "${VENV}"
fi

TORCH_VER=$("${VENV}/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
if [[ "${TORCH_VER}" != 2.6.0* ]]; then
    echo "  Installing torch 2.6.0+cu124..."
    uv pip install --python "${VENV}/bin/python" \
        torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
        --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3
else
    echo "  torch 2.6.0+cu124 already installed"
fi

# ============================================================================
# [2/6] Python 依赖
# ============================================================================
echo "=== [2/6] Python dependencies ==="
if "${VENV}/bin/python" -c "import trimesh, utils3d, imageio" 2>/dev/null; then
    echo "  Already installed"
else
    echo "  Installing..."
    uv pip install --python "${VENV}/bin/python" \
        pillow imageio imageio-ffmpeg tqdm easydict \
        opencv-python-headless scipy ninja trimesh plyfile \
        rembg onnxruntime transformers open3d kiui \
        kornia timm zstandard spconv-cu120 \
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8" \
        2>&1 | tail -5
fi

if ! "${VENV}/bin/python" -c "import flash_attn" 2>/dev/null; then
    echo "  Installing flash-attn..."
    uv pip install --python "${VENV}/bin/python" wheel setuptools 2>/dev/null
    uv pip install --python "${VENV}/bin/python" \
        --no-build-isolation flash-attn==2.7.3 2>&1 | tail -3
else
    echo "  flash-attn already installed"
fi

# ============================================================================
# [3/6] CUDA 扩展（预编译 site-packages）
# ============================================================================
echo "=== [3/6] CUDA extensions ==="
CUDA_SP_TAR="${S3_DATA}/cuda_site_packages.tar"
if "${VENV}/bin/python" -c "import nvdiffrast" 2>/dev/null; then
    echo "  Already installed"
elif s5cmd ls "${CUDA_SP_TAR}" &>/dev/null; then
    echo "  Restoring from S3 tar..."
    s5cmd cat "${CUDA_SP_TAR}" | tar xf - -C "${VENV}/lib/python3.12/site-packages/"
    echo "  Done"
else
    echo "  ERROR: No CUDA ext tar at ${CUDA_SP_TAR}"
    return 1 2>/dev/null || exit 1
fi

# ============================================================================
# [4/6] TRELLIS.2 源码
# ============================================================================
echo "=== [4/6] TRELLIS.2 source ==="
REFERENCE_TAR="${S3_DATA}/trellis2_reference.tar"
if [ -d "${PROJECT_DIR}/third_party/TRELLIS.2/trellis2" ]; then
    echo "  Already present"
elif s5cmd ls "${REFERENCE_TAR}" &>/dev/null; then
    echo "  Restoring from S3 tar..."
    mkdir -p "${PROJECT_DIR}/third_party"
    s5cmd cat "${REFERENCE_TAR}" | tar xf - -C "${PROJECT_DIR}/third_party/"
    echo "  Done"
else
    echo "  ERROR: No tar at ${REFERENCE_TAR}"
    return 1 2>/dev/null || exit 1
fi

# o-voxel (体素化工具)
if ! "${VENV}/bin/python" -c "import o_voxel" 2>/dev/null; then
    OVOXEL_DIR="${PROJECT_DIR}/third_party/TRELLIS.2/o-voxel"
    if [ -d "${OVOXEL_DIR}" ]; then
        echo "  Installing o-voxel..."
        uv pip install --python "${VENV}/bin/python" \
            "${OVOXEL_DIR}" --no-build-isolation --no-deps 2>&1 | tail -1
    fi
fi

# ============================================================================
# [5/6] 预训练权重 + 数据集
# ============================================================================
echo "=== [5/6] Weights & dataset ==="
mkdir -p "${WEIGHTS_LOCAL}"

# TRELLIS.2-4B
TRELLIS2_TAR="${S3_DATA}/TRELLIS.2-4B.tar"
if [ -d "${WEIGHTS_LOCAL}/TRELLIS.2-4B" ]; then
    echo "  TRELLIS.2-4B: present"
elif s5cmd ls "${TRELLIS2_TAR}" &>/dev/null; then
    echo "  TRELLIS.2-4B: restoring (~30s)..."
    s5cmd cat "${TRELLIS2_TAR}" | tar xf - -C "${WEIGHTS_LOCAL}/"
    echo "  Done"
else
    echo "  WARNING: No TRELLIS.2-4B tar"
fi

# TRELLIS-image-large (ss_dec shared component)
TRELLIS1_TAR="${S3_DATA}/TRELLIS-image-large.tar"
if [ -d "${WEIGHTS_LOCAL}/TRELLIS-image-large" ]; then
    echo "  TRELLIS-image-large: present"
elif s5cmd ls "${TRELLIS1_TAR}" &>/dev/null; then
    echo "  TRELLIS-image-large: restoring..."
    s5cmd cat "${TRELLIS1_TAR}" | tar xf - -C "${WEIGHTS_LOCAL}/"
    echo "  Done"
fi

# DINOv3
DINOV3_TAR="${S3_DATA}/dinov3-vitl16.tar"
if [ -d "${WEIGHTS_LOCAL}/dinov3-vitl16-pretrain-lvd1689m" ]; then
    echo "  DINOv3: present"
elif s5cmd ls "${DINOV3_TAR}" &>/dev/null; then
    echo "  DINOv3: restoring..."
    s5cmd cat "${DINOV3_TAR}" | tar xf - -C "${WEIGHTS_LOCAL}/"
    echo "  Done"
fi

# Symlink: pipeline.json references microsoft/TRELLIS-image-large
mkdir -p "${WEIGHTS_LOCAL}/TRELLIS.2-4B/microsoft"
ln -sfn "${WEIGHTS_LOCAL}/TRELLIS-image-large" "${WEIGHTS_LOCAL}/TRELLIS.2-4B/microsoft/TRELLIS-image-large"

# Dataset
DATASET_TAR="${S3_DATA}/alphaimages_v2.tar"
if [ -d "${DATASET_LOCAL}/images" ]; then
    echo "  Dataset: present"
elif s5cmd ls "${DATASET_TAR}" &>/dev/null; then
    echo "  Dataset: restoring..."
    s5cmd cat "${DATASET_TAR}" | tar xf - -C /local-ssd/
    echo "  Done"
fi

# ============================================================================
# [6/6] 验证
# ============================================================================
echo "=== [6/6] Verification ==="
"${VENV}/bin/python" -c "
import torch
print(f'  torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import trellis2
print(f'  trellis2: OK')
import nvdiffrast
print(f'  nvdiffrast: OK')
" || echo "  WARNING: verification failed"

echo ""
echo "=== TRELLIS2 env ready ==="
echo "VENV: ${VENV}"
echo "Weights: ${WEIGHTS_LOCAL}"
echo "Dataset: ${DATASET_LOCAL}/images/"
echo ""
echo "Next: ${VENV}/bin/python scripts/eval/prepare_trellis2_renders.py --n-samples 5"
