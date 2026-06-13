#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PanSR environment setup.
#
# This installs everything into the *currently active* Python environment:
# PyTorch (CUDA 12.4), detectron2 (from source), the PanSR package + deps, and
# the MultiScaleDeformableAttention CUDA op. Create and activate an environment
# first, e.g.:
#
#     conda create -n pansr python=3.10 -y && conda activate pansr
#
# then run:  bash setup.sh
#
# Requires: a CUDA 12.x toolkit (nvcc) and gcc on PATH, and a Python 3.10 *with
# development headers* (Python.h) — a conda Python ships these. The exact
# commands this script runs are also listed under "Manual installation" in the
# README, if you'd rather not run a script.
#
# Configurable via env vars: TORCH_VERSION, TORCHVISION_VERSION,
# TORCH_CUDA_INDEX, TORCH_CUDA_ARCH_LIST.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
export FORCE_CUDA="${FORCE_CUDA:-1}"
# GPU compute capability for the CUDA builds (detectron2 + the MSDeformAttn op).
# If you set TORCH_CUDA_ARCH_LIST it is honored; otherwise it's left unset and
# PyTorch auto-detects it from the visible GPU at build time.
if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  export TORCH_CUDA_ARCH_LIST
fi

# --- Sanity checks --------------------------------------------------------
if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
  echo "ERROR: no active Python environment detected."
  echo "       Create and activate one first, e.g.:"
  echo "         conda create -n pansr python=3.10 -y && conda activate pansr"
  exit 1
fi

PY="$(command -v python)"
echo "==> Installing into: $PY"

echo "==> Checking for Python development headers (Python.h)"
if ! "$PY" - <<'PYEOF'
import os, sys, sysconfig
sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_path("include"), "Python.h")) else 1)
PYEOF
then
  echo "ERROR: Python.h not found — detectron2 and the CUDA op need it to compile."
  echo "       Use a conda Python (ships headers) or 'sudo apt install python3.10-dev'."
  exit 1
fi

# --- Install --------------------------------------------------------------
"$PY" -m pip install -U pip

echo "==> Installing PyTorch ${TORCH_VERSION} (+cu124) and torchvision ${TORCHVISION_VERSION}"
"$PY" -m pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "${TORCH_CUDA_INDEX}"

echo "==> Installing build helpers"
"$PY" -m pip install ninja wheel setuptools cython

# Resolve the CUDA compute capability for the from-source builds below. If the
# user pinned TORCH_CUDA_ARCH_LIST we keep it; otherwise detect it from the
# visible GPU. (Leaving it unset makes PyTorch crash with a cryptic
# "IndexError: list index out of range" when no GPU is visible at build time.)
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  echo "==> Detecting GPU compute capability"
  DETECTED_ARCH="$("$PY" - <<'PYEOF'
import torch
caps = sorted({"%d.%d" % torch.cuda.get_device_capability(i)
               for i in range(torch.cuda.device_count())}) if torch.cuda.is_available() else []
print(";".join(caps))
PYEOF
)"
  if [ -n "$DETECTED_ARCH" ]; then
    export TORCH_CUDA_ARCH_LIST="$DETECTED_ARCH"
    echo "    detected TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
  else
    echo "ERROR: no visible CUDA GPU to auto-detect compute capability from."
    echo "       Make a GPU visible (e.g. export CUDA_VISIBLE_DEVICES=0) or set the"
    echo "       target arch explicitly (e.g. export TORCH_CUDA_ARCH_LIST=8.6 for Ampere),"
    echo "       then re-run."
    exit 1
  fi
fi

echo "==> Installing detectron2 (from source, matched to the installed PyTorch)"
"$PY" -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

echo "==> Installing PanSR + Python dependencies"
"$PY" -m pip install -r requirements.txt
"$PY" -m pip install -e .

echo "==> Building the MultiScaleDeformableAttention CUDA op (arch $TORCH_CUDA_ARCH_LIST)"
( cd pansr/modeling/pixel_decoder/ops && "$PY" setup.py build install )

# --- Verify ---------------------------------------------------------------
echo "==> Verifying imports"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
"$PY" -c "import detectron2; print('detectron2', detectron2.__version__)"
"$PY" -c "import MultiScaleDeformableAttention; print('MSDeformAttn op OK')"
"$PY" -c "import pansr; print('pansr import OK — meta-arch PanSR registered')"

echo ""
echo "Done. Set the dataset root before running:  export LARS_ROOT=/path/to/LaRS/split_v0.9.3"
echo "If torch.cuda.is_available() is False despite GPUs present, set CUDA_VISIBLE_DEVICES."
