#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# PanSR environment setup.
# Creates a ./.venv, installs PyTorch (CUDA 12.4), detectron2 (from source),
# the PanSR package + deps, and compiles the MultiScaleDeformableAttention CUDA op.
#
# Usage:   bash setup.sh
# Requires: a CUDA 12.x toolkit (nvcc), gcc, and a Python 3.10 *with development
#           headers* (Python.h). The venv is created with `uv` if available, else
#           with `conda` (-p ./.venv), else with the stdlib `venv` module.
#
# If detectron2 / the CUDA op fail to build with "fatal error: Python.h: No such
# file or directory", install the Python dev headers (e.g. `sudo apt install
# python3.10-dev`) or point PANSR_PYTHON at an interpreter that has them
# (a conda python ships headers), then re-run.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
# Compute capability for the build (RTX A4500 = Ampere 8.6). Override if your GPU differs.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export FORCE_CUDA="${FORCE_CUDA:-1}"

VENV="$REPO_DIR/.venv"

echo "==> Creating virtual environment at $VENV (Python ${PYTHON_VERSION})"
if [ -n "${PANSR_PYTHON:-}" ]; then
  # Use an explicit interpreter (e.g. a conda python that ships dev headers).
  "$PANSR_PYTHON" -m venv "$VENV"
elif command -v uv >/dev/null 2>&1; then
  uv venv --python "${PYTHON_VERSION}" "$VENV"
elif command -v conda >/dev/null 2>&1; then
  conda create -y -p "$VENV" "python=${PYTHON_VERSION}"
else
  "python${PYTHON_VERSION}" -m venv "$VENV"
fi

PY="$VENV/bin/python"

echo "==> Bootstrapping pip (resolves the PyTorch CUDA index reliably)"
if "$PY" -m pip --version >/dev/null 2>&1; then
  "$PY" -m pip install -U pip
elif command -v uv >/dev/null 2>&1; then
  VIRTUAL_ENV="$VENV" uv pip install pip
else
  "$PY" -m ensurepip --upgrade
fi

echo "==> Preflight: checking for Python development headers (Python.h)"
if ! "$PY" - <<'PYEOF'
import os, sys, sysconfig
hdr = os.path.join(sysconfig.get_path("include"), "Python.h")
sys.exit(0 if os.path.exists(hdr) else 1)
PYEOF
then
  echo "ERROR: Python.h not found for $PY."
  echo "       Install dev headers (e.g. 'sudo apt install python${PYTHON_VERSION}-dev') or set"
  echo "       PANSR_PYTHON to a Python that ships headers (a conda python does), then re-run."
  exit 1
fi

echo "==> Installing PyTorch ${TORCH_VERSION} (+cu124) and torchvision ${TORCHVISION_VERSION}"
"$PY" -m pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "${TORCH_CUDA_INDEX}"

echo "==> Installing build helpers"
"$PY" -m pip install ninja wheel setuptools cython

echo "==> Installing detectron2 (from source, matched to the installed PyTorch)"
"$PY" -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

echo "==> Installing PanSR + Python dependencies"
"$PY" -m pip install -r requirements.txt
"$PY" -m pip install -e .

echo "==> Building the MultiScaleDeformableAttention CUDA op"
( cd pansr/modeling/pixel_decoder/ops && "$PY" setup.py build install )

echo "==> Verifying imports"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"
"$PY" -c "import detectron2; print('detectron2', detectron2.__version__)"
"$PY" -c "import MultiScaleDeformableAttention; print('MSDeformAttn op OK')"
"$PY" -c "import pansr; print('pansr import OK — meta-arch PanSR registered')"

echo ""
echo "Done. Activate with:  source .venv/bin/activate"
echo "Set the dataset root: export LARS_ROOT=/path/to/LaRS/split_v0.9.3"
echo "If your GPUs are hidden (torch.cuda.is_available() == False), set CUDA_VISIBLE_DEVICES."
