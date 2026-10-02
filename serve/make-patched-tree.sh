#!/bin/bash
# Builds the patched vLLM (XPU) tree + venv this repo was validated on:
# vLLM main @ ca65eb67d (2026-10-01) + patches/000*.patch, Python 3.12, uv, requirements/xpu.txt
# (torch 2.14.0+xpu, triton-xpu 3.8.0, vllm-xpu-kernels 0.1.15.4). No C++ build: kernels ship as a wheel.
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
VLLM_DIR=${VLLM_DIR:-$HOME/vllm}
QWEN38_VENV=${QWEN38_VENV:-$HOME/venv-vllm-xpu}
VLLM_COMMIT=${VLLM_COMMIT:-ca65eb67d995f5401c575ec90cba523757127ec7}
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
[ -d "$VLLM_DIR/.git" ] || git clone https://github.com/vllm-project/vllm.git "$VLLM_DIR"
cd "$VLLM_DIR"
git fetch -q origin && git checkout -q "$VLLM_COMMIT"
for p in "$REPO"/patches/000*.patch; do git apply "$p" && echo "applied $(basename "$p")"; done
uv venv -q --python 3.12 "$QWEN38_VENV"
source "$QWEN38_VENV/bin/activate"
uv pip install -q -r requirements/xpu.txt --index-strategy unsafe-best-match
VLLM_TARGET_DEVICE=xpu uv pip install -q --no-build-isolation -e .
# torch 2.14.0+xpu pins oneccl==2022.1.1, which cannot be recorded into XPU graphs (TP + graphs =
# silent garbage). Override to 2022.1.2 like vLLM's docker/Dockerfile.xpu does (FINDINGS §8).
# Must run after the vLLM install, which re-resolves torch's pin.
uv pip install -q "oneccl==2022.1.2" "oneccl-devel==2022.1.2"
python -c "import torch, vllm, importlib.metadata as m; print('vllm', vllm.__version__, 'torch', torch.__version__, 'oneccl', m.version('oneccl'), 'xpus', torch.xpu.device_count())"
