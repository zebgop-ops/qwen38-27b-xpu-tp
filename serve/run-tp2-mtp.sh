#!/bin/bash
# Production launcher: Qwen3.8-27B-FP8 on 2x Intel Arc Pro B70, TP=2, XPU graphs, MTP, 256k context,
# 16-bit KV cache. Needs the patched tree from make-patched-tree.sh and compute-runtime >= 26.35.
# Every knob is env-overridable; defaults are the measured optimum (see RESULTS.md).
set -e
QWEN38_VENV=${QWEN38_VENV:-$HOME/venv-vllm-xpu}
QWEN38_MODEL=${QWEN38_MODEL:-$HOME/models/Qwen3.8-27B-FP8}
QWEN38_SPEC=${QWEN38_SPEC:-3}          # MTP depth. 3 = best prose + concurrency under graphs; 6 = best code (+8%)
QWEN38_MAXLEN=${QWEN38_MAXLEN:-262144}
QWEN38_UTIL=${QWEN38_UTIL:-0.93}       # graphs need ~0.9 GiB/rank headroom; 0.93 still fits a full 262k sequence
QWEN38_PORT=${QWEN38_PORT:-8000}
source "$QWEN38_VENV/bin/activate"

# oneCCL (pip wheel) dlopen()s its plugins by bare name: without this, TP init fails with
# "oneCCL could not be initialized! Could not load any plugin".
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Patch 0002: block-FP8 -> per-channel FP8 + fp8_gemm_w8a16. ~2x faster GEMMs on Xe2, equal accuracy.
export VLLM_XPU_FP8_BLOCK_TO_CHANNEL=${VLLM_XPU_FP8_BLOCK_TO_CHANNEL:-1}
# Patch 0003: FP8 LM head. ~+2% decode, +0.6 GiB/rank KV. Set 0 to keep the checkpoint's bf16 lm_head.
export VLLM_XPU_LMHEAD_FP8=${VLLM_XPU_LMHEAD_FP8:-1}
# Patch 0004: graph-capturable all-reduce/all-gather (symmetric memory + P2P). This is what makes
# XPU graphs work under TP. Disable together with graphs: VLLM_XPU_SYMM_AR=0 QWEN38_GRAPHS=0.
export VLLM_XPU_SYMM_AR=${VLLM_XPU_SYMM_AR:-1}
# Patch 0001 (VLLM_XPU_ISOLATE_WORKERS=1 + CCL_TOPO_P2P_ACCESS=0) is only for compute-runtime < 26.35
# and is incompatible with 0004 (no P2P between isolated workers). Leave it off.

if [ "${QWEN38_GRAPHS:-1}" = 1 ]; then
  # Capture up to 64 tokens: 128/256 cost 2.0/5.7 GiB per rank and no longer leave room for 262k of KV.
  COMPILE_CFG='{"max_cudagraph_capture_size":64}'
else
  COMPILE_CFG='{"cudagraph_mode":"NONE"}'
fi

exec vllm serve "$QWEN38_MODEL" \
  --served-model-name qwen38 \
  --tensor-parallel-size 2 \
  --max-model-len "$QWEN38_MAXLEN" \
  --gpu-memory-utilization "$QWEN38_UTIL" \
  --host 0.0.0.0 --port "$QWEN38_PORT" \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$QWEN38_SPEC}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --compilation-config "$COMPILE_CFG" \
  "$@"
