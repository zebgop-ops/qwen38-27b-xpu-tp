# Qwen3.8-27B-FP8 on 2× Intel Arc Pro B70: TP=2 + XPU graphs + MTP + 256k context on vLLM

Patches, forensics, benchmarks, and a runnable serving setup for
**Qwen/Qwen3.8-27B-FP8** on vLLM's XPU backend with **tensor parallelism
(TP=2), XPU graphs, MTP speculative decoding, and the full 262,144-token
context, all enabled at once**, with a 16-bit KV cache. Developed and validated on
2× Intel Arc Pro B70 (Battlemage BMG-G31, 32 GB each) sharing one bifurcated
PCIe Gen4 x8/x8 slot, on top of vLLM `main` @ `ca65eb67d` (2026-10-01).

Headline numbers (details in **[RESULTS.md](RESULTS.md)**): **87 tok/s
single-stream on code, 64 tok/s on prose**, 371 tok/s at 8-way, 542 tok/s at
16-way, ~2.2k tok/s prefill, 51 tok/s decode with 216k tokens in context, and
GSM8K 95% (same as the unpatched FP8 path). For comparison, stock vLLM on the
same box gets 18 tok/s; with MTP it gets 48 / 37 tok/s.

## Quickstart

```bash
sudo ./system/install-compute-runtime-26.35.sh   # runtime 26.35 is required (see FINDINGS §1)
sudo cp system/80-intel-gpu-no-runtime-pm.rules /etc/udev/rules.d/ && sudo reboot
./serve/make-patched-tree.sh                     # vLLM @ ca65eb67d + patches/, py3.12 venv via uv
QWEN38_MODEL=/path/to/Qwen3.8-27B-FP8 ./serve/run-tp2-mtp.sh
```

`serve/run-tp2-mtp.sh` is the exact production launcher. Every env var and
flag carries a comment explaining why it's load-bearing. Tunables:
`QWEN38_SPEC` (MTP depth, default 3), `QWEN38_MAXLEN`, `QWEN38_UTIL`,
`QWEN38_GRAPHS` (0 = eager), and one env switch per patch.

## The problem this repo solves: XPU graphs under tensor parallelism

Without graphs, TP=2 decode on this box is **launch-bound**. Each MTP step
issues ~300 GEMMs and ~140 oneCCL all-reduces from Python, and per-stream speed
barely drops from 1 to 8 concurrent streams. Graphs are the obvious fix, but on
XPU they cannot be combined with TP. **oneCCL collectives cannot be recorded
into a SYCL graph**:

- Small dense models fail at capture: `onecclAllReduce caught std::exception:
  ext_oneapi_get_graph() can only be called on recording queues`.
- Qwen3.8-27B captures without error and then **silently produces garbage**:
  `itanolleolleolle…`, 0% draft acceptance.

None of the public B70 setups we found had graphs working with TP either; all
of them run with graphs off.

**The fix (patch 0004):** while a graph is being captured, the XPU communicator's
`all_reduce`, `all_gather` and `gather` bypass oneCCL. Each call:

1. stages the input in a **torch symmetric-memory buffer** (`XPUSymmetricMemory`, Level Zero IPC);
2. synchronizes both GPUs with its **device-side barrier kernel**;
3. sums or concatenates the peers' buffers via **P2P loads**, in the same rank order on every rank, so results are bit-identical;
4. barriers again before the buffer is reused.

Every step is an ordinary kernel, so the whole collective records and replays.
It costs **16 µs per all-reduce on replay vs 46 µs for oneCCL**. Eager calls,
such as large prefills, still go through oneCCL.

Overriding `all_reduce` alone isn't enough. The MTP drafter's FULL prefill
graph includes `compute_logits`, whose vocab-parallel all-gather was still
oneCCL. Main-model output stayed perfect while **every draft was rejected**
(acceptance 1.00). An ngram-speculation control proved target-side
verification was fine (3.95 accepted tokens per step under graphs). Covering
`all_gather`/`gather` restored normal acceptance: eager 65 → **graphs 94
tok/s** (k=5, code).

## Patch series

Against vLLM `ca65eb67d`. Each patch is opt-in via an env var; the launcher
sets the production values.

| # | Env | Fix |
|---|-----|-----|
| 0001 | `VLLM_XPU_ISOLATE_WORKERS=1` | Spawn each TP worker with `ZE_AFFINITY_MASK=<its GPU>` (single-device L0 context, device ordinal 0). Works around the **1:1 host-RAM mirror of every device allocation** in multi-device contexts on compute-runtime 26.09–26.31. Superseded by runtime 26.35; off in production; incompatible with 0004 (no P2P). |
| 0002 | `VLLM_XPU_FP8_BLOCK_TO_CHANNEL=1` | Requantize block-FP8 weights to per-channel FP8 at load and run `fp8_gemm_w8a16`. The default block path (dynamic-activation W8A8 `fp8_gemm`) is **no faster than bf16** on Xe2. W8A16 is ~2× faster at every batch size, with equal accuracy. Prefill +55%, decode +15–20%. |
| 0003 | `VLLM_XPU_LMHEAD_FP8=1` | FP8 (W8A16) LM head: +~2% decode, +0.6 GiB/rank KV. |
| 0004 | `VLLM_XPU_SYMM_AR=1` | Graph-capturable all-reduce/all-gather/gather via symmetric memory + P2P (above). |

## Also in here

- **[FINDINGS.md](FINDINGS.md)**: every failure mode hit on the way and how
  each was diagnosed, including the falsified approaches:
  - host-RAM mirror, runtime-PM resume failure, oneCCL plugin loading;
  - `expandable_segments` segfaulting oneCCL;
  - a torch.compile cache that crosses device-isolation modes;
  - piecewise graphs with eager all-reduce;
  - PP=2 with graphs;
  - the oneCCL tuning sweep.
- **`system/`**: verified compute-runtime 26.35 installer, the runtime-PM udev
  rule, and the CPU-governor unit. `serve/` also has a health watchdog for the
  systemd unit.
- **`tools/`**:
  - `bench.py`: decode, concurrency, prefill and long-context decode, with a nonce per request so prefix caching can't flatter the numbers;
  - `gsm8k_eval.py`: GSM8K accuracy check;
  - `symm_allreduce_graph_test.py`: standalone proof of the capturable all-reduce;
  - `fp8_gemm_bench.py` / `fp8_requant_check.py`: the kernel evidence behind 0002;
  - `ccl_latency.py`: oneCCL all-reduce latency.

## Stack

| | |
|---|---|
| Hardware | **GPUs:** 2× Arc Pro B70 32 GB (BMG-G31, `xe` driver, SR-IOV PF), ReBAR on.<br>**Board:** B550I AORUS PRO AX, single x16 slot bifurcated x8/x8 Gen4 on CPU lanes; P2P 14.3 GB/s.<br>**CPU / RAM:** Ryzen 9 3950X, 30 GiB. |
| OS | Ubuntu 26.04.1, kernel 7.0.0-38 |
| GPU runtime | compute-runtime **26.35.39758.10** + IGC 2.41.5 (GitHub release, over `ppa:kobuk-team/intel-graphics` 26.31); Level Zero 1.32 |
| Python stack | **vLLM:** `main` @ `ca65eb67d`.<br>**Kernels and libraries:** torch 2.14.0+xpu, triton-xpu 3.8.0, vllm-xpu-kernels 0.1.15.4, oneCCL 2022.1.1 (pip). |
| Model | `Qwen/Qwen3.8-27B-FP8`: official block-FP8 [128,128]. Hybrid: 48 Gated-DeltaNet + 16 full-attention layers (4 KV heads × 256), 1 MTP layer. 30.9 GB, so it doesn't fit one card and needs TP. |
