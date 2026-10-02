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
./serve/make-patched-tree.sh                     # vLLM @ ca65eb67d + patches/, py3.12 venv, oneCCL 2022.1.2
QWEN38_MODEL=/path/to/Qwen3.8-27B-FP8 ./serve/run-tp2-mtp.sh
```

`serve/run-tp2-mtp.sh` is the exact production launcher. Every env var and
flag carries a comment explaining why it's load-bearing. Tunables:
`QWEN38_SPEC` (MTP depth, default 3), `QWEN38_MAXLEN`, `QWEN38_UTIL`,
`QWEN38_GRAPHS` (0 = eager), and one env switch per patch.

## XPU graphs under tensor parallelism: use oneCCL 2022.1.2

Without graphs, TP=2 decode on this box is **launch-bound**. Each MTP step
issues ~300 GEMMs and ~140 oneCCL all-reduces from Python, and per-stream speed
barely drops from 1 to 8 concurrent streams. Graphs remove that overhead and
give **+33% on code and +45% on prose** single-stream. They only work if the
TP collectives can be recorded into the graph.

**A from-source vLLM XPU install gets a oneCCL that can't be recorded.**
`torch==2.14.0+xpu` pins `oneccl==2022.1.1`, and with that version:

- Small dense models fail at capture: `onecclAllReduce caught std::exception:
  ext_oneapi_get_graph() can only be called on recording queues`.
- Qwen3.8-27B captures without error and then **silently produces garbage**:
  `itanolleolleolle…`, 0% draft acceptance.

**oneCCL 2022.1.2 records correctly**, with default settings: replays are
correct, at 16.2 µs per in-graph collective. vLLM's official XPU Docker image
already installs 2022.1.2 over torch's pin, so Docker users don't see this.
`serve/make-patched-tree.sh` applies the same override, and the launcher
checks the installed version.

**Fallback (patch 0004)** for stacks stuck on oneCCL 2022.1.1: graph-capturable
collectives built on torch symmetric memory. While a graph is being captured,
`all_reduce`, `all_gather` and `gather` stage the input in an
`XPUSymmetricMemory` buffer, sync both GPUs with its device-side barrier
kernel, then sum or concatenate the peer buffers over P2P (same order on every
rank). It's just as fast end to end (RESULTS). The launcher enables it
automatically if oneCCL is older than 2022.1.2. The all-gather part matters
for MTP: the drafter's FULL prefill graph includes `compute_logits`, whose
vocab-parallel all-gather alone was enough to make every draft fail
(acceptance 1.00, FINDINGS §10).

## Patch series

Against vLLM `ca65eb67d`. Each patch is opt-in via an env var; the launcher
sets the production values.

| # | Env | Fix |
|---|-----|-----|
| 0001 | `VLLM_XPU_ISOLATE_WORKERS=1` | Spawn each TP worker with `ZE_AFFINITY_MASK=<its GPU>` (single-device L0 context, device ordinal 0). Works around the **1:1 host-RAM mirror of every device allocation** in multi-device contexts on compute-runtime 26.09–26.31. Superseded by runtime 26.35; off in production; incompatible with 0004 (no P2P). |
| 0002 | `VLLM_XPU_FP8_BLOCK_TO_CHANNEL=1` | Requantize block-FP8 weights to per-channel FP8 at load and run `fp8_gemm_w8a16`. The default block path (dynamic-activation W8A8 `fp8_gemm`) is **no faster than bf16** on Xe2. W8A16 is ~2× faster at every batch size, with equal accuracy. Prefill +55%, decode +15–20%. |
| 0003 | `VLLM_XPU_LMHEAD_FP8=1` | FP8 (W8A16) LM head: +~2% decode, +0.6 GiB/rank KV. |
| 0004 | `VLLM_XPU_SYMM_AR=1` | **Fallback, only for oneCCL < 2022.1.2:** graph-capturable all-reduce/all-gather/gather via symmetric memory + P2P (above). Off in production; the launcher enables it automatically on an old oneCCL. |

## Also in here

- **[FINDINGS.md](FINDINGS.md)**: every failure mode hit on the way and how
  each was diagnosed, including the falsified approaches:
  - host-RAM mirror, runtime-PM resume failure, oneCCL plugin loading;
  - `expandable_segments` segfaulting oneCCL;
  - a torch.compile cache that crosses device-isolation modes;
  - oneCCL 2022.1.1 vs 2022.1.2 under XPU graphs;
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
| Python stack | **vLLM:** `main` @ `ca65eb67d`.<br>**Kernels and libraries:** torch 2.14.0+xpu, triton-xpu 3.8.0, vllm-xpu-kernels 0.1.15.4, **oneCCL 2022.1.2** (pip, overriding torch's 2022.1.1 pin). |
| Model | `Qwen/Qwen3.8-27B-FP8`: official block-FP8 [128,128]. Hybrid: 48 Gated-DeltaNet + 16 full-attention layers (4 KV heads × 256), 1 MTP layer. 30.9 GB, so it doesn't fit one card and needs TP. |
