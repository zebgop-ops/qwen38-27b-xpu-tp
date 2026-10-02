# Results

Model: **Qwen/Qwen3.8-27B-FP8**, the official block-FP8 checkpoint, at
`max_model_len=262144` with a **16-bit KV cache** (`kv_cache_dtype=auto`).

Hardware: 2× Intel Arc Pro B70 (32 GB) on one bifurcated PCIe Gen4 x8/x8 slot,
Ryzen 9 3950X, 30 GiB RAM, Ubuntu 26.04 / kernel 7.0.

Production config: TP=2, MTP `num_speculative_tokens=3`, patches 0002+0003+0004,
XPU graphs (`FULL_AND_PIECEWISE`, `max_cudagraph_capture_size=64`),
`gpu_memory_utilization=0.93`, compute-runtime 26.35.

All numbers come from `tools/bench.py`:
- temperature 0, thinking off, 512 generated tokens with `ignore_eos`;
- a random nonce system message on every request, so prefix caching can't
  short-circuit prefill;
- prefill prompts are real Python source;
- decode tok/s = (completion_tokens − 1) / (t_last − t_first).

Run-to-run noise is ±5% on single-stream decode, because MTP acceptance varies
with tiny numeric differences.

## Production (graphs + symm collectives, MTP k=3)

| | |
|---|---|
| Decode C=1, code / prose | **87.0 / 64.2 tok/s** |
| Decode C=2 / 4 / 8 / 16 (aggregate) | 124 / 221 / **371** / **542 tok/s** |
| Decode with 27k / 110k / 216k tokens in context | 74.7 / 64.7 / **50.7 tok/s** |
| Prefill 906 / 7,107 / 27,100 tok | 2,158 / 2,240 / 2,098 tok/s |
| Prefill 110,236 / 216,161 tok | 1,638 / 1,277 tok/s (TTFT 67 s / 169 s) |
| KV pool | 289,863 tokens (1.09× a full 262k sequence) |
| MTP mean acceptance length | 2.93 (k=3) |
| GSM8K (first 100, greedy, thinking off) | **95%** |
| Peak host RAM during the 216k run | 17.2 GB used, 13.7 GB available |
| First request after start = 109,585-token prefill | OK, 67.2 s; the engine-death case from vllm#58388 doesn't reproduce |

## How we got here (single-stream decode, code / prose tok/s)

| Step | Config | C1 code / prose | C8 agg | Prefill 7k |
|---|---|---|---|---|
| 1 | Stock vLLM, eager, block-FP8 W8A8, no MTP | 18.4 / 18.4 | 143 | 1,340 |
| 2 | + MTP k=3 | 51.6 / 39.7 | 251 | 1,281 |
| 3 | + 0002 (block → per-channel W8A16) | 55.8 / 43.1 | 324 | 1,904 |
| 4 | + 0003 (FP8 LM head), runtime 26.35, shared mode, k=5 | 65.3 / 44.3 | 317 | 2,204 |
| 5 | + 0004 + XPU graphs, k=3 | **87.0 / 64.2** | **371** | 2,240 |

Step 1 at long context: 1,115 tok/s prefill at 110k, 938 at 216k, and 14.6
tok/s decode at 216k. In step 4 the same tests give 1,626 / 1,264 tok/s and
41.6 tok/s.

## Quality A/B (isolated workers, runtime 26.35, MTP k=3, eager)

| | GSM8K | C1 code / prose | C8 | Prefill 7k / 27k | Decode @27k | Acceptance |
|---|---|---|---|---|---|---|
| Stock block-FP8 W8A8 | 96% | 47.8 / 36.5 | 268 | 1,278 / 1,254 | 43.6 | 2.88 |
| 0002 W8A16 | 95% | 55.8 / 43.1 | 324 | 1,904 / 1,875 | 53.3 | 2.79 |
| 0002 + 0003 FP8 LM head | 95% | 56.6 / 44.2 | 324 | 1,925 / 1,894 | 51.1 | 2.76 |
| 0002 + 0003 + 0004 graphs (k=3, shared) | 95% | 87.0 / 64.2 | 371 | 2,240 / 2,098 | 74.7 | 2.93 |

One question in 100 is within noise.

## MTP depth under graphs (shared, 0002+0003+0004, capture ≤ 64)

| k | C1 code | C1 prose | C4 | C8 | C16 | Acceptance |
|---|---|---|---|---|---|---|
| 3 | 88.8 | **66.0** | **218** | **364** | **540** | 2.78 |
| 4 | 91.8 | 63.7 | 203 | 355 | 495 | 3.00 |
| 5 | 93.8 | 62.3 | – | 333 | – | 3.26 |
| 6 | **95.7** | 61.6 | 193 | 306 | 422 | 3.33 |
| 7 | 91.1 | 57.6 | 178 | 290 | 313 | 3.06 |

k=3 is the default:
- it's best on prose and at every concurrency level;
- code is within 7% of the best;
- it stays clear of vllm#54785 (wrong logits with XPU graphs at MTP k ≥ 4).

Capture sizes 128 and 256 cost 2.0 and 5.4–5.75 GiB per rank. That no longer
leaves room for a full 262k sequence, so the engine refuses to start.

Eager, isolated, runtime 26.31, async scheduling, for comparison:
- k=3: 50.8 / 40.5
- k=4: 58.8 / 39.9
- k=5: 60.0 / 43.0

`--async-scheduling` made no difference (50.8 vs 51.6).

## Graphs: the failure modes and the fix (MTP k=5 unless noted)

| Config | Output | Acceptance | C1 code / prose |
|---|---|---|---|
| Graphs, oneCCL inside capture (small dense model) | capture **fails** (`ext_oneapi_get_graph()`) | – | – |
| Graphs, oneCCL inside capture (27B) | **garbage** | 0% | – |
| Piecewise graphs, all-reduce as a splitting op (out-of-place, eager) | **garbage** | 0% | 16.9 / 16.7 |
| 0004 all_reduce only | correct (GSM8K 95%) | **1.00** (all drafts rejected) | 24.2 / 24.1 |
| Same, drafter `enforce_eager` | correct | 1.00 | 24.2 / 24.2 |
| Same, `PIECEWISE` / `FULL_DECODE_ONLY`, k=3 | correct | 1.00 | 26.2 / 25.4 · 27.2 / 27.1 |
| Same, **ngram** speculation (control) | correct | **3.95** (eager 3.90) | – |
| **0004 all_reduce + all_gather + gather** | correct | 3.26 | **93.8 / 62.3** |

The ngram row is the key falsification. ngram drafts come from the prompt on
the CPU, with no draft model. High acceptance under graphs proves that
multi-token verification by the target model is correct. So the fault was the
drafter's own forward, i.e. its logits all-gather.

## Pipeline parallel alternative (PP=2, graphs, no TP collectives)

Settings: `VLLM_PP_LAYER_PARTITION=35,29` (8 attention layers per stage),
FP8 LM head, util 0.96, `max_num_seqs=32`, MTP k=3.

| | PP=2 + graphs | TP=2 + graphs (production) |
|---|---|---|
| C1 code / prose | 48.6 / 38.3 | 87.0 / 64.2 |
| C4 / C8 | 76 / 126 | 221 / 371 |
| Prefill 7k / 27k | **2,385 / 2,848** | 2,240 / 2,098 |
| KV pool | 270,855 | 289,863 |

PP has no per-layer all-reduce, so it prefills faster, but the stages run
serially for decode. Two failed starts on the way: with the even 32/32 split
the last stage holds the LM head and the MTP layer and runs out of KV space
(6.1 GiB available vs 9.45 needed). A 37/27 split puts 9 attention layers on
stage 0, which then needs 10.5 GiB.

## FP8 GEMM kernels (single B70, per-rank TP=2 shapes, µs)

From `tools/fp8_gemm_bench.py`. Cells are block-FP8 W8A8 (vLLM default) / W8A16 per-channel / bf16:

| Shape (N×K) | M=1 | M=16 | M=64 |
|---|---|---|---|
| mlp.gate_up 17408×5120 | 335 / **164** / 301 | 301 / **164** / 304 | 313 / **196** / 341 |
| mlp.down 5120×8704 | 135 / **79** / 153 | 144 / **77** / 154 | 147 / **110** / 171 |
| attn.qkv 7168×5120 | 104 / **59** / 127 | 93 / **61** / 128 | **101** / 113 / 143 |
| attn.o 5120×3072 | 52 / **18** / 40 | 52 / **17** / 41 | 56 / **40** / 50 |
| gdn.in_proj 12288×5120 | 219 / **113** / 213 | 230 / **117** / 215 | 228 / **132** / 241 |
| gdn.out 5120×3072 | 53 / **19** / 38 | 52 / **17** / 39 | 56 / **35** / 49 |

W8A16 reaches 540–920 GB/s, close to the card's memory bandwidth. Block W8A8
runs at about 270 GB/s effective, no better than bf16.

On a real checkpoint weight (layer 10 `gate_proj`, `tools/fp8_requant_check.py`), with relative error against the exact dequantized weight:

| Path | Error | M=1 | M=2048 |
|---|---|---|---|
| Block W8A8 | 0.0264 | 737 µs | 6,561 µs |
| W8A16 per-channel | 0.0265 | **161 µs** | **2,794 µs** |
| W8A8 per-channel | 0.0373 | 249 µs | 6,592 µs |

Requantizing the weight to per-channel FP8 costs 0.0264 relative error, the
same as the activation quantization it removes.

## Collectives

`tools/ccl_latency.py` and `tools/symm_allreduce_graph_test.py`:

| | 10 KiB | 1 MiB | 16 MiB |
|---|---|---|---|
| oneCCL, shared (default SYCL kernels) | 47 µs | 122 µs (8.6 GB/s) | 1,894 µs (8.9 GB/s) |
| oneCCL, isolated workers (0001) + `CCL_TOPO_P2P_ACCESS=0` | 46 µs | 190–200 µs (5.4 GB/s) | 2,680 µs (6.3 GB/s) |
| oneCCL `CCL_ENABLE_SYCL_KERNELS=0` / `CCL_ALLREDUCE=direct\|rabenseifner` | 173–179 µs | ~950 µs | ~15 ms |
| oneCCL `CCL_WORKER_WAIT=0`, `CCL_ZE_IPC_EXCHANGE=pidfd`, `CCL_SYCL_ALLREDUCE_SMALL_THRESHOLD=0` | no change | | |
| **Symmetric-memory all-reduce (0004), 40 KiB, eager** | 65 µs | | |
| **Symmetric-memory all-reduce (0004), 40 KiB, graph replay** | **16.4 µs** | | |

GPU-to-GPU copy runs at 14.3 GB/s and GPU-to-pinned-host at 14.2 GB/s, which
is the Gen4 x8 limit.

Eager decode profile, isolated workers, W8A16, k=3, 61 steps (torch profiler):
- the GPU was busy only 56% of the window;
- of kernel time, 52% was `oneccl_allreduce_pcie` (291 µs average including
  waiting for the peer) and 43% was GEMMs.

That launch-bound and sync-bound profile is what graphs remove.

## Host memory: the multi-device mirror

Host `MemAvailable` drop for GPU allocations (`tools`-style probe):

| | Drop |
|---|---|
| Runtime 26.31, one 4 GiB alloc, both GPUs visible in the process | **4,135 MiB** |
| Same, `ONEAPI_DEVICE_SELECTOR=level_zero:0` | 33 MiB |
| Same, `ZE_AFFINITY_MASK=0` | 36 MiB |
| Runtime 26.31, oneCCL init, 2 ranks × 4 GiB | 8,223 MiB, so 16 GiB per rank gets OOM-killed on a 30 GiB host |
| Runtime 26.31, isolated workers (0001), 2 ranks × 8 GiB | ~1.3 GiB |
| **Runtime 26.35**, 2 × 4 GiB, both GPUs visible | **200 MiB** |
