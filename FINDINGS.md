# Findings: Qwen3.8-27B-FP8 TP=2 on 2× Arc Pro B70 (vLLM XPU)

Every failure hit between a fresh Ubuntu 26.04 install and the production
config, in roughly the order it was hit, with the evidence and the fix. Numbers
are in [RESULTS.md](RESULTS.md).

## 1. Host RAM mirrors every GPU allocation in a multi-device context (runtime ≤ 26.31)

**Symptom.** TP=2 engine init was SIGKILLed by the kernel's global OOM killer
on a 30 GiB host. vLLM itself reported the OOM as an XPU error: "Tried to
allocate 16.00 GiB… 29.62 GiB is free".

**Evidence.** A single `torch.zeros(4 GiB, device="xpu:0")` dropped host
`MemAvailable` by 4,135 MiB whenever the process could see both GPUs. The
memory doesn't show in RSS. The same allocation cost 33 MiB with
`ONEAPI_DEVICE_SELECTOR=level_zero:0` and 36 MiB with `ZE_AFFINITY_MASK=0`. The
data really was in VRAM (520 GB/s device copies, `xpu-smi` showed 4.4 GB used).
This is intel/compute-runtime#980 and intel/torch-xpu-ops#5428, the
multi-device GTT mirror.

**Fixes.**
1. Patch 0001 spawns each TP worker with a single-device mask. This needs
   `CCL_TOPO_P2P_ACCESS=0`, otherwise oneCCL's topology detection hangs when
   each rank sees one device. It loses about 30% of large-message bandwidth
   (6.3 vs 8.9 GB/s); small all-reduces stay at 46 µs.
2. **compute-runtime 26.35.39758.10 fixes it in userspace**: 2 × 4 GiB with both
   GPUs visible costs 200 MiB. Production uses 26.35 without 0001, because
   isolation costs bandwidth and the 0004 fallback needs P2P between ranks.

Caveat: there is a separate kernel-side issue, xe dma-buf export populating
the TT, reported on kernel 7.0 even with 26.35. Its fix is unmerged. We didn't
hit it: host RAM peaked at 17 GB during a 216k-context run. If you do hit it,
move to kernel ≥ 7.1.

## 2. Second B70 dies on runtime-PM resume

About 10 minutes after boot, card 2 logged `Force wake domain 0 failed to ack
wake (-ETIMEDOUT)` and `runtime resume failed`, followed by `Runtime PM usage
count underflow!`. It then disappeared from OpenCL and Level Zero until the
next reboot. Both cards share one x16 slot split x8/x8 (B550I), and the
card's own `lspci` link reads 2.5 GT/s x1. That reading is normal for B70,
because there's a switch on the card; the upstream `e2ff` bridge showed the
real link, Gen4 x8. **Fix:** `system/80-intel-gpu-no-runtime-pm.rules` sets
`power/control=on` and `d3cold_allowed=0` for Intel display-class devices. No
recurrence since.

## 3. oneCCL "Could not load any plugin"

The pip `oneccl` wheel installs `libccl*.so` into `$VIRTUAL_ENV/lib` but
`dlopen()`s its plugins by bare name. `CCL_LOG_LEVEL=info` shows `Cannot open
plugin library: libccl.so.1`. **Fix:** `LD_LIBRARY_PATH=$VIRTUAL_ENV/lib`.

## 4. `PYTORCH_ALLOC_CONF=expandable_segments:True` segfaults oneCCL

With expandable segments, every TP=2 start died with `Segfault encountered` in
`ofi_copy_iov_buf ← xnet_tsendmsg ← atl_ofi::send`. The OFI transport can't
handle virtual-memory-mapped allocations. Don't set it, even though one public
B70 recipe does.

## 5. KV-cache allocation "OOM" with plenty of free VRAM

"Tried to allocate 26.47 GiB… 29.54 GiB free". This was §1 in disguise, not a
max-allocation limit: `clinfo` reports a 30.3 GiB maximum allocation, and a
standalone `torch.zeros(24 GiB)` works.

## 6. The block-FP8 GEMM path is no faster than bf16 on Xe2 (patch 0002)

`XPUFp8BlockScaledMMKernel` quantizes activations per token group on every
call and feeds oneDNN `fp8_gemm`. Microbenchmarks on the per-rank TP=2 shapes
show:
- it runs at ~270 GB/s effective, the same as bf16 matmul;
- `fp8_gemm_w8a16` (weight-only FP8, bf16 activations) runs at 540–920 GB/s;
- W8A16 is 2× faster at M=1–64 and still 2.3× faster at M=2048.

W8A16 only accepts per-channel or per-tensor scales, so 0002 dequantizes each
block-scaled weight at load and requantizes it per output channel.

Accuracy on a real checkpoint weight, relative error against the exact
dequantized weight:
- block W8A8 path: 0.0264, all from activation quantization;
- W8A16 path: 0.0265, all from weight requantization;
- per-channel W8A8 (requantized weight plus activation quantization) is worse,
  at 0.0373, which is why 0002 uses W8A16 for every batch size.

End to end: GSM8K 96% → 95% (one question), prefill +55%, decode +15–20%.

## 7. Where eager TP=2 decode time goes

A torch-profiler trace of eager decode (W8A16, k=3) showed the GPU busy only
56% of the time. 52% of kernel time was `oneccl_allreduce_pcie`, averaging
291 µs per call against 46 µs in isolation, so most of that is one rank
waiting for the other's CPU to launch. 43% was GEMMs.

Per-stream decode barely changed from C=1 to C=8: 49 tok/s per stream at
C=8. So the step time was fixed overhead, not compute or bandwidth:
- ~140 all-reduces and ~300 GEMM launches per MTP step, all dispatched from Python;
- the CPU governor was not the cause (switching to `performance` changed nothing measurable);
- `--async-scheduling` changed nothing;
- the oneCCL environment sweep (RESULTS) found nothing better than the defaults.

The 46 µs floor is oneCCL's SYCL-kernel path over PCIe. Disabling SYCL kernels
falls back to 175 µs.

## 8. oneCCL inside XPU graphs: broken on 2022.1.1, fixed in 2022.1.2

**Root cause (found after §9–10, verified end to end):** the installed oneCCL
was too old. `torch==2.14.0+xpu` pins `oneccl==2022.1.1`, so a from-source
`requirements/xpu.txt` install gets 2022.1.1, and 2022.1.1 can't record
collectives into a SYCL graph. vLLM's `docker/Dockerfile.xpu` installs 2022.1.2
over torch's pin, so the official image works. On the same box,
`tools/ccl_graph_test.py` (64 chained all-reduces plus an all-gather, captured)
gives:

| oneCCL | Capture | Replay | In-graph cost per collective |
|---|---|---|---|
| 2022.1.1 | `ext_oneapi_get_graph()` error | wrong | – |
| 2022.1.1 + `CCL_SYCL_{ALLREDUCE,ALLGATHERV}_SIMPLE_THRESHOLD=4294967296` | same error | wrong | – |
| **2022.1.2** (defaults) | OK | **correct** | **16.2 µs** |
| 2022.1.2 + the SIMPLE thresholds | OK | correct | 16.4 µs |

End to end on Qwen3.8-27B with graphs and MTP k=3, stock oneCCL 2022.1.2 and
patch 0004 on 2022.1.1 match within noise:
- 87.8 / 64.2 vs 87.0 / 64.2 tok/s single-stream;
- 372 / 540 vs 371 / 542 tok/s at C8 / C16;
- GSM8K 95% for both.

Production now runs stock oneCCL 2022.1.2. Patch 0004 is a fallback that the
launcher enables automatically on an older oneCCL. The rest of this section
describes the 2022.1.1 behaviour.

torch's `XPUGraph` records into a **SYCL graph**: the capture stream's queue
goes into `ext_oneapi` recording state. What happened with 2022.1.1:
- **Small dense model (Qwen3-0.6B):** capture throws `onecclAllReduce caught
  std::exception: ext_oneapi_get_graph() can only be called on recording
  queues`, and the engine dies.
- **Qwen3.8-27B:** capture succeeds and the replayed graph produces **garbage**,
  deterministically and identical across runs (`<think> _ / a ${ch …`). With
  graphs on but no MTP the output is the same garbage. Without graphs the
  output is correct.

A torch.compile cache from an isolated-worker run (device `xpu:0` baked in for
rank 1) also gave `Expected all tensors to be on the same device` when reused
in shared mode. **Clear `~/.cache/vllm/torch_compile_cache` whenever you
switch worker isolation.** It initially muddied this diagnosis.

**Falsified: piecewise graphs with the all-reduce as a split point.** We added
`vllm::all_reduce` to `splitting_ops`, so the all-reduce runs eagerly between
graph pieces. Capture worked, and replay produced garbage with 0% acceptance.
vLLM's all-reduce is out-of-place, so every call returns a new address, while
the next captured piece reads the address recorded at capture time. Attention
works as a split point only because its ops write into preallocated outputs
(`*_with_output`).

## 9. Workaround for oneCCL 2022.1.1: symmetric-memory collectives during capture (patch 0004)

This was built before the oneCCL version was identified as the cause (§8). It
still works and is just as fast, so it stays as a fallback for stacks that can't
override torch's pin. torch 2.14's `XPUSymmetricMemory` provides Level Zero IPC buffers, peer
pointers (`get_buffer(peer)`), and **device-side `barrier` / `put_signal` /
`wait_signal` implemented as SYCL kernels**. It ships no collective ops for
XPU, but an all-reduce can be built from those pieces.

Copy into the local buffer, then `barrier(channel=0)`, then `out = buf[0] +
buf[1]` (P2P load, same order on every rank, so bit-identical), then
`barrier(channel=1)`. The second barrier stops a rank from overwriting its
buffer while the peer is still reading it.

`tools/symm_allreduce_graph_test.py` shows:
- correct eager results;
- a graph of 64 chained all-reduces replayed with five different inputs,
  correct every time;
- **16.4 µs per all-reduce on replay**.

In vLLM the symmetric path is used only when
`torch.xpu.is_current_stream_capturing()` is true and the tensor fits the
64 MiB buffer; eager calls keep oneCCL. During capture, `reduce_scatter`,
`all_gatherv` and `broadcast` log a warning if they're ever hit. They aren't
hit for this model.

## 10. MTP acceptance 1.00 under graphs: the drafter's logits gather

With 0004 covering only `all_reduce`:
- main-model output became correct (GSM8K 95%);
- **MTP acceptance was exactly 1.00** (per-position 0.000, 0.000, 0.000);
- decode fell to 24 tok/s, which is graphs without MTP: still +30% over the
  18.4 eager no-MTP baseline.

Falsified, in order:
- **MTP depth:** k=3 instead of 5 → still 1.00. This rules out the k ≥ 4 logits bug, vllm#54785.
- **Drafter graphs:** speculative `enforce_eager: true` → still 1.00. The V2 runner's drafter prefill path still captures.
- **Graph mode:** `PIECEWISE` and `FULL_DECODE_ONLY` → still 1.00.
- **Target verification:** with **ngram speculation** (drafts from the prompt, no draft model) acceptance was 3.95 under graphs vs 3.90 eager. Multi-token verification by the target is correct.

PP=2 with graphs had normal MTP acceptance (2.79). So the fault was in the
drafter, and specific to TP.

**Root cause.** The V2 runner's `AutoRegressiveSpeculator.capture()` records
the draft prefill as one FULL graph: model forward + `compute_logits` + sample.
`compute_logits` gathers the vocab-parallel LM-head output with an all-gather
inherited from `DeviceCommunicatorBase`. That's still oneCCL (2022.1.1), so it
got recorded broken, and every draft came from garbage logits. The target model
computes its logits outside its graph, which is why its outputs were
unaffected. Routing `all_gather` and `gather` through the symmetric buffer
during capture gave normal acceptance (3.26 at k=5) and 93.8 / 62.3 tok/s.
With oneCCL 2022.1.2 the oneCCL all-gather records correctly too, so none of
this applies there.

## 11. One-off copy-engine fault at startup, and the watchdog

Across ~40 engine starts, one hung during weight loading:
- **What the driver logged:** card 1 logged `Engine memory CAT error [18]: class=bcs` (a page fault on the copy engine), then `Engine reset`, then `Timedout job … in VLLM::Worker`. It wrote a devcoredump.
- **What the processes did:** the worker's host-to-device copy never completed (`py-spy --native`: spinning in `ur::level_zero::urEventWait` under `VocabParallelEmbedding.weight_loader`). The other rank waited forever in `shm_broadcast.wait_until_ready`.
- **Recovery:** a plain service restart.
- **Context:** the fault came seconds after another vLLM instance on the same GPUs was SIGKILLed, so stale IPC or VM state from the killed process is a plausible trigger, but it isn't proven. It hasn't recurred.

A hang isn't a failure, so `Restart=on-failure` never fires. `serve/vllm-watchdog.{sh,service,timer}`
restarts the unit when it has been active longer than 15 minutes and `/health`
fails three times in a row. This was verified with a simulated failure.

## 12. Things that didn't help

- **`--async-scheduling`:** ±0.
- **The CPU governor** (`powersave` → `performance`): ±0 measurable. It's kept on for latency hygiene.
- **The oneCCL environment sweep:** see RESULTS.
- **Larger prefill chunks** (`--max-num-batched-tokens 16384`): ±noise.
- **MTP k ≥ 6 under graphs:** single-stream code gains up to +8%, while prose and every concurrency level get worse.
- **Graph capture sizes 128 / 256:** 2.0 / 5.7 GiB per rank, which pushes the KV pool below one full 262k sequence.
- **PP=2 with graphs:** better prefill, much worse decode; see RESULTS.
- **Intel llm-scaler-vllm (0.26.0-b2):** older vLLM, plus a reported MTP + prefix-cache corruption in its GDN kernels. Not tried.
- **IPEX-LLM:** archived.
- **OpenVINO:** has no tensor parallelism, so not tried.
- **FP8 KV cache:** deliberately not used, because this deployment requires a 16-bit KV cache. At 256k, BF16 KV is 8 GiB per rank and fits alongside 14.4 GiB of weights per rank.
