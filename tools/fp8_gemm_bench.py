"""Microbenchmark of the XPU FP8 GEMM kernels on the per-rank (TP2) Qwen3.8-27B linear shapes:
block-scaled W8A8 (vLLM default for block-FP8) vs weight-only W8A16 vs bf16."""
import torch, time
import vllm_xpu_kernels._xpu_C  # noqa: registers torch.ops._xpu_C
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
dev = "xpu:0"; torch.xpu.set_device(0)
# per-rank (TP2) linear shapes of Qwen3.8-27B: (name, N, K)
shapes = [("mlp.gate_up", 17408, 5120), ("mlp.down", 5120, 8704), ("attn.qkv", 7168, 5120), ("attn.o", 5120, 3072), ("gdn.in_proj", 12288, 5120), ("gdn.out", 5120, 3072)]
def bench(fn, it=50):
    for _ in range(5): fn()
    torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.xpu.synchronize(); return (time.perf_counter() - t) / it * 1e6
for M in (1, 4, 16, 64):
    print(f"--- M={M}")
    for name, N, K in shapes:
        w = (torch.randn(N, K, device=dev) / 10).to(torch.float8_e4m3fn)
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        # block-scaled W8A8 path (current): dynamic per-token-group act quant + fp8_gemm
        bs = torch.rand(K // 128, N // 128, device=dev, dtype=torch.float32)  # [k_blocks, n_blocks]
        wkn = w.t()  # [K, N] view
        def blk():
            xq, xs = per_token_group_quant_fp8(x, 128, column_major_scales=False)
            return torch.ops._xpu_C.fp8_gemm(xq, wkn, torch.bfloat16, xs, bs, None)
        # per-channel W8A16 path
        wcs = torch.rand(1, N, device=dev, dtype=torch.float32)
        def w816(): return torch.ops._xpu_C.fp8_gemm_w8a16(x, wkn, wcs, None)
        wb = w.to(torch.bfloat16).t()
        def bf16(): return x @ wb
        tb, tw, tf = bench(blk), bench(w816), bench(bf16)
        gbps = N * K / tw / 1e3
        print(f"{name:12s} N={N:5d} K={K:5d}: block-W8A8 {tb:7.1f} us | W8A16 {tw:7.1f} us ({gbps:5.0f} GB/s) | bf16 {tf:7.1f} us")
