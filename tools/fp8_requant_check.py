"""Accuracy + speed of block-FP8 W8A8 vs per-channel W8A16 vs per-channel W8A8 on a real
Qwen3.8-27B-FP8 weight (layer 10 gate_proj). Usage: QWEN38_MODEL=/path/to/Qwen3.8-27B-FP8 python fp8_requant_check.py"""
import os, torch, time, json, glob
import vllm_xpu_kernels._xpu_C  # noqa
from safetensors import safe_open
dev = "xpu:0"; FP8 = torch.float8_e4m3fn; FMAX = torch.finfo(FP8).max
root = os.environ.get("QWEN38_MODEL", os.path.expanduser("~/models/Qwen3.8-27B-FP8"))
idx = {}
for f in glob.glob(root + "/*.safetensors"):
    with safe_open(f, "pt") as s:
        for k in s.keys(): idx[k] = f
name = [k for k in idx if k.endswith("layers.10.mlp.gate_proj.weight")][0]
def get(k):
    with safe_open(idx[k], "pt", device="cpu") as s: return s.get_tensor(k)
w = get(name).to(dev); s = get(name.replace("weight", "weight_scale_inv")).to(dev)
print(name, tuple(w.shape), w.dtype, "scale", tuple(s.shape))
N, K = w.shape; bn = bk = 128
def dequant(w, s):
    return w.float() * s.repeat_interleave(bn, 0)[:N].repeat_interleave(bk, 1)[:, :K]
wf = dequant(w, s)                                   # reference fp32 weight
amax = wf.abs().amax(1).clamp(min=1e-12); cs = (amax / FMAX).float()
wc = (wf / cs[:, None]).clamp(-FMAX, FMAX).to(FP8)   # per-channel fp8 [N,K]
wkn = wc.t()                                         # [K,N] view
print("requant weight rel err:", ((wc.float() * cs[:, None] - wf).norm() / wf.norm()).item())
def w8a16(x): return torch.ops._xpu_C.fp8_gemm_w8a16(x, wkn, cs.view(1, -1), None)
def w8a8c(x):
    xs = (x.abs().amax(-1, keepdim=True).float().clamp(min=1e-12) / FMAX)
    q = (x.float() / xs).clamp(-FMAX, FMAX).to(FP8)
    return torch.ops._xpu_C.fp8_gemm(q, wkn, torch.bfloat16, xs, cs, torch.Tensor())
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
bs_kn = s.t().contiguous()
def block(x):
    xq, xs = per_token_group_quant_fp8(x, 128, column_major_scales=False)
    return torch.ops._xpu_C.fp8_gemm(xq, w.t(), torch.bfloat16, xs, bs_kn, torch.Tensor())
def bench(fn, x, it=50):
    for _ in range(5): fn(x)
    torch.xpu.synchronize(); t = time.perf_counter()
    for _ in range(it): fn(x)
    torch.xpu.synchronize(); return (time.perf_counter() - t) / it * 1e6
for M in (1, 16, 64, 256, 2048):
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    ref = x.float() @ wf.t()
    row = [f"M={M:5d}"]
    for nm, fn in (("block", block), ("w8a16", w8a16), ("w8a8-chan", w8a8c)):
        y = fn(x).float(); err = ((y - ref).norm() / ref.norm()).item()
        row.append(f"{nm} {bench(fn, x):7.1f}us err {err:.4f}")
    print(" | ".join(row))
