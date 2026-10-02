"""oneCCL all-reduce/all-gather recorded into an XPU graph: correctness + per-call latency. Tells you
whether your oneCCL can be captured (2022.1.1: no, 2022.1.2: yes). Run: torchrun --nproc-per-node 2 ccl_graph_test.py"""
import os, time, torch, torch.distributed as dist
r = int(os.environ["LOCAL_RANK"]); torch.xpu.set_device(r); dev = torch.device("xpu", r)
dist.init_process_group("xccl", device_id=dev)
n = int(os.environ.get("NELEM", 4 * 5120))
x = torch.zeros(n, device=dev, dtype=torch.bfloat16)
g_in = torch.zeros(n // 4, device=dev, dtype=torch.bfloat16)
def ar(t):
    o = t.clone(); dist.all_reduce(o); return o
s = torch.xpu.Stream(); s.wait_stream(torch.xpu.current_stream())
with torch.xpu.stream(s):
    for _ in range(3): ar(x); dist.all_gather_into_tensor(torch.empty(n // 2, device=dev, dtype=torch.bfloat16), g_in)
torch.xpu.current_stream().wait_stream(s); torch.xpu.synchronize()
g = torch.xpu.XPUGraph()
try:
    with torch.xpu.graph(g):
        h = x
        for _ in range(64): h = ar(h) * 0.5
        out = h
        gat = torch.empty(n // 2, device=dev, dtype=torch.bfloat16)
        dist.all_gather_into_tensor(gat, g_in)
except Exception as e:
    print(r, "CAPTURE FAILED:", str(e)[:150]); raise SystemExit
ok = True
for it in range(5):
    v = float(it + 1)
    x.fill_(v if r == 0 else 3 * v); g_in.fill_(10 * r + it)
    g.replay(); torch.xpu.synchronize()
    ok &= abs(out[0].item() - 2 * v) < 1e-2 and gat[0].item() == it and gat[-1].item() == 10 + it
print(r, "graph replay correct:", ok)
torch.xpu.synchronize(); t = time.perf_counter()
for _ in range(50): g.replay()
torch.xpu.synchronize(); print(r, f"in-graph per-collective us: {(time.perf_counter()-t)/50/65*1e6:.1f}")
dist.destroy_process_group()
