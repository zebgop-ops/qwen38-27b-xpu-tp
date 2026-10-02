"""Standalone proof that a torch symmetric-memory all-reduce (copy -> device barrier -> P2P sum -> barrier)
is correct eagerly and when recorded into an XPU graph. Run: torchrun --nproc-per-node 2 symm_allreduce_graph_test.py"""
import os, time, torch, torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
r = int(os.environ["LOCAL_RANK"]); torch.xpu.set_device(r); dev = torch.device("xpu", r)
dist.init_process_group("xccl", device_id=dev)
group = dist.group.WORLD
MAXB = 8 << 20
buf = symm_mem.empty(MAXB // 2, dtype=torch.bfloat16, device=dev)
hdl = symm_mem.rendezvous(buf, group.group_name)
peer = 1 - r
print(r, "rendezvous ok; buffer_ptrs", [hex(p) for p in hdl.buffer_ptrs])
remote = hdl.get_buffer(peer, (MAXB // 2,), torch.bfloat16)
def ar(x):
    n = x.numel()
    buf[:n].copy_(x.view(-1))
    hdl.barrier(channel=0)
    out = buf[:n] + remote[:n]
    hdl.barrier(channel=1)
    return out.view_as(x)
# correctness eager
for n in (5120, 4 * 5120, 64 * 5120):
    torch.manual_seed(0); a = torch.randn(n, device=dev, dtype=torch.bfloat16); torch.manual_seed(1); b = torch.randn(n, device=dev, dtype=torch.bfloat16)
    y = ar(a if r == 0 else b); torch.xpu.synchronize()
    print(r, n, "eager maxerr", (y.float() - (a.float() + b.float())).abs().max().item())
# latency eager
x = torch.randn(4 * 5120, device=dev, dtype=torch.bfloat16)
for _ in range(20): ar(x)
torch.xpu.synchronize(); t = time.perf_counter()
for _ in range(500): ar(x)
torch.xpu.synchronize(); print(r, "eager 40KiB allreduce us:", round((time.perf_counter() - t) / 500 * 1e6, 1))
# graph capture: 64 chained allreduces with changing input
static_in = torch.zeros(4 * 5120, device=dev, dtype=torch.bfloat16)
s = torch.xpu.Stream()
s.wait_stream(torch.xpu.current_stream())
with torch.xpu.stream(s):
    for _ in range(3): y = ar(static_in)
torch.xpu.current_stream().wait_stream(s); torch.xpu.synchronize()
g = torch.xpu.XPUGraph()
with torch.xpu.graph(g):
    h = static_in
    for _ in range(64): h = ar(h) * 0.5     # sum of two equal ranks *0.5 keeps values stable if inputs equal
    static_out = h
ok = True
for it in range(5):
    v = float(it + 1)
    static_in.fill_(v if r == 0 else 3 * v)   # first ar -> 4v, *0.5 -> 2v, then each step: 2*(2v)*0.5 = 2v
    g.replay(); torch.xpu.synchronize()
    exp = 2 * v
    err = (static_out.float() - exp).abs().max().item()
    ok &= err < 1e-2
    print(r, f"graph replay {it}: out={static_out[0].item():.3f} expect {exp:.3f}")
torch.xpu.synchronize(); t = time.perf_counter()
for _ in range(50): g.replay()
torch.xpu.synchronize(); print(r, "graph: per-allreduce us:", round((time.perf_counter() - t) / 50 / 64 * 1e6, 1), "ok" if ok else "WRONG")
dist.destroy_process_group()
