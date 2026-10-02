"""oneCCL (xccl) all-reduce latency/bandwidth at 10 KiB / 1 MiB / 16 MiB.
Shared mode: torchrun --nproc-per-node 2 ccl_latency.py ; isolated: ISO=1 CCL_TOPO_P2P_ACCESS=0 torchrun ..."""
import os, sys
r=int(os.environ["LOCAL_RANK"])
if os.environ.get("ISO")=="1": os.environ["ZE_AFFINITY_MASK"]=str(r); dev=0
else: dev=r
import torch, torch.distributed as dist, time
torch.xpu.set_device(dev); dist.init_process_group("xccl", device_id=torch.device("xpu", dev))
for nbytes in (10240, 1<<20, 16<<20):
    x=torch.randn(nbytes//2, device="xpu", dtype=torch.float16)
    for _ in range(20): dist.all_reduce(x)
    torch.xpu.synchronize(); t0=time.time(); N=200 if nbytes<1e6 else 30
    for _ in range(N): dist.all_reduce(x)
    torch.xpu.synchronize(); dt=(time.time()-t0)/N
    if r==0: print(f"{nbytes/1024:.0f}KiB: {dt*1e6:.0f} us, {nbytes/dt/1e9:.1f} GB/s")
dist.destroy_process_group()
