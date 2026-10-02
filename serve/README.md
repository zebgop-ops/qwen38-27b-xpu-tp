# Serving

- **`run-tp2-mtp.sh`**: the exact production launcher. It sets every engine
  flag and env var, with a comment on why each load-bearing one matters:
  - the oneCCL `LD_LIBRARY_PATH`;
  - the per-patch switches;
  - why the graph capture size stops at 64;
  - why patch 0001 stays off.

  Tunables (env):
  - `QWEN38_MODEL`, `QWEN38_VENV`, `QWEN38_PORT`;
  - `QWEN38_SPEC`: MTP depth, default 3;
  - `QWEN38_MAXLEN`: default 262144;
  - `QWEN38_UTIL`: default 0.93;
  - `QWEN38_GRAPHS`: 1 = XPU graphs, 0 = eager.

  Extra arguments pass through to `vllm serve`.
- **`make-patched-tree.sh`**: clones vLLM at `ca65eb67d`, applies
  `../patches/000*.patch`, and builds a Python 3.12 venv with `uv` from
  `requirements/xpu.txt`. The kernels come as wheels, so there's no C++ build.
  It's verified to reproduce the production tree byte-for-byte.
- **`vllm-qwen38.service`**: systemd unit. Replace `USER` before installing.
- **`vllm-watchdog.{sh,service,timer}`**: restarts the unit if it's active but `/health` keeps failing. This catches startup hangs, which `Restart=on-failure` can't (FINDINGS §11).

```bash
./serve/make-patched-tree.sh
QWEN38_MODEL=/path/to/Qwen3.8-27B-FP8 ./serve/run-tp2-mtp.sh
# eager, no symmetric-memory collectives (stock-equivalent comms):
VLLM_XPU_SYMM_AR=0 QWEN38_GRAPHS=0 ./serve/run-tp2-mtp.sh
```

Model download:
`hf download Qwen/Qwen3.8-27B-FP8 --local-dir ~/models/Qwen3.8-27B-FP8`
(30.9 GB).

Notes:
- **Clear the torch.compile cache when you switch worker isolation.** Delete `~/.cache/vllm/torch_compile_cache` whenever you toggle `VLLM_XPU_ISOLATE_WORKERS`.
- **Disabling thinking:** pass `chat_template_kwargs: {"enable_thinking": false}` per request.
