#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible vLLM endpoint: decode tok/s (single + concurrent) and prefill tok/s."""
import json, time, sys, glob, random, threading, urllib.request, argparse
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
ap.add_argument("--model", default="qwen38")
ap.add_argument("--prefill", default="1000,8000,32000,128000,250000")
ap.add_argument("--conc", default="1,4,8")
ap.add_argument("--gen", type=int, default=512)
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()

def stream(messages, max_tokens, nonce=True):
    if nonce: messages = [{"role": "system", "content": f"Session {random.randrange(1<<60)}."}] + messages
    body = json.dumps({"model": a.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
        "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": max_tokens > 1,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(a.url, body, {"Content-Type": "application/json"})
    t0 = time.time(); tfirst = tlast = None; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]": continue
            d = json.loads(line[6:])
            if d.get("usage"): usage = d["usage"]
            if d.get("choices") and (d["choices"][0]["delta"].get("content") or d["choices"][0]["delta"].get("reasoning_content")):
                tlast = time.time(); tfirst = tfirst or tlast
    return dict(ttft=tfirst - t0, decode_s=tlast - tfirst, prompt=usage["prompt_tokens"], gen=usage["completion_tokens"], total=tlast - t0)

PROMPTS = {
  "code": "Write a complete, well-commented Python implementation of a thread-safe LRU cache with TTL expiry, plus a full pytest test suite.",
  "prose": "Write a long, detailed essay about the history of the printing press and its effects on European society, science and religion.",
}
res = {}
print("== warmup"); stream([{"role": "user", "content": "Say hi."}], 16)
for kind, p in PROMPTS.items():
    r = stream([{"role": "user", "content": p}], a.gen)
    tps = (r["gen"] - 1) / r["decode_s"]; res[f"decode_{kind}"] = tps
    print(f"decode C=1 {kind:5s}: {tps:6.1f} tok/s  ({r['gen']} tok, TTFT {r['ttft']*1e3:.0f} ms)", flush=True)
for c in [int(x) for x in a.conc.split(",") if int(x) > 1]:
    out = []; ths = [threading.Thread(target=lambda i=i: out.append(stream([{"role": "user", "content": list(PROMPTS.values())[i % 2]}], a.gen))) for i in range(c)]
    t0 = time.time(); [t.start() for t in ths]; [t.join() for t in ths]; wall = time.time() - t0
    agg = sum(o["gen"] for o in out) / wall; per = sum((o["gen"] - 1) / o["decode_s"] for o in out) / c
    res[f"conc{c}"] = agg
    print(f"decode C={c}: {agg:6.1f} tok/s aggregate, {per:5.1f} tok/s per stream", flush=True)
src = "".join(open(f, errors="ignore").read() for f in sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))[:4000])
for n in [int(x) for x in a.prefill.split(",")]:
    text = src[: int(n * 3.3)]
    r = stream([{"role": "user", "content": "Here is some source code:\n\n" + text + "\n\nSummarize it in one sentence."}], 1)
    res[f"prefill_{r['prompt']}"] = r["prompt"] / r["ttft"]
    print(f"prefill {r['prompt']:7d} tok: TTFT {r['ttft']:7.2f} s -> {r['prompt']/r['ttft']:7.0f} tok/s", flush=True)
    if n >= 32000:
        # decode speed with a long context already in the KV cache
        r2 = stream([{"role": "user", "content": "Here is some source code:\n\n" + text + "\n\nExplain in detail what this code does."}], 256)
        print(f"  decode @ {r2['prompt']} ctx: {(r2['gen']-1)/r2['decode_s']:6.1f} tok/s", flush=True)
print("RESULT", json.dumps(res))
