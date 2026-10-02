#!/usr/bin/env python3
"""GSM8K accuracy (first N test problems), greedy, thinking off. Saves outputs for cross-config comparison."""
import json, re, sys, urllib.request, concurrent.futures as cf, os
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
tag = sys.argv[2] if len(sys.argv) > 2 else "run"
path = os.path.expanduser("~/gsm8k_test.jsonl")
if not os.path.exists(path):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl", path)
probs = [json.loads(l) for l in open(path)][:N]
def ask(p):
    body = {"model": "qwen38", "temperature": 0, "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": p["question"] + "\nSolve step by step, then finish with a line of the form 'Answer: <number>'."}]}
    r = urllib.request.Request("http://localhost:8000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    txt = json.loads(urllib.request.urlopen(r, timeout=900).read())["choices"][0]["message"]["content"] or ""
    m = re.findall(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", txt)
    pred = m[-1].replace(",", "") if m else None
    gold = p["answer"].split("####")[-1].strip().replace(",", "")
    ok = pred is not None and abs(float(pred) - float(gold)) < 1e-6
    return ok, txt
with cf.ThreadPoolExecutor(8) as ex: res = list(ex.map(ask, probs))
acc = sum(o for o, _ in res) / len(res)
json.dump([t for _, t in res], open(os.path.expanduser(f"~/gsm8k_{tag}.json"), "w"))
print(f"GSM8K[{N}] {tag}: {acc*100:.1f}%")
