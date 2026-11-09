#!/usr/bin/env python3
"""Make the ranked expert list per MoE layer for --moe-stream-vram-experts-file ("tokenwing vram-experts v1").

Counts the experts picked while writing replies (not prompts), most-picked first, ties by expert id.

usage: make_vram_list.py OUT.json RECORDING [RECORDING ...] [--top N]     (a recording is the prefix given to llama-moe-trace; see picks.py)
       --top N: how many experts to keep per layer (default 64)
"""
import json
import sys
import picks

args = sys.argv[1:]
top = 64
if "--top" in args:
    i = args.index("--top"); top = int(args[i + 1]); del args[i:i + 2]
if len(args) < 2:
    sys.exit(__doc__)
out, rec = args[0], args[1:]
cnt, m, tokens, n = picks.count(rec, decode_only=True)
j = {"format": "tokenwing vram-experts v1", "arch": m["arch"], "n_expert": m["n_expert"], "n_expert_used": m["k"],
     "source": "%d recordings, %d decode words" % (n, tokens),
     "counted": "decode picks per layer; most-picked first, ties by expert id", "layers": {}}
for i, l in enumerate(m["moe_layers"]):
    order = sorted(range(m["n_expert"]), key=lambda e: (-int(cnt[i, e]), e))[:top]
    j["layers"][str(l)] = [[e, int(cnt[i, e])] for e in order]
with open(out, "w") as f:
    json.dump(j, f, indent=None, separators=(",", ":"))
print("wrote %s: %d layers x %d experts, %d decode words of %d recordings" % (out, len(m["moe_layers"]), top, tokens, n))
