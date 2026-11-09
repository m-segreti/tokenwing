#!/usr/bin/env python3
"""Make the seed of the usage table of --moe-stream-idle-load ("tokenwing usage v1") from recorded expert picks.

The seed counts the picks per expert per MoE layer. The engine reads it only while its live table (<sidecar>.usage.json) does not exist yet,
and halves every count when it loads a table. Keep the output next to the experts file, as <sidecar>.usage-seed.json.
The counts are scaled so that every layer sums to 16,384 tokens' worth of picks: the live table (+1 per pick, halved at each start) takes over
after a few thousand tokens of real use instead of many restarts.

usage: make_usage_seed.py OUT.json RECORDING [RECORDING ...]     (a recording is the prefix given to llama-moe-trace; see picks.py)
"""
import json
import os
import sys
import numpy as np
import picks

if len(sys.argv) < 3:
    sys.exit(__doc__)
out, rec = sys.argv[1], sys.argv[2:]
cnt, m, tokens, n = picks.count(rec, decode_only=False)
total = m["k"] * 16384
scaled = np.floor(cnt * (total / np.maximum(cnt.sum(axis=1, keepdims=True), 1))).astype(np.int64)
j = {"format": "tokenwing usage v1", "arch": m["arch"], "n_expert": m["n_expert"],
     "counted": "seed: picks per expert per MoE layer, %d tokens of %d recordings, scaled to %d picks per layer" % (tokens, n, total),
     "layers": {str(l): [int(x) for x in scaled[i]] for i, l in enumerate(m["moe_layers"])}}
tmp = out + ".tmp"
with open(tmp, "w") as fo:
    json.dump(j, fo)
    fo.flush(); os.fsync(fo.fileno())
os.replace(tmp, out)
print("wrote %s: %d layers x %d experts, %d tokens, %d recordings, scaled to %d picks per layer" % (out, len(m["moe_layers"]), m["n_expert"], tokens, n, total))
