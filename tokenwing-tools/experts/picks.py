"""Load the experts a model picked, from the files llama-moe-trace writes.

Record with:  llama-moe-trace -m MODEL.gguf -o PREFIX -f PROMPT.txt -n WORDS -c 4096 -b 512 --no-warmup
That writes PREFIX.bin (int32 [tokens][MoE layers][experts per token]) and PREFIX.json (which layers, how many experts, the token phases).
Use many varied prompts: the counts are only as good as the text they come from.
"""
import json
import numpy as np


def load(prefixes):
    """Yields (meta, picks [tokens][layers][k]) per recording; a prefix may also be given as PREFIX.json."""
    for p in prefixes:
        base = p[:-5] if p.endswith(".json") else p
        with open(base + ".json") as f:
            m = json.load(f)
        a = np.fromfile(base + ".bin", dtype=np.int32).reshape(m["n_tokens"], len(m["moe_layers"]), m["k"])
        yield m, a


def count(prefixes, decode_only):
    """Picks per expert per MoE layer over all recordings. Returns (counts [layers][n_expert], first recording's meta, tokens, recordings)."""
    cnt = None; first = None; tokens = 0; n = 0
    for m, a in load(prefixes):
        if first is None:
            first = m
            cnt = np.zeros((len(m["moe_layers"]), m["n_expert"]), np.int64)
        if (m["moe_layers"], m["n_expert"], m["k"]) != (first["moe_layers"], first["n_expert"], first["k"]):
            raise SystemExit("the recordings are not of the same model")
        if decode_only:
            a = a[m["n_prompt"]:]
        tokens += a.shape[0]; n += 1
        for l in range(a.shape[1]):
            cnt[l] += np.bincount(a[:, l, :].ravel(), minlength=m["n_expert"])
    if first is None:
        raise SystemExit("no recordings given")
    return cnt, first, tokens, n
