#!/usr/bin/env python3
"""Summarize a trace written by llama-moe-trace.

usage: read_trace.py <prefix>

Reads <prefix>.bin (int32 [n_tokens, n_moe_layers, k]) and <prefix>.json, checks
that the trace is well formed and prints how the routing is distributed.
"""

import json
import os
import sys

import numpy as np


def load(prefix):
    meta_path = prefix + ".json"
    bin_path = prefix + ".bin"

    for path in (meta_path, bin_path):
        if not os.path.exists(path):
            sys.exit("read_trace.py: missing '%s'" % path)

    with open(meta_path) as f:
        meta = json.load(f)

    n_tokens = meta["n_tokens"]
    k = meta["k"]
    n_moe = len(meta["moe_layers"])

    expected = n_tokens * n_moe * k
    actual = os.path.getsize(bin_path) // 4

    topk = np.fromfile(bin_path, dtype="<i4")

    if actual != expected:
        sys.exit("read_trace.py: '%s' holds %d int32, expected %d" % (bin_path, actual, expected))

    return meta, topk.reshape(n_tokens, n_moe, k)


def main(argv):
    if len(argv) != 2:
        sys.exit(__doc__)

    meta, topk = load(argv[1])

    n_tokens, n_moe, k = topk.shape
    n_expert = meta["n_expert"]

    print("model          : %s (%s)" % (meta["model"], meta["arch"]))
    print("layers         : %d total, %d MoE (%s .. %s)" % (
        meta["n_layer"], n_moe, meta["moe_layers"][0], meta["moe_layers"][-1]))
    print("experts        : n_expert = %d, k = %d, groups = %d of %d" % (
        n_expert, k, meta["n_group_used"], meta["n_expert_groups"]))
    print("n_embd         : %d" % meta["n_embd"])
    print("n_tokens       : %d (n_prompt = %d, n_decode = %d)" % (
        n_tokens, meta["n_prompt"], meta["n_decode"]))
    print("sampling       : %s" % meta.get("sampling", "unknown"))
    print("topk file      : %s" % meta["topk_file"])
    print()

    print("phases         : %d prefill, %d decode" % (
        meta["phases"].count("prefill"), meta["phases"].count("decode")))

    # the trace has to line up with the token list of the json
    if len(meta["token_ids"]) != n_tokens or len(meta["positions"]) != n_tokens:
        sys.exit("read_trace.py: json token arrays do not match n_tokens")
    if sorted(meta["positions"]) != list(range(n_tokens)):
        sys.exit("read_trace.py: positions are not 0 .. n_tokens-1")
    print("positions      : 0 .. %d, in order" % (n_tokens - 1))
    print()

    print("== validity ==")

    lo, hi = int(topk.min()), int(topk.max())
    print("id range       : [%d, %d] (expected [0, %d])" % (lo, hi, n_expert - 1))
    ok_range = lo >= 0 and hi < n_expert
    print("ids in range   : %s" % ("OK" if ok_range else "FAIL"))

    srt = np.sort(topk, axis=2)
    n_dup = int((np.diff(srt, axis=2) == 0).any(axis=2).sum())
    print("rows with dup  : %d of %d" % (n_dup, n_tokens * n_moe))

    # every layer has to hold exactly k ids, one per expert
    complete = topk.shape == (n_tokens, n_moe, k) and n_dup == 0 and ok_range
    print("complete rows  : %s" % ("OK" if complete else "FAIL"))
    print()

    baseline = k / n_expert

    print("== routing per layer ==")
    print("%6s %10s %10s %10s  %s" % ("layer", "overlap", "baseline", "ratio", "top 5 experts (share)"))

    for j in range(n_moe):
        ids = topk[:, j, :]
        prev = ids[:-1]
        cur = ids[1:]

        if prev.shape[0] == 0:
            overlap = float("nan")
        else:
            # |A n B| / k, one boolean row per consecutive pair
            inter = (prev[:, :, None] == cur[:, None, :]).any(axis=2).sum(axis=1)
            overlap = float(inter.mean()) / k

        counts = np.bincount(ids.ravel(), minlength=n_expert)
        top5 = np.argsort(-counts)[:5]
        top5_str = " ".join("%d:%.3f" % (e, counts[e] / ids.size) for e in top5)

        print("%6d %10.4f %10.4f %10.2f  %s" % (
            meta["moe_layers"][j], overlap, baseline,
            overlap / baseline if baseline > 0 else float("nan"), top5_str))

    # every expert has to be reachable, otherwise the trace cannot be used for
    # cache studies
    used = np.bincount(topk.ravel(), minlength=n_expert)
    print()
    print("experts used at least once : %d of %d" % (int((used > 0).sum()), n_expert))
    print("least used expert          : %d (%d times)" % (int(used.argmin()), int(used.min())))
    print("most used expert           : %d (%d times)" % (int(used.argmax()), int(used.max())))

    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
