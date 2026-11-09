#!/usr/bin/env python3
"""Verify a sample of an expert sidecar against the GGUF it was built from.

For each of a sample of layers and experts it reads the three projections from the model (through the
index, from the shard the tensor is in) and from the sidecar, and compares the bytes. Split GGUFs are
covered: the model's bytes live in several files.

Example: python3 verify_sidecar.py index.json experts.bin.json
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_io import ExpertIndex, load_sidecar, read_expert_block, open_direct  # noqa: E402
from expert_repack import cpu_repack_mxfp4                                          # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("index", help="expert index JSON of the model (written by expert_index.py -o)")
    ap.add_argument("sidecar_json", help="the <sidecar>.json next to the sidecar data file")
    ap.add_argument("--layers", type=int, default=8, help="how many layers to sample (default 8)")
    ap.add_argument("--experts", type=int, default=3, help="experts per sampled layer (default 3)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    idx = ExpertIndex.load(args.index)
    meta = load_sidecar(args.sidecar_json)
    print(f"sidecar expert layout: {meta.get('expert_layout', 'plain')}")
    # the engine derives the data file from the metadata path; do the same here
    sidecar = args.sidecar_json[:-len(".json")] if args.sidecar_json.endswith(".json") else args.sidecar_json
    n_expert = meta["n_expert"]
    moe = [il for il in meta["moe_layers"] if str(il) in meta["blocks"]]

    rnd = random.Random(args.seed)
    layers = sorted(rnd.sample(moe, min(args.layers, len(moe))))
    if moe[0] not in layers and moe[-1] not in layers:
        layers = [moe[0]] + layers[:-1] + [moe[-1]]

    fds = [open_direct(s["path"]) for s in idx.shards]
    sf = open_direct(sidecar)
    bad = 0
    try:
        for il in layers:
            for e in [0, n_expert - 1] + sorted(rnd.sample(range(n_expert), max(0, args.experts - 2))):
                # A sidecar can store its experts in llama.cpp's repacked CPU layout, and then the
                # model bytes have to be repacked the same way before they can be compared
                blk = meta["blocks"][str(il)]
                blay = blk.get("expert_layout", meta.get("expert_layout", "plain"))
                n_rows = {p: (idx.slice_size(il, p) // 17) // (idx.tensor(il, p)["shape_ggml"][0] // 32)
                          for p in ("gate", "up", "down")}
                from_model = {}
                for p in ("gate", "up", "down"):
                    fd = fds[idx.shard(il, p)]
                    blob = read_expert_slice_model(fd, idx, il, e, p)
                    if blay != "plain":
                        blob = cpu_repack_mxfp4(blob, n_rows[p])
                    from_model[p] = blob
                from_sidecar = dict(zip(("gate", "up", "down"), read_expert_block(sf, meta, il, e)))
                same = all(from_model[p] == from_sidecar[p] for p in ("gate", "up", "down"))
                nb = sum(len(from_model[p]) for p in from_model)
                print(f"layer {il:3d} expert {e:4d}: {nb:,} B  {'same' if same else 'MISMATCH'}"
                      f"{'' if same else ' ' + str({p: (len(from_model[p]), len(from_sidecar[p])) for p in from_model})}")
                bad += 0 if same else 1
    finally:
        for fd in fds:
            os.close(fd)
        os.close(sf)

    print(f"\n{len(layers)} layers x {args.experts} experts, {bad} mismatches")
    return 1 if bad else 0


def read_expert_slice_model(fd, index, layer, expert, proj):
    import expert_io
    return expert_io.read_expert_slice(fd, index, layer, expert, proj)


if __name__ == "__main__":
    sys.exit(main())
