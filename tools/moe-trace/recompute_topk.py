#!/usr/bin/env python3
"""Recompute the MoE selection of a trace and compare it with the recorded ids.

usage: recompute_topk.py <prefix> <model.gguf>

The router weights (blk.N.ffn_gate_inp.weight) and the selection bias
(blk.N.exp_probs_b.bias) are read from the GGUF, then the selection is redone
for every stored router input exactly the way build_moe_ffn does it in
src/llama-graph.cpp: score -> (expert selection bias) -> group selection
(top 2 sum per group, keep the best n_group_used groups, mask the rest) -> top k.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))

from gguf import GGUFReader  # noqa: E402
from gguf.constants import GGMLQuantizationType  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


# llama_expert_gating_func_type, src/llama-hparams.h
GATING_FUNCS = {1: "softmax", 2: "sigmoid", 3: "softmax_weight", 4: "sqrt_softplus"}


def score(logits, gating):
    if gating == "softmax":
        e = np.exp(logits - logits.max())
        return e / e.sum()
    if gating == "sigmoid":
        return 1.0 / (1.0 + np.exp(-logits))
    if gating == "sqrt_softplus":
        return np.sqrt(np.log1p(np.exp(-np.abs(logits))) + np.maximum(logits, 0))
    raise NotImplementedError("gating func %s is not implemented" % gating)


def select(probs, bias, n_expert_groups, n_group_used, k):
    """the selection path of build_moe_ffn for one token, all shapes are 1d"""
    sel = probs + bias if bias is not None else probs

    if n_expert_groups > 1:
        groups = sel.reshape(n_expert_groups, -1)

        # top 2 per group, summed -> one score per group
        part = np.partition(groups, -2, axis=1)[:, -2:]
        group_scores = part.sum(axis=1)

        keep = np.argsort(-group_scores, kind="stable")[:n_group_used]

        masked = np.full_like(sel, -np.inf).reshape(n_expert_groups, -1)
        masked[keep] = groups[keep]
        sel = masked.reshape(-1)

    # ggml_argsort_top_k sorts descending
    order = np.argsort(-sel, kind="stable")

    return order[:k]


def tensor_f32(reader, name):
    for t in reader.tensors:
        if t.name == name:
            break
    else:
        sys.exit("recompute_topk.py: the model has no tensor '%s'" % name)

    if t.tensor_type != GGMLQuantizationType.F32:
        data = dequantize(t.data, t.tensor_type)
    else:
        data = t.data

    # ggml shapes are ne order, the reader hands out numpy arrays in ne reversed order
    return np.asarray(data, dtype=np.float32).reshape(t.shape[::-1])


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)

    prefix, model_path = argv[1], argv[2]

    with open(prefix + ".json") as f:
        meta = json.load(f)

    hin_path = prefix + ".hin.bin"
    if not os.path.exists(hin_path):
        sys.exit("recompute_topk.py: missing '%s', run with --dump-router-input" % hin_path)

    n_tokens = meta["n_tokens"]
    n_embd = meta["n_embd"]
    k = meta["k"]
    n_expert = meta["n_expert"]
    n_expert_groups = meta["n_expert_groups"]
    n_group_used = meta["n_group_used"]
    moe_layers = meta["moe_layers"]

    recorded = np.fromfile(prefix + ".bin", dtype="<i4").reshape(n_tokens, len(moe_layers), k)
    hin = np.fromfile(hin_path, dtype="<f2").reshape(n_tokens, len(moe_layers), n_embd)

    if os.path.getsize(hin_path) // 2 != n_tokens * len(moe_layers) * n_embd:
        sys.exit("recompute_topk.py: '%s' has the wrong size" % hin_path)

    reader = GGUFReader(model_path)

    def model_meta(key):
        field = reader.get_field(key)
        if field is None:
            sys.exit("recompute_topk.py: '%s' is not in the model" % key)
        return field.contents()

    arch = model_meta("general.architecture")

    for key, want in (("expert_count", n_expert), ("expert_used_count", k),
                      ("expert_group_count", n_expert_groups),
                      ("expert_group_used_count", n_group_used)):
        got = model_meta("%s.%s" % (arch, key))
        if got != want:
            sys.exit("recompute_topk.py: %s.%s = %d, the trace says %d" % (arch, key, got, want))

    gating = GATING_FUNCS.get(int(model_meta("%s.expert_gating_func" % arch)))
    if gating is None:
        sys.exit("recompute_topk.py: unsupported expert_gating_func")
    if gating == "softmax_weight":
        # the score is only used for the weights, the selection uses the raw logits
        gating = "none"

    print("model          : %s (%s)" % (model_path, arch))
    print("gating         : %s, bias for selection: yes" % gating)
    print("groups         : top 2 sum per group, best %d of %d groups, then top %d of %d"
          % (n_group_used, n_expert_groups, k, n_expert))
    print("router input   : %s (float16, %d tokens)" % (hin_path, n_tokens))
    print()

    print("%6s %10s %10s %10s %10s" % ("layer", "rows_eq", "set_eq", "ids", "pos"))

    tot_rows = tot_rows_eq = tot_set = tot_ids = tot_pos = 0
    worst = None

    for j, il in enumerate(moe_layers):
        # [n_expert, n_embd], the reader gives the transposed view of the ggml tensor
        weight = tensor_f32(reader, "blk.%d.ffn_gate_inp.weight" % il)
        bias = tensor_f32(reader, "blk.%d.exp_probs_b.bias" % il)

        if weight.shape != (n_expert, n_embd) or bias.shape != (n_expert,):
            sys.exit("recompute_topk.py: unexpected router shape in layer %d" % il)

        # [n_tokens, n_expert] scores, ggml_mul_mat(gate_inp, cur) is gate_inp^T * cur
        logits = hin[:, j, :] @ weight.T
        probs = np.stack([score(row, gating) for row in logits]) if gating != "none" else logits

        rows_eq = set_eq = ids = pos = 0

        for t in range(n_tokens):
            got = select(probs[t], bias, n_expert_groups, n_group_used, k)
            want = recorded[t, j]

            if np.array_equal(got, want):
                rows_eq += 1
            ids += len(set(got.tolist()) & set(want.tolist()))
            if len(set(got.tolist()) & set(want.tolist())) == k:
                set_eq += 1
            pos += int((got == want).sum())

        rows_eq /= n_tokens
        set_eq /= n_tokens
        ids /= n_tokens * k
        pos /= n_tokens * k

        tot_rows += n_tokens
        tot_rows_eq += rows_eq * n_tokens
        tot_set += set_eq * n_tokens
        tot_ids += ids * n_tokens * k
        tot_pos += pos * n_tokens * k

        if worst is None or ids < worst[1]:
            worst = (il, ids)

        print("%6d %10.4f %10.4f %10.4f %10.4f" % (il, rows_eq, set_eq, ids, pos))

    print()
    print("rows_eq        : the recomputed ids equal the recorded ids in the same order")
    print("set_eq         : the recomputed ids are the same set")
    print("ids            : |A n B| / k")
    print("pos            : fraction of positions that hold the same id")
    print()
    print("overall        : rows_eq %.6f, set_eq %.6f, ids %.6f, pos %.6f over %d rows"
          % (tot_rows_eq / tot_rows, tot_set / tot_rows, tot_ids / (tot_rows * k), tot_pos / (tot_rows * k), tot_rows))
    print("worst layer    : %d with ids %.6f" % worst)

    return 0 if tot_ids / (tot_rows * k) >= 0.99 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
