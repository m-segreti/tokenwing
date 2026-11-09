#!/usr/bin/env python3
"""Trains the ridge correction of the router-trick guess from recordings and writes the correction file.
Built for Ling 3.0 Flash: the model's shape is built in (correction_file.py, correction_fit.py).

Training data = recordings made with `--moe-stream-record DIR --moe-stream-pred-input DIR.pi --moe-stream-pred-input-every 1` and
MOE_STREAM_TEST_REC_RLOGITS=1: per recording `rlogits.f32` (the layers' real router logits = the target)
and `<DIR>.pi.vec.f32` (the guess input of the source layers).  Fit per (target X, lead d), source L = X - d:
    x0 = vec_L / g_{L+1};  f = W_X (g_X x0);  Delta = rlogits_X - f;  B = (G + lam I)^-1 C,  G = sum x0^T x0,  C = sum x0^T Delta
    lam = lam_rel * trace(G)/n_embd;  lam_rel from {1e-4 .. 10} by 4-fold cross-validation BY RECORDING (position in the list mod 4), the
    held-out squared logit error in closed form.  Variant: no intercept (exactly foldable: W' = W_X diag(g_X) + B^T).
Quality gate: a (target, lead) pair is WRITTEN only if, on held-out data, the corrected guess finds at least as
many of the needed experts (picked, not in RAM at guess time, not served by the VRAM pool) as the plain guess - pooled over the held-out folds AND
for every kind of text (math, code, chat, json, story).  Held-out data = each recording scored with the fit that left its fold out (cross-fitting),
so the data that picks lam and the data that gates are the training recordings, never the evaluation recordings.  Pairs that fail are listed in the
header with the counts and are simply absent (the engine then uses the plain guess for them).

usage:  make_correction.py --model GGUF --train LIST --kinds prompts_kinds.json --out FILE [--out-all FILE_ALL] [--work DIR]
        LIST: one recording directory per line (the order defines the folds).  Resumable: exit code 3 = a piece (env PIECE_SECONDS, default 600) ended,
        run again with the same arguments.  --out-all also writes a file with ALL pairs (no gate) for the reproduction check.
        env CORRECTION_TEST_LAYERS=5,6 (test only: source layers).  The tool is CPU only."""
import sys, os, json, time, argparse, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import correction_file as cf
from correction_fit import top8, counts, cold32, sig, Vec, solve_all, ND, NE, LAM_REL, NF, LEADS
from recording import Rec
from gguf import GGUFReader

KINDS = ("math", "code", "chat", "json", "story")


def rec_masks(r, L, X, dd, W):
    a_t = r.pos[X]
    tr = r.truth[:, a_t, :].astype(np.int64)
    T = np.zeros((W, NE), bool); np.put_along_axis(T, tr, True, axis=-1)
    tres = r.tres[:, a_t, :]
    vr = np.zeros((W, NE), bool); np.put_along_axis(vr, tr, (tres == 3) | (tres == 4), axis=-1)
    miss = np.zeros((W, NE), bool); np.put_along_axis(miss, tr, tres == 0, axis=-1)
    rs = r.resident_bits(dd - 1)[:, r.pos[L], :]
    return T, rs, vr, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--kinds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-all", default=None)
    ap.add_argument("--pairs", default=None, help="a file of \"X,d\" lines: write only these pairs (every one must have passed the gate) - a lean file from the trained matrices")
    ap.add_argument("--work", default=None)
    ap.add_argument("--group", type=int, default=2)
    a = ap.parse_args()
    work = a.work or a.out + ".work"
    os.makedirs(work, exist_ok=True)
    train = [l.strip() for l in open(a.train) if l.strip()]
    kinds = json.load(open(a.kinds))
    kind_of = [kinds[os.path.basename(d).split("_", 1)[1]] for d in train]
    ntr = len(train)
    rd = GGUFReader(a.model)
    tn = {x.name: x for x in rd.tensors}
    Wr, Gn = {}, {}
    for X in range(5, 42):
        w = tn["blk.%d.ffn_gate_inp.weight" % X]
        Wr[X] = np.asarray(w.data, np.float32).reshape(tuple(w.shape[::-1]))
        Gn[X] = np.asarray(tn["blk.%d.ffn_norm.weight" % X].data, np.float32)
    assert min(float(np.abs(Gn[X]).min()) for X in Gn) > 1e-6
    LAYERS = [int(x) for x in os.environ["CORRECTION_TEST_LAYERS"].split(",")] if os.environ.get("CORRECTION_TEST_LAYERS") else list(range(5, 41))
    targets = sorted({L + d for L in LAYERS for d in LEADS if L + d <= 41})
    groups = [LAYERS[i:i + a.group] for i in range(0, len(LAYERS), a.group)]
    state_p = work + "/state.json"
    st = json.load(open(state_p)) if os.path.exists(state_p) else {"done": [], "pairs": {}}
    trv = [Vec(d) for d in train]
    recs = [Rec(d) for d in train]
    t_start = time.time()
    budget = float(os.environ.get("PIECE_SECONDS", "600"))
    for gi, Lg in enumerate(groups):
        if gi in st["done"]:
            continue
        if time.time() - t_start > budget:
            print("PIECE_END next group", gi, flush=True)
            sys.exit(3)
        tg = time.time()
        acc = {L: dict(G=np.zeros((NF, ND, ND)), n=np.zeros(NF), C={d: np.zeros((NF, ND, NE)) for d in LEADS if L + d <= 41},
                       S2={d: np.zeros(NF) for d in LEADS if L + d <= 41}) for L in Lg}
        for i, d_ in enumerate(train):
            meta = json.load(open(d_ + "/meta.json"))
            Ls_pos = {L: p for p, L in enumerate(meta["layers"])}
            W = meta["n_words"]
            rl = np.memmap(d_ + "/rlogits.f32", dtype="<f4", mode="r").reshape(W, meta["n_streamed"], NE)
            vecs = trv[i].rows(Lg, W)
            f_ = i % NF
            for li, L in enumerate(Lg):
                x0 = vecs[:, li, :] / Gn[L + 1][None, :]
                A = acc[L]
                A["G"][f_] += (x0.T @ x0).astype(np.float64)
                A["n"][f_] += W
                for dd in LEADS:
                    X = L + dd
                    if X > 41 or X not in Ls_pos:
                        continue
                    f = (x0 * Gn[X][None, :]) @ Wr[X].T
                    De = np.asarray(rl[:, Ls_pos[X], :], np.float32) - f
                    A["C"][dd][f_] += (x0.T @ De).astype(np.float64)
                    A["S2"][dd][f_] += float((De.astype(np.float64) ** 2).sum())
        # ---- lambda by cross-validation, final B on all recordings, and the cross-fitted B per fold for the gate
        Bfold, Bfin, lamsel = {}, {}, {}
        for L in Lg:
            A = acc[L]
            dds = [dd for dd in LEADS if dd in A["C"]]
            Gt = A["G"].sum(0)
            Ct = np.concatenate([A["C"][dd].sum(0) for dd in dds], 1)
            sse = np.zeros((len(dds), len(LAM_REL)))
            for f in range(NF):
                Gtr = Gt - A["G"][f]
                Ctr = Ct - np.concatenate([A["C"][dd][f] for dd in dds], 1)
                sc_l = np.trace(Gtr) / ND
                for li_, lr in enumerate(LAM_REL):
                    Ball = solve_all(Gtr, Ctr, lr * sc_l)
                    for k_, dd in enumerate(dds):
                        B = Ball[:, k_ * NE:(k_ + 1) * NE]
                        Gf, Cf = A["G"][f], A["C"][dd][f]
                        sse[k_, li_] += A["S2"][dd][f] + float(((Gf @ B) * B).sum()) - 2 * float((B * Cf).sum())
            chosen = [int(np.argmin(sse[k_])) for k_ in range(len(dds))]
            for li_ in sorted(set(chosen)):
                ks = [k_ for k_ in range(len(dds)) if chosen[k_] == li_]
                Ball = solve_all(Gt, Ct, LAM_REL[li_] * np.trace(Gt) / ND)
                for k_ in ks:
                    Bfin[(L, dds[k_])] = Ball[:, k_ * NE:(k_ + 1) * NE].astype(np.float32)
                    lamsel[(L, dds[k_])] = LAM_REL[li_]
                for f in range(NF):
                    Gtr = Gt - A["G"][f]
                    Ctr = np.concatenate([A["C"][dds[k_]].sum(0) - A["C"][dds[k_]][f] for k_ in ks], 1)
                    Bf = solve_all(Gtr, Ctr, LAM_REL[li_] * np.trace(Gtr) / ND)
                    for j, k_ in enumerate(ks):
                        Bfold[(L, dds[k_], f)] = Bf[:, j * NE:(j + 1) * NE].astype(np.float32)
        t_fit = time.time() - tg
        # ---- the gate's held-out scoring: recording i with the fit that left fold i % NF out
        gate = {}
        for i, d_ in enumerate(train):
            r, W, f = recs[i], recs[i].W, i % NF
            vecs = trv[i].rows(Lg, W)
            for li, L in enumerate(Lg):
                x0 = vecs[:, li, :] / Gn[L + 1][None, :]
                for dd in LEADS:
                    X = L + dd
                    if X > 41 or X not in r.pos or L not in r.pos:
                        continue
                    T, rs, vr, miss = rec_masks(r, L, X, dd, W)
                    cold = cold32(T)
                    fpl = (x0 * Gn[X][None, :]) @ Wr[X].T
                    bias = np.asarray(tn["blk.%d.exp_probs_b.bias" % X].data, np.float32)
                    cp = counts(sig(fpl) + bias, T, rs, vr, miss, cold)
                    cc = counts(sig(fpl + x0 @ Bfold[(L, dd, f)]) + bias, T, rs, vr, miss, cold)
                    for key in ("all", kind_of[i]):
                        g = gate.setdefault((L, dd, key), [np.zeros(12), np.zeros(12)])
                        g[0] += cp; g[1] += cc
        for L in Lg:
            for dd in LEADS:
                X = L + dd
                if (L, dd) not in Bfin:
                    continue
                per = {}
                ok = True
                for key in ("all",) + KINDS:
                    g = gate.get((L, dd, key))
                    if g is None:
                        continue
                    per[key] = {"need_plain": int(g[0][2]), "need_corr": int(g[1][2]), "need_total": int(g[0][3]),
                                "miss_plain": int(g[0][6]), "miss_corr": int(g[1][6]), "miss_total": int(g[0][7])}
                    if g[1][2] < g[0][2]:
                        per[key]["fails"] = True
                        ok = False
                np.save("%s/Bt_%d_%d.npy" % (work, X, dd), np.ascontiguousarray(Bfin[(L, dd)].T))
                st["pairs"]["%d,%d" % (X, dd)] = {"lam_rel": lamsel[(L, dd)], "gate_pass": ok,
                                                  "gate_pass_pooled_only": not per["all"].get("fails", False), "held_out": per}
        st["done"].append(gi)
        json.dump(st, open(state_p, "w"))
        print("group %d layers %s: fit %.0f s, total %.0f s" % (gi, Lg, t_fit, time.time() - tg), flush=True)
    # ---- assemble
    minfo = cf.model_info(a.model, targets)
    info = {"training": {"n_recordings": ntr, "kinds": dict(collections.Counter(kind_of)), "list": os.path.abspath(a.train),
                         "folds": "recording position mod %d" % NF, "lam_grid": list(LAM_REL), "variant": "no intercept (foldable)"},
            "gate": {"rule": "pair written only if held-out need hits (corrected) >= (plain), pooled and for every kind; margin 0 points",
                     "metric": "need = picked, not in RAM at guess time (res.bits), not served by the VRAM pool (truth_res 3/4)",
                     "data": "cross-fitted on the training recordings (each scored with the fit that left its fold out)"}}
    want_pairs = None
    if a.pairs:
        want_pairs = {tuple(map(int, l.split(","))) for l in open(a.pairs) if l.strip()}
        bad = sorted(k for k in want_pairs if not st["pairs"].get("%d,%d" % k, {}).get("gate_pass"))
        if bad:
            sys.exit("make_correction: these requested pairs are not in the gated set: %s" % bad)
    allp, shp = {}, {}
    for k, v in st["pairs"].items():
        X, dd = map(int, k.split(","))
        Bt = np.load("%s/Bt_%d_%d.npy" % (work, X, dd))
        meta = {"lam_rel": v["lam_rel"], "gate_pass": v["gate_pass"], "held_out": v["held_out"]}
        allp[(X, dd)] = (Bt, meta)
        if v["gate_pass"] and (want_pairs is None or (X, dd) in want_pairs):
            shp[(X, dd)] = (Bt, meta)
    left = {k: v["held_out"] for k, v in st["pairs"].items() if not v["gate_pass"]}
    info["gate"]["left_out"] = sorted(left)
    if want_pairs is not None:
        info["lean"] = {"pairs_requested": sorted("%d,%d" % k for k in want_pairs), "source": os.path.abspath(a.pairs), "note": "a subset of the gated pairs, the same trained matrices"}
    info["gate"]["n_pairs"] = len(allp); info["gate"]["n_written"] = len(shp)
    info["gate"]["n_pass_pooled_only"] = sum(1 for v in st["pairs"].values() if v["gate_pass_pooled_only"])
    sz = cf.write_correction(a.out, minfo, shp, info)
    print("wrote %s: %d of %d pairs, %d bytes (%.3f GiB)" % (a.out, len(shp), len(allp), sz, sz / 2 ** 30))
    if a.out_all:
        info2 = dict(info); info2["gate"] = dict(info["gate"], rule="NOT GATED: every pair written (reproduction check only)")
        sz2 = cf.write_correction(a.out_all, minfo, allp, info2)
        print("wrote %s: %d pairs, %d bytes" % (a.out_all, len(allp), sz2))


if __name__ == "__main__":
    main()
