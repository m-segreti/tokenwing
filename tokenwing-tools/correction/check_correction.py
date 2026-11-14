#!/usr/bin/env python3
"""The correction file: the reader check and the corrupted-file test.

  check_correction.py reproduce --file F --model GGUF --eval LIST [--ref ridge4.json --ref-variant ni70] [--out table.json]
      Loads F with the reference loader (correction_file.load_correction), folds it as the engine does (W' = W_X diag(g_X) + Bt on models with
      ffn_norm, or W' = W_X + Bt on hyper-connection models without ffn_norm; a pair absent from the file = the plain guess), scores the
      held-out recordings of LIST ("set kind dir" per line) and prints need recall / miss coverage / read precision.  With --ref it compares
      every count of the reference run (a saved ridge JSON, the variant given) and prints the differences: for a file with ALL pairs they must be ~0.
  check_correction.py selftest --model GGUF [--other-model GGUF2] --work DIR
      Writes a small file and shows that the loader REFUSES: a truncated file, an extended file, a flipped payload byte, a flipped header byte,
      a wrong magic, an unknown format, a pair outside the file, overlapping pairs, a router hash mismatch, another model (and that the good file loads).
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import correction_file as cf
from correction_fit import Vec, cold32, counts, gate_scores, load_model_routers
from recording import Rec


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def metrics(c):
    return dict(rec=pct(c[0], c[1]), need=pct(c[2], c[3]), miss=pct(c[6], c[7]), prec=pct(c[5], c[4]), reads=c[4] / c[8])


def reproduce(a):
    hdr, mats = cf.load_correction(a.file, a.model)
    mdl = load_model_routers(a.model)
    arch, ne, nd = mdl["arch"], mdl["n_expert"], mdl["n_embd"]
    has_hc_norm = mdl["has_hc_norm"]
    Wr, Bi, Gn = mdl["Wr"], mdl["Bi"], mdl["Gn"]
    moe_set = set(mdl["moe_layers"])
    print("loaded %s: %d pairs, %d bytes, model %s (%s), gate: %s" % (
        a.file, len(mats), os.path.getsize(a.file), hdr["model"]["name"], arch,
        hdr.get("gate", {}).get("rule", "-"),
    ))
    Wp = {}
    for (X, d), Bt in mats.items():
        base_w = Wr[X] if has_hc_norm else Wr[X] * Gn[X][None, :]
        Wp[(X, d)] = base_w + np.asarray(Bt)
    evals = [l.split() for l in open(a.eval) if l.strip()]
    res, res_plain = {}, {}
    for s, kind, d in evals:
        r, v = Rec(d), Vec(d, nd)
        W = r.W
        src_layers = [L for L in mdl["moe_layers"][:-1] if L in r.pos]
        for L in src_layers:
            x0_shared = None if has_hc_norm else (v.row(L, W, lead=1) / Gn[L + 1][None, :])
            for dd in (1, 2):
                X = L + dd
                if X not in moe_set or X not in r.pos:
                    continue
                x_d = v.row(L, W, lead=dd, require_lead2=has_hc_norm) if has_hc_norm else x0_shared
                a_t = r.pos[X]
                tr = r.truth[:, a_t, :].astype(np.int64)
                T = np.zeros((W, ne), bool); np.put_along_axis(T, tr, True, axis=-1)
                tres = r.tres[:, a_t, :]
                vr = np.zeros((W, ne), bool); np.put_along_axis(vr, tr, (tres == 3) | (tres == 4), axis=-1)
                miss = np.zeros((W, ne), bool); np.put_along_axis(miss, tr, tres == 0, axis=-1)
                rs = r.resident_bits(dd - 1)[:, r.pos[L], :]
                cold = cold32(T)
                w_plain = Wr[X] if has_hc_norm else Wr[X] * Gn[X][None, :]
                lg_pl = x_d @ w_plain.T
                lg_co = x_d @ Wp[(X, dd)].T if (X, dd) in Wp else lg_pl
                cp = counts(gate_scores(lg_pl, Bi[X], r.gating, r.gating_id), T, rs, vr, miss, cold, r.K, r.G, r.GU)
                cc = counts(gate_scores(lg_co, Bi[X], r.gating, r.gating_id), T, rs, vr, miss, cold, r.K, r.G, r.GU)
                for key in ((s, kind, dd), (s, "all", dd)):
                    res_plain[key] = res_plain.get(key, 0) + cp
                    res[key] = res.get(key, 0) + cc
    out = {}
    for k, c in sorted(res.items()):
        m = metrics(c)
        mp = metrics(res_plain[k])
        out["|".join(map(str, k))] = c.tolist()
        if k[1] == "all":
            print("%s plain lead %d: need recall %.2f | miss coverage %.2f | recall(all) %.2f | read precision %.2f | reads/row %.2f" % (
                k[0], k[2], mp["need"], mp["miss"], mp["rec"], mp["prec"], mp["reads"],
            ))
            print("%s corr  lead %d: need recall %.2f | miss coverage %.2f | recall(all) %.2f | read precision %.2f | reads/row %.2f" % (
                k[0], k[2], m["need"], m["miss"], m["rec"], m["prec"], m["reads"],
            ))
    if a.out:
        json.dump(out, open(a.out, "w"))
    if a.ref:
        ref = json.load(open(a.ref))["res"]
        worst, n = 0.0, 0
        for k, c in res.items():
            rk = json.dumps([k[0], k[1], k[2], a.ref_variant])
            if rk not in ref:
                continue
            d = np.abs(np.array(ref[rk])[:9] - c[:9]).max()
            worst = max(worst, d); n += 1
            if k[1] == "all":
                print("  vs reference %s lead %d: max |count difference| over items 0-8 = %g (of %d rows; need picks %d)" % (k[0], k[2], d, c[8], c[3]))
        print("REPRODUCE: %d keys compared, worst count difference %g" % (n, worst))
        return worst
    return 0.0


def _rewrite_header(src, dst, edit):
    """copy src to dst with the header JSON edited by `edit(dict)` and its sha256 / sizes refreshed (so the refusal comes from the field, not the hash)"""
    b = open(src, "rb").read()
    jl, ps, fs = struct.unpack("<QQQ", b[8:32])
    h = json.loads(b[64:64 + jl])
    edit(h)
    js = json.dumps(h, sort_keys=True).encode()
    assert 64 + len(js) <= ps
    open(dst, "wb").write(cf.MAGIC + struct.pack("<QQQ", len(js), ps, fs) + hashlib.sha256(js).digest() + js + b"\0" * (ps - 64 - len(js)) + b[ps:])


def selftest(a):
    os.makedirs(a.work, exist_ok=True)
    rng = np.random.default_rng(1)
    mdl = load_model_routers(a.model)
    ne, nd = mdl["n_expert"], mdl["n_embd"]
    layers = mdl["moe_layers"][1:] if len(mdl["moe_layers"]) > 2 else mdl["moe_layers"]
    mi = cf.model_info(a.model, layers)
    tgt = layers[min(2, len(layers) - 1)]
    pairs = {
        (tgt, 1): (rng.standard_normal((ne, nd)).astype(np.float32) * 1e-3, {"lam_rel": 0.1}),
        (tgt, 2): (rng.standard_normal((ne, nd)).astype(np.float32) * 1e-3, {"lam_rel": 1.0}),
    }
    good = a.work + "/good.twcorr"
    size = cf.write_correction(good, mi, pairs, {"selftest": True})
    results = []

    def expect(name, fn, needle):
        try:
            fn()
        except cf.CorrectionFileError as e:
            ok = needle in str(e)
            results.append(ok)
            print("%-34s -> refused (%s): %s" % (name, "OK" if ok else "WRONG MESSAGE", e))
            return
        results.append(False)
        print("%-34s -> NOT REFUSED (bug)" % name)

    hdr, m = cf.load_correction(good, a.model)
    ok = len(m) == 2 and np.array_equal(np.asarray(m[(tgt, 2)]), pairs[(tgt, 2)][0])
    results.append(ok)
    print("%-34s -> loaded, %d pairs, matrices equal the written ones: %s (%d bytes)" % ("good file", len(m), ok, size))
    raw = open(good, "rb").read()

    def variant(name, data):
        p = a.work + "/" + name
        open(p, "wb").write(data)
        return p

    expect("truncated (half)", lambda: cf.load_correction(variant("trunc", raw[:size // 2]), a.model), "truncated or extended")
    expect("truncated (header only)", lambda: cf.load_correction(variant("short", raw[:40]), a.model), "shorter than")
    expect("extended (+4096 bytes)", lambda: cf.load_correction(variant("ext", raw + b"\0" * 4096), a.model), "truncated or extended")
    ps = struct.unpack("<Q", raw[16:24])[0]
    b = bytearray(raw); b[ps + min(12345, size - ps - 1)] ^= 1
    expect("one payload byte flipped", lambda: cf.load_correction(variant("flip", bytes(b)), a.model), "payload hash mismatch")
    b = bytearray(raw); b[200] ^= 1
    expect("one header byte flipped", lambda: cf.load_correction(variant("hflip", bytes(b)), a.model), "header hash mismatch")
    expect("wrong magic", lambda: cf.load_correction(variant("magic", b"XXXXXXXX" + raw[8:]), a.model), "bad magic")
    _rewrite_header(good, a.work + "/fmt2", lambda h: h.update(format=2))
    expect("unknown format 2", lambda: cf.load_correction(a.work + "/fmt2", a.model), "format")
    _rewrite_header(good, a.work + "/off", lambda h: h["pairs"][0].update(offset=size))
    expect("pair offset outside the file", lambda: cf.load_correction(a.work + "/off", a.model), "outside the payload")
    _rewrite_header(good, a.work + "/overlap", lambda h: h["pairs"][1].update(offset=h["pairs"][0]["offset"]))
    expect("two pairs overlap", lambda: cf.load_correction(a.work + "/overlap", a.model), "overlap")
    _rewrite_header(good, a.work + "/hash", lambda h: h["model"].update(router_sha256="0" * 64))
    expect("router hash differs (other quant)", lambda: cf.load_correction(a.work + "/hash", a.model), "another model or quant")
    if a.other_model:
        expect("another model (%s)" % os.path.basename(a.other_model), lambda: cf.load_correction(good, a.other_model), "another model")
    print("SELFTEST: %d of %d checks passed" % (sum(results), len(results)))
    return 0 if all(results) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("reproduce")
    p.add_argument("--file", required=True); p.add_argument("--model", required=True); p.add_argument("--eval", required=True)
    p.add_argument("--ref", default=None); p.add_argument("--ref-variant", default="ni70"); p.add_argument("--out", default=None)
    p = sp.add_parser("selftest")
    p.add_argument("--model", required=True); p.add_argument("--other-model", default=None); p.add_argument("--work", required=True)
    a = ap.parse_args()
    sys.exit(selftest(a) if a.cmd == "selftest" else (1 if (reproduce(a) or 0) > 1e-6 else 0))
