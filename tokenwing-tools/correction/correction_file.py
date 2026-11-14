#!/usr/bin/env python3
"""The correction file: the trained ridge correction of the router-trick guess, one matrix per (target layer, lead).
The model's shape (n_expert, n_embd) is read from the model.  This module is the reference WRITER and the reference LOADER (what the engine must do at load).

    write_correction(path, model_info, pairs, extra)   pairs: {(target, lead): Bt float32 [n_expert, n_embd]} plus per-pair meta
    load_correction(path, gguf_path)                   -> (header dict, {(target, lead): Bt memmap}); raises CorrectionFileError

The guess of target layer X made `lead` layers early from source layer L = X - lead:
  - models with ffn_norm:
      x0 = RMSNorm(residual + shared expert of L)            (the normed base WITHOUT any gain)
      plain logits     = W_X (g_X * x0)                      (g_X = blk.X.ffn_norm.weight)
      corrected logits = plain + x0 @ B = (W_X diag(g_X) + Bt) x0          Bt = B^T  [n_expert, n_embd]
  - hyper-connection models without ffn_norm (hc_ffn_norm):
      x_d = guess_in(X)                                      (the hyper-connection FFN mixer output for target X)
      plain logits     = W_X x_d
      corrected logits = plain + x_d @ B = (W_X + Bt) x_d                  Bt = B^T  [n_expert, n_embd]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "gguf-py"))
import hashlib, json, os, struct
import numpy as np

MAGIC = b"TWCORR01"
FORMAT = 1
ALIGN = 4096


class CorrectionFileError(Exception):
    pass


def _gguf(path):
    from gguf import GGUFReader
    return GGUFReader(path)


def _field(rd, key, default=None):
    f = rd.fields.get(key)
    if f is None:
        return default
    try:
        v = f.parts[f.data[0]]
        if f.types and f.types[0].name == "STRING":
            return bytes(v).decode()
        return v.tolist()[0] if v.size == 1 else v.tolist()
    except Exception:
        return default


def model_info(gguf_path, layers):
    """The identity of a model for the file: architecture, sizes, and a sha256 over the router tensors of `layers` (ffn_gate_inp.weight,
    optional exp_probs_b.bias, and ffn_norm.weight or hc_ffn_norm.weight, as stored, in this order, for X in `layers`)."""
    rd = _gguf(gguf_path)
    tn = {t.name: t for t in rd.tensors}
    arch = _field(rd, "general.architecture", "?")
    h = hashlib.sha256()
    shape = None
    has_hc_norm = None
    for X in layers:
        layer_hc = ("blk.%d.ffn_norm.weight" % X not in tn) and ("blk.%d.hc_ffn_norm.weight" % X in tn)
        if has_hc_norm is None:
            has_hc_norm = layer_hc
        elif layer_hc != has_hc_norm:
            raise CorrectionFileError("layer %d norm kind differs from layer %d" % (X, layers[0]))
        norm_suffix = "hc_ffn_norm.weight" if has_hc_norm else "ffn_norm.weight"
        for nm in ("blk.%d.ffn_gate_inp.weight" % X, "blk.%d.exp_probs_b.bias" % X, "blk.%d.%s" % (X, norm_suffix)):
            if nm not in tn:
                if nm.endswith("exp_probs_b.bias") and arch != "bailingmoe3":
                    continue   # hashed whenever the model has it (as the engine does), required only for bailingmoe3
                raise CorrectionFileError("model has no tensor %s" % nm)
            t = tn[nm]
            allowed = ("F32", "BF16", "F16") if nm.endswith("ffn_gate_inp.weight") else ("F32",)
            if t.tensor_type.name not in allowed:
                raise CorrectionFileError("tensor %s is %s, not %s" % (nm, t.tensor_type.name, "/".join(allowed)))
            h.update(np.ascontiguousarray(np.asarray(t.data)).tobytes())
            if nm.endswith("ffn_gate_inp.weight"):
                shape = [int(x) for x in t.shape][::-1]            # [n_expert, n_embd]
    norm_label = "hc_ffn_norm.weight" if has_hc_norm else "ffn_norm.weight"
    return {"arch": arch, "name": _field(rd, "general.name", ""), "n_layer": _field(rd, "%s.block_count" % arch),
            "n_expert": shape[0], "n_embd": shape[1], "layers": list(layers), "router_sha256": h.hexdigest(),
            "hash_of": "sha256 over, for X in layers: blk.X.ffn_gate_inp.weight, optional blk.X.exp_probs_b.bias, blk.X.%s (raw bytes)" % norm_label}


def write_correction(path, minfo, pairs, extra):
    """pairs: {(target, lead): (Bt float32 [n_expert, n_embd], meta dict)}.  Writes atomically (tmp + rename)."""
    keys = sorted(pairs)
    ne, nd = int(minfo["n_expert"]), int(minfo["n_embd"])
    nbytes = ne * nd * 4
    fold_desc = (
        "W' = W_X + Bt  (applied to guess_in(X) = the hyper-connection FFN mixer output); lead d: source layer = X - d"
        if "hc_ffn_norm.weight" in minfo.get("hash_of", "")
        else "W' = W_X diag(g_X) + Bt  (applied to x0 = the normed base without gain); lead d: source layer = X - d"
    )
    # header first with provisional offsets to learn its size; the JSON length is fixed by padding the offsets to a constant width
    def build(payload_start):
        off = payload_start
        plist = []
        for (X, d) in keys:
            plist.append({"target": X, "lead": d, "source": X - d, "offset": off, "nbytes": nbytes, "shape": [ne, nd], "dtype": "f32", **pairs[(X, d)][1]})
            off += nbytes
        return plist, off
    payload_start = ALIGN * 16
    while True:
        plist, size = build(payload_start)
        hdr = {"format": FORMAT, "model": minfo, "fold": fold_desc,
               "pairs": plist, "payload_start": payload_start, "file_size": size, **extra}
        # payload hash needs the payload: hash the pairs in order (all pairs are contiguous, each nbytes, no padding between)
        h = hashlib.sha256()
        for k in keys:
            h.update(np.ascontiguousarray(pairs[k][0], dtype="<f4").tobytes())
        hdr["payload_sha256"] = h.hexdigest()
        js = json.dumps(hdr, sort_keys=True).encode()
        if 64 + len(js) <= payload_start:
            break
        payload_start += ALIGN * 16
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(MAGIC + struct.pack("<QQQ", len(js), payload_start, size) + hashlib.sha256(js).digest() + js)
        f.write(b"\0" * (payload_start - f.tell()))
        for k in keys:
            f.write(np.ascontiguousarray(pairs[k][0], dtype="<f4").tobytes())
        assert f.tell() == size, (f.tell(), size)
    os.replace(tmp, path)
    return size


def load_correction(path, gguf_path, verify_payload=True):
    """The loader contract.  Refuses (CorrectionFileError, message says why): short file, wrong magic, size differs from the header's
    (truncated / extended), header hash, unknown format, another model (arch, sizes, router hash), a pair table that does not fit the file
    or the model, payload hash.  Returns (header, {(target, lead): memmap [n_expert, n_embd] float32 read-only})."""
    sz = os.path.getsize(path)
    if sz < 64:
        raise CorrectionFileError("file is %d bytes: shorter than the 64-byte preamble" % sz)
    with open(path, "rb") as f:
        pre = f.read(64)
        if pre[:8] != MAGIC:
            raise CorrectionFileError("bad magic %r (not a TokenWing correction file, or another format version)" % pre[:8])
        jl, ps, fs = struct.unpack("<QQQ", pre[8:32])
        if fs != sz:
            raise CorrectionFileError("file is %d bytes but the header says %d (truncated or extended)" % (sz, fs))
        if jl > ps or ps > sz or ps % ALIGN:
            raise CorrectionFileError("header sizes inconsistent (json %d, payload_start %d, file %d)" % (jl, ps, sz))
        js = f.read(jl)
    if hashlib.sha256(js).digest() != pre[32:64]:
        raise CorrectionFileError("header hash mismatch (header bytes changed)")
    try:
        hdr = json.loads(js)
    except Exception as e:
        raise CorrectionFileError("header is not valid JSON: %s" % e)
    if hdr.get("format") != FORMAT:
        raise CorrectionFileError("format %r, this loader reads %d" % (hdr.get("format"), FORMAT))
    mi = hdr["model"]
    cur = model_info_light(gguf_path)
    for k in ("arch", "n_layer", "n_expert", "n_embd"):
        if mi.get(k) != cur[k]:
            raise CorrectionFileError("another model: %s is %r in the file, %r in the model" % (k, mi.get(k), cur[k]))
    real = model_info(gguf_path, mi["layers"])
    if real["router_sha256"] != mi["router_sha256"]:
        raise CorrectionFileError("another model or quant: the router tensors' sha256 differs (file %s.., model %s..)" % (mi["router_sha256"][:12], real["router_sha256"][:12]))
    seen = []
    ne, nd = int(mi["n_expert"]), int(mi["n_embd"])
    nb = ne * nd * 4
    for p in hdr["pairs"]:
        X, d, off = p["target"], p["lead"], p["offset"]
        if p["shape"] != [ne, nd] or p["nbytes"] != nb or p["dtype"] != "f32":
            raise CorrectionFileError("pair (%s, %s): shape %s / %s bytes do not fit the model" % (X, d, p["shape"], p["nbytes"]))
        if off < ps or off % ALIGN or off + nb > sz:
            raise CorrectionFileError("pair (%s, %s): offset %d outside the payload" % (X, d, off))
        if p["source"] != X - d or X not in mi["layers"]:
            raise CorrectionFileError("pair (%s, %s): source layer / target layer not valid" % (X, d))
        seen.append((off, off + nb))
    seen.sort()
    for a, b in zip(seen, seen[1:]):
        if a[1] > b[0]:
            raise CorrectionFileError("two pairs overlap in the file")
    if verify_payload:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            f.seek(ps)
            while True:
                b = f.read(1 << 24)
                if not b:
                    break
                h.update(b)
        if h.hexdigest() != hdr["payload_sha256"]:
            raise CorrectionFileError("payload hash mismatch (matrix bytes changed)")
    out = {}
    for p in hdr["pairs"]:
        out[(p["target"], p["lead"])] = np.memmap(path, dtype="<f4", mode="r", offset=p["offset"], shape=(ne, nd))
    return hdr, out


def model_info_light(gguf_path):
    rd = _gguf(gguf_path)
    arch = _field(rd, "general.architecture", "?")
    t = next(t for t in rd.tensors if t.name.endswith("ffn_gate_inp.weight"))
    sh = [int(x) for x in t.shape][::-1]
    return {"arch": arch, "n_layer": _field(rd, "%s.block_count" % arch), "n_expert": sh[0], "n_embd": sh[1]}
