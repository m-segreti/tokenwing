#!/usr/bin/env python3
"""Loader and helpers for the engine's own recordings (--moe-stream-record). Counting only; nothing here touches the engine."""
import json
import numpy as np

EV_DT = np.dtype([("t", "<i8"), ("type", "u1"), ("flags", "u1"), ("layer", "<i2"), ("expert", "<i2"), ("slot", "<i2"),
                  ("word", "<i4"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4")])


class Rec:
    def __init__(self, d):
        self.d = d
        m = self.meta = json.load(open(d + "/meta.json"))
        self.W, self.Ls, self.E, self.K = m["n_words"], m["n_streamed"], m["n_expert"], m["k"]
        self.layers = list(m["layers"])
        self.pos = {L: a for a, L in enumerate(self.layers)}
        r = m["router"]
        self.G, self.GU = r["n_expert_groups"], r["n_group_used"]
        b = r["bias"]
        self.bias = np.array(json.loads(b) if isinstance(b, str) else b, dtype=np.float32)  # [Ls, E]
        W, Ls, E, K = self.W, self.Ls, self.E, self.K
        self.truth = np.fromfile(d + "/truth.i32", np.int32).reshape(W, Ls, K)
        self.tres = np.fromfile(d + "/truth_res.u8", np.uint8).reshape(W, Ls, K)
        self.tw = np.fromfile(d + "/truth_w.f32", np.float32).reshape(W, Ls, K)
        self.logits = np.memmap(d + "/logits.f32", np.float32, "r", shape=(W, Ls, 4, E))
        self.res = np.fromfile(d + "/res.bits", np.uint8).reshape(W, Ls, 4, E // 8)
        self.t_layer = np.fromfile(d + "/t_layer.i64", np.int64).reshape(W, Ls)
        self.t_layer_end = np.fromfile(d + "/t_layer_end.i64", np.int64).reshape(W, Ls)
        self.t_word = np.fromfile(d + "/t_word.i64", np.int64).reshape(W, 2)
        self.words = np.fromfile(d + "/words.i64", np.int64)
        self._ev = None

    @property
    def events(self):
        if self._ev is None:
            raw = open(self.d + "/events.bin", "rb").read()
            n = int(np.frombuffer(raw[16:24], np.int64)[0])
            self._ev = np.frombuffer(raw[64:64 + 32 * n], EV_DT)
        return self._ev

    def truth_mask(self):
        T = np.zeros((self.W, self.Ls, self.E), bool)
        np.put_along_axis(T, self.truth.astype(np.int64), True, axis=-1)
        return T

    def resident_bits(self, lead_idx):
        """[W, Ls, E] bool: res.bits at axis a, lead index = the pool of layer layers[a]+d as seen from layers[a]"""
        return np.unpackbits(self.res[:, :, lead_idx, :], axis=-1, bitorder="little").astype(bool)

    def scores(self, di, ok_out=False):
        """sigmoid(logit)+bias for lead index di; rows with NaN logits (source not streamed) -> ok False"""
        lg = np.asarray(self.logits[:, :, di, :], dtype=np.float32)
        ok = ~np.isnan(lg).any(-1)
        sc = 1.0 / (1.0 + np.exp(-np.nan_to_num(lg))) + self.bias[None, :, :]
        return (sc, ok) if ok_out else sc

    def ranking(self, sc, n=None):
        """expert order of the router's own selection for scores sc [..., E]: the experts of the best GU groups (group
        score = sum of the top 2), best first; the other experts follow, never inside the 'kept' ranks.
        returns order [..., n] (n default E) and kept [..., E] bool"""
        G, GU, E = self.G, self.GU, self.E
        g = sc.reshape(*sc.shape[:-1], G, E // G)
        gs = np.sort(g, axis=-1)[..., -2:].sum(-1)
        keep = np.argsort(-gs, axis=-1)[..., :GU]
        kg = np.zeros(gs.shape, bool)
        np.put_along_axis(kg, keep, True, axis=-1)
        kept = np.repeat(kg, E // G, axis=-1)
        masked = np.where(kept, sc, -np.inf)
        order = np.argsort(-masked, axis=-1)
        if n is not None:
            order = order[..., :n]
        return order, kept


def merge_pairs():
    pass


def guess_pairs(rec, d):
    """for lead d: list of (a_target, a_source) over streamed targets whose source layer X-d is streamed (the pairs with real
    logits and a residency snapshot taken at the source's own remap)"""
    out = []
    for a, X in enumerate(rec.layers):
        s = X - d
        if s in rec.pos:
            out.append((a, rec.pos[s]))
    return out
