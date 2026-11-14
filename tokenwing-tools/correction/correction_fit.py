#!/usr/bin/env python3
"""The shared maths of make_correction.py and check_correction.py: the ridge correction of the router-trick guess, and the counting.
Reads architecture, MoE layer list, expert count, hidden size, router dtype (F32/BF16/F16), optional bias, and norm kind (ffn_norm or hc_ffn_norm)
from the model GGUF and recordings.

Data per recording (make_correction.py says how to record it):
  - rlogits.f32 [W, Ls, E] = the real router logits of every streamed layer
  - <rec>.pi.vec.f32 [W, n_layer, n_embd] = layer L's lead-1 guess input
  - <rec>.pi.vec2.f32 [W, n_layer, n_embd] = layer L's lead-2 guess input when distinct (hyper-connection models without ffn_norm)
  - logits.f32 = the guess logits; res.bits = the pool state of the target layer at the source's remap.

Per source layer L and distance d (target X = L + d, both streamed):
  - models with ffn_norm:
      x0 = vec_L / g_{L+1}                        the normed base, shared across leads
      f  = W_X (g_X * x0)                         the plain guess logits
      corrected logits = f + x0 B                 (foldable: W' = W_X diag(g_X) + B^T)
  - hyper-connection models without ffn_norm (hc_ffn_norm):
      x_d = vec_L (d = 1) or vec2_L (d = 2)       the per-target hyper-connection FFN mix output
      f   = W_X x_d                               the plain guess logits
      corrected logits = f + x_d B                (foldable: W' = W_X + B^T)
"""
import json
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "gguf-py"))

import correction_file as cf
from gguf import GGUFReader, GGMLQuantizationType

ND, NE, K = 2560, 512, 8
G8, GU = 8, 4
LAM_REL = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
NF = 4
LEADS = (1, 2)
NC = 12


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax_scores(x):
    xm = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(xm)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def sqrt_softplus_scores(x):
    sp = np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -80.0, 20.0))))
    return np.sqrt(np.maximum(sp, 0.0))


def gate_scores(lg, bias=None, gating="sigmoid", gating_id=0):
    if gating == "softmax":
        sc = softmax_scores(lg)
    elif gating_id == 3 or gating == "sqrt_softplus":
        sc = sqrt_softplus_scores(lg)
    else:
        sc = sig(lg)
    return sc + bias if bias is not None else sc


def top_k_select(sc, k=K, n_group=G8, n_group_used=GU):
    n, e_cnt = sc.shape
    if n_group > 1 and 0 < n_group_used < n_group:
        g = sc.reshape(n, n_group, e_cnt // n_group)
        gs = np.sort(g, axis=2)[:, :, -2:].sum(2)
        keep = np.argsort(-gs, axis=1)[:, :n_group_used]
        kg = np.zeros((n, n_group), bool)
        np.put_along_axis(kg, keep, True, axis=1)
        m = np.where(np.repeat(kg, e_cnt // n_group, axis=1), sc, -np.inf)
        return np.argsort(-m, axis=1)[:, :k]
    return np.argsort(-sc, axis=1)[:, :k]


def top8(sc):
    return top_k_select(sc, K, G8, GU)


def counts(sc, T, res, vr, miss, cold, k=K, n_group=G8, n_group_used=GU):
    """sc [n, E] corrected/plain scores; masks [n, E] bool.  returns [NC] float64:
    0 hit_all 1 T_all 2 hit_need 3 need 4 reads(topk not in RAM) 5 reads_needed 6 hit_miss 7 miss 8 n
    9 proxy: hit_cold 10 proxy: cold 11 proxy: entries (top-k entries the layer did NOT pick in the last 32 words)"""
    tk = top_k_select(sc, k, n_group, n_group_used)
    Gm = np.zeros(sc.shape, bool)
    np.put_along_axis(Gm, tk, True, axis=1)
    need = T & ~res & ~vr
    ent = Gm & ~res & ~vr
    pc = T & ~cold
    return np.array([
        (Gm & T).sum(), T.sum(), (Gm & need).sum(), need.sum(),
        ent.sum(), (ent & need).sum(), (Gm & miss).sum(), miss.sum(),
        len(sc), (Gm & pc).sum(), pc.sum(), (Gm & ~cold).sum(),
    ], np.float64)


def cold32(T):
    """[W, E] bool: the layer picked the expert in words w-32 .. w-1"""
    cs = np.cumsum(T.astype(np.int32), axis=0)
    prev = np.zeros_like(cs)
    prev[1:] = cs[:-1]
    lo = np.zeros_like(cs)
    if len(cs) > 33:
        lo[33:] = cs[:-33]
    return (prev - lo) > 0


def load_router_f32(t):
    raw = np.frombuffer(t.data, dtype=np.uint8)
    sh = tuple(int(x) for x in t.shape[::-1])
    if t.tensor_type == GGMLQuantizationType.F32:
        return np.frombuffer(raw, dtype=np.float32).reshape(sh).copy()
    if t.tensor_type == GGMLQuantizationType.BF16:
        u16 = np.frombuffer(raw, dtype=np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32).reshape(sh).copy()
    if t.tensor_type == GGMLQuantizationType.F16:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(sh).copy()
    raise ValueError("unsupported router tensor type %s" % t.tensor_type)


def load_model_routers(gguf_path):
    rd = GGUFReader(gguf_path)
    tn = {x.name: x for x in rd.tensors}
    light = cf.model_info_light(gguf_path)
    arch = light["arch"]
    n_layer, ne, nd = int(light["n_layer"]), int(light["n_expert"]), int(light["n_embd"])
    moe_layers = [X for X in range(n_layer) if ("blk.%d.ffn_gate_inp.weight" % X) in tn]
    if not moe_layers:
        raise ValueError("no MoE router tensors found in %s" % gguf_path)
    has_hc_norm = ("blk.%d.ffn_norm.weight" % moe_layers[0] not in tn) and ("blk.%d.hc_ffn_norm.weight" % moe_layers[0] in tn)
    Wr, Bi, Gn = {}, {}, {}
    for X in moe_layers:
        Wr[X] = load_router_f32(tn["blk.%d.ffn_gate_inp.weight" % X])
        b_nm = "blk.%d.exp_probs_b.bias" % X
        Bi[X] = np.asarray(tn[b_nm].data, np.float32).copy() if b_nm in tn else None
        if not has_hc_norm:
            Gn[X] = np.asarray(tn["blk.%d.ffn_norm.weight" % X].data, np.float32).copy()
    return dict(arch=arch, n_layer=n_layer, n_expert=ne, n_embd=nd, moe_layers=moe_layers,
                has_hc_norm=has_hc_norm, Wr=Wr, Bi=Bi, Gn=Gn)


class Vec:
    def __init__(self, d, n_embd=None):
        self.d = d
        self.pj = json.load(open(d + ".pi.json"))
        self.axis = {L: i for i, L in enumerate(self.pj["axis_layer"]) if L >= 0}
        vec_p = d + ".pi.vec.f32"
        if n_embd is None:
            n_embd = int(self.pj.get("n_embd") or (os.path.getsize(vec_p) // (4 * self.pj["n_words"] * self.pj["n_layer"])))
        self.n_embd = int(n_embd)
        shape = (self.pj["n_words"], self.pj["n_layer"], self.n_embd)
        self.mm = np.memmap(vec_p, dtype="<f4", mode="r").reshape(shape)
        p2 = d + ".pi.vec2.f32"
        self.mm2 = np.memmap(p2, dtype="<f4", mode="r").reshape(shape) if os.path.exists(p2) else None

    def rows(self, layers, W, lead=1, require_lead2=False):
        if lead == 2 and self.mm2 is None:
            if require_lead2:
                raise FileNotFoundError("%s.pi.vec2.f32 is required for lead-2 hyper-connection guess inputs" % self.d)
            src = self.mm
        else:
            src = self.mm2 if lead == 2 else self.mm
        idx = [self.axis[L] for L in layers]
        return np.ascontiguousarray(src[:W][:, idx, :]).astype(np.float32)

    def row(self, L, W, lead=1, require_lead2=False):
        return self.rows([L], W, lead=lead, require_lead2=require_lead2)[:, 0, :]


def solve_all(G, C, lam_abs):
    """(G + lam I) B = C"""
    A = G.copy()
    A[np.diag_indices_from(A)] += lam_abs
    return np.linalg.solve(A, C)
