#!/usr/bin/env python3
"""The shared maths of make_correction.py and check_correction.py: the ridge correction of the router-trick guess, and the counting.
Built for Ling 3.0 Flash: the model's shape (ND, NE, K below) is built in.

Data per recording (make_correction.py says how to record it): rlogits.f32 [W, Ls, E] = the real router logits of every streamed layer;
<rec>.pi.vec.f32 [W, 43, 2560] = layer L's guess input for the router of layer L+1 = RMSNorm(base_L) * g_{L+1}  (base = residual + shared
expert); logits.f32 = the guess logits; res.bits = the pool state of the target layer at the source's remap.
Per source layer L and distance d (target X = L + d, both streamed):
    x0 = vec_L / g_{L+1}                          the normed base, the same for every d (the gain is a diagonal map)
    f  = W_X (g_X * x0)                            the plain guess logits
    t  = rlogits[X]                                the real logits;   Delta = t - f
    corrected logits = f + x0 B   (foldable: W' = W_X diag(g_X) + B^T)
    B = (G + lam I)^-1 C  from sufficient statistics G = sum x0^T x0 (2560^2), C = sum x0^T Delta (2560 x 512); lam = lam_rel * trace(G)/2560.
"""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from recording import Rec

ND, NE, K = 2560, 512, 8
G8, GU = 8, 4
LAM_REL = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
NF = 4
LEADS = (1, 2)


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def top8(sc):
    n = sc.shape[0]
    g = sc.reshape(n, G8, NE // G8)
    gs = np.sort(g, axis=2)[:, :, -2:].sum(2)
    keep = np.argsort(-gs, axis=1)[:, :GU]
    kg = np.zeros((n, G8), bool)
    np.put_along_axis(kg, keep, True, axis=1)
    m = np.where(np.repeat(kg, NE // G8, axis=1), sc, -np.inf)
    return np.argsort(-m, axis=1)[:, :8]


NC = 12
def counts(sc, T, res, vr, miss, cold):
    """sc [n, E] corrected/plain scores; masks [n, E] bool.  returns [NC] float64:
    0 hit_all 1 T_all 2 hit_need 3 need 4 reads(top8 not in RAM) 5 reads_needed 6 hit_miss 7 miss 8 n
    9 proxy: hit_cold 10 proxy: cold 11 proxy: entries (top-8 entries the layer did NOT pick in the last 32 words) - the old study's read entries; 9 / 11 = its read precision"""
    t8 = top8(sc)
    Gm = np.zeros(sc.shape, bool)
    np.put_along_axis(Gm, t8, True, axis=1)
    need = T & ~res & ~vr
    ent = Gm & ~res & ~vr               # entries the guess would read: not in RAM and not a pick served by the VRAM pool (res.bits never holds the VRAM pool's experts)
    pc = T & ~cold                      # proxy need: picked, and not picked by the layer in the last 32 words
    return np.array([(Gm & T).sum(), T.sum(), (Gm & need).sum(), need.sum(), ent.sum(), (ent & need).sum(),
                     (Gm & miss).sum(), miss.sum(), len(sc), (Gm & pc).sum(), pc.sum(), (Gm & ~cold).sum()], np.float64)


def cold32(T):
    """[W, E] bool: the layer picked the expert in words w-32 .. w-1"""
    cs = np.cumsum(T.astype(np.int32), axis=0)
    prev = np.zeros_like(cs)
    prev[1:] = cs[:-1]
    lo = np.zeros_like(cs)
    if len(cs) > 33:
        lo[33:] = cs[:-33]
    return (prev - lo) > 0


class Vec:
    def __init__(self, d):
        self.pj = json.load(open(d + ".pi.json"))
        self.axis = {L: i for i, L in enumerate(self.pj["axis_layer"]) if L >= 0}
        self.mm = np.memmap(d + ".pi.vec.f32", dtype="<f4", mode="r").reshape(self.pj["n_words"], self.pj["n_layer"], ND)

    def rows(self, layers, W):
        idx = [self.axis[L] for L in layers]
        return np.ascontiguousarray(self.mm[:W][:, idx, :]).astype(np.float32)     # [W, len(layers), ND]


def solve_all(G, C, lam_abs):
    """(G + lam I) B = C"""
    A = G.copy()
    A[np.diag_indices_from(A)] += lam_abs
    return np.linalg.solve(A, C)
