#!/usr/bin/env python3
"""Counts on llama-moe-trace recordings of Qwen3.8 (Q9): expert-use spread per layer, the recall of guess input g1 (layer L's own
router input with the router of layer L+d) at d = 1, 2, 3, and missing experts per layer per token for an LRU pool of S slots.

usage: count_q9.py MODEL.gguf OUT.json PREFIX [PREFIX ...]
Counting on real recordings only (no timing model).
"""
import json, sys, os
import numpy as np
sys.path.insert(0, '/path/to/project/llama.cpp-qwen/tokenwing-tools/experts')
from expert_index import GGUFHeader

model, out, prefixes = sys.argv[1], sys.argv[2], sys.argv[3:]
h = GGUFHeader(model)
h.parse()
tens = {t.name: t for t in h.tensors}

def tensor_f32(name):
    t = tens[name]
    assert t.ggml_type == 0, (name, t.ggml_type)   # F32
    with open(model, 'rb') as f:
        f.seek(h.data_offset + t.rel_offset)
        a = np.frombuffer(f.read(t.n_bytes), dtype=np.float32)
    return a.reshape(list(reversed(t.dims)))

recs = []
for p in prefixes:
    m = json.load(open(p + '.json'))
    nl = len(m['moe_layers'])
    ids = np.fromfile(p + '.bin', dtype=np.int32).reshape(m['n_tokens'], nl, m['k'])
    hin = np.fromfile(p + '.hin.bin', dtype=np.float16).reshape(m['n_tokens'], nl, m['n_embd'])
    recs.append((os.path.basename(p), m, ids, hin))
m0 = recs[0][1]
L = len(m0['moe_layers']); E = m0['n_expert']; K = m0['k']
W = [tensor_f32('blk.%d.ffn_gate_inp.weight' % l) for l in m0['moe_layers']]   # [E, n_embd]
print('routers', W[0].shape, 'layers', L, 'experts', E, 'k', K)

res = {'recordings': {}, 'widths': [10, 16, 20]}
widths = res['widths']

# sanity: the router of layer L on the recorded input of layer L gives the recorded picks (d = 0)
# recall[d][w] summed over decode tokens, per layer
rec_sum = {d: {w: np.zeros(L) for w in widths} for d in (0, 1, 2, 3)}
rec_n = {d: np.zeros(L) for d in (0, 1, 2, 3)}
cnt_dec = np.zeros((L, E), np.int64)
for name, m, ids, hin in recs:
    npr = m['n_prompt']
    dec_ids = ids[npr:]; dec_h = hin[npr:].astype(np.float32)
    T = dec_ids.shape[0]
    per = {}
    for l in range(L):
        cnt_dec[l] += np.bincount(dec_ids[:, l, :].ravel(), minlength=E)
        for d in (0, 1, 2, 3):
            if l + d >= L:
                continue
            lg = dec_h[:, l, :] @ W[l + d].T                     # [T, E]
            order = np.argsort(-lg, axis=1)
            truth = dec_ids[:, l + d, :]
            for w in widths:
                top = order[:, :w]
                hit = (top[:, :, None] == truth[:, None, :]).any(axis=1).sum()
                rec_sum[d][w][l] += hit / K
            rec_n[d][l] += T
    res['recordings'][name] = {'n_prompt': npr, 'n_decode': int(T)}
    print(name, 'prompt', npr, 'decode', T)

recall = {}
for d in (0, 1, 2, 3):
    recall[d] = {}
    for w in widths:
        v = rec_sum[d][w] / np.maximum(rec_n[d], 1)
        valid = rec_n[d] > 0
        recall[d][w] = {'mean': float((rec_sum[d][w][valid]).sum() / rec_n[d][valid].sum()),
                        'layers_0_11': float(v[0:12].mean()), 'layers_12_35': float(v[12:36][valid[12:36]].mean()),
                        'layers_36_47': float(v[36:48][valid[36:48]].mean()),
                        'min': float(v[valid].min()), 'min_layer': int(np.argmin(np.where(valid, v, 9))),
                        'per_layer': [round(float(x), 4) for x in v]}
res['g1_recall_decode'] = recall

# spread of expert use per layer (decode picks)
spread = []
for l in range(L):
    c = np.sort(cnt_dec[l])[::-1]; tot = c.sum(); cs = np.cumsum(c) / max(tot, 1)
    spread.append({'used': int((c > 0).sum()), 'top10pct_share': float(cs[E // 10 - 1]), 'top25pct_share': float(cs[E // 4 - 1]),
                   'top50pct_share': float(cs[E // 2 - 1]), 'n_for_50': int(np.searchsorted(cs, 0.5) + 1),
                   'n_for_80': int(np.searchsorted(cs, 0.8) + 1), 'n_for_90': int(np.searchsorted(cs, 0.9) + 1)})
res['spread_decode'] = spread

# missing experts per layer per decode token: one LRU pool of S slots per layer, warmed by the prompt (all rows in order),
# and of the misses the share the d = 1 guess of width 10 (g1) named
def lru_misses(S):
    miss = np.zeros(L); n_dec = 0; covered = np.zeros(L); miss_tok = []
    for name, m, ids, hin in recs:
        npr = m['n_prompt']
        dec_h = hin[npr:].astype(np.float32)
        g1 = {}
        for l in range(1, L):
            lg = dec_h[:, l - 1, :] @ W[l].T
            g1[l] = np.argsort(-lg, axis=1)[:, :10]
        for l in range(L):
            stamp = {}; clock = 0
            for t in range(ids.shape[0]):
                clock += 1
                row = ids[t, l]
                if t >= npr:
                    mset = [e for e in row if e not in stamp]
                    miss[l] += len(mset)
                    if l >= 1 and mset:
                        gs = set(g1[l][t - npr].tolist())
                        covered[l] += sum(1 for e in mset if e in gs)
                for e in row:
                    stamp[int(e)] = clock
                while len(stamp) > S:
                    v = min(stamp, key=stamp.get)
                    del stamp[v]
        n_dec += ids.shape[0] - npr
    return miss / n_dec, covered / n_dec

res['lru'] = {}
for S in (120, 138, 159, 190, 205):
    mpl, cov = lru_misses(S)
    res['lru'][S] = {'misses_per_layer_per_token_mean': float(mpl.mean()), 'misses_per_token_all_layers': float(mpl.sum()),
                     'named_by_g1_d1_w10_share': float(cov[1:].sum() / max(mpl[1:].sum(), 1e-9)),
                     'per_layer': [round(float(x), 3) for x in mpl]}
    print('S', S, 'misses/layer/token %.3f, /token %.1f, g1 named %.3f' % (mpl.mean(), mpl.sum(), res['lru'][S]['named_by_g1_d1_w10_share']))

for d in (0, 1, 2, 3):
    print('g1 d=%d' % d, ' '.join('w%d %.4f' % (w, recall[d][w]['mean']) for w in widths),
          '| L0-11 %.3f L12-35 %.3f L36-47 %.3f min %.3f (layer %d)' % (recall[d][10]['layers_0_11'], recall[d][10]['layers_12_35'],
          recall[d][10]['layers_36_47'], recall[d][10]['min'], recall[d][10]['min_layer']))
sp = res['spread_decode']
print('spread: used %d..%d of %d; top 10%% share %.3f..%.3f (mean %.3f); n for 80%% %d..%d (mean %.0f)' % (
    min(s['used'] for s in sp), max(s['used'] for s in sp), E, min(s['top10pct_share'] for s in sp), max(s['top10pct_share'] for s in sp),
    np.mean([s['top10pct_share'] for s in sp]), min(s['n_for_80'] for s in sp), max(s['n_for_80'] for s in sp), np.mean([s['n_for_80'] for s in sp])))
json.dump(res, open(out, 'w'), indent=1)
