#!/usr/bin/env python3
"""Recall of the engine's guesses from its own recordings (--moe-stream-record): logits.f32 [W, Ls, 4, E] = the guess logits FOR
streamed layer a made d layers before it (lead d, as predict/t91_analyze.py reads them); truth.i32 [W, Ls, k] = the real picks.
All layers streamed (pos = layer). per_layer is indexed by the source layer a - d, as count_q9.py's.
usage: erec_recall.py OUT.json CAND=DIR[,DIR...] ...
"""
import json, os, sys
import numpy as np

out = sys.argv[1]
res = {}
for arg in sys.argv[2:]:
    cand, dirs = arg.split('=', 1)
    acc = {d: {w: [0.0, 0] for w in (10, 16, 20)} for d in (1, 2, 3)}
    per_layer = {d: np.zeros((2, 48)) for d in (1, 2, 3)}
    words = 0
    for dd in dirs.split(','):
        m = json.load(open(os.path.join(dd, 'meta.json')))
        W, Ls, E, k = m['n_words'], m['n_streamed'], m['n_expert'], m['k']
        assert m['layers'] == list(range(Ls)), 'all layers must be streamed'
        lg = np.fromfile(os.path.join(dd, 'logits.f32'), np.float32)[:W * Ls * 4 * E].reshape(W, Ls, 4, E)
        tr = np.fromfile(os.path.join(dd, 'truth.i32'), np.int32).reshape(W, Ls, k)
        words += W
        for d in (1, 2, 3):
            for ls in range(Ls - d):   # source layer ls, target ls + d
                x = lg[:, ls + d, d - 1, :]
                ok = ~np.isnan(x).any(axis=1)
                if not ok.any():
                    continue
                order = np.argsort(-x[ok], axis=1)
                t = tr[ok, ls + d, :]
                for w in (10, 16, 20):
                    hit = (order[:, :w, None] == t[:, None, :]).any(axis=1).sum(axis=1) / k
                    acc[d][w][0] += hit.sum(); acc[d][w][1] += hit.size
                    if w == 10:
                        per_layer[d][0, ls] += hit.sum(); per_layer[d][1, ls] += hit.size
    r = {'words': words}
    for d in (1, 2, 3):
        pl = per_layer[d][0] / np.maximum(per_layer[d][1], 1)
        v = per_layer[d][1] > 0
        r['d%d' % d] = {('w%d' % w): acc[d][w][0] / max(acc[d][w][1], 1) for w in (10, 16, 20)}
        r['d%d' % d]['L0_11'] = float(pl[0:12][v[0:12]].mean())
        r['d%d' % d]['L12_35'] = float(pl[12:36][v[12:36]].mean())
        r['d%d' % d]['L36_47'] = float(pl[36:48][v[36:48]].mean())
        r['d%d' % d]['per_layer'] = [round(float(x), 3) for x in pl]
    res[cand] = r
    print(cand, 'words', words, ' | '.join('d%d w10 %.4f w16 %.4f w20 %.4f (L0-11 %.3f L12-35 %.3f L36-47 %.3f)' % (
        d, r['d%d' % d]['w10'], r['d%d' % d]['w16'], r['d%d' % d]['w20'], r['d%d' % d]['L0_11'], r['d%d' % d]['L12_35'],
        r['d%d' % d]['L36_47']) for d in (1, 2, 3)))
json.dump(res, open(out, 'w'), indent=1)
