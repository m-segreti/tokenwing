#!/usr/bin/env python3
# Tabulates pairs5.sh: per prompt and arm, decode ms/token (eval time line), SSD wait and needed reads per decode token;
# the per-round difference against no guess, mean and spread
import re, glob, statistics as st
def get(f):
    t = open(f, errors='replace').read()
    ms = float(re.findall(r'eval time = +[0-9.]+ ms / +\d+ runs +\( *([0-9.]+) ms per token', t)[-1])
    d = re.findall(r'decode tokens = \d+, misses = ([0-9.]+)/token.*?stall = ([0-9.]+) ms/token', t)[-1]
    return ms, float(d[1]), float(d[0])
for p in ('05_math_word_problem', '03_python_code'):
    R = {a: [get(f) for f in sorted(glob.glob('p5_%s_%s_r*.log' % (a, p)))] for a in ('noguess', 'g1', 'g2')}
    print(p, {a: len(v) for a, v in R.items()})
    for a in ('noguess', 'g1', 'g2'):
        v = R[a]
        print('  %-8s ms/token %s mean %.2f | SSD wait %s mean %.2f | needed reads mean %.2f' % (a, ' '.join('%.2f' % x[0] for x in v),
              st.mean(x[0] for x in v), ' '.join('%.2f' % x[1] for x in v), st.mean(x[1] for x in v), st.mean(x[2] for x in v)))
    for a in ('g1', 'g2'):
        n = min(len(R[a]), len(R['noguess']))
        dm = [R[a][i][0] - R['noguess'][i][0] for i in range(n)]
        ds = [R[a][i][1] - R['noguess'][i][1] for i in range(n)]
        print('  %s - noguess per round: ms/token %s (mean %+.2f, %d of %d lower) | SSD wait %s (mean %+.2f)' % (a, ' '.join('%+.2f' % x for x in dm),
              st.mean(dm), sum(1 for x in dm if x < 0), n, ' '.join('%+.2f' % x for x in ds), st.mean(ds)))
    n = min(len(R['g1']), len(R['g2']))
    dm = [R['g2'][i][0] - R['g1'][i][0] for i in range(n)]
    print('  g2 - g1 per round: ms/token %s (mean %+.2f, %d of %d lower)' % (' '.join('%+.2f' % x for x in dm), st.mean(dm), sum(1 for x in dm if x < 0), n))
