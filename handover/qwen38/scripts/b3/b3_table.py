#!/usr/bin/env python3
"""Per-run numbers and averages from traces/qwen38/b3/<prefix>_run*.json: b3_table.py PREFIX [PREFIX ...]"""
import glob, json, sys

D = '/path/to/project/traces/qwen38/b3/'


def g(d, *ks):
    for k in ks:
        d = (d or {}).get(k) if isinstance(d, dict) else None
    return d


for pre in sys.argv[1:]:
    rows = []
    for f in sorted(glob.glob(D + pre + '_run*.json')):
        j = json.load(open(f))
        st = next((v for k, v in j.items() if k.startswith('story') and 'timings' in v), None)
        pk = j.get('p10k') if 'timings' in j.get('p10k', {}) else None
        r = {'run': f.split('/')[-1][:-5]}
        if st:
            r.update(reply=g(st, 'timings', 'predicted_per_second'), n=g(st, 'timings', 'predicted_n'),
                     p99=g(st, 'gaps', 'p99_ms'), s1=g(st, 'gaps', 'slowest1pct_mean_ms'), mean_gap=g(st, 'gaps', 'mean_ms'),
                     ssd=g(st, 'timings', 'moe_stream', 'reply', 'stall_ms_per_token'), sp=g(st, 'timings', 'prompt_per_second'))
        if pk:
            r.update(p10k=g(pk, 'timings', 'prompt_per_second'), p10k_gib=(g(pk, 'timings', 'moe_stream', 'prompt', 'mib_read') or 0) / 1024,
                     p10k_ssd=g(pk, 'timings', 'moe_stream', 'prompt', 'stall_ms_per_token'), after=g(pk, 'timings', 'predicted_per_second'),
                     after_ssd=g(pk, 'timings', 'moe_stream', 'reply', 'stall_ms_per_token'), p10k_text=pk.get('content', '')[:60])
        rows.append(r)
        print(pre, json.dumps(r))
    def avg(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return (sum(v) / len(v), min(v), max(v), len(v)) if v else None
    for k in ('reply', 'p99', 's1', 'ssd', 'sp', 'p10k', 'p10k_gib', 'p10k_ssd', 'after', 'after_ssd'):
        a = avg(k)
        if a:
            print('  %-9s mean %.2f  min %.2f  max %.2f  (n=%d)' % (k, *a))
