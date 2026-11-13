#!/usr/bin/env python3
"""Requests for b3_srv.sh: REQS (env) in order, results to the JSON file in argv[1], a summary line per request on stdout."""
import json, os, sys, time, urllib.request

Q = '/path/to/project/traces/qwen38'
URL = 'http://127.0.0.1:%s' % os.environ.get('PORT', '8093')


def post(path, body, timeout=7200):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)


def completion(prompt, n, stream=False, cache=False, timeout=7200):
    body = {'prompt': prompt, 'n_predict': n, 'temperature': 0, 'top_k': 1, 'ignore_eos': True, 'cache_prompt': cache,
            'stream': stream}
    t0 = time.time()
    if not stream:
        r = json.loads(post('/completion', body, timeout).read())
        r['wall_s'] = time.time() - t0
        return r
    times, text, last = [], [], None
    with post('/completion', body) as resp:
        for raw in resp:
            line = raw.decode('utf-8', 'replace').strip()
            if not line.startswith('data: '):
                continue
            ev = json.loads(line[6:])
            now = time.time()
            if ev.get('content'):
                times.append(now)
                text.append(ev['content'])
            if ev.get('stop'):
                last = ev
    r = last or {}
    r['content'] = ''.join(text)
    r['wall_s'] = time.time() - t0
    gaps = [(b - a) * 1000.0 for a, b in zip(times, times[1:])]
    if gaps:
        s = sorted(gaps)
        n1 = max(1, len(s) // 100)
        r['gaps'] = {'n': len(s), 'mean_ms': sum(s) / len(s), 'p50_ms': s[len(s) // 2], 'p99_ms': s[min(len(s) - 1, int(0.99 * len(s)))],
                     'slowest1pct_mean_ms': sum(s[-n1:]) / n1, 'max_ms': s[-1]}
    return r


def summary(k, v):
    t = v.get('timings', {})
    ms = t.get('moe_stream', {}) or {}
    pr, rp = ms.get('prompt', {}) or {}, ms.get('reply', {}) or {}
    g = v.get('gaps', {})
    line = '%s prompt_n %s prompt %.1f tok/s | predicted_n %s reply %.2f tok/s' % (
        k, t.get('prompt_n'), t.get('prompt_per_second', 0), t.get('predicted_n'), t.get('predicted_per_second', 0))
    if g:
        line += ' | gaps mean %.1f p99 %.1f slowest1%% %.1f max %.1f ms' % (g['mean_ms'], g['p99_ms'], g['slowest1pct_mean_ms'], g['max_ms'])
    if ms:
        line += ' | prompt SSD wait %.2f ms/tok, %.1f GiB read | reply SSD wait %.2f ms/tok, %.1f MiB/tok' % (
            pr.get('stall_ms_per_token', 0), pr.get('mib_read', 0) / 1024, rp.get('stall_ms_per_token', 0), rp.get('mib_per_token', 0))
    print(line)
    print('   text:', repr(v.get('content', '')[:200]))
    sys.stdout.flush()


out = {}
story = open(Q + '/rec/06_story.txt').read()
p8k = open(Q + '/rec/p8k.txt').read()
for k in os.environ.get('REQS', 'story512 p10k').split():
    try:
        if k.startswith('story'):
            out[k] = completion(story, int(k[5:]), stream=True)
        elif k.startswith('p10k'):
            out[k] = completion(p8k, int(k[4:]) if len(k) > 4 else 32, timeout=int(os.environ.get('P10K_TIMEOUT', '7200')))
        elif k == 'session':
            a = completion(p8k, 400, cache=True)
            out['session1'] = a
            summary('session1', a)
            follow = p8k + a.get('content', '') + '<|im_end|>\n<|im_start|>user\nNow give a short summary of your answer in three sentences.<|im_end|>\n<|im_start|>assistant\n'
            out['session2'] = completion(follow, 200, cache=True)
            summary('session2', out['session2'])
            continue
        elif k == 'detok':   # the story recording's decode words as text (to see where a run's text leaves the recording)
            rec = json.load(open(Q + '/rec/06_story.json'))
            ids = rec['token_ids'][rec['n_prompt']:]
            out[k] = json.loads(post('/detokenize', {'tokens': ids}).read())
            print('detok: recording text', repr(out[k].get('content', '')[:200]))
            continue
        elif k == 'chat':
            body = {'messages': [{'role': 'user', 'content': 'What is 17 times 23? Answer briefly.'}], 'max_tokens': 400, 'temperature': 0}
            r = json.loads(post('/v1/chat/completions', body).read())
            out[k] = r
            m = r['choices'][0]['message']
            print('chat reasoning_content:', repr((m.get('reasoning_content') or '')[:300]))
            print('chat content:', repr(m.get('content', '')[:300]))
            continue
        summary(k, out[k])
    except Exception as e:
        print(k, 'FAILED', repr(e))
        out[k] = {'error': repr(e)}
    json.dump(out, open(sys.argv[1], 'w'), indent=1)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
