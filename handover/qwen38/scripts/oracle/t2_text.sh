#!/usr/bin/env bash
# The perfect-guess pair, step 2: the recording's decode words as text (llama-server's /detokenize, CPU-only, memory cgroup), checked
# by tokenizing it back the way llama-moe-decode-batch's --force-text does (no special tokens, no BOS), and the prompt's tokens the way
# its -f prompt is tokenized.   t2_text.sh (under tools/model-lock.sh) -> traces/qwen38/rec2/06_story.decode.txt
R=/path/to/project; source $R/bench/queue/qwen38/t2_flags.sh
B=$R/llama.cpp-qwen/build/bin
systemd-run --user --scope -q -p MemoryMax=$(t2_cap) -p MemorySwapMax=0 -- env CUDA_VISIBLE_DEVICES= $B/llama-server -m $T2_MODEL \
  --moe-stream model --load-mode dio -fit off -ngl 0 -c 2048 --moe-stream-slots 16 --host 127.0.0.1 --port 8094 --no-warmup > $Q/rec2/text_srv.log 2>&1 &
SP=$!
for _ in $(seq 1 600); do curl -s http://127.0.0.1:8094/health 2>/dev/null | grep -q '"ok"' && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
python3 - <<'PY'
import json, urllib.request
Q = '/path/to/project/traces/qwen38'
def post(p, b):
    return json.loads(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8094' + p, json.dumps(b).encode(), {'Content-Type': 'application/json'})).read())
m = json.load(open(Q + '/rec2/06_story.json'))
ids = m['token_ids']; npr = m['n_prompt']
dec = ids[npr:]
txt = post('/detokenize', {'tokens': dec})['content']
back = post('/tokenize', {'content': txt, 'add_special': False, 'parse_special': False})['tokens']
same = back == dec
first = next((i for i, (a, b) in enumerate(zip(back, dec)) if a != b), min(len(back), len(dec)))
print('decode words %d, text %d chars; tokenized back %d tokens, equal: %s (first difference at %d)' % (len(dec), len(txt), len(back), same, first))
open(Q + '/rec2/06_story.decode.txt', 'w').write(txt)
p = open(Q + '/rec/06_story.txt').read()
for name, s in (('as read', p), ('without the last newline', p[:-1] if p.endswith('\n') else p)):
    t = post('/tokenize', {'content': s, 'add_special': True, 'parse_special': True})['tokens']
    print('prompt %s: %d tokens, equal to the recording\'s %d: %s' % (name, len(t), npr, t == ids[:npr]))
print('decoded text start:', repr(txt[:300]))
PY
kill $SP 2>/dev/null; wait $SP 2>/dev/null
