#!/usr/bin/env bash
# One pass with Flint's case-1 preset flags that apply to qwen4exp (llama-server, 65k, q8_0 KV): a story (400 forced words) and
# the 10k-token prompt (32 words after it). Run through tools/model-lock.sh. NAME = output prefix
set -u
Q=/path/to/project/traces/qwen38
B=/path/to/project/llama.cpp-qwen/build/bin
O=$Q/preset
NAME=${1:-preset}
AVAIL_K=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
CAP_M=$(( AVAIL_K/1024 - 1536 )); [ $CAP_M -gt 24576 ] && CAP_M=24576
echo "cap ${CAP_M}M (MemAvailable $((AVAIL_K/1024))M)" > $O/$NAME.log
COMMON="--moe-stream-ram-free 0.5GiB --moe-stream-overlap 1 --moe-stream-pred-shared 1 --moe-stream-chunk-prefetch 512
 --moe-stream-io-wake chunk --moe-stream-promote 1 --moe-stream-guess-cap 2 --moe-stream-policy decay --moe-stream-prompt-end recency
 --moe-stream-cpu-skip-zero 1 --moe-stream-cpu-x4 1 --moe-stream-cpu-fuse 1 --moe-stream-cpu-dispatch 1 --moe-stream-plan 1
 --moe-stream-drop-stale 1 --moe-stream-needed-early 1 --moe-stream-lookahead 2 --moe-stream-guess-combined 1 --moe-stream-proj-ready 1
 --moe-stream-wordstart 2 --moe-stream-protect-named 1 --moe-stream-guess-inflight 4 --moe-stream-cpu-down-scale 1 --moe-stream-cpu-topk 1
 --moe-stream-cpu-q8-tiled 1 --moe-stream-cpu-fa 1 --moe-stream-cpu-mmid-merge 1 --moe-stream-cpu-iqk 8
 --moe-stream-layer-major 1 --moe-stream-wave-plan 1 --moe-stream-read-ahead 1 --moe-stream-idle-load 2 --moe-stream-window 1 --moe-stream-prompt-defer 1"
GPU="--moe-stream-gpu-waves 512 --moe-stream-gpu-share-q8 1 --moe-stream-gpu-skip-zero 1 --moe-stream-async-h2d 1 --moe-stream-gpu-wave-skip-zero 1
 --moe-stream-prompt-split 1 --moe-stream-vram-free 200MiB --moe-stream-vram-experts-file $Q/qwen38.vram-experts.json --moe-stream-vram-follow 1
 --moe-stream-vram-parallel 1 --moe-stream-vram-copyback 1 --moe-stream-vram-end-fill 1 --moe-stream-copy-ahead 1 --moe-stream-vram-lend-prompt 1 --backend-sampling"
echo "flags: $COMMON $GPU" >> $O/$NAME.log
systemd-run --user --scope -q -p MemoryMax=${CAP_M}M -p MemorySwapMax=0 -- \
  env GGML_CUDA_COPY_KERNEL=65536 GGML_CUDA_POOL1D_ROW=1 GGML_CUDA_FUSE_AMAX=1 GGML_CUDA_MOE_MARK_IDS=1 GGML_CUDA_FUSE_NORM_QUANT=1 \
  $B/llama-server -m $Q/Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf --load-mode dio -fit off \
  -ngl 99 -c 65536 -ctk q8_0 -ctv q8_0 -fa on -np 1 -t 8 -b 2048 -ub 2048 --cache-ram 0 --ctx-checkpoints 2 --moe-stream model \
  $COMMON $GPU --host 127.0.0.1 --port 8092 --no-warmup -v >> $O/$NAME.log 2>&1 &
SP=$!
echo $SP > $O/$NAME.pid
for i in $(seq 1 600); do
  curl -s http://127.0.0.1:8092/health 2>/dev/null | grep -q '"ok"' && break
  kill -0 $SP 2>/dev/null || { echo "server died"; break; }
  sleep 1
done
python3 - "$O/$NAME.json" <<'PY'
import json, sys, urllib.request
Q='/path/to/project/traces/qwen38'
def req(prompt, n):
    body=json.dumps({'prompt': prompt, 'n_predict': n, 'temperature': 0, 'top_k': 1, 'ignore_eos': True, 'cache_prompt': False}).encode()
    r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8092/completion', body, {'Content-Type': 'application/json'}), timeout=3000)
    return json.loads(r.read())
out={}
out['story400']=req(open(Q+'/rec/06_story.txt').read(), 400)
out['p10k']=req(open(Q+'/rec/p8k.txt').read(), 32)
for k,v in out.items():
    t=v.get('timings',{}); ms=t.get('moe_stream',{})
    print(k, 'prompt_n', t.get('prompt_n'), 'prompt %.1f tok/s' % t.get('prompt_per_second',0), 'predicted_n', t.get('predicted_n'),
          'reply %.2f tok/s' % t.get('predicted_per_second',0), 'prompt SSD wait %.2f ms/token' % ms.get('prompt',{}).get('stall_ms_per_token',0),
          'reply SSD wait %.2f ms/token' % ms.get('reply',{}).get('stall_ms_per_token',0))
    print('   text:', repr(v.get('content','')[:160]))
json.dump(out, open(sys.argv[1],'w'), indent=1)
PY
kill $SP; wait $SP 2>/dev/null
echo "preset done"
