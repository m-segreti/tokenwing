#!/usr/bin/env bash
# CPU path, the frozen gate's command (120 slots, --no-repack, 32 words): all same-output settings on must give the gate's text;
# then + the rounding-level CPU settings (q8-tiled, cpu-fa). Shared build unless BIN is set. Run under tools/model-lock.sh.
set -u
R=/path/to/project
W=$R/traces/qwen38
export BIN=${BIN:-$R/llama.cpp/build/bin}
cd $W
COMMON="-f $W/prompt_chat.txt -c 4096 --ignore-eos -t 8 -ngl 0 --no-repack -n 32 --moe-stream-slots 120"
SAME="--moe-stream-overlap 1 --moe-stream-pred-shared 1 --moe-stream-chunk-prefetch 512 --moe-stream-io-wake chunk --moe-stream-promote 1
 --moe-stream-guess-cap 2 --moe-stream-policy decay --moe-stream-prompt-end recency --moe-stream-cpu-skip-zero 1 --moe-stream-cpu-fuse 1
 --moe-stream-cpu-dispatch 1 --moe-stream-plan 1 --moe-stream-drop-stale 1 --moe-stream-needed-early 1 --moe-stream-lookahead 2
 --moe-stream-guess-combined 1 --moe-stream-proj-ready 1 --moe-stream-wordstart 2 --moe-stream-protect-named 1 --moe-stream-guess-inflight 4
 --moe-stream-cpu-topk 1 --moe-stream-cpu-mmid-merge 1 --moe-stream-wave-plan 1 --moe-stream-prompt-defer 1
 --moe-stream-cpu-weighted-sum 1 --moe-stream-cpu-small-ops 1 --moe-stream-cpu-state-inplace 1"
bash $W/qrun.sh b3/cpu_same_on CUDA_VISIBLE_DEVICES= -- $COMMON $SAME
bash $W/qrun.sh b3/cpu_same_on_round CUDA_VISIBLE_DEVICES= -- $COMMON $SAME --moe-stream-cpu-q8-tiled 1 --moe-stream-cpu-fa 1
for n in cpu_same_on cpu_same_on_round; do
    echo "$n rc=$(cat $W/b3/$n.rc) md5=$(md5sum < $W/b3/$n.txt | cut -c1-32)"
done
echo "gate (all off): 0a3c44f700c1e2a72ee787d66b72a88d"
