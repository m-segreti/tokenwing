#!/usr/bin/env bash
# The perfect-guess pair, step 1: a new story recording (llama-moe-trace) with llama.cpp-qwen and the pair's own placement and flags.
#   t2_rec.sh   (under tools/model-lock.sh) -> traces/qwen38/rec2/06_story.{bin,json,log}
R=/path/to/project; source $R/bench/queue/qwen38/t2_flags.sh
B=$R/llama.cpp-qwen/build/bin
rm -f $T2_MODEL.usage.json
echo "cap $(t2_cap)" > $Q/rec2/06_story.log
env $T2_ENV systemd-run --user --scope -q -p MemoryMax=$(t2_cap) -p MemorySwapMax=0 -- \
  $B/llama-moe-trace -m $T2_MODEL $T2_FLAGS -f $Q/rec/06_story.txt -n 600 -o $Q/rec2/06_story >> $Q/rec2/06_story.log 2>&1
echo "rc=$?" | tee $Q/rec2/06_story.rc
