#!/usr/bin/env bash
# SSD wait and needed reads with no guess / g1 / g2 / g2 + lookahead 2, the GPU command (whole layers 0,1,2,4 pinned,
# 190 slots pinned), 512 forced words, arms interleaved per prompt; the GPU power check before and after
Q=/path/to/project/traces/qwen38
M=/path/to/project
L=$M/tools/model-lock.sh
$M/tools/gpu-cap-check.sh || { echo "GPU CHECK FAILED - stop"; exit 1; }
for p in 01_casual_chat 05_math_word_problem; do
  for a in noguess g1 g2 g2la2; do
    case $a in
      noguess) A=(--moe-stream-prefetch 0);;
      g1) A=(--moe-stream-pred-shared 0);;
      g2) A=(--moe-stream-pred-shared 1);;
      g2la2) A=(--moe-stream-pred-shared 1 --moe-stream-plan 1 --moe-stream-drop-stale 1 --moe-stream-lookahead 2);;
    esac
    RUN_TIMEOUT=900 $L Qwen-helper $Q/qrun.sh q10/ab_${a}_$p -- -ngl 99 -c 4096 -n 512 --ignore-eos -b 2048 -ub 512 -t 8 \
      --moe-stream-gpu-layers-list 0,1,2,4 --moe-stream-slots 190 -f $Q/rec/$p.txt "${A[@]}" -v
  done
done
$M/tools/gpu-cap-check.sh || echo "GPU CHECK AFTER FAILED - the runs are void"
echo ALLDONE
