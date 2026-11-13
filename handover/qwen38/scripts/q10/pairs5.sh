#!/usr/bin/env bash
# 5 rounds x (no guess, g1, g2) interleaved, math and code prompts, 512 forced words, the GPU command of ssd_ab.sh;
# the GPU power check before and after
Q=/path/to/project/traces/qwen38
M=/path/to/project
L=$M/tools/model-lock.sh
$M/tools/gpu-cap-check.sh || { echo "GPU CHECK FAILED - stop"; exit 1; }
for r in 1 2 3 4 5; do
  for p in 05_math_word_problem 03_python_code; do
    for a in noguess g1 g2; do
      case $a in
        noguess) A=(--moe-stream-prefetch 0);;
        g1) A=(--moe-stream-pred-shared 0);;
        g2) A=(--moe-stream-pred-shared 1);;
      esac
      RUN_TIMEOUT=900 $L Qwen-helper $Q/qrun.sh q10/p5_${a}_${p}_r$r -- -ngl 99 -c 4096 -n 512 --ignore-eos -b 2048 -ub 512 -t 8 \
        --moe-stream-gpu-layers-list 0,1,2,4 --moe-stream-slots 190 -f $Q/rec/$p.txt "${A[@]}" -v
    done
  done
done
$M/tools/gpu-cap-check.sh || echo "GPU CHECK AFTER FAILED - the runs are void"
echo ALLDONE
