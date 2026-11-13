#!/usr/bin/env bash
# Engine recordings for the guess-input recall: all 48 layers streamed, 150 slots pinned, 512 forced words, one run per
# (candidate, prompt). g1 = --moe-stream-pred-shared 0, g2 = 1, g3 = 1 + MOE_STREAM_TEST_GUESS_IN_ATTN=1.
# erec_all.sh N_WORDS PROMPT... ; outputs q10/er_<cand>_<prompt>{,.txt,.log,.rc}
Q=/path/to/project/traces/qwen38
L=/path/to/project/tools/model-lock.sh
NW=$1; shift
for p in "$@"; do
  for c in g1 g2 g3; do
    case $c in
      g1) E=(); A=(--moe-stream-pred-shared 0);;
      g2) E=(); A=(--moe-stream-pred-shared 1);;
      g3) E=(MOE_STREAM_TEST_GUESS_IN_ATTN=1); A=(--moe-stream-pred-shared 1);;
    esac
    N=q10/er_${c}_${p}
    rm -rf $Q/$N
    RUN_TIMEOUT=900 $L Qwen-helper $Q/qrun.sh $N "${E[@]}" -- -ngl 99 -c 4096 -n $NW --ignore-eos -b 2048 -ub 512 -t 8 \
      --moe-stream-gpu-layers 0 --moe-stream-slots 150 -f $Q/rec/$p.txt --moe-stream-record $Q/$N "${A[@]}" -v
  done
done
echo ALLDONE
