#!/usr/bin/env bash
# The guess never changes the maths: the CPU gate (frozen without a guess) with the guess off / g1 / g2, Flint with and
# without pred-shared against the shared build, Ling-tiny check.py
Q=/path/to/project/traces/qwen38
M=/path/to/project
L=$M/tools/model-lock.sh
G=(-ngl 0 --no-repack -c 4096 -n 32 --ignore-eos -t 8 --moe-stream-slots 120 -f $Q/prompt_chat.txt)
$L Qwen-helper $Q/qrun.sh q10/cpu_noguess CUDA_VISIBLE_DEVICES= -- "${G[@]}" --moe-stream-prefetch 0
$L Qwen-helper $Q/qrun.sh q10/cpu_g1 CUDA_VISIBLE_DEVICES= -- "${G[@]}"
$L Qwen-helper $Q/qrun.sh q10/cpu_g2 CUDA_VISIBLE_DEVICES= -- "${G[@]}" --moe-stream-pred-shared 1
$L Qwen-helper $Q/qrun.sh q10/cpu_g2_la2 CUDA_VISIBLE_DEVICES= -- "${G[@]}" --moe-stream-pred-shared 1 --moe-stream-plan 1 --moe-stream-drop-stale 1 --moe-stream-lookahead 2
$L Qwen-helper $Q/flint_ab2.sh $M/llama.cpp/build/bin q10/flint_shared_ps1 --moe-stream-pred-shared 1
$L Qwen-helper $Q/flint_ab2.sh $M/llama.cpp-qwen/build/bin q10/flint_q10_ps1 --moe-stream-pred-shared 1
$L Qwen-helper $Q/flint_ab2.sh $M/llama.cpp-qwen/build/bin q10/flint_q10 
$L Qwen-helper python3 $M/llama.cpp-qwen/tokenwing-tools/testing/check.py --model /path/to/models/Ling-3.0-tiny-MXFP4_MOE.gguf \
  --build $M/llama.cpp-qwen/build --no-ctest --no-build -j 8 --work $Q/q10/check-tiny > $Q/q10/check-tiny.log 2>&1; echo "check.py rc=$?"
cd $Q/q10 && md5sum cpu_noguess.txt cpu_g1.txt cpu_g2.txt cpu_g2_la2.txt flint_shared_ps1.txt flint_q10_ps1.txt flint_q10.txt
echo ALLDONE
