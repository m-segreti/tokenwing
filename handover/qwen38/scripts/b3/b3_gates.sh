#!/usr/bin/env bash
# The frozen Qwen3.8 texts (traces/qwen38/gate_md5.json) on a given build: BIN=<build>/bin bash b3_gates.sh
# Runs through traces/qwen38/qrun.sh (memory cgroup); call it under tools/model-lock.sh. Outputs traces/qwen38/b3/gate_*.
set -u
R=/path/to/project
W=$R/traces/qwen38
export BIN=${BIN:-$R/llama.cpp/build/bin}
cd $W
COMMON="-f $W/prompt_chat.txt -c 4096 --ignore-eos -t 8"
bash $W/qrun.sh b3/gate_cpu_120 CUDA_VISIBLE_DEVICES= -- $COMMON -ngl 0 --no-repack -n 32 --moe-stream-slots 120
bash $W/qrun.sh b3/gate_cpu_120_ub12 CUDA_VISIBLE_DEVICES= -- $COMMON -ngl 0 --no-repack -n 32 -b 2048 -ub 12 --moe-stream-slots 120
bash $W/qrun.sh b3/gate_gpu_150 X=1 -- $COMMON -ngl 99 -n 64 -b 2048 -ub 512 --moe-stream-gpu-layers 0 --moe-stream-slots 150
for n in gate_cpu_120 gate_cpu_120_ub12 gate_gpu_150; do
    echo "$n rc=$(cat $W/b3/$n.rc) md5=$(md5sum < $W/b3/$n.txt | cut -c1-32)"
done
echo "expected: cpu 0a3c44f700c1e2a72ee787d66b72a88d (both), gpu 6417ddef0d82258c961f2a7449ddc811"
