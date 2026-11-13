#!/usr/bin/env bash
# After b3_series_gpu.sh: the perfect-guess run, plain llama.cpp on the GPU cases (power check before / after), then CPU-only
# (ours and plain, 11 GiB cap and the laptop's free RAM). Not under model-lock.sh (each run takes the lock itself).
set -u
R=/path/to/project
cd $R
L=$R/traces/qwen38/b3/series_rest.log
chk () { local v; v=$(tools/gpu-cap-check.sh 2>&1 | tail -1); echo "$(date +%T) $1: $v" | tee -a $L; echo "$v" | grep -q "OK" || { echo "GPU check not OK - stopping" | tee -a $L; exit 4; }; }
run () { local n=$1; shift; env "$@" tools/model-lock.sh Qwen-b3 bash bench/queue/qwen38/b3_srv.sh $n > /dev/null 2>&1; echo "$(date +%T) $n: $(tr '\n' ' ' < traces/qwen38/b3/$n.out | cut -c1-700)" | tee -a $L; }
OR="--moe-stream-oracle $R/traces/qwen38/rec/06_story --moe-stream-oracle-lead 48"
chk before
run c1_oracle QWEN_CASE=1 MOE_STREAM_TEST_ORACLE_LIST=1 EXTRA="$OR" REQS="detok story512"
run p1_run1 PLAIN=1 QWEN_CASE=1 P10K_TIMEOUT=900 REQS="story96 p10k"
run p1_run2 PLAIN=1 QWEN_CASE=1 REQS="story96"
run p2_run1 PLAIN=1 QWEN_CASE=2 VRAM_HOLD_MIB=8192 MEMCAP=11G P10K_TIMEOUT=900 REQS="story96 p10k"
run p2_run2 PLAIN=1 QWEN_CASE=2 VRAM_HOLD_MIB=8192 MEMCAP=11G REQS="story96"
chk after
# CPU-only (CUDA hidden: no power check needed)
run c3cap_run1 QWEN_CASE=3 MEMCAP=11G REQS="story128 p10k"
run c3cap_run2 QWEN_CASE=3 MEMCAP=11G REQS="story128"
run c3cap_run3 QWEN_CASE=3 MEMCAP=11G REQS="story128"
run c3free_run1 QWEN_CASE=3 REQS="story128 p10k"
run c3free_run2 QWEN_CASE=3 REQS="story128"
run c3free_run3 QWEN_CASE=3 REQS="story128"
run p3cap_run1 PLAIN=1 QWEN_CASE=3 MEMCAP=11G P10K_TIMEOUT=900 REQS="story96 p10k"
run p3cap_run2 PLAIN=1 QWEN_CASE=3 MEMCAP=11G REQS="story96"
run p3free_run1 PLAIN=1 QWEN_CASE=3 P10K_TIMEOUT=900 REQS="story96 p10k"
run p3free_run2 PLAIN=1 QWEN_CASE=3 REQS="story96"
echo "$(date +%T) rest done" | tee -a $L
