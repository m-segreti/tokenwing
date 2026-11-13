#!/usr/bin/env bash
# GPU series: power check, case 1 x5, case 2 (8 GB free, 11 GiB cap) x3, the long session, power check.
# Do NOT run it under model-lock.sh: each run takes the lock itself (b3_srv.sh under tools/model-lock.sh).
set -u
R=/path/to/project
cd $R
L=$R/traces/qwen38/b3/series_gpu.log
chk () { local v; v=$(tools/gpu-cap-check.sh 2>&1 | tail -1); echo "$(date +%T) $1: $v" | tee -a $L; echo "$v" | grep -q "OK" || { echo "GPU check not OK - stopping" | tee -a $L; exit 4; }; }
run () { local n=$1; shift; env "$@" tools/model-lock.sh Qwen-b3 bash bench/queue/qwen38/b3_srv.sh $n > /dev/null 2>&1; echo "$(date +%T) $n: $(tr '\n' ' ' < traces/qwen38/b3/$n.out | cut -c1-600)" | tee -a $L; }
chk before
for i in 1 2 3 4 5; do run c1_run$i QWEN_CASE=1 REQS="story512 p10k"; done
run c1_session QWEN_CASE=1 REQS="session"
chk middle
for i in 1 2 3; do run c2_run$i QWEN_CASE=2 VRAM_HOLD_MIB=8192 MEMCAP=11G REQS="story512 p10k"; done
chk after
