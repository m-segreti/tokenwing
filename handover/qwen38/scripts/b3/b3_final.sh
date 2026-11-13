#!/usr/bin/env bash
# Last runs: the perfect-guess pair (all 48 layers streamed, as the story recording: the oracle needs the same streamed layers),
# the preset as written (65k, the printed command, one chat request), the 262k start (case 1, one chat request).
# Not under model-lock.sh (each run takes the lock itself).
set -u
R=/path/to/project
cd $R
L=$R/traces/qwen38/b3/series_final.log
chk () { local v; v=$(tools/gpu-cap-check.sh 2>&1 | tail -1); echo "$(date +%T) $1: $v" | tee -a $L; echo "$v" | grep -q "OK" || { echo "GPU check not OK - stopping" | tee -a $L; exit 4; }; }
run () { local n=$1; shift; env "$@" tools/model-lock.sh Qwen-b3 bash bench/queue/qwen38/b3_srv.sh $n > /dev/null 2>&1; echo "$(date +%T) $n: $(tr '\n' ' ' < traces/qwen38/b3/$n.out | cut -c1-900)" | tee -a $L; }
OR="--moe-stream-oracle $R/traces/qwen38/rec/06_story --moe-stream-oracle-lead 48"
chk before
run c1g0_base   QWEN_CASE=1 EXTRA="--moe-stream-gpu-layers 0" REQS="detok story512"
run c1g0_oracle QWEN_CASE=1 MOE_STREAM_TEST_ORACLE_LIST=1 EXTRA="--moe-stream-gpu-layers 0 $OR" REQS="detok story512"
chk after
run c1_262k QWEN_CASE=1 SIZE=262k REQS="chat story64"
# the preset as written: the printed command itself (port 8080), one chat request
tools/model-lock.sh Qwen-b3 bash -c '
  R=/path/to/project; O=$R/traces/qwen38/b3
  CMD=$(source $R/presets/qwen38.sh; preset_command 65k)
  echo "cmd: $CMD" > $O/asis.log
  eval "exec $CMD" >> $O/asis.log 2>&1 &
  SP=$!
  for _ in $(seq 1 600); do curl -s http://127.0.0.1:8080/health 2>/dev/null | grep -q "\"ok\"" && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
  REQS=chat PORT=8080 python3 $R/bench/queue/qwen38/b3_client.py $O/asis.json > $O/asis.out 2>&1
  kill $SP; wait $SP 2>/dev/null'
echo "$(date +%T) asis: $(tr '\n' ' ' < traces/qwen38/b3/asis.out | cut -c1-600)" | tee -a $L
echo "$(date +%T) final done" | tee -a $L
