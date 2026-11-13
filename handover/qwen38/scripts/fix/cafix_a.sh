#!/usr/bin/env bash
# The copy-ahead fix's checks (a) + (b): the 10k prompt through the case-1 preset in llama-server, built from llama.cpp-qwen,
# copy-ahead 3 / 1 (two sets) with --backend-sampling on and off, and 2 (one set). Each run under the model lock; the live usage
# table deleted before each so that the set sizes are comparable.   Usage: cafix_a.sh  (detached; log traces/qwen38/b3/cafix_a.log)
R=/path/to/project
Q=$R/traces/qwen38
export PRESET_BIN=$R/llama.cpp-qwen/build/bin REQS="${REQS:-p10k300}"
one () {   # name, copy-ahead value, backend sampling 1|0
    rm -f $Q/Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf.usage.json
    local drop=""; [ "$3" = 0 ] && drop="--backend-sampling"
    echo "=== $1 (copy-ahead $2, backend sampling $3) $(date +%T)"
    DROP="$drop" EXTRA="--moe-stream-copy-ahead $2 ${MORE:-}" RUN_TIMEOUT=2400 $R/tools/model-lock.sh Qwen-fix bash $R/bench/queue/qwen38/b3_srv.sh $1
    grep -h "copy-ahead [0-9]: one set\|two copy sets\|one copy set\|non-finite\|output check" $Q/b3/$1.log | cut -c1-220
    grep -o "copy calls ([0-9]* of them zeros[^)]*)" $Q/b3/$1.log
}
RUNS=${RUNS:-"fix_ca3_bs 3 1,fix_ca3_nobs 3 0,fix_ca1_bs 1 1,fix_ca1_nobs 1 0,fix_ca2_bs 2 1"}
IFS=, read -ra LIST <<< "$RUNS"
for a in "${LIST[@]}"; do one $a; done
echo "done $(date +%T)"
