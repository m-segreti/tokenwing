#!/usr/bin/env bash
# The perfect-guess pair, step 3: today's guess (base) against perfect guesses (oracle: the new recording, lead 48 = one word ahead,
# fed into the waiting list) on the story's 512 forced words (llama-moe-decode-batch --force-text = the recording's own words), all 48
# layers streamed, the same RAM slots pinned in both arms (T2_SLOTS, default 200), arms interleaved. Each run under the model lock;
# the GPU power check before and after the series.   t2_pair.sh [N_PAIRS=3] (SMOKE=1: one 32-word run per arm)
R=/path/to/project; source $R/bench/queue/qwen38/t2_flags.sh
B=$R/llama.cpp-qwen/build/bin; O=$Q/oracle
NP=${1:-3}; NG=512; PFX=""
[ "${SMOKE:-0}" = 1 ] && { NP=1; NG=32; PFX="smoke_"; }
$R/tools/gpu-cap-check.sh 2>&1 | tail -1 | tee $O/${PFX}cap_before.txt
grep -q "^gpu-cap-check: OK" $O/${PFX}cap_before.txt || { echo "GPU power check not OK - stopping"; exit 4; }
one () {   # name, oracle 0|1
    local extra="" envs=""
    [ "$2" = 1 ] && { extra="--moe-stream-oracle $Q/rec2/06_story --moe-stream-oracle-lead 48"; envs="MOE_STREAM_TEST_ORACLE_LIST=1"; }
    rm -f $T2_MODEL.usage.json
    $R/tools/model-lock.sh Qwen-fix bash -c "env $T2_ENV $envs systemd-run --user --scope -q -p MemoryMax=$(t2_cap) -p MemorySwapMax=0 -- \
        $B/llama-moe-decode-batch -m $T2_MODEL $T2_FLAGS $extra -f $Q/rec/06_story.txt --force-text $Q/rec2/06_story.decode.txt \
        --gen $NG --no-batch > $O/$1.txt 2> $O/$1.log; echo \$? > $O/$1.rc"
    echo "$1 rc=$(cat $O/$1.rc) $(grep -o 'gen: [0-9]* calls of 1 token, [0-9.]* ms/call, remap op [0-9.]* ms/token, stall [0-9.]* ms/token' $O/$1.log)"
    grep -h "oracle list: \|oracle: .* ids compared" $O/$1.log | cut -c1-260
}
for i in $(seq 1 $NP); do
    one ${PFX}base_$i 0
    one ${PFX}oracle_$i 1
done
$R/tools/gpu-cap-check.sh 2>&1 | tail -1 | tee $O/${PFX}cap_after.txt
echo "done $(date +%T)"
