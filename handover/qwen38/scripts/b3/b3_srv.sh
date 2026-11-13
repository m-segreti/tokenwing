#!/usr/bin/env bash
# One llama-server run of Qwen3.8 for the first numbers: start the server, send the requests, stop it by PID.
#   b3_srv.sh NAME            (run it under tools/model-lock.sh; outputs traces/qwen38/b3/NAME.{log,json,out})
# Environment:
#   QWEN_CASE=1|2|3   the preset's case (presets/qwen38.sh); 3 = CPU-only (CUDA hidden)
#   SIZE=65k          the preset's size
#   PLAIN=1           plain llama.cpp instead: the same build, no --moe-stream flag, the ORIGINAL two-file GGUF mapped (mmap),
#                     its own automatic fit on the GPU (-fit on, -ngl not set), -ngl 0 for case 3; same -c / K/V / batch / template
#   MEMCAP=<size>     the cgroup cap (default: the preset's MemTotal - 1 GiB)
#   VRAM_HOLD_MIB=N   tools/vram_hold leaves N MiB of VRAM free for this run
#   EXTRA="..."       more server flags (appended); DROP="--a --b" removes flags (with their value) from the command
#   REQS="story512 p10k"  requests, in order: storyN (the ~100-token story prompt, N forced words, streamed: per-token times),
#                     p10k / p10kN (the 10,469-token prompt + 32 / N words), session (p10k + 400 words, then a second request
#                     that continues the same chat), chat (one /v1/chat/completions request through the template)
#   PORT=8093
set -u
R=/path/to/project
O=$R/traces/qwen38/b3
NAME=$1
PORT=${PORT:-8093}
SIZE=${SIZE:-65k}
LOG=$O/$NAME.log
: > $LOG
cap=${MEMCAP:-$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) - 1048576 ))K}
HOLD_PID=""
if [ -n "${VRAM_HOLD_MIB:-}" ]; then
    $R/tools/vram_hold "$VRAM_HOLD_MIB" > $O/$NAME.vramhold 2>&1 &
    HOLD_PID=$!
    for _ in $(seq 1 100); do grep -q "holding\|nothing to hold" $O/$NAME.vramhold 2>/dev/null && break; sleep 0.1; done
    grep -q "holding" $O/$NAME.vramhold || { echo "$NAME: vram_hold did not hold: $(cat $O/$NAME.vramhold)"; kill $HOLD_PID 2>/dev/null; exit 3; }
    echo "vram_hold: $(cat $O/$NAME.vramhold)" >> $LOG
fi
if [ "${PLAIN:-0}" = 1 ]; then
    M=/path/to/models/Qwen3.8-Flash-Next-3.5bpw-00001-of-00002.gguf
    ub=2048
    ngl=""; envs=""
    [ "${QWEN_CASE:-1}" = 3 ] && { ngl="-ngl 0"; envs="CUDA_VISIBLE_DEVICES="; }
    ctx=$(source $R/presets/qwen38.sh; preset_ctx $SIZE)
    CMD="systemd-run --user --scope -q -p MemoryMax=$cap -p MemorySwapMax=0 -- env $envs $R/llama.cpp/build/bin/llama-server -m $M --load-mode mmap $ngl"
    CMD="$CMD -c $ctx -ctk q8_0 -ctv q8_0 -fa on -np 1 -t 8 -b $ub -ub $ub --cache-ram 0 --ctx-checkpoints 2"
    CMD="$CMD --jinja --chat-template-file $R/traces/qwen38/qwen38-chat-template.jinja --host 127.0.0.1 --port $PORT"
else
    CMD=$(source $R/presets/qwen38.sh; PRESET_PORT=$PORT PRESET_MEMCAP=$cap preset_command $SIZE)
fi
for d in ${DROP:-}; do CMD=$(echo "$CMD" | sed -E "s/ $d( [^-][^ ]*)?( |$)/ /"); done   # DROP="--flag ..." removes flags (and their value)
CMD="$CMD --no-warmup ${EXTRA:-}"
echo "cmd: $CMD" >> $LOG
echo "start $(date +%T); MemAvailable $(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo) MiB; cap $cap" >> $LOG
eval "exec $CMD" >> $LOG 2>&1 &
SP=$!
echo $SP > $O/$NAME.pid
ok=0
for _ in $(seq 1 1200); do
    curl -s http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"ok"' && { ok=1; break; }
    kill -0 $SP 2>/dev/null || break
    sleep 1
done
if [ $ok = 1 ]; then
    echo "ready $(date +%T)" >> $LOG
    REQS="${REQS:-story512 p10k}" PORT=$PORT python3 $R/bench/queue/qwen38/b3_client.py $O/$NAME.json > $O/$NAME.out 2>&1
    echo "client rc=$? $(date +%T)" >> $LOG
else
    echo "$NAME: the server did not start (see $LOG)" | tee $O/$NAME.out
fi
kill $SP 2>/dev/null; wait $SP 2>/dev/null
[ -n "$HOLD_PID" ] && { kill $HOLD_PID 2>/dev/null; wait $HOLD_PID 2>/dev/null; }
echo "end $(date +%T)" >> $LOG
cat $O/$NAME.out
