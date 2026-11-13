#!/usr/bin/env bash
# One Qwen3.8 run from the TokenWing file, inside a memory cgroup (MemAvailable - 1.5 GiB, at most 24G).
#   qrun.sh NAME [env VAR=VAL ...] -- [llama-completion args...]
# Output: NAME.txt (text), NAME.log (stderr), NAME.rc
set -u
W=/path/to/project/traces/qwen38
B=${BIN:-/path/to/project/llama.cpp-qwen/build/bin}
N=$1; shift
ENVS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do ENVS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
AVAIL_K=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
CAP_M=$(( AVAIL_K/1024 - 1536 )); [ $CAP_M -gt 24576 ] && CAP_M=24576
echo "cap ${CAP_M}M (MemAvailable $((AVAIL_K/1024))M)" > $W/$N.log
env "${ENVS[@]}" systemd-run --user --scope -q -p MemoryMax=${CAP_M}M -p MemorySwapMax=0 -- \
  $B/llama-completion -m $W/Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf --moe-stream model --load-mode dio -fit off \
  --temp 0 --top-k 1 --no-warmup -no-cnv -st --no-display-prompt "$@" > $W/$N.txt 2>> $W/$N.log
RC=$?; echo $RC > $W/$N.rc; echo "$N rc=$RC"
