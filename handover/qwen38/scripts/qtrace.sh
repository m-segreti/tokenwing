#!/usr/bin/env bash
# llama-moe-trace on the TokenWing file, CPU only, inside a memory cgroup. qtrace.sh NAME [args]
set -u
W=/path/to/project/traces/qwen38
B=/path/to/project/llama.cpp-qwen/build/bin
N=$1; shift
AVAIL_K=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
CAP_M=$(( AVAIL_K/1024 - 1536 )); [ $CAP_M -gt 24576 ] && CAP_M=24576
echo "cap ${CAP_M}M" > $W/$N.log
CUDA_VISIBLE_DEVICES= systemd-run --user --scope -q -p MemoryMax=${CAP_M}M -p MemorySwapMax=0 -- \
  $B/llama-moe-trace -m $W/Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf --moe-stream model --load-mode dio -fit off -ngl 0 \
  --no-repack -o $W/$N "$@" >> $W/$N.log 2>&1
RC=$?; echo $RC > $W/$N.rc; echo "$N rc=$RC"
