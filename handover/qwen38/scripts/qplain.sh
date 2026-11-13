#!/usr/bin/env bash
# Plain llama.cpp (this build, no streaming) on the ORIGINAL two-file GGUF, CPU only, mapped from the file (the page cache holds
# what fits under the cgroup cap; the model is never loaded whole), inside a memory cgroup. qplain.sh NAME [llama-completion args]
set -u
W=/path/to/project/traces/qwen38
B=${BIN:-/path/to/project/llama.cpp-qwen/build/bin}
N=$1; shift
AVAIL_K=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
CAP_M=$(( AVAIL_K/1024 - 1536 )); [ $CAP_M -gt 24576 ] && CAP_M=24576
echo "cap ${CAP_M}M (MemAvailable $((AVAIL_K/1024))M)" > $W/$N.log
CUDA_VISIBLE_DEVICES= systemd-run --user --scope -q -p MemoryMax=${CAP_M}M -p MemorySwapMax=0 -- \
  $B/llama-completion -m /path/to/models/Qwen3.8-Flash-Next-3.5bpw-00001-of-00002.gguf --load-mode mmap -fit off -ngl 0 \
  --temp 0 --top-k 1 --no-warmup -no-cnv -st --no-display-prompt "$@" > $W/$N.txt 2>> $W/$N.log
RC=$?; echo $RC > $W/$N.rc; echo "$N rc=$RC"
