#!/usr/bin/env bash
# The copy-ahead check: Qwen3.8's case-1 preset flags in llama-completion (65k, q8_0 K/V, -ub 2048) on the 10,469-token prompt.
#   cafix_run.sh NAME CA [N_PREDICT] [extra args...]     (under tools/model-lock.sh; outputs traces/qwen38/cafix/NAME.{txt,log,rc})
# Environment: BIN (default: llama.cpp-qwen/build/bin), BS=0 drops --backend-sampling, PROMPT (default rec/p8k.txt),
#              ENVS="VAR=VAL ..." extra environment (test switches), KEEP_USAGE=1 keeps the live usage table.
set -u
R=/path/to/project
Q=$R/traces/qwen38
N=$1; CA=$2; NP=${3:-32}; shift 3 2>/dev/null || shift $#
export BIN=${BIN:-$R/llama.cpp-qwen/build/bin}
[ "${KEEP_USAGE:-0}" = 1 ] || rm -f $Q/Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf.usage.json
bs="--backend-sampling"; [ "${BS:-1}" = 0 ] && bs=""
common="--moe-stream-ram-free 0.5GiB --moe-stream-overlap 1 --moe-stream-pred-shared 1 --moe-stream-chunk-prefetch 512"
common="$common --moe-stream-io-wake chunk --moe-stream-promote 1 --moe-stream-guess-cap 2 --moe-stream-policy decay --moe-stream-prompt-end recency"
common="$common --moe-stream-cpu-skip-zero 1 --moe-stream-cpu-fuse 1 --moe-stream-cpu-dispatch 1 --moe-stream-plan 1"
common="$common --moe-stream-drop-stale 1 --moe-stream-needed-early 1 --moe-stream-lookahead 2 --moe-stream-proj-ready 1"
common="$common --moe-stream-wordstart 2 --moe-stream-protect-named 1 --moe-stream-guess-inflight 4 --moe-stream-cpu-topk 1 --moe-stream-cpu-q8-tiled 1 --moe-stream-cpu-fa 1 --moe-stream-cpu-mmid-merge 1"
common="$common --moe-stream-wave-plan 1 --moe-stream-idle-load 2 --moe-stream-prompt-defer 1"
gpu="--moe-stream-gpu-waves 512 --moe-stream-gpu-share-q8 1 --moe-stream-gpu-skip-zero 1 --moe-stream-async-h2d 1 --moe-stream-gpu-wave-skip-zero 1 --moe-stream-prompt-split 1"
gpu="$gpu --moe-stream-vram-free 200MiB --moe-stream-vram-experts-file $Q/qwen38.vram-experts.json --moe-stream-vram-follow 1"
gpu="$gpu --moe-stream-vram-parallel 1 --moe-stream-vram-copyback 1 --moe-stream-vram-end-fill 1 --moe-stream-copy-ahead $CA --moe-stream-vram-lend-prompt 1"
if [ -n "${EVALCB:-}" ]; then   # the eval callback tool (every tensor's sum printed) instead of llama-completion, same cgroup rule as qrun.sh
  AVAIL_K=$(awk '/MemAvailable/ {print $2}' /proc/meminfo); CAP_M=$(( AVAIL_K/1024 - 1536 )); [ $CAP_M -gt 24576 ] && CAP_M=24576
  env GGML_CUDA_COPY_KERNEL=65536 GGML_CUDA_POOL1D_ROW=1 GGML_CUDA_FUSE_AMAX=1 GGML_CUDA_MOE_MARK_IDS=1 GGML_CUDA_FUSE_NORM_QUANT=1 ${ENVS:-} \
    systemd-run --user --scope -q -p MemoryMax=${CAP_M}M -p MemorySwapMax=0 -- $BIN/llama-eval-callback \
    -m $Q/Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf --moe-stream model --load-mode dio -fit off \
    -ngl 99 -c 65536 -ctk q8_0 -ctv q8_0 -fa on -t 8 -b 2048 -ub 2048 -f ${PROMPT:-$Q/rec/p8k.txt} $common $gpu "$@" > $Q/cafix/$N.txt 2> $Q/cafix/$N.log
  echo "$N rc=$?"; exit 0
fi
$Q/qrun.sh cafix/$N GGML_CUDA_COPY_KERNEL=65536 GGML_CUDA_POOL1D_ROW=1 GGML_CUDA_FUSE_AMAX=1 GGML_CUDA_MOE_MARK_IDS=1 GGML_CUDA_FUSE_NORM_QUANT=1 ${ENVS:-} -- \
  -ngl 99 -c 65536 -ctk q8_0 -ctv q8_0 -fa on -t 8 -b 2048 -ub 2048 -n $NP -f ${PROMPT:-$Q/rec/p8k.txt} $common $gpu $bs "$@"
echo "--- $N text:"; head -c 600 $Q/cafix/$N.txt; echo
grep -h 'copy-ahead [0-9]: one set\|prompt eval time\|llama_moe_stream: copy-ahead: on\|MOE_STREAM_TEST' $Q/cafix/$N.log | cut -c1-400
