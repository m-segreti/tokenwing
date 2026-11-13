# The case-1 preset's engine flags (presets/qwen38.sh, llama-server only flags left out) for the perfect-guess pair's tools; sourced.
# T2_CA = the copy-ahead value (default the preset's).
Q=/path/to/project/traces/qwen38
T2_ENV="GGML_CUDA_COPY_KERNEL=65536 GGML_CUDA_POOL1D_ROW=1 GGML_CUDA_FUSE_AMAX=1 GGML_CUDA_MOE_MARK_IDS=1 GGML_CUDA_FUSE_NORM_QUANT=1"
T2_FLAGS="--moe-stream model --load-mode dio -fit off -ngl 99 -c 65536 -ctk q8_0 -ctv q8_0 -fa on -t 8 -b 2048 -ub 2048"
T2_FLAGS="$T2_FLAGS --moe-stream-ram-free 0.5GiB --moe-stream-overlap 1 --moe-stream-pred-shared 1 --moe-stream-chunk-prefetch 512"
T2_FLAGS="$T2_FLAGS --moe-stream-io-wake chunk --moe-stream-promote 1 --moe-stream-guess-cap 2 --moe-stream-policy decay --moe-stream-prompt-end recency"
T2_FLAGS="$T2_FLAGS --moe-stream-cpu-skip-zero 1 --moe-stream-cpu-fuse 1 --moe-stream-cpu-dispatch 1 --moe-stream-plan 1"
T2_FLAGS="$T2_FLAGS --moe-stream-drop-stale 1 --moe-stream-needed-early 1 --moe-stream-lookahead 2 --moe-stream-proj-ready 1"
T2_FLAGS="$T2_FLAGS --moe-stream-wordstart 2 --moe-stream-protect-named 1 --moe-stream-guess-inflight 4 --moe-stream-cpu-topk 1 --moe-stream-cpu-q8-tiled 1 --moe-stream-cpu-fa 1 --moe-stream-cpu-mmid-merge 1"
T2_FLAGS="$T2_FLAGS --moe-stream-wave-plan 1 --moe-stream-idle-load 2 --moe-stream-prompt-defer 1"
T2_FLAGS="$T2_FLAGS --moe-stream-gpu-waves 512 --moe-stream-gpu-share-q8 1 --moe-stream-gpu-skip-zero 1 --moe-stream-async-h2d 1 --moe-stream-gpu-wave-skip-zero 1 --moe-stream-prompt-split 1"
T2_FLAGS="$T2_FLAGS --moe-stream-vram-free 200MiB --moe-stream-vram-experts-file $Q/qwen38.vram-experts.json --moe-stream-vram-follow 1"
T2_FLAGS="$T2_FLAGS --moe-stream-vram-parallel 1 --moe-stream-vram-copyback 1 --moe-stream-vram-end-fill 1 --moe-stream-copy-ahead ${T2_CA:-2} --moe-stream-vram-lend-prompt 1"
# the oracle needs every MoE layer streamed (its trace covers all 48) - the same placement in the recording and in both arms
T2_FLAGS="$T2_FLAGS --moe-stream-gpu-layers 0 --moe-stream-slots ${T2_SLOTS:-200}"
T2_MODEL=$Q/Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf
t2_cap () { local a; a=$(awk '/MemAvailable/ {print $2}' /proc/meminfo); local c=$(( a/1024 - 1536 )); [ $c -gt 24576 ] && c=24576; echo ${c}M; }
