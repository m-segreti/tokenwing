# Presets for Qwen3.8-Flash-Next 3.5bpw (Grawnch's quant, arch qwen4exp) - read by tools/run ("run qwen38 65k").
# The commands are a PROPOSAL until the user decides them (reports/qwen38-b3.md has every flag's check and the first numbers).
# One command per machine kind (QWEN_CASE, as Flint's FLINT_CASE) and context size, q8_0 K/V cache, llama-server:
#   QWEN_CASE=1 (default)  this laptop: GPU (12 GB) + all free RAM - the engine's own whole-layer fit + the VRAM expert pool
#   QWEN_CASE=2            a small GPU (8 GB free) + a 16 GB machine (checked with tools/vram_hold 8192 and an 11 GiB cap)
#   QWEN_CASE=3            CPU-only (CUDA hidden)
# Checked sizes: 65k and 262k (the others are the same command with another -c, not checked).
# Base = Flint's preset (presets/flint.sh) with this model's data files: the VRAM list traces/qwen38/qwen38.vram-experts.json
# (the TokenWing file holds the weights only), the usage seed found next to the model, and the official chat template
# (the file's own template is empty). Whole MoE layers: none chosen by hand - the engine's automatic fit (65k, case 1: layers 0, 1, 2).
# LEFT OUT, nothing to act on for this model (reports/qwen38-b3.md, "flags"):
#   --moe-stream-layer-major / -window / -read-ahead: layer-major prompts are not built for qwen4exp yet (the window and the read-ahead
#     act only in layer-major calls; the read-ahead in calls of 4096+ tokens, and without layer-major a call is one -ub 2048 ubatch)
#   --moe-stream-cpu-iqk: ik's kernels cover none of this file's expert formats (Q3_K / Q4_0 / Q2_0) - 0 experts computed with them
#   --moe-stream-cpu-x4: an MXFP4-only kernel;  --moe-stream-cpu-down-scale: mistral4 / mellum / flint only - 0 layer builds
#   --moe-stream-conv-silu: Ling's KDA graph only (qwen4exp's GDN conv already feeds its SiLU directly)
#   --moe-stream-guess-combined: 0 layers build their guesses in one step (with --moe-stream-pred-shared every target layer mixes its
#     own guess input, so the stacked step does not apply) - start-up line "guess-combined: 0 streamed layers ..."
#   --moe-stream-fused-route: needs expert groups (this model has none);  --moe-stream-correction: refused (no ffn_norm)
# RAM: the pools take all free RAM minus 0.5 GiB at start (close other apps first); KEEP_FREE=1GiB run qwen38 65k keeps more free.
# Knobs for checks: PRESET_BIN (the build's bin), PRESET_PORT (8080), PRESET_MEMCAP (the cgroup cap; default MemTotal - 1 GiB).

PRESET_SIZES="4k 8k 16k 32k 65k 131k 262k"
ROOT=${ROOT:-/path/to/project}

preset_ctx () {   # size -> tokens
    case "$1" in
        4k) echo 4096 ;; 8k) echo 8192 ;; 16k) echo 16384 ;; 32k) echo 32768 ;; 65k|64k) echo 65536 ;; 131k|128k) echo 131072 ;; 262k|256k) echo 262144 ;;
        *) return 1 ;;
    esac
}

preset_command () {
    local ctx; ctx=$(preset_ctx "$1") || return 1
    local c=${QWEN_CASE:-1}
    local ub=2048; case "$1" in 262k|256k) ub=1024 ;; esac   # at 262k the 2048 batch's attention mask alone is ~1 GB of VRAM
    # copy-ahead: the engine's own rule (1 = two sets when one set costs at most a quarter of the VRAM pool's room, else one) on this
    # laptop, as Flint's case 1; one set (2) on the small card, as Flint's case 2 (more VRAM experts per layer). Two sets gave garbage on
    # this model until the tail waves kept the quantized matmul's row padding clean (reports/qwen38-b3.md section 9): now every value
    # answers correctly (the 10k prompt with 1, 2 and 3, backend sampling on and off). NEEDS that fix: until branch qwen-fix is merged
    # into moe-lab, run this preset with PRESET_BIN=$ROOT/llama.cpp-qwen/build/bin (the shared build can still pick two sets and garble)
    local ca=1; [ "$c" = 2 ] && ca=2
    local cap=${PRESET_MEMCAP:-$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) - 1048576 ))K}
    local port=${PRESET_PORT:-8080}
    local q=$ROOT/traces/qwen38
    local common="--moe-stream-ram-free ${KEEP_FREE:-0.5GiB} --moe-stream-overlap 1 --moe-stream-pred-shared 1 --moe-stream-chunk-prefetch 512"
    common="$common --moe-stream-io-wake chunk --moe-stream-promote 1 --moe-stream-guess-cap 2 --moe-stream-policy decay --moe-stream-prompt-end recency"
    common="$common --moe-stream-cpu-skip-zero 1 --moe-stream-cpu-fuse 1 --moe-stream-cpu-dispatch 1 --moe-stream-plan 1"
    common="$common --moe-stream-drop-stale 1 --moe-stream-needed-early 1 --moe-stream-lookahead 2 --moe-stream-proj-ready 1"
    common="$common --moe-stream-wordstart 2 --moe-stream-protect-named 1 --moe-stream-guess-inflight 4 --moe-stream-cpu-topk 1 --moe-stream-cpu-q8-tiled 1 --moe-stream-cpu-fa 1 --moe-stream-cpu-mmid-merge 1"
    common="$common --moe-stream-wave-plan 1 --moe-stream-idle-load 2 --moe-stream-prompt-defer 1"
    local base="${PRESET_BIN:-$ROOT/llama.cpp/build/bin}/llama-server -m $q/Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf --load-mode dio -fit off"
    base="$base -ngl 99 -c $ctx -ctk q8_0 -ctv q8_0 -fa on -np 1 -t 8 -b $ub -ub $ub --cache-ram 0 --ctx-checkpoints 2"
    base="$base --jinja --chat-template-file $q/qwen38-chat-template.jinja --moe-stream model"
    if [ "$c" = 3 ]; then
        echo "systemd-run --user --scope -q -p MemoryMax=$cap -p MemorySwapMax=0 -- env CUDA_VISIBLE_DEVICES=" \
             "$base --moe-stream-gpu-layers 0 $common --moe-stream-cpu-weighted-sum 1 --moe-stream-cpu-small-ops 1 --moe-stream-cpu-state-inplace 1 --host 127.0.0.1 --port $port"
    else
        echo "systemd-run --user --scope -q -p MemoryMax=$cap -p MemorySwapMax=0 -- env GGML_CUDA_COPY_KERNEL=65536 GGML_CUDA_POOL1D_ROW=1 GGML_CUDA_FUSE_AMAX=1 GGML_CUDA_MOE_MARK_IDS=1 GGML_CUDA_FUSE_NORM_QUANT=1" \
             "$base $common --moe-stream-gpu-waves 512 --moe-stream-gpu-share-q8 1" \
             "--moe-stream-gpu-skip-zero 1 --moe-stream-async-h2d 1 --moe-stream-gpu-wave-skip-zero 1 --moe-stream-prompt-split 1" \
             "--moe-stream-vram-free 200MiB --moe-stream-vram-experts-file $q/qwen38.vram-experts.json --moe-stream-vram-follow 1" \
             "--moe-stream-vram-parallel 1 --moe-stream-vram-copyback 1 --moe-stream-vram-end-fill 1 --moe-stream-copy-ahead $ca --moe-stream-vram-lend-prompt 1 --backend-sampling --host 127.0.0.1 --port $port"
    fi
}
