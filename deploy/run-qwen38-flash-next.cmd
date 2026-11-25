@echo off
setlocal

if not defined CONTAINER_NAME set "CONTAINER_NAME=morrigan-tokenwing-qwen38"
if not defined MODEL_DIR set "MODEL_DIR=E:\tokenwing\models\qwen38-flash-next"
if not defined CACHE_DIR set "CACHE_DIR=E:\tokenwing\cache\qwen38-flash-next"
if not defined MODEL_FILE set "MODEL_FILE=Qwen3.8-Flash-Next-WV-v5-3.5bpw.tokenwing-00001-of-00002.gguf"
if not defined MMPROJ_FILE set "MMPROJ_FILE=mmproj-Qwen3.8-Flash-Next-Q8_0.gguf"
if not defined HOST_PORT set "HOST_PORT=8084"
if not defined USE_CONTROL_SIDECARS set "USE_CONTROL_SIDECARS=1"

if "%USE_CONTROL_SIDECARS%"=="1" (
  set "SIDECAR_ARGS=--moe-stream-vram-experts-file /models/Qwen3.8-Flash-Next-WV-v5-3.5bpw.vram-experts.json --moe-stream-correction /models/Qwen3.8-Flash-Next-WV-v5-3.5bpw.correction"
) else (
  set "SIDECAR_ARGS="
)

if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%"
if errorlevel 1 exit /b 1

docker inspect "%CONTAINER_NAME%" >nul 2>&1 && docker rm -f "%CONTAINER_NAME%" >nul

docker run -d ^
  --name "%CONTAINER_NAME%" ^
  --restart unless-stopped ^
  --gpus all ^
  --memory 46g ^
  --memory-swap 46g ^
  -p "127.0.0.1:%HOST_PORT%:8080" ^
  -e GGML_CUDA_COPY_KERNEL=65536 ^
  -e GGML_CUDA_POOL1D_ROW=1 ^
  -e GGML_CUDA_FUSE_AMAX=1 ^
  -e GGML_CUDA_MOE_MARK_IDS=1 ^
  -e GGML_CUDA_FUSE_NORM_QUANT=1 ^
  -v "%MODEL_DIR%:/models" ^
  -v "%CACHE_DIR%:/cache" ^
  --health-cmd "curl -f http://localhost:8080/health || exit 1" ^
  --health-interval 15s ^
  --health-timeout 10s ^
  --health-retries 120 ^
  --health-start-period 30m ^
  morrigan/tokenwing:c702b7e-cuda13-sm86 ^
  -m /models/%MODEL_FILE% ^
  --mmproj /models/%MMPROJ_FILE% ^
  --no-mmproj-offload ^
  --image-min-tokens 1024 ^
  --load-mode dio ^
  -fit off ^
  -ngl 99 ^
  -c 131072 ^
  -ctk q8_0 ^
  -ctv q8_0 ^
  -np 1 ^
  -t 8 ^
  -tb 15 ^
  -b 1024 ^
  -ub 1024 ^
  --cache-ram 0 ^
  --cache-disk /cache ^
  --cache-disk-gb 16 ^
  --cache-disk-checkpoints 2 ^
  --ctx-checkpoints 2 ^
  --moe-stream model ^
  --moe-stream-ram-free 2GiB ^
  --moe-stream-overlap 1 ^
  --moe-stream-pred-shared 1 ^
  --moe-stream-chunk-prefetch 512 ^
  --moe-stream-io-wake chunk ^
  --moe-stream-promote 1 ^
  --moe-stream-guess-cap 2 ^
  --moe-stream-policy decay ^
  --moe-stream-prompt-end recency ^
  --moe-stream-cpu-skip-zero 1 ^
  --moe-stream-fused-route 1 ^
  --moe-stream-cpu-x4 1 ^
  --moe-stream-cpu-fuse 1 ^
  --moe-stream-conv-silu 1 ^
  --moe-stream-gpu-share-q8 1 ^
  --moe-stream-cpu-dispatch 1 ^
  --moe-stream-plan 1 ^
  --moe-stream-drop-stale 1 ^
  --moe-stream-needed-early 1 ^
  --moe-stream-lookahead 2 ^
  --moe-stream-guess-combined 1 ^
  --moe-stream-proj-ready 1 ^
  --moe-stream-wordstart 2 ^
  --moe-stream-protect-named 1 ^
  --moe-stream-guess-inflight 4 ^
  --moe-stream-vram-free 200MiB ^
  %SIDECAR_ARGS% ^
  --moe-stream-gpu-skip-zero 1 ^
  --moe-stream-vram-follow 1 ^
  --moe-stream-vram-parallel 1 ^
  --moe-stream-vram-copyback 1 ^
  --moe-stream-async-h2d 1 ^
  --moe-stream-layer-major 1 ^
  --moe-stream-wave-plan 1 ^
  --moe-stream-read-ahead 1 ^
  --moe-stream-idle-load 2 ^
  --moe-stream-window 1 ^
  --moe-stream-vram-end-fill 1 ^
  --moe-stream-prompt-defer 1 ^
  --moe-stream-gpu-wave-skip-zero 1 ^
  --moe-stream-prompt-split 1 ^
  --moe-stream-vram-lend-prompt 1 ^
  --moe-stream-cpu-down-scale 1 ^
  --moe-stream-cpu-topk 1 ^
  --moe-stream-cpu-q8-tiled 1 ^
  --moe-stream-cpu-fa 1 ^
  --moe-stream-cpu-mmid-merge 1 ^
  --moe-stream-cpu-iqk 8 ^
  --moe-stream-layer-major-batch 32768 ^
  --moe-stream-hc-share 1 ^
  --moe-stream-rs-window 1 ^
  --moe-stream-ple-late 1 ^
  --moe-stream-hc-silu 1 ^
  --moe-stream-hc-q8 1 ^
  --moe-stream-rs-inplace 1 ^
  --backend-sampling ^
  --host 0.0.0.0 ^
  --port 8080

if errorlevel 1 exit /b 1
docker ps --filter "name=^/%CONTAINER_NAME%$"
