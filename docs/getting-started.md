# Getting started

This page takes you from nothing to a running server: build TokenWing, get a model, start `llama-server` with the command for your PC, and check that it works. The commands are for Linux.

**Always start from a full command.** Every mechanism of the engine is a flag, and most flags are off unless you give them. Without its settings the engine runs with most of its speed-ups off: a long prompt on Ling, for example, is several times slower. The commands below set every flag for one kind of PC. Copy one whole, and do not start from a short command and add flags to it.

## 1. Build

You need CMake, a C++ compiler and Git, and for an NVIDIA card the CUDA Toolkit 13.x.

```bash
git clone https://github.com/runmyai/tokenwing && cd tokenwing
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j
```

For a PC without an NVIDIA card leave out `-DGGML_CUDA=ON`. [Building](building.md) covers AMD and Windows, and the options that shorten the build, such as building for your GPU only. The programs are in `build/bin/`. If you downloaded a release zip instead, use the path of its `llama-server` where the commands below say `./build/bin/llama-server`.

The commands assume a CUDA 13 build. If you build with CUDA 12, change `--moe-stream-async-h2d 1` to `--moe-stream-async-h2d 0` in them.

## 2. Get a model

TokenWing runs two models. Each is one GGUF file: the original model's quantization, laid out for TokenWing. Every tensor starts on a 4 KiB boundary, so the engine can read an expert from the SSD straight into memory, and the file ends with data the engine reads at start, such as which experts are used most in each layer. A plain GGUF of the same model from another source is refused with one line, so use these files.

| Model | File | Size on disk | Download |
|---|---|---|---|
| [Flint 40B](https://huggingface.co/runmyai/Flint-40B-TokenWing) | `flint-UD-Q4_K_XL.tokenwing.gguf` | 25.1 GB | `runmyai/Flint-40B-TokenWing` |
| [Ling 3.0 Flash](https://huggingface.co/runmyai/Ling-3.0-Flash-TokenWing) | `Ling-3.0-flash-MXFP4_MOE.tokenwing.gguf` | 70.2 GB | `runmyai/Ling-3.0-Flash-TokenWing` |

```bash
hf download runmyai/Flint-40B-TokenWing --local-dir $HOME/models/flint
hf download runmyai/Ling-3.0-Flash-TokenWing --local-dir $HOME/models/ling
```

Or download the file from the model's page into a folder of your choice and change the path in the command.

**Put the file on an internal NVMe SSD.** The engine reads experts from it while the model runs, so a slow disk, such as a USB disk, slows the load and every read of an expert that is not in RAM. The disk needs room for the file, 25.1 GB for Flint and 70.2 GB for Ling, and for the files the engine writes next to it, such as the usage table `<model>.gguf.usage.json`. The commands below expect the folders `$HOME/models/flint` and `$HOME/models/ling`.

## 3. Start the server

Before you start:

- **Close other big programs.** The engine fills the RAM that is free when it starts and keeps `--moe-stream-ram-free` of it free for everything else. Raise that value if your PC needs more.
- **Set the memory limit.** The commands start the server with `systemd-run`, which gives it its own memory limit and no swap, so that only the server is stopped if something goes wrong. The engine reads the limit and sizes its RAM cache under it. This line sets it to your RAM minus 1 GiB:

  ```bash
  CAP=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) - 1048576 ))K
  ```

  On a PC with 32 GB of RAM that is about 30G.

Run the command from the `tokenwing` folder. The server answers on `http://127.0.0.1:8080`: a chat page in the browser, and an OpenAI-compatible API. It listens on this PC only.

Every command is for a context length of 65k (`-c 65536`). The model's K/V cache lives in VRAM and grows with the context length, and what it takes is no longer free for experts. So another length needs other values, not only another `-c`. See [Other context lengths](#other-context-lengths).

### An NVIDIA card with 12 GB of VRAM and 32 GB of RAM

The 12 GB card keeps whole MoE layers and a cache of the most-used experts in VRAM, and the RAM holds the rest.

Flint 40B:

```bash
systemd-run --user --scope -q -p MemoryMax=$CAP -p MemorySwapMax=0 -- env \
  GGML_CUDA_COPY_KERNEL=65536 GGML_CUDA_POOL1D_ROW=1 GGML_CUDA_FUSE_AMAX=1 \
  GGML_CUDA_MOE_MARK_IDS=1 GGML_CUDA_FUSE_NORM_QUANT=1 ./build/bin/llama-server -m $HOME/models/flint/flint-UD-Q4_K_XL.tokenwing.gguf \
  --load-mode dio -fit off -ngl 99 -c 65536 -ctk q8_0 -ctv q8_0 -fa on -np 1 -t 8 -b 2048 -ub 2048 \
  --cache-ram 0 --ctx-checkpoints 2 --moe-stream model --moe-stream-gpu-layers-list 43,44,45,46,47,48 \
  --moe-stream-ram-free 0.5GiB --moe-stream-overlap 1 --moe-stream-pred-shared 1 \
  --moe-stream-chunk-prefetch 512 --moe-stream-io-wake chunk --moe-stream-promote 1 \
  --moe-stream-guess-cap 2 --moe-stream-policy decay --moe-stream-prompt-end recency \
  --moe-stream-cpu-skip-zero 1 --moe-stream-cpu-x4 1 --moe-stream-cpu-fuse 1 \
  --moe-stream-cpu-dispatch 1 --moe-stream-plan 1 --moe-stream-drop-stale 1 \
  --moe-stream-needed-early 1 --moe-stream-lookahead 2 --moe-stream-guess-combined 1 \
  --moe-stream-proj-ready 1 --moe-stream-wordstart 2 --moe-stream-protect-named 1 \
  --moe-stream-guess-inflight 4 --moe-stream-cpu-down-scale 1 --moe-stream-cpu-topk 1 \
  --moe-stream-cpu-q8-tiled 1 --moe-stream-cpu-fa 1 --moe-stream-cpu-mmid-merge 1 \
  --moe-stream-cpu-iqk 8 --moe-stream-layer-major 1 --moe-stream-wave-plan 1 \
  --moe-stream-read-ahead 1 --moe-stream-idle-load 2 --moe-stream-window 1 \
  --moe-stream-prompt-defer 1 --moe-stream-gpu-waves 512 --moe-stream-gpu-share-q8 1 \
  --moe-stream-gpu-skip-zero 1 --moe-stream-async-h2d 1 --moe-stream-gpu-wave-skip-zero 1 \
  --moe-stream-prompt-split 1 --moe-stream-vram-free 200MiB --moe-stream-vram-experts-file model \
  --moe-stream-vram-follow 1 --moe-stream-vram-parallel 1 --moe-stream-vram-copyback 1 \
  --moe-stream-vram-end-fill 1 --moe-stream-copy-ahead 1 --moe-stream-vram-lend-prompt 1 \
  --backend-sampling --host 127.0.0.1 --port 8080
```

Ling 3.0 Flash:

```bash
systemd-run --user --scope -q -p MemoryMax=$CAP -p MemorySwapMax=0 -- env \
  GGML_CUDA_COPY_KERNEL=65536 GGML_CUDA_POOL1D_ROW=1 GGML_CUDA_FUSE_AMAX=1 \
  GGML_CUDA_MOE_MARK_IDS=1 GGML_CUDA_FUSE_NORM_QUANT=1 ./build/bin/llama-server -m $HOME/models/ling/Ling-3.0-flash-MXFP4_MOE.tokenwing.gguf \
  --load-mode dio -fit off -ngl 99 -c 65536 -ctk q8_0 -ctv q8_0 -np 1 -t 8 -b 2048 -ub 2048 \
  --cache-ram 0 --ctx-checkpoints 2 --moe-stream model --moe-stream-ram-free 0.5GiB \
  --moe-stream-overlap 1 --moe-stream-pred-shared 1 --moe-stream-chunk-prefetch 512 \
  --moe-stream-gpu-waves 512 --moe-stream-gpu-layers-list 2,3 --moe-stream-io-wake chunk \
  --moe-stream-promote 1 --moe-stream-guess-cap 2 --moe-stream-policy decay \
  --moe-stream-prompt-end recency --moe-stream-cpu-skip-zero 1 --moe-stream-fused-route 1 \
  --moe-stream-cpu-x4 1 --moe-stream-cpu-fuse 1 --moe-stream-conv-silu 1 --moe-stream-gpu-share-q8 1 \
  --moe-stream-cpu-dispatch 1 --moe-stream-plan 1 --moe-stream-drop-stale 1 \
  --moe-stream-needed-early 1 --moe-stream-lookahead 2 --moe-stream-guess-combined 1 \
  --moe-stream-proj-ready 1 --moe-stream-wordstart 2 --moe-stream-protect-named 1 \
  --moe-stream-guess-inflight 4 --moe-stream-vram-free 200MiB --moe-stream-vram-experts-file model \
  --moe-stream-correction model --moe-stream-gpu-skip-zero 1 --moe-stream-vram-follow 1 \
  --moe-stream-vram-parallel 1 --moe-stream-vram-copyback 1 --moe-stream-async-h2d 1 \
  --moe-stream-layer-major 1 --moe-stream-wave-plan 1 --moe-stream-read-ahead 1 \
  --moe-stream-idle-load 2 --moe-stream-window 1 --moe-stream-vram-end-fill 1 \
  --moe-stream-prompt-defer 1 --moe-stream-gpu-wave-skip-zero 1 --moe-stream-prompt-split 1 \
  --moe-stream-vram-lend-prompt 1 --moe-stream-cpu-down-scale 1 --moe-stream-cpu-topk 1 \
  --moe-stream-cpu-q8-tiled 1 --moe-stream-cpu-fa 1 --moe-stream-cpu-mmid-merge 1 \
  --moe-stream-cpu-iqk 8 --moe-stream-layer-major-batch 32768 --backend-sampling --host 127.0.0.1 \
  --port 8080
```

### An NVIDIA card with 8 GB of VRAM

No whole layers stay in VRAM (`--moe-stream-gpu-layers 0`): the VRAM holds the most-used experts and the RAM the rest.

Flint 40B:

```bash
systemd-run --user --scope -q -p MemoryMax=$CAP -p MemorySwapMax=0 -- env \
  GGML_CUDA_COPY_KERNEL=65536 GGML_CUDA_POOL1D_ROW=1 GGML_CUDA_FUSE_AMAX=1 \
  GGML_CUDA_MOE_MARK_IDS=1 GGML_CUDA_FUSE_NORM_QUANT=1 ./build/bin/llama-server -m $HOME/models/flint/flint-UD-Q4_K_XL.tokenwing.gguf \
  --load-mode dio -fit off -ngl 99 -c 65536 -ctk q8_0 -ctv q8_0 -fa on -np 1 -t 8 -b 2048 -ub 2048 \
  --cache-ram 0 --ctx-checkpoints 2 --moe-stream model --moe-stream-gpu-layers 0 \
  --moe-stream-ram-free 1GiB --moe-stream-overlap 1 --moe-stream-pred-shared 1 \
  --moe-stream-chunk-prefetch 512 --moe-stream-io-wake chunk --moe-stream-promote 1 \
  --moe-stream-guess-cap 2 --moe-stream-policy decay --moe-stream-prompt-end recency \
  --moe-stream-cpu-skip-zero 1 --moe-stream-cpu-x4 1 --moe-stream-cpu-fuse 1 \
  --moe-stream-cpu-dispatch 1 --moe-stream-plan 1 --moe-stream-drop-stale 1 \
  --moe-stream-needed-early 1 --moe-stream-lookahead 2 --moe-stream-guess-combined 1 \
  --moe-stream-proj-ready 1 --moe-stream-wordstart 2 --moe-stream-protect-named 1 \
  --moe-stream-guess-inflight 4 --moe-stream-cpu-down-scale 1 --moe-stream-cpu-topk 1 \
  --moe-stream-cpu-q8-tiled 1 --moe-stream-cpu-fa 1 --moe-stream-cpu-mmid-merge 1 \
  --moe-stream-cpu-iqk 8 --moe-stream-layer-major 1 --moe-stream-wave-plan 1 \
  --moe-stream-read-ahead 1 --moe-stream-idle-load 2 --moe-stream-window 1 \
  --moe-stream-prompt-defer 1 --moe-stream-gpu-waves 512 --moe-stream-gpu-share-q8 1 \
  --moe-stream-gpu-skip-zero 1 --moe-stream-async-h2d 1 --moe-stream-gpu-wave-skip-zero 1 \
  --moe-stream-prompt-split 1 --moe-stream-vram-free 200MiB --moe-stream-vram-experts-file model \
  --moe-stream-vram-follow 1 --moe-stream-vram-parallel 1 --moe-stream-vram-copyback 1 \
  --moe-stream-vram-end-fill 1 --moe-stream-copy-ahead 2 --moe-stream-vram-lend-prompt 1 \
  --backend-sampling --host 127.0.0.1 --port 8080
```

Ling 3.0 Flash:

```bash
systemd-run --user --scope -q -p MemoryMax=$CAP -p MemorySwapMax=0 -- env \
  GGML_CUDA_COPY_KERNEL=65536 GGML_CUDA_POOL1D_ROW=1 GGML_CUDA_FUSE_AMAX=1 \
  GGML_CUDA_MOE_MARK_IDS=1 GGML_CUDA_FUSE_NORM_QUANT=1 ./build/bin/llama-server -m $HOME/models/ling/Ling-3.0-flash-MXFP4_MOE.tokenwing.gguf \
  --load-mode dio -fit off -ngl 99 -c 65536 -ctk q8_0 -ctv q8_0 -np 1 -t 8 -b 2048 -ub 2048 \
  --cache-ram 0 --ctx-checkpoints 2 --moe-stream model --moe-stream-ram-free 1GiB \
  --moe-stream-overlap 1 --moe-stream-pred-shared 1 --moe-stream-chunk-prefetch 512 \
  --moe-stream-gpu-waves 512 --moe-stream-gpu-layers 0 --moe-stream-io-wake chunk \
  --moe-stream-promote 1 --moe-stream-guess-cap 2 --moe-stream-policy decay \
  --moe-stream-prompt-end recency --moe-stream-cpu-skip-zero 1 --moe-stream-fused-route 1 \
  --moe-stream-cpu-x4 1 --moe-stream-cpu-fuse 1 --moe-stream-conv-silu 1 --moe-stream-gpu-share-q8 1 \
  --moe-stream-cpu-dispatch 1 --moe-stream-plan 1 --moe-stream-drop-stale 1 \
  --moe-stream-needed-early 1 --moe-stream-lookahead 2 --moe-stream-guess-combined 1 \
  --moe-stream-proj-ready 1 --moe-stream-wordstart 2 --moe-stream-protect-named 1 \
  --moe-stream-guess-inflight 4 --moe-stream-vram-free 200MiB --moe-stream-vram-experts-file model \
  --moe-stream-correction model --moe-stream-gpu-skip-zero 1 --moe-stream-vram-follow 1 \
  --moe-stream-vram-parallel 1 --moe-stream-vram-copyback 1 --moe-stream-async-h2d 1 \
  --moe-stream-layer-major 1 --moe-stream-wave-plan 1 --moe-stream-read-ahead 1 \
  --moe-stream-idle-load 2 --moe-stream-window 1 --moe-stream-vram-end-fill 1 \
  --moe-stream-prompt-defer 1 --moe-stream-gpu-wave-skip-zero 1 --moe-stream-prompt-split 1 \
  --moe-stream-vram-lend-prompt 1 --moe-stream-cpu-down-scale 1 --moe-stream-cpu-topk 1 \
  --moe-stream-cpu-q8-tiled 1 --moe-stream-cpu-fa 1 --moe-stream-cpu-mmid-merge 1 \
  --moe-stream-cpu-iqk 8 --moe-stream-layer-major-batch 32768 --backend-sampling --host 127.0.0.1 \
  --port 8080
```

### An NVIDIA card with 6 GB of VRAM and 16 GB of RAM

Flint 40B:

```bash
systemd-run --user --scope -q -p MemoryMax=$CAP -p MemorySwapMax=0 -- env \
  GGML_CUDA_COPY_KERNEL=65536 GGML_CUDA_POOL1D_ROW=1 GGML_CUDA_FUSE_AMAX=1 \
  GGML_CUDA_MOE_MARK_IDS=1 GGML_CUDA_FUSE_NORM_QUANT=1 ./build/bin/llama-server -m $HOME/models/flint/flint-UD-Q4_K_XL.tokenwing.gguf \
  --load-mode dio -fit off -ngl 99 -c 65536 -ctk q8_0 -ctv q8_0 -fa on -np 1 -t 8 -b 2048 -ub 2048 \
  --cache-ram 0 --ctx-checkpoints 2 --moe-stream model --moe-stream-gpu-layers-list 47,48 \
  --moe-stream-ram-free 0.5GiB --moe-stream-overlap 1 --moe-stream-pred-shared 1 \
  --moe-stream-chunk-prefetch 512 --moe-stream-io-wake chunk --moe-stream-promote 1 \
  --moe-stream-guess-cap 2 --moe-stream-policy decay --moe-stream-prompt-end recency \
  --moe-stream-cpu-skip-zero 1 --moe-stream-cpu-x4 1 --moe-stream-cpu-fuse 1 \
  --moe-stream-cpu-dispatch 1 --moe-stream-plan 1 --moe-stream-drop-stale 1 \
  --moe-stream-needed-early 1 --moe-stream-lookahead 2 --moe-stream-guess-combined 1 \
  --moe-stream-proj-ready 1 --moe-stream-wordstart 2 --moe-stream-protect-named 1 \
  --moe-stream-guess-inflight 4 --moe-stream-cpu-down-scale 1 --moe-stream-cpu-topk 1 \
  --moe-stream-cpu-q8-tiled 1 --moe-stream-cpu-fa 1 --moe-stream-cpu-mmid-merge 1 \
  --moe-stream-cpu-iqk 8 --moe-stream-layer-major 1 --moe-stream-wave-plan 1 \
  --moe-stream-read-ahead 1 --moe-stream-idle-load 2 --moe-stream-window 1 \
  --moe-stream-prompt-defer 1 --moe-stream-gpu-waves 512 --moe-stream-gpu-share-q8 1 \
  --moe-stream-gpu-skip-zero 1 --moe-stream-async-h2d 1 --moe-stream-gpu-wave-skip-zero 1 \
  --moe-stream-prompt-split 1 --moe-stream-vram-free 200MiB --moe-stream-vram-experts-file model \
  --moe-stream-vram-follow 1 --moe-stream-vram-parallel 1 --moe-stream-vram-copyback 1 \
  --moe-stream-vram-end-fill 1 --moe-stream-copy-ahead 2 --moe-stream-vram-lend-prompt 1 \
  --backend-sampling --host 127.0.0.1 --port 8080
```

### CPU only

`CUDA_VISIBLE_DEVICES=` hides the GPU from the engine. The CPU computes every expert, and the RAM also holds the weights a GPU would hold.

Flint 40B:

```bash
systemd-run --user --scope -q -p MemoryMax=$CAP -p MemorySwapMax=0 -- env CUDA_VISIBLE_DEVICES= ./build/bin/llama-server \
  -m $HOME/models/flint/flint-UD-Q4_K_XL.tokenwing.gguf --load-mode dio -fit off -ngl 99 -c 65536 \
  -ctk q8_0 -ctv q8_0 -fa on -np 1 -t 8 -b 2048 -ub 2048 --cache-ram 0 --ctx-checkpoints 2 \
  --moe-stream model --moe-stream-gpu-layers 0 --moe-stream-ram-free 0.5GiB --moe-stream-overlap 1 \
  --moe-stream-pred-shared 1 --moe-stream-chunk-prefetch 512 --moe-stream-io-wake chunk \
  --moe-stream-promote 1 --moe-stream-guess-cap 2 --moe-stream-policy decay \
  --moe-stream-prompt-end recency --moe-stream-cpu-skip-zero 1 --moe-stream-cpu-x4 1 \
  --moe-stream-cpu-fuse 1 --moe-stream-cpu-dispatch 1 --moe-stream-plan 1 --moe-stream-drop-stale 1 \
  --moe-stream-needed-early 1 --moe-stream-lookahead 2 --moe-stream-guess-combined 1 \
  --moe-stream-proj-ready 1 --moe-stream-wordstart 2 --moe-stream-protect-named 1 \
  --moe-stream-guess-inflight 4 --moe-stream-cpu-down-scale 1 --moe-stream-cpu-topk 1 \
  --moe-stream-cpu-q8-tiled 1 --moe-stream-cpu-fa 1 --moe-stream-cpu-mmid-merge 1 \
  --moe-stream-cpu-iqk 8 --moe-stream-layer-major 1 --moe-stream-wave-plan 1 \
  --moe-stream-read-ahead 1 --moe-stream-idle-load 2 --moe-stream-window 1 \
  --moe-stream-prompt-defer 1 --moe-stream-cpu-weighted-sum 1 --moe-stream-cpu-small-ops 1 --host 127.0.0.1 \
  --port 8080
```

### What the flags do

The full list, with what each flag measured, is in [Settings](settings.md). In short:

| Part of the command | What it does |
|---|---|
| `systemd-run ... env GGML_CUDA_...` | `systemd-run` sets the memory limit. The `GGML_CUDA_` variables switch on small CUDA speed-ups. |
| `-m`, `--load-mode dio`, `--moe-stream model` | Load the model file with direct reads, and stream the experts from the model file itself. |
| `-c`, `-ctk q8_0 -ctv q8_0`, `-fa on`, `-np 1`, `-t 8` | Context length, the K/V cache stored as 8-bit values, Flash Attention (in the Flint commands), one server slot, 8 CPU threads. |
| `-b 2048 -ub 2048`, `--cache-ram 0`, `--ctx-checkpoints 2` | The prompt batch, and how the server keeps chat states in RAM. |
| `--moe-stream-ram-free`, `--moe-stream-idle-load 2`, `--moe-stream-policy decay`, `--moe-stream-protect-named` | How much RAM stays free, filling the RAM with experts before the model counts as loaded, and which experts stay in RAM. |
| `--moe-stream-plan`, `-drop-stale`, `-lookahead`, `-needed-early`, `-guess-*`, `-wordstart`, `-pred-shared`, `-correction` | Predicting the experts of the next layers from their own routers, and keeping the useful predictions. |
| `--moe-stream-overlap`, `-proj-ready`, `-chunk-prefetch`, `-io-wake`, `-promote` | How the SSD reads are issued: a needed read goes before a predicted one, and the CPU starts on experts that have landed. |
| `--moe-stream-gpu-layers`, `-gpu-layers-list`, `-vram-free`, `-vram-experts-file`, `-vram-follow`, `-vram-parallel`, `-vram-copyback` | What the VRAM holds: whole layers, and a cache of experts sized from the free VRAM that follows the text. |
| `--moe-stream-gpu-waves`, `-gpu-share-q8`, `-gpu-skip-zero`, `-gpu-wave-skip-zero`, `-async-h2d` | Work on the GPU: the expert matmuls of large prompt batches, and the copies between the CPU's work and the GPU's. |
| `--moe-stream-cpu-*` | CPU kernels for the experts in RAM. |
| `--moe-stream-layer-major`, `-wave-plan`, `-read-ahead`, `-window`, `-prompt-defer`, `-prompt-split`, `-copy-ahead`, `-vram-end-fill`, `-vram-lend-prompt` | Prompts: a long prompt passes through one layer before the next starts, so each layer's experts are read about once. |
| `--backend-sampling` | Tokens are picked on the GPU. |
| `--host 127.0.0.1 --port 8080` | Where the server listens. |

### Other context lengths

Change `-c` to the length you want: 4096, 8192, 16384, 32768, 65536, 131072 or 262144. The 12 GB and 6 GB commands change with the length in other places too, because a longer K/V cache takes VRAM from the experts. The table gives the values for each length.

| `-c` | Flint, 12 GB card | Flint, 6 GB card | Ling, 12 GB card |
|---|---|---|---|
| 4096 to 32768 | no change | no change | remove `--moe-stream-layer-major-batch 32768` |
| 65536 | the command above | the command above | the command above |
| 131072 | `--moe-stream-gpu-layers-list 46,47,48` | `--moe-stream-gpu-layers 0` in place of `--moe-stream-gpu-layers-list 47,48` | no change |
| 262144 | `--moe-stream-gpu-layers-list 47,48`, `-b 1024 -ub 1024` | `--moe-stream-gpu-layers 0` in place of `--moe-stream-gpu-layers-list 47,48`, `-b 1024 -ub 1024` | `-b 1024 -ub 1024` |

The CPU-only Flint command changes only `-c`, and `-b 1024 -ub 1024` at 262144. The 8 GB commands are given for 65536 only.

## 4. What you see while it loads

The server loads the weights that every token uses, then fills the RAM with experts, and only then does the model count as loaded (`--moe-stream-idle-load 2`). So the load takes longer than plain llama.cpp's, and the first request does not start cold. The progress lines are on the server's output:

```
llama_moe_stream: loading: experts into RAM 0 % (0 of N, 0.0 of X GB, 0.0 s)
llama_moe_stream: loading: experts into RAM 10 % (...)
...
llama_moe_stream: loading: experts into RAM 100 % (...)
llama_moe_stream: loading: experts into RAM done: ... experts read into the RAM pools of ... streamed layers, ... of ... slots filled; ...
```

A line comes at 0 %, at every 10 % and at 100 %. The `done` line says what was read. If every expert fits in RAM it says so, and after the load the SSD is read no more. If not, it says the RAM holds the most-used experts of each layer, and the rest are read from the SSD while the model runs. The server's health check (`GET /health`) passes only after the `done` line.

The engine prints its plan at the start: the RAM slots per layer, and the VRAM it keeps. Every line that starts with `llama_moe_stream:` comes from the engine. When it has to use a slower path for lack of RAM or VRAM, it prints one line saying what it did.

## 5. Check that it works

Send a request, or open `http://127.0.0.1:8080` in a browser:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Why is the sky blue?"}]}'
```

A sensible answer means the model runs. Garbage or no output is an error.

To test a build on your machine, run `tokenwing-tools/testing/check.py` with any small model that has routed experts and fits in your RAM:

```bash
python3 tokenwing-tools/testing/check.py --model SMALL.gguf --build build
```

[Testing](testing.md) lists the steps it runs and what each proves.

### Read the speed lines

After each request the server prints its timings, and one line from the engine:

```
prompt eval time = ... ms / N tokens (... ms per token, ... tokens per second)
       eval time = ... ms / N tokens (... ms per token, ... tokens per second)
      total time = ... ms / N tokens
    picked words = ... on the GPU (backend sampling), ... on the CPU
      moe stream: L layers x S slots, this request P prompt + D decode tokens, hit rate H%, M misses/decode token, X MiB/decode token read, stall Y ms/decode token, Z ms/prompt token
```

| Value | Means |
|---|---|
| `prompt eval time` | How fast the prompt was processed, in tokens per second. For a long prompt its time is your time to first token (TTFT). |
| `eval time` | How fast the reply was written, in tokens per second. |
| `picked words` | Where the reply's tokens were picked: on the GPU with `--backend-sampling`, or on the CPU. |
| `L layers x S slots` | The streamed layers, and the expert slots each holds in RAM. |
| `hit rate` | The share of expert lookups that found the expert in RAM. |
| `misses/decode token`, `MiB/decode token read` | The experts and bytes read from the SSD for each reply token. |
| `stall` | The time the engine waited for the SSD, per reply token and per prompt token. The prediction's job is to keep it low. |

`GET /moe-stream/stats` returns the same figures live, per layer too, and the final response of a request carries them in `timings.moe_stream`. With `--moe-stream-timing 1` they also say where each reply token's time went: waiting for the SSD, the CPU, the GPU, and the hand-off between them. It is off in the commands above.

What to expect:

- Reply speed falls as a chat fills a long context.
- After a very long prompt (16k tokens and more), the first reply tokens are slower while the RAM cache moves from the prompt's experts to the reply's.
- On a PC with an 8 GB NVIDIA card and 16 GiB of free RAM, the 8 GB commands gave Flint 45.8 tokens/s (plain llama.cpp: 25.5) and Ling 22.1 tokens/s (plain llama.cpp: 6.7), at 65k context. Speed depends on the machine: with more RAM, the RAM holds more experts.
- Speed far below that can mean a flag was lost in copying. A missing flag turns a speed-up off and nothing complains. On a laptop, also check that it is plugged in and that the GPU is not held at a low power limit.

## 6. When something fails

Open an issue at [github.com/runmyai/tokenwing/issues](https://github.com/runmyai/tokenwing/issues) and send:

- the server's whole output, from the start to the error,
- the command you ran,
- your GPU, your RAM and your SSD, and the output of `./build/bin/llama-server --version`.

Before that, look at what the output says:

- A line starting with `llama_moe_stream:` says what the engine chose or why it fell back to a slower path.
- A model file that is refused with one line is not a TokenWing file. Use the file from the table in [Get a model](#2-get-a-model).
- If the GPU runs out of memory at start, use a shorter context, or the values for your length in [Other context lengths](#other-context-lengths).
- If the server stops with no error message, the system probably ran out of memory. Close other programs, or raise `--moe-stream-ram-free` so more RAM stays free, and start again.

[Hardware](hardware.md) says what each part of your PC does for the speed, and [Inside TokenWing](how-it-works.md) covers the whole pipeline.
