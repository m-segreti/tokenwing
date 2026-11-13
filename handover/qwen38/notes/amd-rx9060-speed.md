# AMD RX 9060 XT: Ling's long-prompt crash and the reply-token GPU spots (ROCm / HIP)

Status: done for this round (see "Not done").

Machine: rented Radeon RX 9060 XT 16 GB (gfx1200, RDNA4), HIP 7.16, Ryzen 5 3500, `-t 4`. Engine `moe-lab` = `308bc4b42`,
work branch `amd-speed` in `llama.cpp-amd`. Flint `flint-UD-Q4_K_XL.tokenwing.gguf` (whole-layer layout: 27 MoE layers whole on
the GPU, 20 in RAM, the flags in `/root/tw/up/prof_args.txt`); Ling `Ling-3.0-flash-MXFP4_MOE.tokenwing.gguf` (`up/cases_ling.json`).

Tools added (MoE_Lab only, not engine code): `traces/amd/rx9060/kt_stats.py` (per-kernel table from a rocprofv3 kernel trace:
calls, mean / median us, idle gap before each kernel) and on the machine `/root/tw/up/kt.sh NAME [VAR=VAL ...]` (rocprofv3
`--kernel-trace` of 64 Flint reply tokens, CUDA graphs as in the real run -> `/root/tw/results7/NAME.txt`). Unlike
`GGML_CUDA_PROFILE` (event to event, graphs off) these are the kernels' own GPU times.

## 0. The crash: Ling's 8,092-token prompt (first priority)

**What happened.** Ling on this card answered the story and the sum, then llama-server aborted on the 8,092-token prompt, in both
runs of the queued job: `ROCm error: out of memory ... ggml_cuda_pool_leg::alloc <- launch_fattn<512,2,16> <-
ggml_cuda_flash_attn_ext_tile_case<576,512>`. The VRAM plan left 0.31 GiB free (2 whole layers, 71 VRAM experts x 38 layers).

**The cause (proved).** HIP builds have no VMM pool (`GGML_HIP_NO_VMM=ON`), so the ops' temporary buffers come from the legacy
pool: a device malloc at run time; a freed buffer stays in the pool and goes to any later request that fits (best fit), even a
much smaller one. Ling's attention (576-wide K, 512-wide V, MLA) runs the flash-attention **tile** kernel on RDNA4 (NVIDIA runs the
MMA kernel with stream-k, whose scratch is small). The tile kernel's launcher (`launch_fattn`, non-stream-k branch) picks
`parallel_blocks` = the blocks per CU (2 here) and then needs a scratch of `parallel_blocks x the whole output`:
2 x (2048 rows x 32 heads x 512 values x 4 B) = **256 MiB (+5 % look-ahead)** per prompt ubatch, against ~0.3 GiB free.
- The new one-line log on the fixed build prints exactly this request: `launch_fattn: AMD: flash attention of a 2048-row call runs 1
  parallel blocks instead of 2 (their scratch 256.0 MiB would exceed GGML_HIP_FA_TMP_MAX_MIB 64)`.
- With the cap switched off (`GGML_HIP_FA_TMP_MAX_MIB=0`, same build) the crash comes back: `long expect '4417': FAIL`,
  `stopped (rc=-6)`, `ROCm error: out of memory ... in function alloc at ggml-cuda.cu:565` - this time in `ggml_cuda_mul_mat_q`'s
  pool request right after: the 256 MiB the attention took stays in the legacy pool, the next matmul's buffers no longer fit.
- The engine's growth estimate counts 40 MiB for the scratch pool (`vx_setup: ... 40.0 MiB scratch pool (not mapped at start)`,
  calibrated on NVIDIA where the peak is 36-38 MiB); the AMD tile path alone wanted 6.7 times that.

**The fix (which of the lead's three):** none of the three as written - the need itself is removed. Under AMD (`GGML_CUDA_CC_IS_AMD`
at run time) `launch_fattn` keeps the parallel-blocks scratch under 64 MiB (`GGML_HIP_FA_TMP_MAX_MIB`, 0 = no cap): it lowers
`parallel_blocks` (here to 1, which writes the output directly, no scratch, no combine kernel). It only acts where the output is
over 64 MiB / 2 = a batch of >= 1024 rows (Ling) - there the output tiles alone fill the GPU many times over, so nothing is lost; a
reply token (one row) is untouched. One line per process when it acts. NVIDIA builds unchanged (the check is a vendor check; it
would help an NVIDIA tile / vec kernel case the same way - not enabled there).
- (2) accounting: NOT changed - the 40 MiB scratch term of the growth estimate stays; with the cap, the attention's part is <= 64 MiB.
- (3) a failed scratch allocation ending the call (decode 3) instead of `ggml_abort`: NOT done (the pool's malloc failure aborts
  inside `CUDA_CHECK`; making it recoverable needs a try-alloc path through the backend).

**Check (`up/cases_ling.json --only ling_tw_t4`, the FA cap on, the two speed changes below switched off):**
```
[19:10:36]   measured  TTFT 1.88 s | prompt 48 tok 26 tok/s | reply 300 tok 19.90 tok/s | end length
[19:10:41]   sum       expect '43': ok
[19:11:13]   long      TTFT 25.68 s | prompt 8092 tok 315 tok/s | reply 93 tok 12.81 tok/s | end stop
[19:11:13]   long      expect '4417': ok
[19:11:15]   ok; peak RSS 13.33 GiB, peak VRAM of the server None MiB; stopped (rc=0)
[19:11:15]   check: ok
```
(the run with everything on: section "Before / after".)

## 1. The "down" group

**What makes it slow (rocprofv3, the kernels' own times, CUDA graphs on as in the real run):** it is the matvec kernel itself, not
a copy, a gap or a second kernel. `mul_mat_vec_q` with workgroup 32x8, grid 65536x48 (2048 rows x 6 experts) = the down projection:
**81.6 us mean, 95.3 us median, 46.3 calls/token = 3.78 ms/token** (the down quantize kernel next to it: 1.3 us). Flint's experts
have K = 512 (n_ff), so a down row is 2 blocks of 256 values: the RDNA4 table runs 8 warps on one row, and only 16 lanes (IQ4_XS)
or 32 lanes (Q5_K) of the 256 threads have work; 12,288 blocks of 256 threads, the time follows the block count (gate: K = 2048,
3,072 blocks, 31 us although it reads twice the weights). The brief's earlier test (1 warp, 1 row per block) kept 12,288 blocks and
so changed nothing.

**Fix:** on RDNA4, a one-column row whose K loop fits one warp's lanes (`blocks_per_row x (qi/vdr) <= warp size`, rows a multiple
of 8) runs the existing short-row instance (`small_k`) with **1 warp per block and 8 rows per block** (1,536 blocks for down). The
generic code had switched `small_k` off for all of RDNA. `GGML_HIP_MMVQ_SHORT_ROWS=0` gives the old shape. Other GPUs unchanged
(the RDNA4 table is only chosen for RDNA4).

**Result (same build, `kt.sh`):** the down kernel 81.6 -> **13.25 us mean** (median 95.3 -> 14.3), 3.78 -> 0.60 ms/token. Node
profile (`prof.sh`, us per call, `off` = the switches at 0 in the same build):

| node | base | off | new |
|---|---|---|---|
| MUL_MAT_ID ffn_moe_down iq4_xs | 107.2 | 105.2 | 24.8 |
| MUL_MAT_ID ffn_moe_down q5_K | 108.6 | 106.7 | 29.2 |
| MUL_MAT_ID ffn_moe_vram_down q5_K | 68.1 | 67.0 | 19.2 |
| GPU node ms/token | 20.67 | 20.80 | 17.40 |

## 2. The router's logits

**What path:** BF16 n_expert x n_embd (256 x 2048) times one vector goes to `mul_mat_vec_f` (not hipBLAS): 256 threads per row,
each thread reads 4 bytes (2 values) per step, 4 dependent steps. rocprofv3: **23.18 us mean, 78.8 calls/token (the router + the
guess rows) = 1.83 ms/token** for 1 MiB of weights.

**Fix:** under HIP, a one-column BF16 product without fusion whose rows / channels / y are 16-byte aligned reads 16 bytes (8
values) per thread and step (`x8` instance of the same kernel; a 2048-wide row = one step). `GGML_HIP_MMVF_BF16_X8=0` gives the old
loop. NVIDIA path unchanged (the code is inside `#if defined(GGML_USE_HIP)`).

**Result:** kernel 23.18 -> **8.71 us mean** (1.83 -> 0.69 ms/token); node profile: `ffn_moe_logits` 31.87 -> 19.52 us,
`ffn_moe_logits_pred_comb` (2-4 guess rows) 55.42 -> 26.62 us, `ffn_moe_logits_pred` 30.65 -> 16.77 us. The rest of the node time
is the profile's own event + launch gap.

## 3. Other nodes, and the HIP wait mode

Not worked on (time). What the data says:
- After the two fixes the kernel time is 16.83 -> 12.63 ms/token (rocprofv3, 64 tokens), and the trace's gaps between kernels are
  16.4-17.0 ms/token: the GPU is idle more than it computes (the waits for the CPU experts are inside that number). The biggest
  kernels left are bandwidth-bound dense matvecs (2048-row Q8_0, 22 us; the vocabulary matvec 1.05 ms once per token; 320 GB/s
  against the laptop's 896 GB/s explains their ~2x).
- Still slow by shape: the VRAM-pool down in Q8_0 (66 us, 1 call/token; K = 512 needs 64 lanes, so the short-row rule does not
  take it) - small.
- Wait mode (the lead's run, Flint `-t 4`, one run each): HIP default 39.5, spin 40.8, yield 39.3 tok/s. +3 % from one run each is
  inside this machine's run-to-run spread (Flint story with the same build: 41.7 and 46.8 tok/s), so I did not make spin the
  default; the experiment (`GGML_HIP_SCHEDULE`) stays only in the machine's copy, not in the branch.
- Ling's reply-token split on this card (the lead's numbers: 55.3 ms/token = SSD wait 13.5, CPU 13.5, waits between CPU and GPU
  24.4, GPU nodes 31.8 overlapping): the CPU-GPU waits are the largest part; not looked at.

## What changed (files)
Branch `amd-speed` in `llama.cpp-amd` (from `moe-lab` `308bc4b42`), commits `cc684afa7` (the two speed changes; its message still
says "not measured" - measured afterwards, numbers here) and `fd4e3e2b3` (the crash fix + `docs/settings.md`):
- `ggml/src/ggml-cuda/fattn-common.cuh`: the AMD cap of the parallel-blocks scratch (`GGML_HIP_FA_TMP_MAX_MIB`).
- `ggml/src/ggml-cuda/mmvq.cu`: RDNA4 short rows (`GGML_HIP_MMVQ_SHORT_ROWS`).
- `ggml/src/ggml-cuda/mmvf.cu`: HIP BF16 one-column 16-byte loads (`GGML_HIP_MMVF_BF16_X8`).
- `docs/settings.md` (engine): the three switches. MoE_Lab: `docs/flags.md` (three rows), `docs/other-machines.md` (three
  support-table rows), this report.
The machine's `/root/tw/src` has the same three files (md5 checked) plus the lead's `GGML_HIP_SCHEDULE` experiment in
`ggml-cuda.cu`; its `mmvq.cu` experiment was replaced by the branch's file.

## Before / after

Flint, real runs (`up/cases_threads.json --only flint_tw_t4`, same build, `off` = `GGML_HIP_MMVQ_SHORT_ROWS=0
GGML_HIP_MMVF_BF16_X8=0`, interleaved off / on / off / on; the FA cap is on in all four):

| run | story reply tok/s | sum | 7.7k prompt tok/s | long reply tok/s | checks |
|---|---|---|---|---|---|
| off 1 | 38.99 | ok | 1422 | 35.71 | ok |
| on 1 | 46.83 | ok | 1282 | 39.86 | ok |
| off 2 | 38.76 | ok | 1279 | 34.81 | ok |
| on 2 | 41.71 | ok | 1279 | 40.36 | ok |

Story replies: off 38.9 -> on 44.3 tok/s mean (+14 %; the two "on" runs differ by 5 tok/s, so the size of the gain is not exact;
the long prompt's replies agree: 35.3 -> 40.1). Prompt: a tie (1279 vs 1279 in pair 2; pair 1's off run is the outlier).
`llama-completion` 160 tokens in profile mode: 44.60 (off) -> 40.60 (new) ms/token; kernel trace mode 64 tokens: 31.16 (base build)
-> 27.77 ms/token.

Ling, everything on (`up/cases_ling.json --only ling_tw_t4`): story 20.18 tok/s, `sum expect '43': ok`, `long ... prompt 8092 tok
318 tok/s ... expect '4417': ok`, `check: ok` (switches off, FA cap on: 19.90 tok/s, 315 tok/s, ok; the queued job before the fix:
18.10 / 19.46 tok/s and the abort).

Portable check on the final build (`check.py --model Ling-3.0-tiny ... --build build-hip --no-build --no-ctest`):
```
  OK   streaming with a pool smaller than the prompt needs (waves)  (rc 0)
  OK   streamed run on the GPU  (ROCm0: AMD Radeon RX 9060 XT (16304 MiB, 16236 MiB free))
Everything passed.
```
(The final build differs from the measured one only by a comment in `fattn-common.cuh`.)

Rounding: the new kernels sum in another order, so texts differ at rounding level; every answer check is correct.

Qwen3.8-Flash-Next: the short-row rule applies to any expert whose K fits one warp (K-quants 32 lanes per 512 values); its router
is F32 or BF16 depending on the file (BF16 takes the x8 path); the FA cap helps any model on the tile kernel.

## Not done / follow-ups
- Accounting (the lead's (2)): measured and handled in round 2, section A.
- A failed scratch allocation still aborts (the lead's (3)); the legacy pool's best fit also hands a big cached buffer to a small
  request (seen with the cap off: the abort moved to `ggml_cuda_mul_mat_q`).
- The wait mode: paired repeats before deciding.
- Ling's CPU-GPU waits (24.4 ms/token on this card).
- VRAM-pool down in Q8_0 still 66 us (needs a 2-warp short-row shape).
- `GGML_HIP_MMVQ_SHORT_ROWS` measured on RDNA4 only; RDNA3 / RDNA2 have their own tables (to carry over and measure).

## Time spent
About 100 minutes of work (plus ~25 minutes blocked while the queued job's download stalled).

---

# Second round (lead's items A, B, D, E; C not reached)

The lead's own check on the first round's build: 16k and 32k prompts pass on both models (Ling 16,143 tok 344 tok/s, 32,127 tok
310 tok/s; Flint 15,382 tok 1,342 tok/s, 30,649 tok 1,397 tok/s; planted fact ok in all four).

## A. The scratch room under HIP

**Measured (new tool `GGML_CUDA_POOL_LOG=1`: a line at every new peak of the device memory the legacy pool holds; the engine's
`vx_end_state` line now also shows it).** 16 GB card, the lead's long cases (`up/cases_long.json`), `results9/pool/`:

| model | reply tokens | 8k prompt | after 16k + 32k prompts | largest single request |
|---|---|---|---|---|
| Flint (27 whole layers) | 0.3 MiB | 1.5 MiB | **58.3 MiB** (8 buffers) | 28.4 MiB |
| Ling (2 whole layers) | 2.7 MiB | 4.5 / 8.6 MiB | **85.4 MiB** (12 buffers) | 47.3 MiB |

The engine counts 32 MiB (Flint) / 40 MiB (Ling) for the scratch pool (`ubatch x hidden size x 8 bytes`, calibrated on NVIDIA's
VMM pool). The legacy pool held about twice the largest request because it keeps every size it was asked for: a longer prompt asks
again with more rows (Flint: 4.7, 9.5, 14.2, 28.4 MiB - the quantized activations of the expert matmuls, which grow with the call)
and the older, smaller buffers stay held next to the new one.

**First attempt, measured and reverted: a bigger growth term.** Counting `min(n_batch, n_ctx) x hidden x 1.25 B` (Flint 160 MiB,
Ling 100 MiB, since Flint's layer-major batch is the whole 65,536 context) made the reserve right but cost Flint its VRAM pool on
the lead's 16 GB layout: `growth 210.0 MiB` -> `VRAM experts off` (the 405-place tail, 835 MiB, no longer fit in the room left
next to 27 whole layers), and Flint's long prompts fell from 1,350 / 1,404 to **367 / 363 tok/s** (16k / 32k, same answers;
`results9/reserve/`). Ling lost one VRAM expert per layer (71 -> 70), prompts unchanged.

**What is built instead: the legacy pool gives back outgrown buffers (HIP builds only).** When no cached buffer fits a request of
>= 4 MiB, the cached buffers of 1 MiB up to that size are freed first (once per new size, never during a graph capture: a device
sync, then the captured CUDA graphs are captured again). Measured on the 8 GB-card runs (same prompts, `results9/small2/`):
Flint **28.8 MiB** held at most (4 buffers; was 58.3), Ling **47.7 MiB** (5 buffers; was 85.4) - about the largest request.
So the growth term is unchanged: Flint's 28.8 MiB is inside its 32 MiB; Ling's 47.7 MiB is 7.7 MiB over its 40 MiB at a 32k prompt,
which the 200 MiB keep-free covers. No new setting (`GGML_CUDA_POOL_LOG` is a measuring tool). NVIDIA builds unchanged (the
trimming is under `GGML_USE_HIP`; the VMM pool reports 0 held bytes and the engine reads the count only when the pool is not VMM).
Files: `ggml/src/ggml-cuda/ggml-cuda.cu` (pool counter, trimming, `ggml_backend_cuda_pool_held`), `src/llama-moe-stream.cpp/.h`
(end-state line), `docs/settings.md`; commit `198637724`.

Seen on the way (a follow-up, not fixed): a few MiB less room switches the whole VRAM expert pool off when the tail does not fit
(no smaller tail is tried), and prompts get 4x slower - the cliff above.

## B. An 8 GB card on this machine

`kit/bench/pod2/vram_hold.cu` now also builds with hipcc (`hipcc -O2 -x hip`; the CUDA calls mapped to HIP; MoE_Lab
`bench/pod2/vram_hold.cu`); `vram_hold: holding 8372 MiB, 7864 MiB of 16304 MiB free`. run_cases.py's own hold works without
nvidia-smi. Cases `up/cases_8g.json` (written by `up/mk8g.py`): the `flint_tw_t4` / `ling_tw_t4` commands with
`--moe-stream-gpu-layers 0` (Flint also `--moe-stream-copy-ahead 2`), turns: story 300, sum, 8k, 32k. Final ROCm build:

| model | placement | load | story tok/s | sum | 8k prompt | reply after 8k | 32k prompt | reply after 32k | checks |
|---|---|---|---|---|---|---|---|---|---|
| Flint | 0 whole layers, VRAM experts 46 per layer x 47 layers = 4.381 GiB | 14 s | 30.17 | ok | **55 tok/s** | **1.73 tok/s** | 149 tok/s | **1.37 tok/s** | ok |
| Ling | 0 whole layers, VRAM experts 23 per layer x 40 layers = 3.065 GiB | 18 s | 13.38 | ok | 341 tok/s | 10.06 | 340 tok/s | 9.20 | ok |

No abort, every answer correct. **Flint on the 8 GB card is very slow after a long prompt** (an earlier run without the sum / 8k
turns: 32k prompt 124 tok/s, reply 1.01 tok/s, 989 ms/token while the SSD wait was only 27 ms/token). The cause found so far: RAM,
not VRAM. 47 streamed layers x 139 slots = 11.66 GiB of pools on a 15.5 GiB machine with no process memory cap (`RAM limit:
MemAvailable 14514 MiB - keep-free 1024 MiB ... no process memory cap`); the long prompt's own memory pushed the process to a peak
RSS of 14.38 GiB, the machine has 4 GiB of swap and swapped (`pswpin` / `pswpout` in the millions of pages), the tail-wave copies ran
at ~130 MB/s (`9885.2 MiB in 74338 ms of copy calls`). The engine gives slots back for a prompt's own memory only against a memory cap
(none here). Not fixed (not small): the follow-up is to apply the same give-back against MemAvailable when there is no cap, or to
count the long prompt's host memory in the RAM plan. Ling (fewer slots, 13.3 GiB peak) is fine.

## D. Vulkan on the real card

Build: `cmake -B build-vk ... -DGGML_VULKAN=ON` needed two packages the container did not have: `glslc` and `spirv-headers`
(installed with apt in the rented container, as the lead's item said; nothing else changed there). Build ok (548 steps).
- `check.py --model Ling-3.0-tiny ... --build build-vk --no-build --no-ctest`: `OK streamed run on the GPU (Vulkan0: AMD Radeon RX
  9060 XT (RADV GFX1200) ...)` ... **`Everything passed.`**
- Flint (`flint_tw_t4`, `results_vk/`): loaded in 14 s, story **23.51 tok/s**, sum ok, 8k prompt **670 tok/s** (reply 22.87),
  `check: ok`.
- Ling (`ling_tw_t4`): loaded in 12 s, story **10.96 tok/s**, sum ok, 8k prompt **177 tok/s** (reply 8.06), `check: ok`.

The engine's "off" lines under Vulkan (both models unless named):
- `--moe-stream-gpu-share-q8 1: no CUDA / ROCm backend, ignored`
- `--moe-stream-vram-follow: off - the backend of Vulkan0 has no device-wide wait; the GPU pool keeps a fixed set (no tail, no copy-back)`
- `needed-early: the host producer stays (the host-side stand-in producer): the GPU backend has no guess-mark functions`
- `--moe-stream-prompt-split: off - it needs needed-early's GPU producer (above)`
- `vram-parallel: off - the GPU producer of --moe-stream-needed-early is not active ...`
- Flint: `the VRAM pool is off - a layer-major call's tensors (a GPU buffer of their own) fit the free VRAM (245 MiB, less 512.0 MiB
  of growth) for 0 tokens: n_batch 65536 -> 2048 (a longer prompt comes in several calls)`
- **Ling: `the GPU expert pool is off: allocating 7.695 GiB on Vulkan0 failed - the routed experts run from RAM`** (and with it VRAM
  follow, vram-parallel, copy-back, the end fill, copy-ahead, the tail): `free after the context 8.031 of 15.922 GiB` stays unused.
  RADV's `maxMemoryAllocationSize` is `0xfffffffc` (4 GiB), so one 7.7 GiB buffer cannot exist; Flint's 1.1 GiB pool fits. No crash
  (the fallback works); a follow-up: build the VRAM pool from several buffers of at most the device's largest allocation under Vulkan.

No crash, no wrong answer.

## E. Plain llama.cpp on Ling (bounded)

The ROCm build with no `--moe-stream` flag (`-c 65536 -ctk q8_0 -ctv q8_0 -np 1 -t 4 -b 2048 -ub 2048 --cache-ram 0
--ctx-checkpoints 2`, its own layer fit), Ling's file, one chat request with `max_tokens 48`:
- load: `/health` ok after **79 s**;
- prompt: the first 8 tokens in 129 s (`prompt processing, n_tokens = 8, progress = 0.29, t = 129.25 s / 0.06 tokens per second`);
- then the server **aborted**: `ggml_cuda_compute_forward: MUL_MAT failed` / `ROCm error: no kernel image is available for
  execution on the device` (`ggml-cuda.cu:2833`, in `process_ubatch` of the same request), 209 s after the start. No reply token.
  Not ours to fix in this round (plain llama.cpp, no engine code on the path); which matmul it was is not proven (the node is not
  named). A guess to check: a 20-row F32 / BF16 matmul that goes to hipBLAS(Lt), for which this ROCm install has one gfx1200
  library file (`/opt/rocm/lib/hipblaslt/library`: 1 file for gfx1200, rocBLAS: 110). Log: `/root/tw/results9/plain_ling.log`.

## C. Not reached
The wait-mode pairs and the look at Ling's CPU-GPU waits were not done (time).

## Final table: AMD RX 9060 XT 16 GB, ROCm and Vulkan

Final code (`amd-speed` = `198637724`), `-t 4`, run_cases.py (story 400 tokens for Flint / 300 for Ling, the sum check, the 8k prompt
with the planted fact); ROCm rows from `results9/final_*.out` (20:40-20:42), Vulkan rows from `results9/vk_*.out` (20:37-20:40),
8 GB rows from section B. One run each.

| model | build | layout | load | reply tok/s | 8k prompt tok/s | checks |
|---|---|---|---|---|---|---|
| Flint | ROCm | 27 whole layers + the VRAM pool | 13 s | 45.01 | 1258 | sum ok, 8k ok |
| Ling | ROCm | 2 whole layers + 70-71 VRAM experts / layer | 19 s | 20.10 | 324 | sum ok, 8k ok |
| Flint | Vulkan | 27 whole layers + 27 VRAM experts / layer (no follow, no tail) | 14 s | 23.51 | 670 | sum ok, 8k ok |
| Ling | Vulkan | 2 whole layers, VRAM pool off (4 GiB allocation limit) | 12 s | 10.96 | 177 | sum ok, 8k ok |
| Flint | ROCm, 8 GB held | no whole layers, 46 VRAM experts / layer | 14 s | 30.17 | 55 (RAM swapping, see B) | sum ok, 8k ok, 32k ok |
| Ling | ROCm, 8 GB held | no whole layers, 23 VRAM experts / layer | 18 s | 13.38 | 341 | sum ok, 8k ok, 32k ok |
| Ling | ROCm, plain llama.cpp | its own layer fit, no streaming | 79 s | none (abort) | - | - |

Portable checks on the final code: ROCm `check.py` "Everything passed." (`results9/check3.log`), Vulkan "Everything passed."
(`results9/check_vk.log`). The machine's `/root/tw/src` = the branch (md5 of every changed file checked) plus the lead's
`GGML_HIP_SCHEDULE` experiment in `ggml-cuda.cu`.

## Round 2: not done / follow-ups
- Flint on an 8 GB card swaps after long prompts on a 15.5 GiB machine with no memory cap (section B): the prompt's own host memory
  is not given back against MemAvailable when there is no cap.
- The VRAM pool switches itself off entirely when its tail does not fit (no smaller tail tried): a few MiB less room = 4x slower
  prompts (section A).
- Vulkan: the VRAM expert pool as one buffer fails above the device's largest allocation (4 GiB on RADV); Ling loses its pool.
- Plain llama.cpp on Ling aborts with `no kernel image is available` in a MUL_MAT (not ours; the node not identified).
- A failed scratch allocation still aborts (first round's (3)).
- C (wait-mode pairs, Ling's CPU-GPU waits) not done.

Round 2 time: about 75 minutes of work.
