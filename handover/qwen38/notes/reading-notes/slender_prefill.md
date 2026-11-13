# SlenderEngine (branch speed-fork, 8af6fd84) - how a PROMPT (prefill) is processed for Qwen3.8-Flash-Next

Read-only study. All paths are relative to `traces/refs/slender/` (the fork) unless they start with `strata-0.2.11/` (the base).
"(inferred)" marks things not stated in code or docs that I concluded; "(not verified)" marks things I did not open.
Sources read in full or in the relevant parts: `src/prefill/prefill.cpp` (run_impl, Stager, carve, plan code), `src/prefill/kernels.cu`
(HC / GDN / router / combine), `src/prefill/gemm.cu` (dense GEMM dispatch), `src/prefill/moe_mmq.cu` + `moe_fused_iq.cu` (type coverage),
`src/core/expert_source.cpp` (file reads), `src/program/generate.cpp` (chunk choice, loan, room, refill, checkpoints, fork defaults),
`src/core/{native_head,dflash2,mtp,nv_defaults,native_dense,conversation_state}.cpp`, `src/ngram/ple_reader.cpp`, `src/kernels/ngram.cpp`,
`src/platform/direct_file.cpp`, `src/kernels/cuda/qsa_prompt_attn.cu`, the headers, `docs/fork/PROMPT-ROOM.md`, `PERFORMANCE.md`, `FORK.md`.
`diff -r strata-0.2.11/src slender/src` lists 15 changed `.cpp/.cu` files (prefill.cpp, gemm.cu, expert_source.cpp, layer.cpp, generate.cpp,
pool.cpp, iq_avx512.cpp, native_dense.cpp, native_head.cpp, nv_defaults.cpp (+test), mtp.cpp, dflash2.cpp, conversation_state.cpp, pool_tasks_test.cpp)
plus headers (`ahead_trace.hpp`, `fork_defaults.hpp` new).

Model constants used by the prompt path (`src/prefill/prefill.cpp:208-209`): n_embd N=2560, 4 hyper-connection streams HC=4 (residual row D=10240 floats
per token), HC low-rank LR=320, top-K=10 of NE=512 experts, 48 layers (`g.n_layers`), GDN value heads HV=48, key heads 16, S=128. Every 4th layer is a
full-attention ("QSA") layer: layers 3,7,...,47 = 12 QSA layers, 36 GDN layers (`include/strata/core/geometry.hpp:26,65-76`). Every layer has a routed MoE half
(24,576 experts total, `docs/fork/PROMPT-ROOM.md:31`) plus a shared expert.

---------------------------------------------------------------------------------------------------------------------------------

## 0. Summary of what the fork changed vs Strata 0.2.11 for the prompt path (from `diff`)

Code changes that touch prompts (everything else in the diff is decode / CPU kernels / HIP-only):

| Change | Where (fork) | Default |
|---|---|---|
| Prompt room: lend the DFlash2 pack, the draft head and the output head (and MTP experts) as extra buffer space for one long prompt; relayout with up to 4 extra regions; host buffers sized for the larger chunk | `src/prefill/prefill.cpp:656-677` (Alloc rooms), `:1813-1840` (tmax), `:2159-2210` (relayout overloads); `include/strata/prefill/prefill.hpp:86-94`; `src/core/dflash2.cpp:770-815` (room_lend/reclaim); `src/core/native_head.cpp:285-340`; `src/program/generate.cpp:11179-11250, 16278-16323, 17006-17055, 17432-17480` | on in the pack defaults with `--serve` + `--draft dflash2` / `--mtp` (`generate.cpp:3395-3405`); MTP_ROOM_MIN existed in 0.2.11, DF2_ROOM_* and *_ROOM_CHUNK/_HEAD are new |
| In-place prompt stager: stager thread reads an expert straight into its pinned buffer (no scratch + memcpy); +8 KiB buffer; `skip_of` offset | `src/prefill/prefill.cpp:706-709, 735, 819-829, 889`; `src/core/expert_source.cpp:4072-4088` (copy_blob_aligned), `:3155-3156, 3180-3185` (Windows read_direct in_place) | `STRATA_STAGER_IN_PLACE=1` in fork pack defaults (`generate.cpp:3371-3374`); Windows only effective |
| Stager threads 3 (was `min(4, hw/4)`, or 32 with a file tier) via `g_fork_pack_defaults`; fork also sets `STRATA_STAGER_THREADS=3`, `_BATCH=1`, `_RING=128` | `prefill.cpp:1873-1875`; `generate.cpp:3375-3379` | on in pack defaults |
| Lend cap 90 % (0.2.11: 85 % unless >=90 % of expert bytes are pinned) | `generate.cpp:6291-6295, 8844, 9836-9841` | 90 in pack defaults |
| `STRATA_DIRECT_BATCH=N` (reads published per batch; 0.2.11 hard-coded 16) | `expert_source.cpp:2389-2392` | code default 16; fork pack default 1 |
| Second-SSD mirror: each unbuffered read >=256 KiB is split between two drives | `expert_source.cpp:2824-2841, 3217-3227` | off (`STRATA_EXPERTS_MIRROR`) |
| Parallel refill of the lent slots (`STRATA_REFILL_PINNED=N`) | `generate.cpp:525-599, 17300-17311` | off (user launcher sets 4) |
| Layer-major: CPU computes few-row experts (`STRATA_LM_CPU_ROWS`); smaller super-chunk when pinned RAM is short | `prefill.cpp:6897-7000, 7115-7165, 6576-6597` | off |
| `STRATA_DENSE_ARENA`: dense weights in one allocation (saves 2 MiB granule rounding -> more expert-cache slots -> bigger chunk) | `src/core/native_dense.cpp:176-196, 321-360` | on in pack defaults |
| `STRATA_HC_UPMIX=1` default (code is 0.2.11's) | `generate.cpp:3383` | on in pack defaults (0.2.11: off, except sm_121) |
| `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_PREFILL_CPU_SHARE_MAX=1024`, `STRATA_PF_SMALL=0` (answer-preserving, every build) | `generate.cpp:3359-3361` | on |
| sm_89 table (decode kernels only) | `src/core/nv_defaults.cpp:41-52` | on for RTX 40 |
| Test-only: `STRATA_ROUTE_DUMP`, `STRATA_PREFILL_DUMP_R_STEP`, `STRATA_AHEAD_TRACE` | `prefill.cpp:535-562`, `expert_source.cpp:81-132` | off |
| HIP-only: `STRATA_PF_W16_LAYER_MIB` (`gemm.cu:1413-1454`), `gemm.layer_cache_release()` | | off / no effect on CUDA |
| `STRATA_HEAD_FAST_LOAD`, `STRATA_HEAD_ALIAS_NARROW`, `STRATA_DF2_PARK` | `native_head.cpp:19-122`, `generate.cpp:11063-11076, 17700-17717` | off |

The prompt kernels themselves (HC fusion, GDN, MMQ, QSA, PLE reader) are Strata 0.2.11's, unchanged.

---------------------------------------------------------------------------------------------------------------------------------

## a. The overall loop

### a.1 Request to chunks

* A request's prompt is cut into segments at [`reread_to`, root, message boundary, turn, **n-1**] (`generate.cpp:18415`). The loop over segments is `generate.cpp:18444-18527`.
  A segment of <= `--short-read` (default 64, `generate.cpp:1040`) tokens is read through the decode-style verify windows, not batched (`windows_ok`, `generate.cpp:17088-17095`);
  a longer one goes to the batched path `read_part -> Prefill::run` after `lend(...)` (`:18466`). The **last prompt token** always goes through the first verify window
  (that produces the first reply logits); the batched prefill never runs the LM head. (`generate.cpp:18541-18549`)
* `Prefill::run` (`prefill.cpp:7927-8005`) calls `run_impl`; the layer-major walk is only taken if `STRATA_PREFILL_LAYER_MAJOR` resolves to on, which is off everywhere by default
  (`include/strata/core/sm120_series.hpp:131`: `prefill_layer_major_default` returns false) and the fork sets `STRATA_PF_SMALL=0` so the 0.2.11 "small card turns layer-major on" rule is off
  (`generate.cpp:10852-10897`, `:3361`). So the default is **chunk-major**: a chunk of C tokens goes through all 48 layers, then the next chunk.
* Chunk loop: `for (c0 = 0; c0 < n; c0 += C)` at `prefill.cpp:3437`; `C = forced ? forced : planned ? planned : m.T` (`:3353-3354`), `T = min(C, n-c0)` (`:3454`). The last chunk is the remainder.
  (A layer-split "stage helper"/"pipeline" exists - multi-GPU, irrelevant for one card.)

### a.2 How the chunk size is chosen
1. `--prefill auto` / `auto:N` (`generate.cpp:3701-3720`): N (8192 | 16384 | 32768) is the ceiling `prefill_auto_max`; `STRATA_PREFILL_AUTO_MAX` also sets it. Fork pack defaults set
   `STRATA_PREFILL_AUTO_MAX=16384` when `--resident-budget-gib` is given and the PC has >= 28 GB RAM (`generate.cpp:3408`). On NVIDIA cc >= 8.0 a *bare* `--prefill auto` may reach 32768
   (`generate.cpp:4048-4073`, `STRATA_NV_CHUNK=0` keeps 8192); a RAM guard steps 32768 -> 16384 -> 8192 (`:9865-9895`). Never past the context (`:9846`).
2. The ceiling is shrunk to what the expert cache can lend: a bisection on a 256-token grid for the largest chunk whose buffers (`Prefill::bytes_needed`) fit in the cache's tail with
   `k + 128 <= slots` and `k <= LEND_PCT % of slots` (`generate.cpp:9903-9988`; kAutoLendPct 90 in the fork, 85 in 0.2.11 `:9836-9841`). The streamed ring is part of the same budget: the
   scan wants the largest chunk that still leaves the ring full (`:9932-9982`; "ring full" = `ring_cap_for(small)`; `prefill.cpp:2925-2944`).
3. Optional host-aware refinement `host_plan_chunk` (`generate.cpp:10004-10060`, `STRATA_PREFILL_HOST_PLAN=0` off): prices chunk x ring against RAM page cache and drive speed; with unbuffered reads the page cache is
   counted as 0 (`:10022`).
4. Per request: a prompt shorter than the chunk gets `min(chunk, round_up_256(tokens))` (`generate.cpp:9794-9800`); `lend()` relayouts the loan for just that chunk (`:17414-17554`). `STRATA_PREFILL_EQUAL=1`
   (off) evens chunk sizes (`:9808-9818`). Doc numbers: a 50K prompt in chunks of 8,448 = 6 passes, 10,240 = 5 passes (`docs/fork/PROMPT-ROOM.md:36`).

### a.3 What runs where (one chunk, one layer)
Everything compute is on the GPU stream `m.cs`; the host thread only enqueues, builds routing tables (MMQ path) and waits. Host threads: expert **stager** threads ("strata-stage", `prefill.cpp:756-761`), the expert
**issuer** thread ("strata-issue", `prefill.cpp:3873-3958`, default on `STRATA_PREFILL_ISSUER`, `:3799-3802`), the PLE gather thread (`std::async`, `:3571-3575, 3619-3623`) + PLE reader worker, optional CPU-share pool thread.
Streams / events (`struct Prefill::Impl`, `prefill.cpp:1130-1317`):
* `m.cs` compute stream; `m.copy` = expert host->device copy stream (non-blocking); optional `m.kv_copy` (KV prefetch, `STRATA_KV_PREFETCH` off), `m.ple_copy` (`STRATA_PLE_COPY_STREAM` off).
* `copied[slot]` recorded on `m.copy` after each expert DMA (compute waits on it with `cudaStreamWaitEvent`, `prefill.cpp:3970-3982`); `used[slot]` recorded on `m.cs` when the compute has consumed a ring slot
  (copy stream waits on it before overwriting, `:3779, 3935`); `dma_done[j % kRing]` frees a pinned stager buffer once its DMA is queued/complete (`:893-898`).

Per layer `l` (loop `prefill.cpp:4010`), in this order:
1. (layer 1 only) PLE block: key GEMM from the PLE rows, value GEMM, post-ops (`:4026-4084`).
2. `half = 0`: **hyper-connection read** (HC norm row-scales, down GEMM 10240->320, SiLU, up GEMM 320->10240 + mix, inject GEMM 10240->4) -> `mixed` (T x 2560, FP32/BF16/FP16 images) -> the mixer: GDN (36 layers) or QSA (12 layers) -> `bo`
   -> **HC write** (`R += bo * 2*sigmoid(inj/4)` per stream, fused with the next half's norm).
3. `half = 1`: HC read again with the layer's FFN HC weights -> router + shared expert + routed experts + `moe_combine` -> HC write (fused with the *next layer's* attention-half norm).
After the last layer: `ple_land()` safety, join the issuer, `on_chunk(R, T, p0)` after a `cudaStreamSynchronize` (MTP draft-layer K/V or DFlash2 taps, checkpoints, progress `PP ...` line)
(`prefill.cpp:6120-6243`). At the end of the run: sync `cs` and `copy` and `kv_copy` (`:6266-6282`).

Router: BF16 GEMM of the BF16 image of `mixed` with `ffn_gate_inp.weight` -> logits T x 512 (`prefill.cpp:4534`), then `route` = softmax over 512, top-10 by repeated warp argmax (ties -> lower index), weights renormalised by their sum
(`src/prefill/kernels.cu:2218-2255`). No threshold, no skipping.

---------------------------------------------------------------------------------------------------------------------------------

## b. Expert handling in a prompt

### b.1 Where expert bytes come from (tiers)
Per (layer, expert) at the start of a chunk (`prefill.cpp:3730-3760`):
1. **VRAM expert cache slot** (`m.host_res[l*NE+e] >= 0`, `m.cache->device_slot(...)`): used in place (resident). During a prompt the tail of the cache is *lent* (marked `kNotResident` in `host_res`, `generate.cpp:17524-17532`),
   so up to 90 % (fork) of the cache's slots are not resident while a long prompt runs. Doc: ~4 GiB of experts fit in VRAM; the other ~43 GiB are RAM tier (10 GiB) + files (`docs/fork/PROMPT-ROOM.md:31-34`).
2. **Pinned RAM arena = "RAM tier / complement"** (`--resident-budget-gib N`: the N GiB of experts the GPU does not hold that the usage profile ranks hottest are copied into RAM; `docs/DETAILS.md:251`). `ExpertSource::pinned(l,e)` is true when the blob
   is in page-locked, CUDA-registered memory (`expert_source.cpp:5634-5647`); then the blob pointer is DMA'd directly (`prefill.cpp:3781-3783`). The fork pins only a prefix when RAM is tight (`STRATA_PARTIAL_PIN`, `expert_source.cpp:5645`).
   Experts of lent slots are only in RAM if the budget covered them after the ranked ones ("the lend region comes after the ranked experts", `generate.cpp:10524`; `expert_source.hpp:156-160`) - otherwise they come from the file at prompt time and again at refill.
3. **RAM tier but not pinned** (the part of the arena the OS would not let the driver pin): `pinned()` false but `transient()` false -> stager job that `memcpy`s from the arena into a pinned stage buffer (`prefill.cpp:3754-3757, 820`).
4. **File tier** = `experts.bin` of the pack (per expert one contiguous blob `[gate rows | up rows | down rows]`, `include/strata/prefill/moe_fused_iq.hpp:1-5`) on the SSD, read with unbuffered reads (b.2) into a pinned stage buffer by a stager job (`ExpertSource::transient()` true, `expert_source.cpp:3974-3982`).
   (A GGUF "read in place" variant exists too; `role_ptr_` non-empty, not the fork's mirror/in-place path.)
5. RAM tier is **not** updated by prompt reads: `through_context` ignores >16-token calls "a prompt's chunks would churn the tier" (`expert_source.cpp:2435`); the prompt stager reads via `copy_blob_aligned/copy_blobs` into private buffers.

### b.2 The stager and the SSD reads
`struct Stager` (`prefill.cpp:695-908`):
* `kRing` pinned host buffers (default 16, `STRATA_STAGER_RING`; with a file tier and the var unset: 4 x threads, `prefill.cpp:1875`; fork pack default `STRATA_STAGER_RING=128`, `generate.cpp:3375`). Each buffer = `MAXBLOB()` (+8192 if in_place), allocated with `cudaHostAlloc` (pageable fallback) in `Stager::init` (`:729-763`). `MAXBLOB` = largest expert blob of the pack (`prefill.cpp:211`).
* Worker threads: fork pack default 3 (`STRATA_STAGER_THREADS=3`); 0.2.11 default with a file tier = 32 threads, batch 8, ring = 4 x threads (`prefill.cpp:1853-1884`); fork also sets `STRATA_STAGER_BATCH=1` so a thread claims 1 job at a time (`claim`, `:842-852`).
* Job j = a layer's j-th non-pinned expert in launch order; lands in host buffer `j % kRing`; free again once the DMA of job `j - kRing` is queued (`issued` watermark) and `ready` flags (`:795-806`).
* **In-place read (fork, `STRATA_STAGER_IN_PLACE=1`)**: `copy_blob_aligned` reads the expert directly into the pinned buffer at its sector offset (4 KiB-aligned handle read, `expert_source.cpp:4072-4088`); the buffer reader then returns `buf + skip` (`prefill.cpp:889`). Without it
  (0.2.11): `copy_blobs` merges neighbouring blobs into <= 32 MiB requests, reads into a thread-local scratch and memcpy's into the stage buffer (`expert_source.cpp:4041-4067, 3351-3355`).
* Windows read API (`FileExpertSource::read_direct`, `expert_source.cpp:3119-3357`): handles opened `CreateFileW(..., FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)` (`:2813-2814`); each batch = one overlapped `ReadFile` per request, **all issued before the first wait**, then `GetOverlappedResult(TRUE)` per request
  (`:3241-3254, 3337-3349`); windows are 4 KiB sector-aligned; windows closer than 1 MiB (same role) merge up to 32 MiB (`:3126, 3187-3198`), except in-place ones which are a request of their own (`:3180-3186`). With `STRATA_EXPERTS_MIRROR` a request >= 256 KiB is split between the two drives
  (share 0.5, `:3216-3229`). No IOCP in this path (events). With the fork's 3 stager threads x 1 blob each, **roughly 3 reads in flight (6 I/Os with the mirror)** from the stager (inferred from thread count and batch 1), plus refill reads.
* Linux variant (not the fork's target): `O_DIRECT` fds (`:2850`) + Linux AIO (`io_setup/io_submit`, depth 256 per thread, merging only touching windows; `:3358-3440`). There is **no io_uring** anywhere (`grep` empty). mmap is used for the mapped file tier when not unbuffered (`:873, 1574-1584, 1607`).

### b.3 The staging ring in VRAM
* A chunk of >= `stream_all_min()` tokens (default **1024**, `STRATA_PREFILL_STREAM_MIN`; `prefill.cpp:258-262`) takes the "streamed walk": `stream_all = m.ring > STAGE && T >= stream_all_min() && m.src` (`:3671`). `ring_slots(T)` returns the 8-slot `STAGE` below that size (`:438`).
* Ring slots are `stage_dev[i]`, each `MAXBLOB()` bytes in the loan (`carve`, `prefill.cpp:2099-2135`; one allocation when the prompt path owns its buffers). IQ3_S pack: header comment says 2.54 MiB per blob / "199 slots" for the byte budget (`include/strata/prefill/prefill.hpp:190`, `prefill.cpp:336`);
  arithmetic from the formats: gate/up 1280 rows x 2560 IQ3_S + down 2560 rows x 640 IQ4_NL = 2.22 MiB (inferred; layers with other types are bigger).
* Slot count (`ring_slots`, `prefill.cpp:403-439`): `STRATA_PREFILL_RING` if set; else `g_ring_override`; else, with the byte-budget scan, the loan's leftover (`g_ring_budget`, `set_ring_budget`); else 96 when less than 90 % of experts are pinned ("unpinned arm"), 384 / 1024 when >= 90 %
  pinned. Capped by `ring_cap()` = 512 (1024 only for the Q2_0 pack with fused kernels, `:370`). Docs: fork default ring **96**; the user's launcher sets `STRATA_PREFILL_RING=192` and `STRATA_PREFILL_LEND_PCT=95` (`docs/fork/PERFORMANCE.md:187-193`).
  `set_pinned_share` is computed from `srcp->pinned(l,e)` over all experts (`generate.cpp:9755-9767`). With a 10 GiB RAM tier of 47 GiB, share < 0.9 -> 96 slots (inferred; matches the doc).
* **Fill order**: the whole chunk's stream is planned up front as one sequence `seq` over layers LB..LE-1: for each layer, every expert id 0..511 that is **not VRAM-resident** (and not on a peer GPU), in **ascending expert id**, routed or not (`prefill.cpp:3712-3767`). Entry k -> slot `k % ring`;
  copy k is issued once entry `k - ring` has been consumed: the issuer thread keeps `a_issued < a_consumed + ring` (`:3916-3919`), i.e. the ring is a **sliding window across layer boundaries**: while a layer's attention half runs, the next layer's first experts already arrive. The copy for an expert
  is `cudaMemcpyAsync(stage_dev[sl], pinned_blob_or_stage_buffer, blob_bytes, H2D, m.copy)` + `cudaEventRecord(copied[sl], m.copy)` (`:3942-3951`).
* So for a chunk with T >= 1024: **each non-resident expert is copied exactly once per layer per chunk (one pass over the experts per chunk)**, whether or not the chunk's tokens route to it. An expert nobody routed to only gives its slot back (`release_to`, `:5947-5956`). With 1024 tokens x top-10 of 512 nearly every expert is routed
  anyway (comment `:4720-4721`). `STRATA_PREFILL_ROUTED_STREAM=1` (opt-in, `prefill.cpp:3819-3838`, `routed_stream.hpp`) lets the issuer drop not-yet-issued entries no token routes to; default off.
* A chunk below 1024 tokens uses the **staged walk** instead: the host downloads the routed ids after a `cudaStreamSynchronize` (`prefill.cpp:4764-4821`), groups, and stages **only routed non-resident experts** (id order, `order`) with 8 transfers ahead (`STAGE`) or `STAGE_GRP`=32 with MMQ grouping (`:5539-5548, 5589-5617, 5915-5941`);
  with the CPU share armed (below 1024 tokens by default) the CPU computes some of the few-row experts (`:4769-4806, 5024-5094`; `cpu::MAXT` = 8, `include/strata/kernels/cpu/expert.hpp:143`).
* Which experts are *grouped*: the MoE walk consumes `order` = routed experts in id order; MMQ groups are **16 consecutive routed experts** (`MMQ_GROUP`, `prefill.cpp:1479`), each with its own contiguous block of rows (rows sorted by expert id, within an expert in token order: `slot_h/src_h` built at `:4964-4970`).
  Optional row-count sorting of resident groups (`STRATA_MMQ_RESIDENT_SORT_NE`, `STRATA_MMQ_GROUP_SORT`) is off.

### b.4 The expert GEMMs (kernels)
Three paths, chosen per layer (`prefill.cpp:4619-4660`):

1. **MMQ (default on a card that is not an 80+-SM sm_120 / sm_121)** - llama.cpp's `mul_mat_q` int8 tensor-core kernels (ggml-cuda `mmq.cuh`, MIT) compiled from the pinned llama.cpp checkout (`src/prefill/moe_mmq.cu:1-9`, header `include/strata/prefill/moe_mmq.hpp`). For a group of 16 experts:
   `quantize_scatter` makes each token's activation row q8_1 once and writes it to its 10 rows (`prefill.cpp:5127`), `gather_native_group` copies the 16 blobs (resident or streamed ring slots) into group buffers `grp_gu/grp_d` with a zeroed 4 KiB tail because MMQ over-reads (`moe_mmq.cu:680-700`, `prefill.cpp:5639-5673`,
   `MMQ_TAIL` `:1483`), then gate/up product `[1280 x 2560]` (`:5837-5843`), `swiglu` (split halves, FP32 `moe_mmq.cu:141-147`), H quantized to q8_1 (`:5852`), down product `[2560 x 640]` (`:5853-5860`) into `Dm` rows. **One wait on the group's last streamed copy** and one
   release event per group (`:5642-5645, 5666-5670`); `STRATA_PREFILL_GROUP_GATHER=0` restores per-expert (`:5535-5538`). Resident experts are also gathered D2D (the pointer-table `STRATA_MMQ_DIRECT` that reads them in place is a build option `STRATA_MMQ_TAB`=OFF and runtime off, `CMakeLists.txt:151`, `moe_mmq.cu:619-631`).
   After all groups: `moe_combine` (`kernels.cu:2315-2330`): `bo[t] = sum_k w[t,k]*Dm[slot[t,k]] + shared[t]*sigmoid(sg[t])`.
2. **Fused native int8 kernels** `fused::experts_native` (`src/prefill/moe_fused_iq.cu`, header `moe_fused_iq.hpp`): activations quantized per 32 values in natural order, weight blocks decoded to int8 in shared memory, int32 dot -> FP32 (`moe_fused_iq.cu:1-12`); no host sync, grouping on the GPU (`fused::group`), batches of up to
   `kMaxBatch` experts per launch, at most ring/3 streamed per batch (`prefill.cpp:4676-4754`). Covered: gate/up IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS (Q2_0 gate/up only with `STRATA_PF_FUSED_Q2N`), down Q2_0, IQ4_NL (`moe_fused_iq.hpp:12-13`, `moe_fused_iq.cu:1873-1890`).
   **Default on**: CUDA sm_120 cards with >= 80 SMs (RTX 5080/5090/PRO) for IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS gate/up with IQ4_NL / Q2_0 down (`moe_fused_iq.cu:2011-2018`), and sm_121 via the nv table (`src/core/nv_defaults.cpp:66-75`). **Not default on sm_89 / sm_120 with < 80 SMs / most other cards** -> they use MMQ unless `STRATA_PF_FUSED=1`.
   The fork's pack defaults do not set it (`generate.cpp:3359-3405`).
3. **FP16 path (fallback)**: per expert `iq_dequant_gu_f16` / `iq_dequant_f16` into `dq_gu/dq_d` (FP16), `gemm.f16` (cuBLAS GemmEx FP16 in, FP32 out), `swiglu_interleaved`, `gemm.f16` (`prefill.cpp:5871-5899`). Used for layers MMQ does not cover (`mmq_plan`, `prefill.cpp:1523-1595`) - e.g. IQ1_M, Q3_K, Q4_0, Q2_K-like formats not in MMQ's list.
   (`STRATA_PF_WMMA` FP16 tiles: off.)

Quant types:
* **MMQ `supported()`** (`moe_mmq.cu:319-347`): Q2_0 (ggml type 42, the Strata 2-bit type), IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, **IQ3_S**, IQ4_NL, IQ4_XS, Q8_0; with `STRATA_MMQ_KQUANTS` build (CMake ON, `CMakeLists.txt:134`) and runtime `STRATA_MMQ_KQ=1`: Q4_K, Q5_K, Q5_1, Q6_K (`moe_mmq.cu:41-55, 335-339`). `STRATA_MMQ_KQ=1` is a default on NVIDIA cc >= 8.0 (`include/strata/core/nv_defaults.hpp:36-46`, `nv_defaults.cpp:135-139`) - this is a K-quant prompt change that moves answers (KL 0.024-0.035 vs the FP16 path per the header comment).
  NVFP4 (w4a8 default) also; Q5_0 only with `STRATA_ORCA_Q4KS_MMQ`. Not in MMQ: Q3_K, Q4_0, Q4_1, Q2_K, IQ1_M.
* **FP16 dequant dispatch** (`src/kernels/cuda/iq_kernels.cu:3315-3345, 3381-3388`): IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ1_M, IQ4_XS, Q3_K, Q2_0, Q4_K, Q5_K, Q6_K, Q5_1, Q5_0, Q4_0, Q4_1, NVFP4, Q8_0, BF16, TQ variants. **Q2_K is absent** (no `case 10`) (inferred: unsupported).
* Fused native: see 2. (K-quant fused is gfx11-only.)
* The expert layout `layout.native` requires every layer's experts to be 2560 x 640 (`prefill.cpp:1788-1793`).

### b.5 Dense (non-expert) projections in a prompt
`native_proj` -> `Gemm::native` (`gemm.cu:1363-1547`): the GGUF-quantized dense weight (attention, GDN, shared expert, ...) is **dequantized to FP16 into a 32 M-element scratch** (`strata::kernels::dequant_f16`, row slices if larger) and multiplied by FP16 activations with `cublasGemmEx` (FP16 x FP16, FP32 accumulate, `gemm.cu:1171-1255`).
Router / HC / indexer / PLE-key weights are BF16: `bf16_proj` -> `Gemm::bf16` (BF16 images of the activations, `prefill.cpp:3040-3064`). The FP16-weight cache `STRATA_PF_W16_CACHE_MIB` is off unless Linux CUDA + batch slots (`gemm.cu:1464-1485`).

---------------------------------------------------------------------------------------------------------------------------------

## c. "Prompt room", the VRAM split for a prompt, and the refill

### c.1 VRAM during a prompt
* Permanent VRAM (not lent): dense weights (NativeDense; with `STRATA_DENSE_ARENA=1` in one allocation, `native_dense.cpp:176-196`), output head (`native_head`, 497 MiB per the doc), the drafter's weights (DFlash2 pack 323 MiB + draft head 81 MiB, or MTP's 512 experts 675 MiB, `docs/fork/PROMPT-ROOM.md:46-50`), GDN/QSA session state, the K/V window (`--kv-resident`, default window 32K per the doc), streaming K/V staging, the DFlash2 tap staging, and the **expert cache**.
* The prompt path's buffers (emb, R, xn16, lo, gated, inj, mixed*, steps, the shared region (max of GDN set / QSA set / MoE set), the two dequant slots, MMQ group buffers, **the staging ring**, `ple_emb`, KV staging) are carved by `Prefill::carve` (`prefill.cpp:2002-2157`) out of a **borrowed region = the last slots of the expert cache** ("the loan"); `Alloc::take` is a 256-byte-aligned bump allocator (`:648-686`).
  A comment gives the per-token cost: ~680 KB a token before the "attention half and MoE half share their buffers" rule saved ~260 KB (`prefill.cpp:1453-1457`).
* At start the loan is sized for the chunk found in a.2; each request relayouts for just its needs, starting at `slots - need` (`generate.cpp:17489-17523`); lent slots' (layer, expert) pairs are put in `p.lent` and `host_res[i] = kNotResident` (`:17524-17532`), then the residency table is uploaded (`res_upload`, `:17537`).
  Slots from the cache tail are the *lowest-ranked* experts the GPU held (inferred from "profile ranks hottest first").

### c.2 The room (fork)
Idea (`docs/fork/PROMPT-ROOM.md:42-63`): while a long prompt is read the DFlash2 pack, the draft head and the output head are idle; their VRAM is treated as more buffer space, at the same addresses, and written back before they are needed. Nothing is freed; CUDA graphs stay valid.
Code path:
1. **Start-up sizing** (`generate.cpp:11179-11250`): needs one prompt path with a loan on the drafter's card, no layer split, no `--prefill-head-alias`. The largest chunk `c` in 256 steps up to `max(prefill_chunk, prefill_auto_max)` with
   `bytes_needed(c) + margin <= loan + pack_bytes + heads` where margin = 64 MiB per region (3 regions with heads) (`:11226-11231`). Printed as "a roomed prompt reads in chunks of up to N tokens". `Prefill::set_max_chunk(pr_chunk_max)` sizes the host buffers (token ids, hand-off, PLE rows, steps; `prefill.cpp:1813-1840, 1895-1938`) once.
2. **Per request** with >= `ROOM_MIN` (4096) new tokens (`generate.cpp:17006-17055`): wait until the drafter is idle; `df2.room_lend` makes a **pageable `malloc` RAM copy of the pack (once, ~400 MiB) and of the owned draft head** and marks them lent (`dflash2.cpp:770-795`); `native_head.room_lend` syncs the device and marks the head lent (`native_head.cpp:291-299`);
   for MTP, `mtp.experts_lend` does a **pinned D2H snapshot of the 512 experts (675 MiB) at every lend** (`mtp.cpp:587-613`). While lent, `df2.context/draft`, `NativeHead::run` and the verify windows refuse to run (`dflash2.cpp:1096, 1147`; `native_head.cpp:339-342`). The prompt's DFlash2 tap rows go to a RAM vector instead (`generate.cpp:17033-17048`).
3. **Layout**: `lend()` calls `Prefill::relayout(want_r, loan_start, loan_bytes, err, rooms[], room_bytes[], n)` with regions in order: output head, pack, draft head (`generate.cpp:17445-17456`). `Alloc::take` first fits the loan, else the first region with room (`prefill.cpp:667-677`) - largest buffers first, no straddling. If it does not fit, the chunk steps down by 256 and retries (`generate.cpp:17456-17462`).
4. **Reading**: normal chunk loop with the larger chunk.
5. **Close** before the first verify window or when the prompt is done (`generate.cpp:17214, 17271, 18532-18534`; `df2_room_finish`, `:16278-16319`): prompt path idle (`Prefill::idle`), ring count restored, pack **copied back from the RAM copy to the same address** (`dflash2.cpp:797-816`), the output head **re-read from the model file** into the same address (`native_head.cpp:301-340`: `upload_direct` = 4 threads, 2 x 16 MiB pinned buffers, unbuffered, **Windows only**; elsewhere a plain `cudaMemcpy` from the mapped GGUF), held tap rows handed to the drafter.
   Doc: head re-read about 130-150 ms; `STRATA_DF2_ROOM_VERIFY=1` checksums the head after reload (`native_head.cpp:270-280, 322-331`).
Limits (doc + code): one GPU, `--serve` only; not with a layer split / `--prefill-head-alias` / MTP + `--batch` (`generate.cpp:11200-11203`).

### c.3 Refill after a prompt
* `refill()` runs before any verify window (`generate.cpp:18457-18460`) and at the end of the request (`:18528-18531`), then `res_upload()`; it restores `host_res` for each lent pair (`:17308, 17317, 17326`).
* **Default**: single host thread, per lent slot: `blob_stable(l,e)` (RAM-tier pointer if the expert is in the RAM tier, else the file read into a stage buffer via `staged_blob`) then `cudaMemcpyAsync(slot, blob, H2D, stream 0)` (`fill_slot_queued`, `expert_cache.cpp:839-855`), then one `sync_queued` (`generate.cpp:17312-17341`).
* Opt-in `STRATA_REFILL_PINNED=N` (1-16): N threads, each reading runs of 2 blobs via `copy_blobs` into its own two pinned buffers and copying up on its own stream (`generate.cpp:525-599`; not a fork default, not in `apply_fork_defaults`).
* Timing (docs, one PC; i7-14700KF + RTX 4070 Ti + two 990 PRO): after a 50K prompt **1,048-1,059 ms default vs 405-414 ms with REFILL_PINNED=4**; after short prompts 190 -> 70 ms (`docs/fork/PERFORMANCE.md:187`). The code comment says the lent experts are "nearly all file reads (~1 GB, 380 ms on one thread)" (`generate.cpp:527-529`).
* Source of truth for refilled experts = the same `ExpertSource` (RAM tier or experts.bin); "from where" = mostly the SSD, since the lent experts are the GPU tier's, which the RAM copy complements.

---------------------------------------------------------------------------------------------------------------------------------

## d. Fused hyper-connection kernels in prompts (4 mHC streams)

Residual state `R` = T x 4 x 2560 floats (`D = N*HC`). The embedding is broadcast to the 4 streams (`gr_broadcast`, `kernels.cu:307-312`, call `prefill.cpp:3567`). Per half (`prefill.cpp:4102-4187`):
1. `gr_norm_rs` (or the fused write of the previous half): RMSNorm over each stream row with `hc_*_norm.weight` -> only the row scale `rs` (T x 4) and a BF16/FP16 image `xn16` are written; the FP32 normalized copy is never materialised ("F-1") (`kernels.cu:129-147`; `STRATA_GR_UNFUSED=1` keeps the old two-kernel form, `prefill.cpp:638-642`).
2. down GEMM `xn16 (T x 10240) x hc_down^T -> lo (T x 320)` (BF16 GEMM, `bf16_proj`, `prefill.cpp:4142`); `gr_silu`: `lo/4`, SiLU -> `lo16` (`kernels.cu:269-278`).
3. **up GEMM + mix**: default = `bf16_proj(lo16, hc_up) -> gated (T x 10240 FP32)` then `gr_mix_r`: `x_c = R*rs*w`, `mixed = (sum_c x_c * sigmoid(gated_c)) / 4`, writing `mixed` FP32 + BF16 image + FP16 image (`kernels.cu:148-169`).
   **Fused (`STRATA_HC_UPMIX=1`, fork pack default; code = 0.2.11)**: `gr_upmix` runs the up projection (BF16 mma.m16n8k16, FP32 accumulate over K=320, cp.async double buffering) and the same mix epilogue in registers in **one kernel**, so `gated` (T x 10240 FP32, 336 MB at 8192 tokens) is never written or read (`kernels.cu:2618-2729` CUDA kernel `gr_upmix_cuda_kernel` at `:2648`; launch `:2833-2862`;
   use in `prefill.cpp:4145-4171`). Conditions: sm_80+, `T >= 33` (`hc_upmix_min_t`, `prefill.cpp:574-580`), BF16 up weight, no BF16x2 low parts. `STRATA_HC_UPMIX_CHECK=N` compares with the pair on the first N reads. Doc: 50K +4.3 %, logits byte-identical (`docs/fork/PROMPT-ROOM.md:136-141`).
4. inject GEMM `xn16 x hc_inject^T -> inj (T x 4)` (`prefill.cpp:4174`).
5. After the block (attention or MoE) writes `bo (T x 2560)`: **`gr_write_norm_rs`** = `R[t,c,:] += bo[t,:] * 2*sigmoid(inj[t,c]/4)` per stream **and in the same kernel** the *next* half's RMSNorm scale and BF16/FP16 image (`kernels.cu:174-206`, call `prefill.cpp:6105`) - the next HC read then skips its own norm pass (`normed = true`). Not used on the stage's last half, before layer 1's PLE block, or under a control vector (fused variant `gr_write_cvec_norm_rs`, `STRATA_CVEC_FUSE`, off).
Other fusions on the path: `STRATA_QSA_FUSE_NR` (RMS+RoPE for K, Q, indexer-Q in one pass, `rms_rope_rows`, `kernels.cu:2418-2430, 3375`), `STRATA_QSA_FUSE_GATE` (attention gate epilogue, see f), the stream-final residual rows go to `on_chunk`. Opt-in HIP/gfx11-oriented ones (`STRATA_PF_HCDOWN`, `STRATA_HCD_EXACT`, `STRATA_PF_PAD`) are off (`prefill.cpp:587-599, 620-635`).

---------------------------------------------------------------------------------------------------------------------------------

## e. GDN (Gated DeltaNet) state in a prompt

Per GDN layer (`prefill.cpp:4188-4238`):
* Projections from `mixed_h` (FP16): `attn_qkv` (-> T x 10240 = q 2048 + k 2048 + v 6144), `attn_gate` (z, T x 6144) via dequant+FP16 GEMM; `ssm_alpha`, `ssm_beta` via BF16 GEMM on `mixed_bf` (`:4199-4209`).
* `gdn_gates`: `gate = softplus(alpha+dt) * ssm_a`, `beta = sigmoid(beta_raw)` (`kernels.cu:315-324`).
* `gdn_conv`: causal 4-tap depthwise conv + SiLU over the 10240 channels; the 3 inputs before the chunk are the **conv history carried in the layer state**; tiled over tokens (`CONV_TILE=64`), then `gdn_conv_hist_kernel` writes the last 3 inputs back, then `gdn_l2_kernel` L2-normalises q and k heads (`kernels.cu:343-360, 464-476, 2899-2926`).
* `gdn_recurrence` (`kernels.cu:2929-3123`): **default on NVIDIA = a token-by-token kernel that walks the chunk inside one launch**: `gdn_rec_kh_kernel` (one thread handles the 3 value heads sharing a key head; the S x S state slice is held in registers; q/k/v inputs come in blocks of tokens via cp.async; per token the delta-rule update `s = g*s + k*delta`, `o += s*q`; `kernels.cu:755-830`) where `gdn_keyhead_ok()` (sm_80+ and all 64 blocks resident, `:901-918`), else `gdn_rec_cols_pipe_kernel`/`cols`. The output gated RMSNorm (`silu(z)`-gated per head, to FP16 `y_h`) is `gdn_out_norm_kernel` (`:2200`, `:3119-3120`), then `ssm_out` projection -> `bo`.
  The **chunked/WY kernels** (`gdn_rec_chunked` `kernels.cu:1231`, 32-token chunks, FP32; `gdn_rec_tc` `:1308`, tensor-core scan; `src/prefill/gdn_chunk.cu` = Volta/SM60 experimental) are **opt-in**: `STRATA_GDN_CHUNKED=1` / `STRATA_GDN_TC=1`, chunks >= 128 tokens (`kGdnChunkedMin/kGdnTcMin = 128`, `kernels.cu:934-935, 3036-3095`); neither is set by the fork or the sm_89 table.
* **State across chunks**: `ss.gdn_state` is device memory, one block per GDN layer = `ssm_state_size * ssm_v_heads * ssm_state_size` floats (128*48*128, FP32) followed by `ssm_conv_channels*(d_conv-1)` floats (the conv history) (`prefill.cpp:3338-3339, 4197-4198`). The kernel loads its slice into registers at the start of each chunk and writes it back at the end; the next chunk (and decode) continues from it. The PLE history (`ss.ple.hist`) and `ss.ple_prev` (the two previous token ids) are carried likewise (`prefill.cpp:3340, 3628, 6246-6247`).
  Whole-layer state size = ~3.1 MiB; 36 layers ~ 112 MiB (inferred; the docs say ~118 MB per checkpoint).
* **Checkpoints / snapshots (conversation cache)**: the batched path calls `on_chunk` after each chunk with the stream synchronised; in `generate.cpp:12728-12813` it takes a **periodic checkpoint when `done >= pp_next_check`** (`--prompt-cache-every`, default 16384 tokens, `generate.cpp:1023`), plus an optional tail checkpoint (`prompt_cache_tail`), and `checkpoint_at(...)` at the cut points (root / message / turn / pin, `:18513-18523`). Never mid-super-chunk in layer-major (`sp.layer_major_mid()`).
  A checkpoint (`ConversationCheckpoint`, `include/strata/core/conversation_cache.hpp:24-33`) stores: the token ids, the **GDN state (all GDN layers)**, the **PLE history**, and per QSA layer the indexer `tails`, `dead` and `block_pos` (`src/core/conversation_state.cpp:290-383`). It does **not** copy the K/V pools - the positional cells below the checkpoint stay in the session (`generate.cpp:2308`, `conversation_state.cpp:392-429`).
  Save = synchronous D2H copy by default; `STRATA_CKPT_ASYNC=1` stages it through a device buffer and a background thread in 4 MiB pieces; `STRATA_CKPT_VRAM_SLOTS=N` keeps device copies (`conversation_state.cpp:188-214, 305-365`), both off. Whole-conversation disk snapshots (`conversation_snapshot_sources`, includes K/V blocks) are used for parking (fork diff: draft layer optional, `conversation_state.cpp:619-690`).

---------------------------------------------------------------------------------------------------------------------------------

## f. Full attention (QSA: sparse "indexer" attention) in prompts

Per QSA layer (`prefill.cpp:4239-4524`):
1. Projections: K (`attn_k`, 2 KV heads x 256) and V via dequant+FP16 GEMM; Q (`attn_q`, 24 heads x (256 q + 256 gate), `Qf` T x 12288); indexer K (`indexer.k_proj`, 128) and indexer Q (`indexer.q_proj`, 4 heads x 128) via BF16 GEMM (`:4251-4257`).
2. `rms_rope_rows` (default fused; `STRATA_QSA_FUSE_NR=0` -> `rms_rows`+`rope`+`split_q`): RMSNorm then RoPE on the first 64 dims (NEOX pairs) for K, Q, indexer-Q at positions `p0..p0+T` (`:4259-4338`).
3. **KV write**: `kv_append` quantises/stores the chunk's K/V into the layer pool (int8 with a scale per 64 values with `--kv int8`; `kv_append_q4` for q4_0, hybrid K8V4; optional FWHT rotation `kv_rot`) (`:4294-4323`, kernel `kernels.cu:2483`). With **KV streaming** (`st.kv_mode == 1`, `--kv-resident`), the layer's authoritative K/V is a pinned host copy: the prefix cells `[0,p0)` are staged to a VRAM identity pool with `kv_stage_from_host` once per layer and chunk (the **whole** prefix, not only selected cells; `:4265-4281`), the chunk's cells are written to the host copy, the staging pool and the resident VRAM slots at once.
   Because the pool holds the quantised values, the prompt's own attention reads the same quantised K/V that decode will (inferred from the data flow).
4. **Indexer**: `native_qsa_indexer_append_batch` appends the chunk's indexer keys (one pooled key per `idx_block`=4 cells; tail/dead/pooled/block_pos buffers) in three launches (`:4343-4358`).
5. **Sparse selection**: for query batches of `sel_batch` = 256: `qsa_block_scores_tc` (block scores = sum over 4 indexer heads of relu(q.k_pooled) per block, 3xTF32 tensor cores, FP32-level accuracy, not bitwise vs the warp kernel) then `qsa_block_topk` (4-pass radix select over the query's blocks, each weighted by its cell count; the ids = selected cells ascending) (`:4370-4387`, `include/strata/kernels/qsa_select.hpp:1-48`).
   Budget: `idx_top_k=2048` cells + `idx_block-1` = width `min(n_kv, 2051)` (`include/strata/kernels/qsa.hpp:96-136`); below ~2051 cells the selection is the identity (= dense attention). This is the model's own architecture, not a fork shortcut. Each query uses its own `n_bid` from its step record, so it is causal inside the chunk.
6. **Attention**: `qsa_prompt_attn_batch_gated` - one block per (query, KV head) walks the selected cells in chunks of 32 with an online softmax; 12 query heads per KV head; **default on sm_80+ = INT8 tensor-core kernel `prompt_attn_i8v3`** (K, V as stored int8 codes; q and p*v_scale split into 3 8-bit parts so ~24 bits are kept; `STRATA_PROMPT_ATTN_IMMA=0` returns to the FP16 v2 kernel) (`src/kernels/cuda/qsa_prompt_attn.cu:1085-1111, 2553-2562`; `include/strata/core/sm120_series.hpp:93-96`).
   With `STRATA_QSA_FUSE_GATE` (default on) the output gate `sigmoid(gate)*O` is applied in the epilogue and `attn_h` (FP16) is written directly (`prefill.cpp:4490-4492`; on CUDA the "gated" kernel may still return false and `gate_attn` runs, `:4521`). Then `attn_output` projection -> `bo`.
   Fallback `qsa_decode_attn_batch` 32 queries at a time (`:4513-4518`).

---------------------------------------------------------------------------------------------------------------------------------

## g. The n-gram per-layer embedding (PLE) rows in a prompt

* What it is: 16 rows of 160 values per token (`PLE_N_HEADS=16`, `PLE_HEAD_DIM=160`; 8 bigram heads + 8 trigram heads) from a table of **320,001,536 rows** (`include/strata/kernels/ngram.hpp:31-47`), IQ4_NL = 90 B/row ~ 28.8 GB (other formats: FP8 160 B, Q5_1 120 B, Q8_0 170 B, BF16 320 B). The row indices are hashes of (token, the 2 previous tokens): `ngram_rows(...)` (`ngram.hpp`, per token, `prefill.cpp:3397-3402`).
  It is applied once per prompt chunk at **layer 1** (`prefill.cpp:4031-4084`): key GEMM (2560 -> 10240), value GEMM, `native_ple_postops_batch` with the history conv (`STRATA_PLE_BATCH`, default on except CUDA 12.8 on sm_120; `prefill.cpp:62-98`).
* Gather (the part that touches the SSD): `ple_gather(c0, buf)` (`prefill.cpp:3393-3413`) computes the T x 16 row ids and calls `PleTable::gather_batch(rows, T, ple_emb_host[buf])`. It runs in a `std::async` thread: the **first chunk's gather starts at the chunk start beside layer 0** (`:3571-3575`), the next chunk's is started from `ple_land` (at layer 1 of the current chunk) so it overlaps the 47 other layers (`:3613-3623`), into the other of **two pinned buffers** of T x 2560 floats (`ple_emb_host[2]`, `:1923-1938`). `ple_land` waits for the future, uploads T*2560*4 B (105 MB at T=10,240) with `cudaMemcpyAsync` (`:3592-3610`).
  Opt-in `STRATA_PLE_COMPACT=1` uploads the raw 90-byte rows and dequantises on the GPU (7x fewer bytes; only when the table is mapped in RAM, `ngram.cpp:714-732`); `STRATA_PLE_COPY_STREAM=1` its own stream - both off (`prefill.cpp:192-201`).
* Table access mode (`generate.cpp:740-745, 5508-5651`): default `--ple-io direct`, `--ple-inflight 256`, `--ple-row-cache 1048576` rows (~95 MB for 90 B rows), I/O worker thread on, keep-alive read after 100 ms idle (`STRATA_SSD_KEEPALIVE`). If `ple_resident_admit` (table + **all** experts (47 GiB) + 8 GiB headroom fit in available RAM) the table is instead mapped and locked in RAM
  (`--ple-io ram` by policy, `generate.cpp:5543-5587`, `include/strata/core/ple_residency.hpp:23-28`) - not possible on a 32 GB PC (inferred), so on such PCs it is direct SSD reads. A slow-probe falls back to mmap only if `STRATA_PLE_AUTO_MMAP=1` (`:5591-5623`).
* Direct reads (`PleReader`, `src/ngram/ple_reader.cpp`): `issue(rows, n, out)` (`:422-490`): per row: out-of-range -> zeros; **row-cache hit** (8-way set-associative `RowCache`, round-robin replacement, bounded by rows, `:36-72`) -> memcpy; else a job for the 4 KiB page that holds it (2 pages if the row straddles a boundary); **rows on the same page share one job** (`by_page`, `dedup_rows`) - this is the only dedup (identical rows in one call map to the same page too); jobs **sorted by file offset** so a chunk reads in near-sequential order (`:473-474`); a worker thread submits them keeping up to 256 requests in flight (`pump`, `:148-178`), reaps completions
  (`worker_loop`, `:299-327`), copies the row out of the 2-page slot buffer, inserts it into the cache (`finish`, `:200-251`). `collect` blocks on the ticket. Then the rows are dequantised on the CPU thread (`impl_->decode`, `ngram.cpp:670-673`).
  A chunk of 10,240 tokens = 163,840 row lookups -> up to 163,840 random 4 KiB reads before cache/page dedup (arithmetic; the hit rate is not stated in the code).
* APIs: `DirectFile` (`src/platform/direct_file.cpp`): **Windows** `CreateFileW(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS)`, one IO completion port (concurrency 1), a pool of 4 submit threads (`STRATA_IO_THREADS`, default 4) issuing overlapped `ReadFile`, completions via `GetQueuedCompletionStatusEx` (`:159-256`); **Linux/POSIX** `open(O_DIRECT)` and a pool of 16 threads each doing a blocking `pread` (thread count = queue depth) (`:263-347`); macOS `F_NOCACHE`. No mmap in direct mode (the header mapping is dropped after parse, `ngram.cpp:392-409`).

---------------------------------------------------------------------------------------------------------------------------------

## h. SSD / file reads in a prompt - summary

| Reads | API | Parallelism |
|---|---|---|
| Experts from `experts.bin` (file tier) | Windows: `FILE_FLAG_NO_BUFFERING\|FILE_FLAG_OVERLAPPED` + overlapped `ReadFile` batches (`expert_source.cpp:2813, 3243-3254`); Linux: `O_DIRECT` + libaio `io_submit` (`:2850, 3358-3440`); fallback mmap | stager threads (fork 3, one blob each, in-place into pinned buffers); 0.2.11: 32 threads x batch 8 merged reads; optional second drive halves reads |
| Experts for the **refill** of lent slots | same `read_direct` through `blob_stable` / `copy_blobs` | one thread by default; `STRATA_REFILL_PINNED=N` threads |
| PLE rows | `DirectFile` (Windows IOCP overlapped / POSIX pread pool, 4 KiB pages, 256 in flight) | one worker thread + 4 (Windows) / 16 (Linux) I/O threads |
| Output head after the room | Windows unbuffered `ReadFile` x 4 threads x 4 MiB into 2 x 16 MiB pinned buffers (`native_head.cpp:62-103`); else memcpy from mapping | 4 threads |
| Model load / token embedding | mmap (not in the prompt loop) | - |
No io_uring. The RAM arena and the cache fill before the prompt are not part of the prompt loop.

---------------------------------------------------------------------------------------------------------------------------------

## i. Steps that can change the answers vs a full exact computation (default state in the fork)

Nothing in the **batched prefill** drops or skips routed experts: the router takes top-10 of 512 and every routed expert is computed (b, `kernels.cu:2218-2255`). Items that differ from a bit-exact reference, with default state:

1. **Rounding-level (always on)**: expert products use q8_1-quantised activations and quantised H (MMQ, same as llama.cpp) or per-32 int8 activations (fused); dense projections dequantise to FP16 and multiply FP16 x FP16 (inputs saturated to +-65504, `kernels.cu:65-70`); the router / HC / indexer GEMMs take **BF16-rounded activations in prefill but FP32 in decode** (`prefill.cpp:444-458`; the BF16x2 correction `STRATA_PREFILL_BF16X2` is off) so near-tied routing can differ from a decode-style computation; KV is stored int8/q4 with `--kv int8` etc. (user choice, same for decode); QSA scores use 3xTF32 (not bitwise). The **chunk size** changes cuBLAS/MMQ tile shapes and thus rounding (fork doc: first-token KL 0.0001-0.0024 between 6- and 7-pass chunking, 0.0003-0.0022 with the room, same top token, `docs/fork/PERFORMANCE.md:199`, `PROMPT-ROOM.md:119-125`).
2. **CPU share** (`STRATA_PREFILL_CPU_SHARE`): for chunks < `STRATA_PREFILL_CPU_SHARE_MAX` tokens the idle CPU pool computes some few-row, RAM-resident experts with its own activation format instead of the GPU (`prefill.cpp:4769-4806, 5024-5094`; KL ~0.004 below 1024 tokens per the code comment, `:2950-2956`). **Armed by default** where the engine decides (CUDA, one GPU, no batch slots, host memory OK, GPU cache not holding every expert: `generate.cpp:10705-10712`) for chunks below the limit; the fork lowers the limit from 3072 to 1024 (`generate.cpp:3360`), so it applies to short prompts/tails (<1024 tokens), "the 0.1.40.3 bytes" with `STRATA_PREFILL_CPU_SHARE=0`. (armed-by-default status is inferred from `arm_cpu_share` + `share_default_ok`; I did not run it.)
3. **0.2.11's small-card prompt rule** (`STRATA_PF_SMALL`): on one card whose default chunk < 8192 it turned the fused experts off and layer-major on for prompts of >= 2 chunks (different summation order, `generate.cpp:10852-10897`). **Off** in the fork (`generate.cpp:3361`).
4. **Expert skip** (`STRATA_ROUTE_TAIL_SKIP`, 0.2.11 default 7 on CUDA unless >= 90 % of experts are cached): only in verify windows (decode and <= 64-token prompt parts / the last prompt token), not in batched chunks (`generate.cpp:12559-12605`; `grep ROUTE_TAIL` finds nothing in `src/prefill`). **Fork default 0** (`generate.cpp:3359`). Doc: skip 7 gives KL 0.081 on their PC (`docs/fork/PERFORMANCE.md:323`). `STRATA_ROUTE_TAIL_SKIP_FILE=R` (changes answers, off).
5. **K-quant MMQ** (`STRATA_MMQ_KQ=1`, default on for cc >= 8.0 when the pack has Q4_K/Q5_K/Q5_1/Q6_K experts): not relevant to an IQ3_S pack (`nv_defaults.hpp:36-46`).
6. **STRATA_KVA** (approximate late-layer prompt read: layers >= 36 run state-only for all but the last 2048 positions, MoE half and transfers skipped): **opt-in, off** (`include/strata/prefill/kva.hpp:1-21, 77`).
7. **Layer-major** (`STRATA_PREFILL_LAYER_MAJOR`): experts' contributions summed per expert group in id order (not in router k order) -> FP32 rounding; opt-in, off (`prefill.cpp:6379-6396`). `STRATA_HEAD_ALIAS_NARROW` + layer-major differs (KL 0.02 at 50K) and stays off (`docs/fork/PERFORMANCE.md:469`).
8. **Small prompt parts through the verify windows** (<= 64 tokens, `--short-read`): decode arithmetic instead of batched arithmetic (not a precision loss, different rounding) (`generate.cpp:17088-17095`).
9. Research hooks (`STRATA_DBG_ROUTE_OVERRIDE`, `STRATA_DBG_FEAT`) replace routing from a file - off (`prefill.cpp:4537-4596`).
10. The prompt room itself: contents are restored byte for byte (checksum option); only the chunk size changes the rounding (a above).
Fork's own statement: "Skip 0: every expert is computed, so answers are the full model's" (`docs/fork/PERFORMANCE.md:175`) - true for the expert-skip switch; the points above are rounding-level.

---------------------------------------------------------------------------------------------------------------------------------

## j. Settings / env vars on the prompt path, with defaults (SlenderEngine vs Strata 0.2.11)

"pack defaults" = Windows + NVIDIA + `--pack` (set only if the variable is unset; `generate.cpp:3343-3416`). `STRATA_FORK_DEFAULTS=0` removes all fork defaults.

| Variable / flag | 0.2.11 default | Fork default | Meaning / where |
|---|---|---|---|
| `--prefill auto[:N]` | user flag | same | chunk ceiling `generate.cpp:3701-3720` |
| `STRATA_PREFILL_AUTO_MAX` | 8192 (NVIDIA bare auto -> up to 32768) | 16384 if `--resident-budget-gib` + >= 28 GB RAM | `generate.cpp:3408, 4048-4073` |
| `STRATA_PREFILL_LEND_PCT` | 85 (90 if >= 90 % pinned) | 90 | max share of cache slots lent, `generate.cpp:9836-9841` |
| `STRATA_PREFILL_RING` | unset (rule: 96/384/1024, byte-budget) | unset (user launcher: 192) | ring slots, `prefill.cpp:403-439` |
| `STRATA_RING_BYTES` | on | on | byte-budget ring, `prefill.cpp:393-396` |
| `STRATA_PREFILL_STREAM_MIN` | 1024 (or CPU-share limit) | 1024 | min chunk to stream all experts, `prefill.cpp:258` |
| `STRATA_PREFILL_CPU_SHARE` / `_MAX` | armed by default; limit 3072 | armed by default; `_MAX=1024` | `prefill.cpp:278-305, 2946-2974`; `generate.cpp:3360` |
| `STRATA_PREFILL_ISSUER` | on | on | issuer thread `prefill.cpp:3799` |
| `STRATA_PREFILL_ROUTED_STREAM` (+`_DEPTH`) | off | off | drop unrouted ring entries `prefill.cpp:3819` |
| `STRATA_PREFILL_COPY_EVENT_EVERY` / `_COPY_WAIT_DEDUPE` | 1 / off | same | event batching (Windows WDDM) `prefill.cpp:3803-3867` |
| `STRATA_PREFILL_GROUP_GATHER` | on | on | one gather/wait per 16-expert group `prefill.cpp:5535` |
| `STRATA_PREFILL_MMQ` | on (if built) | on | `prefill.cpp:1523-1530` |
| `STRATA_PF_FUSED` | unset: on only for sm_120 >= 80 SMs / sm_121 | same | fused native experts `moe_fused_iq.cu:1988-2030`; sm_89 -> MMQ |
| `STRATA_MMQ_KQ` | on for cc >= 8.0 | same | K-quant MMQ `nv_defaults.hpp:46` |
| `STRATA_MMQ_DIRECT` (+build `STRATA_MMQ_TAB`) | off / not built | same | MMQ reads resident experts in place `moe_mmq.cu:619` |
| `STRATA_PF_GROUP_SCRATCH`, `STRATA_PF_HC_REGION`, `STRATA_PF_WMMA`, `STRATA_MMQ_WS` | off | off | buffer/kernel variants `prefill.cpp:1505, 1604, 1640` |
| `STRATA_PF_SMALL` | rule on | `0` | `generate.cpp:3361, 10852` |
| `STRATA_PREFILL_LAYER_MAJOR` | off | off | `sm120_series.hpp:131` |
| `STRATA_STAGER_THREADS` / `_BATCH` / `_RING` | file tier: 32 / 8 / 4xthreads; else min(4,hw/4) / 1 / 16 | 3 / 1 / 128 | `prefill.cpp:1853-1884`, `generate.cpp:3375-3379` |
| `STRATA_STAGER_IN_PLACE` | off | 1 | `prefill.cpp:732` |
| `STRATA_DIRECT_IN_PLACE` | on (Windows) | on | stage buffers sector aligned `expert_source.cpp:1967` |
| `STRATA_DIRECT_BATCH` | 16 (hard-coded) | 1 | `expert_source.cpp:2389-2392` |
| `STRATA_UNBUFFERED_LOAD` | off | 1 | unbuffered file tier `generate.cpp:3371` |
| `STRATA_EXPERTS_MIRROR` (+`_SHARE`) | n/a | off | second drive `expert_source.cpp:2824, 3219` |
| `STRATA_REFILL_PINNED` | n/a | off (launcher 4) | `generate.cpp:531` |
| `STRATA_RAM_THROUGH`, `STRATA_KV_LEND`, `STRATA_PARTIAL_PIN`, `STRATA_RAM_ADAPT*`, `STRATA_COLD_*` | off | 1 (pack defaults) | decode-side RAM tier / lend regions, `generate.cpp:3371-3380` |
| `STRATA_DENSE_ARENA` | off | 1 | `native_dense.cpp:181` |
| `STRATA_HC_UPMIX` | off (sm_121 on) | 1 | `generate.cpp:3383`, `prefill.cpp:568-571` |
| `STRATA_QSA_FUSE_NR`, `STRATA_QSA_FUSE_GATE` | on | on | `prefill.cpp:605-612` |
| `STRATA_PROMPT_ATTN_IMMA` | on cc >= 8.0 | on | `qsa_prompt_attn.cu:2553-2562` |
| `STRATA_GDN_CHUNKED`, `STRATA_GDN_TC`, `STRATA_GDN_KEYHEAD` | off, off, on | same | `kernels.cu:3036-3040, 901` |
| `STRATA_PLE_BATCH` | on (except CUDA 12.8 + sm_120) | same | `prefill.cpp:62-98` |
| `STRATA_PLE_COMPACT`, `_COPY_STREAM` | off | off | `prefill.cpp:192-201` |
| `--ple-io`, `--ple-inflight`, `--ple-row-cache`, `STRATA_SSD_KEEPALIVE`, `STRATA_IO_THREADS`, `STRATA_PLE_AUTO_RAM` | direct, 256, 2^20 rows, 100 ms, 4/16, on | same | `generate.cpp:740-745, 5509-5587`, `direct_file.cpp:63` |
| `STRATA_DF2_ROOM_MIN`/`_CHUNK`/`_HEAD`; `STRATA_MTP_ROOM_MIN` (0.2.11 had only this)/`_CHUNK`/`_HEAD` | not present (MTP_ROOM_MIN off) | 4096 / 1 / 1 with `--serve --draft dflash2` (MTP: with `--mtp`, no `--batch`) | `generate.cpp:3395-3405` |
| `STRATA_DF2_ROOM_VERIFY` | n/a | off | head checksum `native_head.cpp:270-280` |
| `--short-read` | 64 | 64 | `generate.cpp:1040` |
| `--prompt-cache-every` | 16384 | 16384 | periodic checkpoints `generate.cpp:1023` |
| `STRATA_ROUTE_TAIL_SKIP` | 7 (CUDA, conditions) | 0 | verify windows only |
| `STRATA_KVA` | off | off | `kva.hpp` |
| `STRATA_TRACE`, `STRATA_PREFILL_TIMING`, `STRATA_PF_STEP_SYNC`, `STRATA_LOAN_TRACE` | off | off | diagnostics |

---------------------------------------------------------------------------------------------------------------------------------

## k. Licence headers of the files read

* Repo licence: `LICENSE` = MIT, "Copyright (c) 2026 Niko1221 and the Strata contributors" and "Copyright (c) 2026 Rafid Ahsan" (the fork owner). `FORK.md:485`: "The license is upstream's MIT license, unchanged".
* **No licence/copyright header** in the project's own files I read: `src/prefill/prefill.cpp`, `gemm.cu`, `kernels.cu`, `moe_mmq.cu`, `moe_fused_iq.cu`, `moe_fused.cu`, `moe_ws*.cu`, `gdn_chunk.cu`, `gdn_tc.cuh`, `mmq_kq_*.cu/.cpp`, `src/core/{expert_source,native_head,dflash2,native_dense,nv_defaults,conversation_state}.cpp`, `src/ngram/ple_reader.cpp`, `src/kernels/ngram.cpp`, `src/platform/direct_file.cpp`, `src/program/generate.cpp`, `src/kernels/cuda/qsa_prompt_attn.cu`, `qsa_select.cu`, the headers I opened.
* Files with an **embedded MIT notice naming other holders** (compatible with the repo's MIT, but different holder lines):
  - "MIT License, Copyright (c) 2023-2026 The ggml authors": `src/kernels/cuda/native_gdn.cu:6-7`, `native_gdn_preprocess.cu:5-6`, `native_qsa_indexer.cu:1-4` (adapted from llama.cpp `3cf03257`), `native_qsa_score.cu:3-4`, `native_gr_postops.cu:4-5`, `native_router.cu:3-4`, `src/prefill/moe_mmq_tab.cuh:27-28`; `third_party/ggml/LICENSE` is MIT "The ggml authors".
  - "MIT License, Copyright (c) 2026 gufo contributors": `src/prefill/moe_f16_kq.cu:13-14`, `moe_f16_common.cuh:11-12` (gfx11-only, opt-in).
  - `src/prefill/moe_mmq.cu:1-3`: compiled against llama.cpp's ggml-cuda sources (MIT) taken from the pinned checkout at build time; `kernels.cu:2422`: idea credit to halo-box/strix-llama.cpp #213 (MIT); `moe_fused_iq.cu:1268`: idea credit to asf0/libstrix (MIT).
  - Many `src/dsv41/*` (DeepSeek path, not read) carry "SPDX/Copyright" strings (`grep`), not part of the Qwen prompt path.
* Not source-code licences but worth knowing: the DFlash2 drafter model files are **CC BY-NC-ND (non-commercial)** (`docs/fork/PERFORMANCE.md:224`); the prompt room's DF2 variants need that drafter.
* No file I read carries a header that contradicts the repo's MIT licence.

---------------------------------------------------------------------------------------------------------------------------------

## Not verified / open points
* I did not read `ExpertCache` sizing, the complement-arena loader, `fused::group`/`experts_native` kernel bodies, `qsa_block_scores_tc` / `qsa_block_topk` kernel bodies, `native_ple_postops_batch`, `moe_combine4`, nor the multi-GPU/peer code. Statements about those come from comments/headers.
* The "expert cache ~4 GiB / chunk 8,448 at ~410 KB per token" reconciliation is my arithmetic from `bytes_needed_impl` (`prefill.cpp:2858-2918`), not a measured value.
* Whether the CPU share is actually armed on the user's PC depends on `share_default_ok()` (H11, free RAM) - not evaluated.
* Windows-specific paths (in-place unbuffered reads, mirror, `upload_direct`) were read, not run; the Linux read path (libaio, copy via scratch) differs.
* Measured numbers quoted (refill ms, pass counts, KL, tok/s) are the fork docs' own, on i7-14700KF / 32 GB / RTX 4070 Ti 12 GB / two 990 PRO / Windows 11; I reproduced none.
