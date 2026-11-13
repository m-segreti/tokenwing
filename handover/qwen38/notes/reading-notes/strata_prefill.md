# Strata main: how a PROMPT (prefill) is processed for Qwen3.8-Flash-Next

Source read: `traces/refs/strata` = Strata main `13eedf55` (after v0.2.13.1, engine 0.1.59), compared with
`traces/refs/strata-0.2.11`. Read-only study; nothing was built or run. Paths are relative to `strata/` unless stated.
`pf` = `src/prefill/prefill.cpp`, `gen` = `src/program/generate.cpp`. "(inferred)" marks anything not directly read in code
or stated in a doc. "Doc stale" marks places where a doc and the code disagree (code wins here).

Model shape the prompt path is compiled for (`include/strata/core/geometry.hpp:25-60`): n_embd 2560, 48 layers (every 4th
is full attention = 12 "QSA" layers at layers 3, 7, ..., 47; the other 36 are Gated DeltaNet), 4 hyper-connection streams
(`hc` 4, low rank 320), 512 routed experts top-10 + 1 shared expert, expert width n_ff 640, GDN state 128 x 48 value heads,
conv channels 10240, QSA 24 query heads / 2 KV heads / head_dim 256, indexer 4 heads x 128, one pooled indexer key per 4
cells, selection budget 2048 cells (width = min(n_kv, 2048 + 4 - 1), `include/strata/kernels/qsa.hpp:131-140`: below ~2051 cells
the selection is the identity, i.e. dense attention).

---------------------------------------------------------------------------------------------------------------------------

## 0. One-paragraph picture

A request reaches `generate.cpp`'s serve loop. The prompt is cut into parts (reuse point, system-prompt root, turn boundary,
last token). Parts of <= 64 tokens go through the decode "verify windows"; longer parts go to `Prefill::run`. Before the
first batched part, the engine **lends the top slots of the VRAM expert cache** to the prompt path as its buffer region
(`lend`, gen:17153), and **refills those slots from RAM** when the prompt is done (`refill`, gen:17126). `Prefill::run_impl`
(pf:3314) walks the prompt in chunks (up to 8192, on NVIDIA cc >= 8.0 up to 32768, tokens), each chunk through all 48 layers; per layer two "halves" (attention half, FFN half), each
wrapped in a hyper-connection read and write. Dense projections are tensor-core GEMMs (quantized weights dequantized to FP16
or BF16 weights as they are). The routed experts are grouped by expert and run as int8 tensor-core kernels (llama.cpp MMQ,
or Strata's own fused kernels). For a chunk of >= `stream_all_min` tokens **every expert the GPU does not hold, routed or not,
is DMA'd once per layer per chunk** through a ring of VRAM staging slots from the pinned RAM arena (or from SSD/GGUF files via
copy threads). The 28.8 GB layer-1 n-gram (PLE) table is read as 16 rows x 90 B per token from SSD (direct reads) or RAM on a
side thread, one chunk ahead. After each chunk the draft (MTP) layer's K/V is built from the chunk's final residual and
conversation checkpoints (118 MB each) are saved every 16384 tokens. No CUDA graphs are used in the prompt path (grep: 0 hits in
`src/prefill`); everything is eager kernel launches on a compute stream plus a copy stream.

---------------------------------------------------------------------------------------------------------------------------

## a. The loop

### a.1 Where prefill is called (generate.cpp)

* The serve loop builds `cuts = {reread_to, root_at, message_at, turn_at, n-1 (+pin_at)}` (gen:18034). `root_at` = first
  `<|im_start|>` after at least `--prompt-cache-root` (2048) tokens, only for reads starting at 0 (gen:17998-18003); `turn_at` =
  last turn token (gen:17977-17989); `message_at` only with `STRATA_CACHE_MESSAGE_BOUNDARY=1` (gen:18004-18010).
* For each part `[at, to)` (gen:18063): `win = windows_ok(at, to)` = part <= `--short-read` (default **64**, gen:965) and no picture
  rows (gen:16858-16864). `win` parts: `refill()` first, then `read_windows` (decode verify windows of S tokens; the decode path,
  incl. its CPU-miss experts). Otherwise `lend(to-at)` (gen:18075) then `read_part` -> `sp.run(ids+q, r-q, q)` (gen:17386, wrapped
  in `run_conc`/pieces for `--batch`). The **last prompt token (n-1) always goes through the first verify window** (gen:18150-18154),
  not through `Prefill`.
* `Prefill::run` (pf:7941): on CUDA, if the layer-major rule says so (section b.9) it runs `run_layer_major`; otherwise
  `run_impl` (pf:8007), then drains the layer-split chain.
* Between chunks the engine calls `on_chunk` (gen:12658-12743): builds the MTP draft layer's prompt K/V from the chunk's final
  residual rows (`Prefill::draft_kv`, pf:2263: Q8_0 MMQ + the prompt GEMMs, "not bit-identical to the drafter's own pass ... never
  the target's tokens", `prefill.hpp:58-65`; falls back to `mtp.prefill`), prints `PP done total ms tok/s`, and takes a periodic
  checkpoint (section e).

### a.2 Chunk size

* `--prefill N`: fixed chunk N. `--prefill auto` (what setup writes, `docs/DETAILS.md:44-46`): the largest chunk whose buffers the expert cache can
  lend (gen:9829-10028 `lend_slots`, `plan_lend`; a bisection on the 256-token grid, `biggest_chunk` gen:9943). Default ceiling 8192
  (gen:9886); **on NVIDIA with every visible card cc >= 8 a bare `--prefill auto` may reach 32768** (gen:3947-3972, `include/strata/core/nv_chunk.hpp`;
  `STRATA_NV_CHUNK=0` keeps 8192, `=16` 16384; a layer split goes back to 8192; a RAM guard steps 32768 -> 16384 -> 8192 when the extra loan does not
  fit, gen:9905-9935). gfx12 Linux has the same raise (gen:3974-3993). The lend is capped at `kAutoLendPct` = 90 % of the cache slots when the
  arena is (nearly) all pinned, 85 % otherwise (gen:9876-9881).
* Per request the chunk is `min(configured chunk, round_up_256(part length))` (`request_chunk`, gen:9834-9840; `planned_chunk`,
  gen:9863). A part that fits one chunk stays one chunk. `STRATA_PREFILL_EQUAL=1` (opt-in) evens chunk sizes (gen:9841-9858).
  On Linux a **host-aware planner** (default on) may pick a smaller chunk/ring when the file set outgrows the page cache
  (`host_plan_chunk`, gen:10044+; `STRATA_PREFILL_HOST_PLAN=0` off; `docs/DETAILS.md:308-312`).
* **Small-card rule** (gen:10922-10968, `include/strata/core/sm120_series.hpp:143-165` (struct at 152, rule at 158)): one CUDA card, a native pack, and the default chunk would be
  < 8192 -> the engine sets the fused-expert override off and `STRATA_PREFILL_LAYER_MAJOR=1` (from two chunks), and re-sizes the chunk
  for that layout. `STRATA_PF_SMALL=0` turns it off.
* A layer split (`--layer-split`) cuts the layers over GPUs; each stage has its own `Prefill`, hands chunk rows to the next
  over pinned host buffers; `--prefill-pipe` (opt-in) sizes chunks for the pipeline (pf:2781-2873). Not detailed here.

### a.3 One chunk through the 48 layers (`run_impl`)

Chunk loop `for (c0 = 0; c0 < n; c0 += C)` (pf:3479), `T = min(C, n - c0)` (pf:3496). Cancel checks `should_stop` (pf:3480).

1. **Embedding** (pf:3549-3609): token ids up as one pinned upload (`tok_up`), one batched gather
   (`native_embed()->gather_dev` or `embedding_gather_dev`), then `gr_broadcast` copies the row into all 4 streams -> residual
   `R [T, 4, 2560]` fp32. Picture rows (`embd_rows`) go through a pinned staging ring (pf:3581-3608).
2. **PLE rows** gather started on a thread (pf:3613-3617), see d.4.
3. **QSA step records** (`qsa_step_fill`, pos/n_kv/n_bid/width per token) uploaded (pf:3672-3676).
4. **Expert stream plan** (when `stream_all`), issuer thread (pf:3750-4028), see b.
5. **Layer loop** `for l in [LB, LE)` (pf:4052). At layer 1 the PLE block runs (pf:4068-4126). Each layer has `half` 0 (attention)
   and 1 (FFN) (pf:4134):
   * **Hyper-connection read** (pf:4144-4224): `gr_norm_rs` (RMS row scale + BF16/FP16 image of `R`; skipped when the
     previous half's write already normed it, `normed`, pf:4165-4167) -> BF16 GEMM "down" (10240 -> 320) -> `gr_silu` ->
     BF16 GEMM "up" (320 -> 10240, `gated`) -> BF16 GEMM "inject" (10240 -> 4) -> `gr_mix_r`: `mixed[t,d] = mean_c xn * sigmoid(gated)`
     in fp32 plus BF16 and FP16 images (`include/strata/prefill/kernels.hpp:14-70`). Optional fused variants (`STRATA_HC_UPMIX`, `STRATA_PF_HCDOWN`,
     `STRATA_HCD_EXACT`) are opt-in.
   * **Half 0, GDN layer** (pf:4230-4280): `attn_qkv` and `attn_gate` GEMMs (native quant weights -> FP16 scratch -> GEMM, `native_proj`
     pf:3067), BF16 `ssm_alpha/beta`, `gdn_gates`, `gdn_conv` (4-tap causal conv + SiLU + L2 norm of q/k, history carried), `gdn_recurrence`,
     `ssm_out` GEMM into `bo`.
   * **Half 0, QSA layer** (pf:4281-4635): K/V/Q projections, indexer K/Q projections, fused norm+rope (`rms_rope_rows`),
     KV append (+ staging for KV streaming), indexer append, block scores + top-k selection, tensor-core attention with gate
     epilogue, `attn_output` GEMM into `bo`. See d.
   * **Half 1, MoE** (pf:4636-6180): router GEMM (BF16, 512 logits) -> `route` (softmax over 512, top-10, ties to lower id, weights
     renormalised over the ten; pf:4645-4646, `kernels.hpp:100`); shared expert (gate/up/down native GEMMs + sigmoid-gated by a BF16 scalar
     projection, pf:4708-4717); routed experts (b); `moe_combine` writes `bo = shared*sigmoid(sg) + sum_k w_k * Dm[slot_k]` (pf:6149-6153).
   * **Hyper-connection write** (pf:6181-6250): `gr_write_norm_rs` = `R[t,c,:] += bo[t,:] * 2*sigmoid(inj[t,c]/hc)` fused with the next
     half's norm (F-2); a steered layer (control vector) uses `gr_write_cvec_norm_rs` or a separate `cvec_apply`.
6. After the layers: `ple_land` safety (pf:6252), join the issuer (pf:6253-6267), `stats`, layer-split hand-off (pf:6271-6319), debug dumps, then
   `cudaStreamSynchronize(cs)` and `on_chunk`/`on_stage_chunk` (pf:6362-6373). After the last chunk the PLE history/prev tokens
   are stored in the session (pf:6377-6378) and the copy stream is drained (pf:6397-6405).

### a.4 Streams, events, threads, graphs

* Streams: `m.cs` (compute, the engine's stream), `m.copy` (non-blocking, expert DMA; created pf:1876), optional `kv_copy`
  (`STRATA_KV_PREFETCH=1`, pf:3365), `ple_copy` (`STRATA_PLE_COPY_STREAM=1`), a dequant side stream (`STRATA_DEQUANT_SIDE=1`, off).
* Events per ring slot: `copied[i]` (recorded on `copy` after the DMA; compute waits on it) and `used[i]` (recorded on `cs` after the
  slot's data was gathered/consumed; the copy stream waits on it before overwriting the slot) (pf:1883-1886, 3821, 3830, 4860, 5797).
* Host threads: the launching thread; `strata-issue` (issuer; `STRATA_PREFILL_ISSUER=0` = inline) enqueues DMAs (pf:3914-4000); `strata-stage`
  pool (`Stager`, pf:712-905; 2-4 threads normally, 32 for GGUF-in-place on SSD, pf:1909) copies unpinned/transient experts into pinned
  buffers; a PLE gather thread (`std::async`, pf:3613, 3661); optional CPU-expert job (`cpu_fut`, pf:5187); optional `strata-hand`
  (layer split). Waits sleep rather than spin on Linux (`stager_sleep`, pf:253).
* GPU vs CPU work: GPU does all dense/attention/expert math. CPU does: PLE index hashing + SSD/RAM row gather; expert staging copies;
  the **per-MoE-layer host grouping of (token, k) pairs by expert on the MMQ path** (the ids are downloaded to mapped pinned memory and the
  stream is synchronised once per MoE layer, pf:4870-4934; the fused path groups on the GPU and has no per-layer sync, pf:4787-4867);
  optionally CPU experts (b.8). `STRATA_PREFILL_TIMING=1` prints the GPU-time-by-phase table (pf:3104-3153, 6448-6463).

---------------------------------------------------------------------------------------------------------------------------

## b. Experts in a prompt

### b.1 Where an expert's bytes can live (the "tiers")

* **VRAM expert cache** (static profile-ranked residency table `host_res[layer*512+e]` = slot or -1; plus the adaptive tier in decode). A resident expert is read in place
  (fused kernels get its slot pointer; MMQ gathers it into a group buffer, or with `STRATA_MMQ_DIRECT=1` reads it in the slot) (pf:4825-4832, 6101-6106).
* **Pinned RAM arena** (all experts copied into page-locked RAM, 23-50 GB; `docs/DETAILS.md:242-285`): `ExpertSource::pinned(l,e)` -> the DMA
  goes straight from the arena to the VRAM ring slot (`cudaMemcpyAsync(stage_dev[sl], blob, bytes, H2D, m.copy)`, pf:3823, 3985).
* **Unpinned RAM / file tier** (the arena could not pin everything; or low-RAM modes `--resident-budget-gib`, `--mmap-experts`; GGUF read in place):
  `Stager` threads copy the blob into a pinned bounce buffer first (`jobs`, pf:3781-3800), then DMA (`stager->wait(job)` + `issued_one`, pf:3826-3828).
  For GGUF-in-place with SSD-resident pages the stager uses 32 threads, a 4x-thread ring and 8-blob batched reads (pf:1895-1917, `STRATA_STAGER_*`).
* **SSD** is only ever reached through the file tier (above) or through the PLE table; the engine has no separate SSD expert cache design
  for prompts - pages come through the OS page cache / `copy_blobs`.
* `--peer-device` / layer-split helper GPUs and the CPU pool are further sources (b.8).

### b.2 Which experts: every non-resident expert, in id order, for big chunks

* `stream_all = ring > STAGE && T >= stream_all_min() && src != nullptr` (pf:3713). `stream_all_min()` is **1024** by default (pf:273-277, `STRATA_PREFILL_STREAM_MIN`) but
  **3072 when the default CPU share is armed** (pf:2981-3008, `share_rules::default_share_limit()` = 3072, `include/strata/prefill/share_rules.hpp:19-24`; `docs/DETAILS.md:1053` still says 1,024: doc stale).
* For such a chunk the plan `seq` lists, **for every layer in order and for every expert id 0..511 that is not resident in VRAM (and not held by a peer)**,
  one entry (pf:3772-3803) - regardless of whether any token of the chunk routes to it. Experts are therefore **copied once per layer per chunk**, in
  fixed (layer, id) order, "nearly all 512 are routed at such a chunk" (pf:240-242). Unrouted entries are released without compute (pf:6076-6085).
* `STRATA_PREFILL_ROUTED_STREAM=1` (opt-in, `include/strata/prefill/routed_stream.hpp`): after a layer's router runs, the issuer drops not-yet-issued entries that no token routes to
  (bit-identical results; "47 % of the streamed experts at 1,045 tokens, 35 % at 2K, 15 % at 8,192" on a 3090).
* A chunk **below** `stream_all_min` takes the **staged walk** (pf:4869-4934 + 6044-6070): the router ids are downloaded and the host sync done, then only experts with at least one routed
  row (`cnt[e] > 0`), non-resident, are staged through a small ring of `STAGE` = 8 slots (`STAGE_GRP` = 32 with MMQ groups; pf:227-231, 5677).
* Cost of the big-chunk scheme: the whole non-resident set crosses PCIe once per chunk (this is why a bigger chunk, or layer-major, helps; comment pf:6535-6546: "~25-34 GB of expert blobs per chunk on Q2_0").

### b.3 The streaming ring: slots, sizes, copies

* A ring slot is **one whole blob of the pack's largest layer** (`MAXBLOB()` = `expert_layout().max_blob`, pf:226; a native pack's blobs differ per layer). Blob sizes quoted in docs: Q2_0 1,382,400 B (pf:399), IQ2_XS 1.51 MB, IQ3_S 2.54 MiB, NVFP4 2.76 MB
  (`docs/NVFP4.md`, pf:176-179 header).
* Slot count `ring_slots(T)` (pf:429-465): T < stream_all_min -> `STAGE` (8); else `STRATA_PREFILL_RING` if set; else a layer split's override (96) or the auto scan's byte-budget ring; else
  the 0.1.39 rule: **1024 slots if the fused kernels cover the pack (1024 only reachable for the non-native Q2_0 pack; `ring_cap()` is 512 for native packs), 384 if the arena is (nearly) fully pinned and the fused ring does not apply, 96 if a large share of the experts goes through unpinned host copies**
  (pf:459, 396). Capped by `ring_cap()` (1024 for fused Q2_0, else 512, pf:396). With `STRATA_RING_BYTES` (default on) the budget is in bytes: `1024|384 x 1,382,400 B / MAXBLOB`, never past the cap (pf:400-428);
  the auto chunk scan trades chunk size against ring size (gen:9972-10022, "the largest chunk that still leaves the ring full"). On gfx12 HIP 96/192/256 (pf:431-452).
* Entry k lands in slot `k % ring`; the issuer may issue entry k only after entry `k - ring` was consumed (`a_consumed`, pf:3958); the copy stream waits on `used[slot]` (pf:3977); the compute stream waits on `copied[slot]` (pf:4012-4024).
  The issuer is a thread so that host copies of unpinned blobs never stall kernel launches (comment pf:3837-3840).
* The ring lives in the **lent region of the VRAM expert cache** (b/c); an optional extra "room" of ring slots can point at the MTP drafter's 512 decode-only experts for one prompt (`STRATA_MTP_ROOM_MIN`, `prefill.hpp:135-156`; off by default).
* PCIe: plain `cudaMemcpyAsync` H2D from pinned RAM on the copy stream. **`cudaMemcpyBatchAsync` / `STRATA_DMA_BATCH` (docs/BATCHED_DMA.md) is not used by the prompt path**: grep shows it only in `src/core/verify.cpp`
  and `include/strata/core/dma_batch.hpp` (decode verifier groups; doc says "a verifier group's independent expert uploads").

### b.4 Expert GEMM kernels (CUDA path) and what each covers

Gate/up/down are int8-tensor-core products with activations rounded to int8 (not FP16 weights), except the FP16 fallback.

| Path | When | Quant formats (code) |
|---|---|---|
| **Strata fused int8 kernels** (`src/prefill/moe_fused.cu` Q2_0 pack; `moe_fused_iq.cu` native packs) | only `stream_all` chunks, single GPU, no peer (pf:4751-4753). Q2_0 pack (Strata blob layout): on by default on sm_80+ since 0.1.36 (`fused::enabled()`, `STRATA_PF_FUSED=0` off). Native packs: **default only on a cc 12.0 card with >= 80 SMs** and only for gate/up IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS with Q2_0/IQ4_NL down (`moe_fused_iq.cu:2046-2087`); on sm_121 (GB10) `STRATA_PF_FUSED=1` is set by default (`src/core/nv_defaults.cpp:60-70`); else opt-in `STRATA_PF_FUSED=1` | gate/up: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS (+ Q2_0 with `STRATA_PF_FUSED_Q2N=1` or Turing); down: Q2_0, IQ4_NL (`gu_covered`/`d_covered`, `moe_fused_iq.cu:1906-1925`). HIP-only extras: Q4_K/Q5_K gate/up, Q5_1/Q8_0/Q5_0 down (gfx11/gfx12, opt-in or gfx1151 Windows default) |
| **llama.cpp MMQ** (`moe_mmq.cu`, ggml-cuda `mmq.cuh`, activations q8_1, tiles of 16-expert groups after a `gather_native_group` copy into group buffers `grp_gu/grp_d`; `MMQ_GROUP`=16, pf:1474) | default for every layer that has an MMQ tile on every visible GPU (`mmq_plan`, pf:1528-1632; `mmq::fits`, `moe_mmq.cu:354`) | Q2_0 (ggml type 42), IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q8_0 (`moe_mmq.cu:324-352`); **K-quants Q4_K, Q5_K, Q5_1, Q6_K only with `STRATA_MMQ_KQ=1`** through a separate `libstrata_kq` library - **on by default for NVIDIA cc >= 8.0** (`nv_kq_default_env`, `src/core/nv_defaults.cpp:112-127`, gen:3926-3942; `docs/QUANTS.md` still says opt-in: doc stale); Q5_0 only in a build with `STRATA_ORCA_Q4KS_MMQ` (`docs/ORCA_Q4_K_S.md`); **NVFP4** `nvfp4_mode()`: w4a8 default (int8 tensor cores, q8_1 activations), `STRATA_PREFILL_NVFP4=w4a4` (Blackwell FP4 x FP4, activations rounded to NVFP4; default on sm_121), `fp16` = reference path |
| **IQ4 w4a8 kernel** (`moe_iq4_w4a8.cu`, new since 0.2.11) | opt-in `STRATA_IQ4_W4A8=1`/`2`, sm_80..sm_88 (sm_120 only where the prompt runs on MMQ) | IQ4_XS and IQ4_NL products, MMQ's tiling-mode bits ("Same bits as MMQ"), declines stream-k-split launches |
| **weight-stationary** (`moe_ws*.cu`) | `STRATA_MMQ_WS=1`, GP100 (sm_60 experimental build) only | IQ2_XXS..IQ3_S gate/up, Q2_0/IQ4_NL down |
| **FP16 WMMA tiles** (`gemm_iq_f16_grouped`) | `STRATA_PF_WMMA=1`, all cards cc >= 7.0 (e.g. Volta; also takes IQ1_M) | layers MMQ has no tile for (pf:1510-1527, 1549-1557) |
| **dequantize to FP16 + cuBLAS** (`blob_dequant_f16` / `iq_dequant_gu_f16` + `Gemm::f16`, one expert at a time, 2 FP16 scratch slots `dq_gu/dq_d`, pf:6000-6028) | fallback for layers with no MMQ/fused coverage (IQ1_M, K-quants without kq library, ...) | everything `iq_dequant_f16` accepts (`is_iq`, `iq_kernels.cu:3381`: IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS, IQ1_M, Q2_0, **Q3_K**, Q4_K, Q5_K, Q6_K, Q5_1/Q5_0, **Q4_0**, Q4_1, Q8_0, NVFP4, trellis types); "an expert as ~10 MB of FP16" (`docs/QUANTS.md`) |

Notes:
* **Q3_K and Q4_0 as *routed-expert* prompt formats**: no MMQ, fused or WMMA coverage exists for them in `src/prefill` (grep). They appear in `iq_row_bytes` /
  `is_iq` (generic dequant) and Q4_0 has CPU expert kernels (`docs/UNSLOTH_Q4.md:248`); that they would run through the FP16 dequant path is **(inferred)** - not verified that the pack
  loader admits them as expert types. Q3_K appears in code as the Q2_0 file's token_embd (`iq_kernels.cu:3049`).
* APR Hadamard-folded layers (TQ trellis types) take MMQ (the fused kernels do not rotate); NVFP4 keeps 4 floats of per-expert tail scales `{s_gate, s_up, s_down, 0}` applied inside the combine (`row_sd`).
* The expert's down output goes to `Dm` rows (`T*K*2560` fp32) in "expert order"; `slot_dev[t*K+k]` maps back; `moe_combine` weights and sums them (pf:6149-6153). `STRATA_PF_DM16` (gfx11 F16 kernels) writes FP16 rows (new in main).
* Fused launches: batches of <= 128 experts (`kMaxBatch`, `moe_fused.hpp:33`), each batch limited to `max(1, ring/3)` streamed experts so the next batch's blobs arrive while one computes (pf:4801); the batch's last blob is the only wait.

### b.5 Ordering inside a layer

Experts are processed in id order (routed ones only in the MMQ path, `order` pf:5101-5105). Resident and streamed experts interleave in that order; MMQ groups of 16 are gathered with **one** launch and **one** wait on the last copy of the group (`gather_native_group`, pf:5768-5802), and
the group's ring slots are released by one event. Optional `STRATA_MMQ_GROUP_SORT` / `STRATA_MMQ_RESIDENT_SORT_NE` sort resident groups by row count (opt-in).

### b.6 Is any expert dropped / approximated in the batched path?

No: every routed (token, expert) pair of the chunk is computed by the GPU, the CPU share, a peer GPU, or a helper. `STRATA_ROUTE_TAIL_SKIP` is a **decode/verify-window** feature (see f.9). The only dropping in prefill is the unrouted-entry drop of `STRATA_PREFILL_ROUTED_STREAM` (copy only) and KVA (opt-in, f.7).

### b.7 Resident-expert special paths

`STRATA_MMQ_DIRECT=1` (needs a `-DSTRATA_MMQ_TAB=ON` build, 36-38 MiB VRAM per GPU; opt-in) reads resident experts in their slots (bitwise); `ws_layer` weight-stationary (GP100).

### b.8 Does a CPU part compute experts during prompts? Yes, for smaller chunks (default on, with conditions)

* `STRATA_PREFILL_CPU_SHARE` (`pf:278-331, 2981-3009, 4882-5048, 5153-5223, 6114-6139`): for a **staged** chunk (below `stream_all_min`) the decode CPU expert pool (idle during a prompt) computes some of the **non-resident experts that few tokens route to**
  (cand = routed by <= `MAXT` tokens, fewest first, up to a measured share `g/(c+g)` of the experts that would be streamed; `auto` measures CPU ms/expert and GPU ms/streamed expert with CUDA events and also checks that
  sharing pays via a ratio gate, pf:4901-4986). The CPU quantizes its tokens' activations in its own format (`native_quant_act`, `ActQ` for Q2_0) and writes unweighted rows into the tail of `Dm`; they join `moe_combine` as any other row.
* **Default**: armed (as `auto`) when `arm_cpu_share(share_pool, by_default)` is called with `by_default` = one GPU, no layer split, not `--no-pool`, no `--batch` (unless `STRATA_PREFILL_CPU_SHARE_BATCH`), `share_has_work` (some experts are not in VRAM) and the RAM
  verdict of `share_rules::default_share_memory_verdict` is OK (gen:10767-10778, `share_rules.hpp:51-71`). CUDA builds and gfx12 HIP. Off: `STRATA_PREFILL_CPU_SHARE=0`. Needs a native pack (`cpu_maybe` requires `lay.native`, pf:4882).
  Code limit is **3072 tokens** (see b.2); the code message says "answers can differ slightly: mean KL ~0.004 below 1024 tokens, ~0.011 at 1-3K" (pf:2985-2988).
* A chunk >= `stream_all_min` never uses the CPU pool (`pool_hold` requires `!stream_all`, pf:3720). One stage at a time takes the pool on a layer split (`g_cpu_pool_busy`, pf:334).
* Side effect (inferred from pf:3008): arming the default share also raises `g_stream_min_share` to 3072 even for packs that cannot use the CPU share.
* Other CPU/GPU helpers: `--peer-device` (second GPU computes the experts it holds with P2P, `PeerPrefill` pf:911+), layer-split helper (`set_stage_helper`, `STRATA_PREFILL_HELP=1`, opt-in), `STRATA_PF_PEER_CPU=1` (new in main: CPU share also beside a peer GPU).

### b.9 Layer-major prefill (reads each layer's experts once per "super-chunk")

`pf:6535-7900`. Off by default everywhere (`prefill_layer_major_default` returns false, `sm120_series.hpp:131`; a layer split reads 32K prompts 37-55 % *later* layer-major, owner decision 2026-11-02). On when `STRATA_PREFILL_LAYER_MAJOR=1`
or via the **single-card small-cache rule** (a.2). For a super-chunk of S tokens (target 8192 explicit, 32768 only if "by default"; pf:6592-6594, 6687; S a multiple of the chunk; fits VRAM minus a 96 MiB (explicit) / 512 MiB margin, pf:6585-6591, 6702)
it runs, layer by layer: every sub-chunk's attention half + router + shared expert (state advances exactly as chunk-major), parks the FFN inputs and routing, **then the layer's experts once for all S tokens** (each routed expert staged once, its rows of all sub-chunks in one MMQ group,
results added per token into an accumulator in group order -> another fp32 summation order, so not bit-identical). Residual rows live in VRAM (`STRATA_PREFILL_LM_R=vram`) or pinned host. Declines (with a log line): image prompts, KVA on, DFlash2 taps on a split, peer GPU,
`stop_at_chunk`, fused layout, APR/NVFP4/IQ1_M layers (pf:6615-6700). `run_multi` reads several prompts in one pass (batch co-reads). The small-card rule sets the env var with `setenv(...,0)`, which probably also makes `lm_env_explicit()` true, i.e. 8192-token super-chunk and 96 MiB margin rather than the "by default" 32768/512 MiB **(inferred, pf:7973-7974, gen:10941-10947)**.

---------------------------------------------------------------------------------------------------------------------------

## c. VRAM layout during a prompt; what is lent and restored

* **Borrow, not reserve**: with `pf_borrow` (profile present, no `--no-prefill-borrow`) the expert cache is sized at full size first and the prompt path carves **all** its device buffers from the tail (top slots) of that cache (`Prefill::init(..., borrow, borrow_bytes)`, pf:1807-2043;
  `carve`, pf:2047-2202; comment gen:6030-6035). `bytes_needed` (pf:2891-2953) mirrors the carve order so the lend is priced exactly (~680 KB per prompt token + 160 MiB fixed is the planning estimate for an *own* allocation, gen:6046).
* What is carved, in order (pf:2003-2192): GEMM FP16 scratch 32 M elements (64 MiB) + cuBLAS workspace 32 MiB (+ optional 2nd scratch); `emb`, `R` (T x 4 x 2560 fp32), `grs`, `xn16`, `lo`, `lo16`, `gated`, `inj`, `mixed`, `mixed_bf`, `mixed_h`, QSA `steps_dev`; **one shared region** sized as the max of the GDN set, the QSA set and the MoE set
  (they are never live together: "~260 KB of the ~680 KB a prompt token cost", comment pf:1448-1452); two FP16 dequant slots `dq_gu/dq_d` (1280x2560 + 2560x640 halves); `ids_identity` + bounds + the 16-expert MMQ group buffers `grp_gu/grp_d`; the streaming ring (`stage_slots` x `MAXBLOB`) ;
  `ple_emb` (T x 2560 fp32), `ple_norm`; with KV streaming a per-layer **KV staging pool** (`take_stage`, pf:1320-1345). Host side: pinned stager buffers, mapped pinned routing tables (`grp_host`), PLE double buffers, CPU-share buffers.
* The **KV cache of the context itself, the GDN/conv/indexer/PLE running state, the dense weights (attention, hyper-connection, router, shared expert) and the output head stay in VRAM untouched**; only expert-cache slots above `first` are lent. (`--prefill-head-alias`, opt-in: the native output head's 417 MiB is
  unloaded during a prompt and lent too, reloaded in ~255 ms before the first decode window; `docs/PREFILL_HEAD_ALIAS.md`, gen:7389-7411, 18142.)
* **Lend** (gen:17153-17237): size = chunk the part needs (`want`, <= configured chunk); `Prefill::relayout(want, device_slot(first), bytes)` re-carves in the tail; every expert whose slot index >= `first` is **marked non-resident** (`host_res[i] = kNotResident`, remembered in `p.lent` as (expert index, slot)) and the table is re-uploaded (`res_upload`).
  So a short prompt lends few slots; "a request lends only what its own prompt needs".
* **Restore** (`refill`, gen:17126-17148; `refill_issue/refill_wait` 17085-17121): for every lent (expert, slot): `srcp->blob_stable(l,e)` (pinned arena / RAM copy) -> `fill_slot_queued` = `cudaMemcpyAsync` H2D on the default stream (`src/core/expert_cache.cpp:852-868`) into the **same slot**; then `sync_queued`, `host_res[i] = slot`, `res_upload`. It runs
  before any verify window (`win && refill`, gen:18067) and after the last part (gen:18137), including on cancel. So **the decode expert cache is restored byte for byte with the same experts in the same slots**: it is *not* re-ranked from the prompt's routing. (Adaptive-tier swaps made before the prompt are preserved because the lent list records the experts actually resident at lend time.)
  Cost is part of prompt time: "refill ~180 ms", "a few dozen tokens pay ~0.6 s" (`PREFILL_HEAD_ALIAS.md`, comment gen:16843-16848).
* The prompt's routing can seed a RAM-side policy (`on_routes`, `STRATA_RAM_ADAPT2`, off; gen:13886-13891, pf:4936) - not VRAM.
* The MMQ row table (`ids_identity`) lives in the borrowed slots, so it is rewritten with `mmq::iota` at the start of every prompt (pf:3421-3423).
* `STRATA_MTP_ROOM_MIN`: opens/closes the drafter-experts room around a prompt (`room_close`, gen:18141).

---------------------------------------------------------------------------------------------------------------------------

## d. Hyper-connections, GDN, QSA, PLE in the prompt path

### d.1 Hyper-connection kernels (prefill, not `fused_gr.cu`)

The prompt kernels are in `src/prefill/kernels.cu` (declared `include/strata/prefill/kernels.hpp:14-70`): `gr_norm_rs`, `gr_mix_r`, `gr_silu`, `gr_write_norm_rs`, `gr_write_cvec_norm_rs`, `gr_broadcast`, optional `gr_upmix` (`STRATA_HC_UPMIX`, sm_80+, T >= 33). The three projections are `Gemm::bf16` (cuBLAS BF16 x BF16 -> fp32,
`gemm.hpp`; a Q8_0 HC weight set, default on CUDA for Unsloth `UD-*` files, is dequantized to BF16 per half, pf:4151-4159, gen `STRATA_HC_Q8` default). `src/kernels/cuda/fused_gr.cu` (read as asked) is the **decode / verify-window** fused kernels (single token and multi-row; 3818 lines) - it is not used by the batched prompt path, but **is** used by the <= 64-token "short read" windows.

### d.2 GDN (Gated DeltaNet) in prompts; state carry

* Projections: `attn_qkv` (-> conv channels 10240), `attn_gate`, `ssm_alpha/beta` (BF16, 48 value heads), `gdn_gates` (softplus/sigmoid) (pf:4241-4253).
* `gdn_conv` (`kernels.cu:2948`): 4-tap causal conv + SiLU over the chunk with a **3-row history carried in the session** (`conv` = tail of the per-layer state block), a history-update kernel, then per-token L2 norm of the q/k heads.
* `gdn_recurrence` (`kernels.cu:2978-3172`): the delta-rule recurrence **walks the chunk's tokens in order inside one launch**, state `[128, 48, 128]` fp32 in registers, updated in place in `ss.gdn_state` (pf:4239-4240) so the next chunk (and decode) continues from it. CUDA dispatch: default = key-head kernel
  `gdn_rec_kh_kernel` on sm_80+ cards that hold its 64 CTAs at once (`STRATA_GDN_KEYHEAD=0` off), else the software-pipelined column kernel; **Linux CUDA: the engine sets `STRATA_GDN_TC=1` and `STRATA_GDN_CONVL2=2` by default** (new in main: `apply_gdn_defaults_cuda`, `src/core/arch_defaults.cpp`, gen:3913; Windows not) = chunked scan on tensor cores for chunks >= 128 tokens (needs cc >= 8 and ~99-122 KB shared memory per CTA; falls back to the old kernels) and a one-pass conv/SiLU/L2 kernel. KL of TC vs plain: "mean 0.0019, max 0.0125; prompt +4.4 % at 4K, +5.5 % 16K, +5.7 % 32K"
  (`include/strata/core/arch_defaults.hpp:90-94`, l620 measurement). `STRATA_GDN_CHUNKED=1` (opt-in on CUDA; auto on gfx12 Linux) = 32-token chunked WY form; `src/prefill/gdn_chunk.cu` (64-token chunks) is Volta-only. The output RMS-norm * sigmoid(z) is fused in the kernel or follows (`gdn_out_norm`); `ssm_out` GEMM follows.
* KVA's "state-only" form (f.7) runs the same recurrence without the output norm/out-projection.

### d.3 Full attention (QSA) in prompts

* Projections K, V, Q (`Qf` = q | gate per head, 24 x 512), indexer K (BF16 -> 128) and indexer Q (BF16 -> 4 x 128). `rms_rope_rows` fuses K/Q/indexer-Q RMS norm + rotary on the first 64 dims (default on; same bits as the separate kernels).
* **KV cache writes**: `kv_append` writes the chunk's K/V cells (`p0..p0+T-1`) into the paged pools (int8 + fp16 scale per 64 by default `--kv int8`; FP16, Q4_0 rotated `kv_append_q4`, or K8V4 hybrid; Hadamard `fwht256` for q4/hybrid), and with KV streaming also into the host copy and the staging pool (pf:4336-4366).
* **Indexer cache writes**: `native_qsa_indexer_append_batch` (pf:4390) appends the chunk's 128-d indexer keys in 3 launches: the unfinished block's raw keys (`idx_tail`, 3 x 128 fp32), the `dead` key, completed blocks' pooled keys (`idx_pooled`, one per 4 cells) and `idx_block_pos` - leaving the exact state n single appends would.
* **Selection** (pf:4412-4498): `qsa_block_scores_tc` (3xTF32 tensor-core scores, FP32-level, **not bitwise** vs the per-token kernel; below sm_80 only with `STRATA_SELECT_SIMT=1`) over `sel_batch` = 256 queries at a time, `qsa_block_topk` (4-pass radix select, blocks weighted by cell count, cells ascending, ties to lowest index; the incomplete tail block always chosen).
  `STRATA_SELECT_TWO_PASS=1` (opt-in; the register's "default on" note is wrong, `qsa_select.cu:3785-3792`) scores in single-pass TF32 with an error bound and re-scores only near the cut-off: same cells as 3xTF32 by construction.
* **Attention** (`qsa_prompt_attn_batch(_gated)`, `src/kernels/cuda/qsa_prompt_attn.cu`): one CTA per (query, KV head) over its selected cells in chunks of 32, online softmax, m16n8k16 FP16 MMAs with FP32 accumulate over the 12 query heads of the KV head; q and p split in FP16 hi+lo; int8 KV codes are exact in FP16, scales applied in FP32;
  **INT8-KV prompt attention on INT8 tensor cores (`STRATA_PROMPT_ATTN_IMMA`) is on by default for cc >= 8.0** (`sm120_series.hpp:93-96`). Gate `sigmoid(gate) * O` is applied in the kernel epilogue (`qsa_fuse_gate`, default on, where no KV rotation). Not bitwise equal to the decode kernel (header comment). Q4_0 KV uses the old decode kernel in 32-query batches or the tensor kernel's mode 4.
* Then `attn_output` GEMM -> `bo`.

### d.4 PLE (layer-1 per-layer-embedding / n-gram table): how rows are read for a chunk

* Table: `per_layer_token_embd`, **320,001,536 rows x 90 B (IQ4_NL) = 28.8 GB** (26.8 GiB; 160 B/row FP8, 320 B BF16, 170 B Q8_0, 120 B Q5_1, 90 B NVFP4 variants) (`ple_reader.hpp:3`, `ngram.hpp:50-56`). 16 rows per token = 2 n-gram orders x 8 hash heads (`PLE_N_HEADS` 16, `PLE_HEAD_DIM` 160 -> one 2560 vector).
* **Which rows**: `ngram_rows(token, two previous tokens, consts)` hashes (token, prev, prev-prev) to 16 row indices (`ple_gather`, pf:3435-3455). The two tokens before the chunk come from `ss.ple_prev` (session). Depends on tokens only, so the next chunk's rows are known one chunk ahead.
* **Reading**: `PleTable::gather_batch(rows, T, out)` (`src/kernels/ngram.cpp:649`): in Direct mode **one `PleReader` request for all 16*T rows**: rows -> 4 KiB pages, **deduplicated by page**, **sorted by file offset**, issued with up to 256 reads in flight, via a worker I/O thread; a row cache (1,048,576 rows = ~95 MB, 8-way set-associative) absorbs repeats ("within one long prompt 20-34 % of rows recur"). Batches of <= 64 rows use a sort-and-merge stack path (new in main); thousands of rows use a hash by page (`ple_reader.cpp:431-545`).
  Mmap mode gathers row by row with a WILLNEED / PrefetchVirtualMemory hint per 4096-row window (`ngram.cpp:701-720`).
* **Where the table lives**: default `--ple-io direct` = unbuffered SSD reads, table never in RAM; **`ple_auto_ram_allowed` residency policy** (gen:5462-5511): on a PC with RAM >= table + expert arena + 8 GiB the engine switches itself to `--ple-io ram` (table mlocked in RAM; measured PLE block 845 ms direct / 451 mmap / 132 resident on an 8K prompt, `docs/DETAILS.md`; "on the 72 GB card prompts +16.5 % at 1K, +30.6 % at 4K, +16.8 % at 8K"); a start probe may warn or choose mmap on slow drives (gen:5521-5560).
* **Overlap**: the gather for chunk 0 starts before layer 0 (`ple_next = std::async`, pf:3613) and is awaited only at layer 1 (`ple_land`, pf:4068-4071) - "gathering them first left the GPU idle for the whole read (~0.4 s of a 32K prompt)" (pf:3610-3612); `ple_land` uploads the rows (pinned double buffer `ple_emb_host[2]`, `cudaMemcpyAsync` H2D) and immediately starts the **next chunk's** gather into the other buffer
  (pf:3620-3669). `STRATA_PLE_COMPACT=1` (opt-in) uploads the raw 90 B rows (7x fewer bytes) and dequantizes IQ4_NL on the device; `STRATA_PLE_COPY_STREAM=1` puts that upload on its own stream.
* **Layer-1 block** (pf:4072-4126): batched (`STRATA_PLE_BATCH`, default on except a CUDA 12.8 build on cc 12.0, pf:68-91): key projection GEMM (BF16 or native) and value projection GEMM over the chunk, then `native_ple_postops_batch` (causal conv over the previous 9 normalised rows, i.e. `ss.ple.hist` = 9 x 10240 fp32 carried across chunks, normalisation, gating and addition into the residual rows `R`); else token by token. The gathered rows are exact dequantisations (same floats as the host path).

---------------------------------------------------------------------------------------------------------------------------

## e. Conversation cache / snapshots

* **What a checkpoint is** (`ConversationCheckpoint`, `src/core/conversation_state.cpp:113-135`, comment gen:2227-2245): only the **running state**, because the KV cells are positional and are not copied: the 36 GDN states (817,152 floats = 3,268,608 B per layer -> **117,669,888 B**), each QSA layer's indexer tail (3 x 128 fp32) / `dead` (128 fp32) / block position, the PLE history (9 x 10240 fp32 = 368,640 B) - **~118 MB** per checkpoint (DETAILS: "about 118 MB each").
  A checkpoint is valid only while the positional cells below it still hold its tokens, so the loop keeps only checkpoints that are prefixes of the session's tokens. Taken after a synchronised chunk, in RAM (host vectors; `STRATA_CKPT_ASYNC=1`, `STRATA_CKPT_VRAM_SLOTS=N` opt-ins move them card-to-card / keep the last N in VRAM).
* **When saved**: `--prompt-cache N` (default **6** in RAM, `gen:928`; 0 = off): (1) at each **turn boundary** (last `<|im_start|>`: the conversation so far), (2) at the **system-prompt root** (first turn boundary, >= `--prompt-cache-root` 2048 tokens, kept for good), (3) **every `--prompt-cache-every` 16384 prompt tokens at a chunk boundary** (gen:12707-12741; the read does not split - it just saves at the first chunk end at or past the mark;
  "The interval is not a piece-size setting", `docs/PROMPT_CACHE_TAIL.md`), (4) optionally `--prompt-cache-tail`: one extra checkpoint at the first chunk boundary within one chunk of the prompt's end (evicted first, root never dropped; off by default, needs `--prompt-cache` > 0 and `-every` > 0; gen:12710-12741),
  (5) optionally `STRATA_CACHE_MESSAGE_BOUNDARY=1` (opt-in, off by default): one more cut at the previous turn boundary before the final <= 1024-token message when the prefix is >= 8192 tokens (`include/strata/program/message_boundary.hpp`; **it splits chunks** - extra cut in `cuts`, gen:18034; excluded with `multi_gpu`; historical validation: first complete 100-record edit 17.236 -> 11.527 s with 16K->32K chunks combined, two modified RTX 3080 20 GB, `docs/MESSAGE_BOUNDARY_CACHE.md`; "not a measured speedup of this port"). Also `strata_prefix` / `pin=N` request keys pin a document checkpoint (needs `--prompt-cache >= 3`).
  `--prompt-cache-spread` (opt-in) thins kept checkpoints geometrically. Eviction: `conv_cache::eviction_victim` with tail/pinned flags (gen:12641).
* **Parked conversations** (the conversation cache): a conversation that is replaced is **parked in host RAM with its K/V** (retained pages, checkpoints, draft-layer K/V) so alternating requests (sub-agents, controller/worker) reuse their prompts; budget auto = 40 % of RAM left above `--conversation-cache-min-free-mib` (2560) once the model is loaded, <= 8192 MiB in 256 MiB steps, off below 1024; `--conversation-cache-slots` default **4**; not kept across restarts (`docs/DETAILS.md:715-777`).
  `STRATA_PARK_ASYNC`, `STRATA_PARK_VRAM_MIB`, `STRATA_CKPT_POOL`, `STRATA_CKPT_GATHER` (opt-in) move parking copies card-to-card / to background threads (measured on a CMP 170HX, "not measured by us").
* **Disk**: `--slot-save-path DIR` + `POST /slots/0?action=save|restore`: the live conversation (deepest checkpoint + KV), "about 1.2 GB per 63K-token conversation", format Strata's own, bound to model files / `--kv` / rope / version (`docs/DETAILS.md:779-813`).
* **KV sizes**: not stated per token in the prefill docs; KV streaming costs "about 13.7 KB of RAM per context token (1.7 GB at 128K)" (`docs/DETAILS.md:236`) and the disk figure above implies ~19 KB/token for the saved int8 KV (arithmetic, inferred).
* `docs/BATCHED_DMA.md` is **not** about prefill or snapshots: it is the decode verifier's `cudaMemcpyBatchAsync` experiment (`STRATA_DMA_BATCH`, default 0; "does not justify changing the default").
* `docs/PROMPT_CACHE_TAIL.md` and `MESSAGE_BOUNDARY_CACHE.md`: summarised above; both are opt-in and may change chunk/checkpoint geometry (rounding-level differences, "no bitwise parity claimed").

---------------------------------------------------------------------------------------------------------------------------

## f. Things in the prompt path that change answers vs exact computation

"Exact" is hard to define here: the prompt path is a different arithmetic from decode in every default configuration (FP16/BF16 GEMMs, int8 activations, TF32 scores). The repo's own yardstick is first-token KL against the FP16 prompt path (typically 0.01-0.02 "for every path including the default"; `sm120_series.hpp:152-153`). Items:

1. **Default, always**: dense projections via dequantized FP16 weights / FP16 activations (`gemm.hpp`), HC GEMMs BF16, expert products int8 with q8_1 (MMQ) or int8-per-32 (fused) activations (the fused ones "differ from MMQ about as much as MMQ differs from FP32", `moe_fused.hpp:16-18`). BF16-remainder second GEMMs (`STRATA_PREFILL_BF16X2`) are opt-in.
2. **CPU expert share** - default on in the regime of b.8 (CUDA, one GPU, native pack, chunk < 3072 tokens, RAM verdict ok): the CPU computes some experts with its own activation quantisation. Off: `STRATA_PREFILL_CPU_SHARE=0`.
3. **GDN tensor-core scan + one-pass conv** - default on for Linux CUDA (new in main). `STRATA_GDN_TC=0`/`STRATA_GDN_CONVL2=0` off. Runtime-gated by cc >= 8.
4. **QSA scoring/attention**: 3xTF32 block scores, tensor-core attention, INT8-tensor-core attention for int8 KV (default for cc >= 8.0). Selection ties/rounding can differ from decode; the selection itself (top 2048 cells) is the model's own sparse attention, not an approximation of it.
5. **KV precision**: default `--kv int8` (8-bit codes + fp16 scale per 64); `--kv q4_0` / `k8v4` opt-in ("perplexity +8-12 %" for q4_0, `docs/DETAILS.md:233`). KV streaming keeps the full KV in RAM, no loss.
6. **K-quant MMQ** (`STRATA_MMQ_KQ=1`, default on for NVIDIA cc >= 8 since round 0212): "first-token KL 0.024 / 0.035", prompt +26..38 % on an RTX PRO 6000 (`nv_defaults.hpp:41-45`).
7. **Layer-major** (small-card rule or opt-in): different summation order for the expert sum.
8. **KVA** ("faster long prompts", `STRATA_KVA=1`, **off by default**): the last 12 layers (split 36) of every cold chunk before the final 2048 tokens only advance their GDN/conv state and QSA K/V/indexer keys, built from the residual that entered layer 24 ("identity" mode); their attention output and MoE half are skipped (and their expert transfers - `kva_skips_layer`, pf:3767-3775).
   Needs Qwen3.8-Flash-Next shape (48 layers / 512 experts), one GPU, no batch slots / peer / DFlash2, a cold read of >= 4096 tokens (`docs/KVA.md`). Measured KL 0.018-0.020 nats at first generated token; split 24 dropped records in long extraction tasks and failed agent runs, split 36 "found the same 585 of 606 facts". "It can still lose details" (release notes 0.2.7).
9. **Short reads and the last token use the decode verify windows**, not the prompt kernels: decode's MMVQ kernels, CPU experts for misses, and **`STRATA_ROUTE_TAIL_SKIP` (default 7 on CUDA serve, off when every expert is resident / >= 90 % cached / NVFP4 / batch / peer)**: "a missed expert that every token of a verify window routes at rank 7 or lower is neither copied nor computed; its contribution is zero, without renormalisation" (`src/core/expert_source.cpp:4961-4975` and `tail_skip_rank()` at 4964, gen:12821-12867). That this applies to the windows that read a <= 64-token prompt part is **(inferred)** from the shared plan code (`read_windows` -> `ver.run`).
10. **Draft layer**: its K/V come from the prompt GEMM path (not bit-identical to the drafter's own pass) - affects draft acceptance only, never the target's logits (`prefill.hpp:58-65`). Under KVA the draft cells of approximated chunks are skipped.
11. `STRATA_ROUTE_PRIOR` / `STRATA_ROUTE_RESIDENT` (bias routing toward resident experts) are decode-only (not referenced in `src/prefill`) and off by default.
12. No sliding window, no context truncation in the prompt path (the cap is `--max-context`); selection width 2048 cells is part of the model definition.

---------------------------------------------------------------------------------------------------------------------------

## g. Settings and env vars for prefill (defaults as read in code; "doc stale" where docs differ)

CLI: `--prefill N|auto|auto:N` (default none; setup writes `auto`), `--prefill-pipe R` (0), `--no-prefill-borrow` (off), `--prefill-until N`, `--prefill-head-alias` (off, serve + one GPU), `--short-read N` (64), `--prompt-cache N` (6), `--prompt-cache-every N` (16384), `--prompt-cache-root N` (2048), `--prompt-cache-tail` (off), `--prompt-cache-spread` (off), `--turn-token` (248045),
`--conversation-cache-mib` (auto), `--conversation-cache-slots` (4), `--ple-io direct|mmap|ram` (direct, then policy), `--ple-row-cache` (1048576), `--ple-inflight` (256), `--kv int8|q4_0|k8v4|fp16` (int8), `--kv-streaming`, `--kv-resident`, `--peer-device`, `--layer-split`, `--peer-prefill-rows`, `--slot-save-path`.

| Variable | Default | Effect |
|---|---|---|
| `STRATA_PREFILL_AUTO_MAX`, `STRATA_NV_CHUNK` | 8192; NVIDIA cc>=8 bare auto 32768 | auto chunk ceiling (`=0` 8192, `=16` 16384) |
| `STRATA_PREFILL_STREAM_MIN` | 1024 (3072 with default CPU share) | chunk size from which every non-resident expert streams |
| `STRATA_PREFILL_CPU_SHARE` (`auto`/x/0), `_CPU_SHARE_MAX` | auto where armed; limit 3072 (code; doc says 1024) | CPU experts for staged chunks; `=0` off |
| `STRATA_PREFILL_RING`, `STRATA_RING_BYTES`, `STRATA_GFX12_RING` | byte-budget ring (384/1024 slots at Q2_0 blob size) | ring slots / byte budget |
| `STRATA_PREFILL_ISSUER` (1), `STRATA_PREFILL_COPY_EVENT_EVERY` (1), `STRATA_PREFILL_COPY_WAIT_DEDUPE` (0), `STRATA_PREFILL_ROUTED_STREAM` (0) + `_DEPTH` (16) | as given | issuer thread, event sparsity (WDDM), routed-only stream |
| `STRATA_STAGER_THREADS`, `_RING`, `_BATCH`, `_SSD`, `_SLEEP` | 2-4 threads / 16 deep (RAM profile); 32 / 4x / batch 8 (SSD profile); sleeping waits on Linux | unpinned/transient expert copy threads |
| `STRATA_PF_FUSED` (0/1), `STRATA_PF_FUSED_NATIVE`, `STRATA_PF_FUSED_Q2N`, `STRATA_PF_BATCH`, `STRATA_PF_FUSED_NT/F16/WS` | Q2_0 pack on sm_80+: on; native packs: on for cc 12.0 >= 80 SMs and sm_121 | fused int8 expert kernels |
| `STRATA_PREFILL_MMQ` | on | `=0`: FP16 path everywhere (A/B) |
| `STRATA_MMQ_KQ` / `STRATA_NV_KQ_DEFAULT` | on for NVIDIA cc>=8 (code); docs/QUANTS.md says opt-in | K-quant MMQ prompt kernels |
| `STRATA_PREFILL_NVFP4` | w4a8 (w4a4 on sm_121) | NVFP4 prompt precision |
| `STRATA_IQ4_W4A8` (0), `STRATA_MMQ_DIRECT` (0, build flag), `STRATA_MMQ_GROUP_SORT`, `STRATA_MMQ_WS`, `STRATA_PF_WMMA` | off | experts options |
| `STRATA_PREFILL_LAYER_MAJOR` (unset -> off except small-card rule), `STRATA_PREFILL_LM_TOKENS` (8192), `_LM_R` (auto), `_LM_MARGIN_MIB` (96 / 512), `_LM_EVEN` (on), `_LM_FORCE`, `STRATA_PF_SMALL` (rule on) | as given | layer-major walk |
| `STRATA_GDN_TC`, `STRATA_GDN_CONVL2` | 1 / 2 on Linux CUDA (engine-set), unset on Windows | GDN tensor-core scan, one-pass conv |
| `STRATA_GDN_CHUNKED` | CUDA: off (opt-in, 128+ SMs); gfx12 Linux: on | chunked recurrence |
| `STRATA_PROMPT_ATTN_IMMA` | on cc>=8 (code; register says opt-in) | int8-KV prompt attention on INT8 tensor cores |
| `STRATA_SELECT_TWO_PASS` (+`_MIN` 4096 blocks, `_CHECK`) | off (opt-in) | two-precision QSA block scorer |
| `STRATA_QSA_FUSE_GATE` / `_FUSE_NR` | on | attention-gate epilogue, fused norm+rope |
| `STRATA_PLE_BATCH` (on except CUDA 12.8 on cc 12.0), `STRATA_PLE_COMPACT` (0), `STRATA_PLE_COPY_STREAM` (0), `STRATA_PLE_AUTO_RAM` (policy on) | as given | PLE block/rows |
| `STRATA_HC_Q8` | on for CUDA UD-* files | Q8_0 hyper-connection projections |
| `STRATA_HC_UPMIX`, `STRATA_PF_HCDOWN`, `STRATA_HCD_EXACT`, `STRATA_PF_DM16`, `STRATA_PF_GROUP_SCRATCH`, `STRATA_PF_HC_REGION`, `STRATA_EMB_REUSE_ACCOUNT`, `STRATA_PREFILL_EQUAL`, `STRATA_PREFILL_BF16X2` | off | opt-in switches |
| `STRATA_KVA` (0), `STRATA_KVA_SPLIT` (36), `_SUFFIX` (2048), `_MIN_PROMPT` (4096), `_SKIP`, `_ONLY`, `_DRAFT_FLOOR` | off | KVA approximate long prompts |
| `STRATA_PREFILL_HOST_PLAN` (on, Linux file tier), `STRATA_SSD_GBPS`, `STRATA_PREFILL_WARM`, `STRATA_PREFILL_LEND_PCT` (90/85) | as given | host-aware chunk planner, warm-up, lend cap |
| `STRATA_KV_PREFETCH` (0), `STRATA_DEQUANT_SIDE` (0), `STRATA_PIPED_CONCURRENT*`, `STRATA_BATCH_*` | off | misc/batch |
| `STRATA_CACHE_MESSAGE_BOUNDARY` (0), `STRATA_CKPT_REREAD`, `STRATA_CKPT_ASYNC/POOL/VRAM_SLOTS`, `STRATA_PARK_*` | off | cache options |
| `STRATA_PREFILL_TIMING`, `STRATA_PF_ALLOC_TRACE`, `STRATA_PF_STEP_SYNC`, `STRATA_MANIFEST`, `STRATA_TRACE` | off | diagnostics |
| `STRATA_DEFAULTS=026` | unset | restores 0.2.6 behaviour for most of the 0.2.7+ defaults (NV chunk ceiling, GDN etc.) |
| `data/settings-register.json` / `docs/SETTINGS.md` | - | generated register (not always accurate: it lists STRATA_SELECT_TWO_PASS as default on and STRATA_PROMPT_ATTN_IMMA as opt-in, both contradicted by code) |

---------------------------------------------------------------------------------------------------------------------------

## h. Published prompt speeds, with machine, settings, and backing files

All "tok/s" are prompt (prefill) throughput unless stated; numbers are the authors', not reproduced here.

1. **Main table, `docs/DETAILS.md:43-63`, raw `bench/results/2026-09-29-speed-0126/{README.md,matrix.json}`**: RTX 5070 12 GB (PCIe 5.0 x16), Ryzen 5 7600, 64 GB DDR5-5200, Windows 10/11, engine **0.1.26** (old), `--prefill auto`, int8 KV above 4K, KV streaming from 64K,
   MTP spec 4, greedy, 256 tokens, one code-agent prompt per length. 1K / 4K / 32K / 64K / 128K / 262K:
   Q2_0 536 / 1,299 / 2,171 / 2,126 / 2,107 / 1,304; IQ2_XS 534 / 1,256 / 2,092 / 1,754 / 1,752 / 1,181; IQ3_XXS 482 / 1,007 / 1,745 / 1,609 / 1,602 / -; IQ3_S 427 / 913 / 1,624 / 1,640 / 1,443 / -; Coder 656 / 1,583 / 2,177 / 2,236 / 2,208 / 1,034. 262K not re-measured since 0.1.22/0.1.14.
   `docs/MODELS.md:339`: "4K prompt reads at 910 to 1,580 tokens/s on the RTX 5070; a 32K prompt takes about 15 seconds with Q2_0".
2. **Release notes 0.2.7** (`docs/release-notes-0.2.7.md`): one Radeon AI PRO R9700 (32 GB), 125 GB RAM, Linux, Qwen IQ3_S: 8K prompt 2,047 -> 2,505 tok/s (+22 %), 32K 2,491 -> 2,802 (+12 %); Strix Halo (Ryzen AI Max+ 395, 128 GB, Windows) Qwen IQ3_S 4K **755**, 32K **792** tok/s; RTX PRO 5000 (72 GB) IQ3_S prompts +1.2 % at 4K, +1.5 % 8K, +1.6 % 16K;
   two RTX A4000 layer-major: first word after a 32K prompt 27.7 s instead of 141.8 s, 20K 17.5 vs 98.8 s, 8K 15.2 vs 41.8 s (5 paired runs, IQ3_XXS, most experts streamed from RAM); `--ple-io ram` on the 72 GB card prompts +16.5 % at 1K, +30.6 % 4K, +16.8 % 8K; `--prefill-pipe` on two A4000 +15 % at 4K, +23 % at 8K.
3. **`bench/results/2026-10-20-0.2.1-bench/README.md`** (engine 0.1.46, `strata bench run`, IQ3_XXS, int8 KV, MTP spec 4): P2003 = RTX 3060 12 GB + Core Ultra 7 265: `prompt-4k` **962** tok/s (TTFT 4.28 s), `prompt-16k` **1,103** (14.9 s); z440 = 2x RTX A4000 + Xeon E5-2643 v3 layer split: **1,158** (3.56 s) / **1,852** (8.9 s). Runs: `p2003-rtx3060/`, `z440-2xa4000/` (run1-3 each).
4. **Community table `bench/results/COMMUNITY.md`** (submitter medians, 4K / 32K / 128K unless stated): RTX 5090 + 9950X3D IQ3_S 262K **2,473 / 4,168 / 4,450** (0.1.33; folder `2026-10-02-community-rtx-5090-iq3s`); same box dev run IQ3_S 4K 4,308 (0.1.34) / 4,846 (0.1.38) / **5,361** (`STRATA_PF_FUSED=1`) (`2026-10-03-dev-5090`);
   2x RTX PRO 4500 Swift IQ3_XXS **2,960 / 5,065 / 5,762** (0.1.36); RTX 4090 + 7950X IQ3_XXS 200K 1,966 / 2,974 / 3,085 (0.1.39); RTX 2080 Ti 497 / 637 / 566 (0.1.33); RX 9070 XT ~600; R9700 Windows IQ2_XS 856 / 1,182 / 1,232 (0.1.38).
5. **`docs/QUANTS.md`** (R9700 Linux, EPYC 7452, HIP, engine 0601072d): 32K prompt IQ3_S 1,555; UD-Q4_K_XL 1,110-1,320; UD-Q5_K_XL 810-1,260; UD-Q6_K_XL 1,020; Q8_0 1,230-1,300 (run-to-run up to 2x).
6. **KVA** (`docs/KVA.md`, opt-in): Tesla P100 IQ3_XXS 14K 323.8 -> 363.5 tok/s (+12 %), 30K 336.1 -> 401.9 (+20 %) at split 36; split 24 chunk 1024: 5.6K 203 -> 266, 14.6K 202.5 -> 311, 30K 199.7 -> 338; R9700 IQ3_S 27K needle 2,411 -> 3,086 (+28 %); RTX 5070 IQ3_S +11 % at 14K, +18 % at 30K.
7. **`--prefill-head-alias`** (`docs/PREFILL_HEAD_ALIAS.md`): RTX PRO 2000 Blackwell Laptop 8 GB, Windows 11, Core Ultra 9 285H, 64 GB, IQ3_XXS: 29,854-token prompt: chunk 256 185.5 tok/s ... chunk 2048 (alias) 667.1; production agent loop median prefill 256 -> 583 tok/s (+125 % within context bands); short reads +0.6-0.75 s. Port check 0.1.43.1: cold 29,842-token prompt 276.5/277.1 -> 643.1/649.7 tok/s.
8. **Message-boundary cache** (`docs/MESSAGE_BOUNDARY_CACHE.md`): two modified RTX 3080 20 GB, Xeon E5-2686 v4, IQ3_S 131K ctx: first 100-record edit 17.236 -> 11.527 s; marked as combined policy + chunk-size effect, "not a measured speedup of this pure-upstream port".
9. **Other** in `docs/DETAILS.md` opt-in tables: `STRATA_PREFILL_ROUTED_STREAM` (RTX 3090 IQ3_XXS: 23 % fewer experts streamed for 2K, a 1,790-token follow-up 18 % less time); `STRATA_SELECT_TWO_PASS` (RTX 3090: 256K prompt 1,887 -> 1,984 tok/s); `STRATA_GDN_TC` (CMP 170HX: prompts +3.9 % 4K ... +6.7 % 52K); chunk raise on NVIDIA (A6000 UD-Q4_K_XL: 20K +43.4 %, 32K +22.5 %); fused kernels on GB10 1.12-1.15x (`nv_defaults.cpp`).
   The `README.md` prompt table (RTX 5070 ~110 tok/s etc.) is for **DeepSeek-V4.1-Flash**, not Qwen3.8-Flash-Next.

---------------------------------------------------------------------------------------------------------------------------

## i. What changed between v0.2.11 and main in prompt-related code (from `diff -r`)

`src/prefill` (files that differ: gdn_chunk.cu, gdn_tc.cuh, gemm.cu, kernels.cu, moe_f16_common.cuh, moe_f16_kq.cu/.hpp, moe_fused.cu, moe_fused_iq.cu, moe_mmq.cu, prefill.cpp; new: `moe_iq4_w4a8.cu`, `mmq_ids_margin.hpp`, `moe_iq4_w4a8.hpp`, `timer_marks.hpp`):

1. **Geometry made a compile-time struct, no behaviour change for Flash-Next**: every literal (2560, 640, 1280, 10, 512, 12288, ...) in prefill.cpp, kernels.cu, moe_*.cu, gdn_chunk.cu, qsa_prompt_attn.cu, fused_gr.cu, native_ple_postops.cu replaced by `kCompiledGeometry` constants with `static_assert`s (a "shape engine" for other model shapes; `--geometry`). The Strata Q2 blob layout is now written in terms of E/F.
2. **Bug fix: MMQ row-table over-read** (PR #5108): the row table `ids_identity` is now allocated with a margin (`mmq_ids_table_rows`, `mmq_ids_margin.hpp`; `roundup(max_rows, J) + J`) because llama.cpp's stream-k fixup reads a whole J-column tile past the end (Xid 31 / "prefill copy_i32: an illegal memory access" on a long prompt). Applied in `carve`, `bytes_needed_impl`, `bind_stage_helper`, `lm_run`, `draft_kv` (`8 * hc` bytes per row), `gemm.cu` native MMQ path. The iota now writes the whole padded table.
3. **New opt-in kernels/switches**: `STRATA_IQ4_W4A8` (IQ4_XS/IQ4_NL int8 pipelined kernel, MMQ-identical bits, `moe_iq4_w4a8.cu` 495 lines, hooked in `moe_mmq.cu`'s `Context::run` and `launch_splits_k`); `STRATA_SELECT_TWO_PASS` (+749 lines in `qsa_select.cu`, `qsa_two_pass*.hpp`, parity test); `STRATA_PF_PEER_CPU`; `STRATA_PF_DM16` + `moe_combine16` (gfx11/gfx1151 FP16 down rows; defaults in `arch_defaults.cpp`); Q5_0 down in the native fused/F16 kernels (UD-IQ4_XS-v31: 33 of 48 layers; gfx1151 Windows default, gfx12 opt-in).
4. **Diagnostics**: `STRATA_PF_ALLOC_TRACE`, `STRATA_MANIFEST` path report of which expert path each layer took (`run_manifest.hpp`), `PfTimer` no longer records a second consecutive mark in the same phase (`timer_marks.hpp`, PR #4006, big under WDDM), `WatchedStream` so the stall report can ask whether the prompt's stream is finished (#5718 follow-up).
5. **GDN dispatch**: `gdn_chunked_auto` (gfx12: chunked scan from 30,000-token chunks; `STRATA_GDN_CHUNKED_MIN`); `gdn_tc.cuh` scratch assert only for Flash-Next shape; main `generate.cpp` sets **`STRATA_GDN_TC=1`, `STRATA_GDN_CONVL2=2` as engine defaults on Linux CUDA** and `STRATA_GDN_CHUNKED=1` on Linux when all cards are gfx12 (owner yes 2026-11-07; before these were a launcher's env). 0.2.11 had no such engine default.
6. **hipBLASLt table switch** (gemm.cu, HIP): `STRATA_HIPBLASLT_TABLE` auto/on/off, only gfx1151 takes the packaged table by default.
7. `generate.cpp` (prompt-relevant, 654 changed lines): `STRATA_HC_Q8` default on for CUDA when the GGUF carries Q8_0 HC projections; PLE hash constants read from the model file (`ple.layer_multipliers/head_vocab_sizes/head_offsets`) and checked against the table; file-tier defaults (io prefetch, look-ahead) switch off while every expert is in VRAM and back on when not; automatic RAM copy honours `STRATA_RESIDENT_PIN`; `--draft-cap` auto per card; stall report asks the prompt stream; AMD GPU pause counter. `ple_reader.cpp`: sort-and-merge stack path for <= 64-row batches (decode/prefetch), hash path unchanged for prompts. `conversation_heap_test.cpp` new.
8. Docs: DETAILS.md small edits (file-tier idle, TWO_PASS paragraph, W4A8 row); `SETTINGS.md` + `data/settings-register.json` generated; `CAPABILITIES.md`, `work/README.md` added.
9. **Not changed since 0.2.11**: the main prompt loop structure, lend/refill, KVA, layer-major rule, share rules, `share_rules.hpp`, `moe_mmq.hpp` API apart from the margin, `qsa_prompt_attn` algorithm.

---------------------------------------------------------------------------------------------------------------------------

## j. Licence headers of files read

* Repo licence: `LICENSE` = MIT, "Copyright (c) 2026 Niko1221 and the Strata contributors".
* **No licence header** (project files under the repo MIT): `src/prefill/prefill.cpp`, `kernels.cu`, `gemm.cu`, `gdn_chunk.cu`, `moe_fused.cu`, `moe_fused_iq.cu`, `moe_iq4_w4a8.cu`, `include/strata/prefill/*.hpp`, `src/program/generate.cpp`, `src/ngram/ple_reader.cpp`, `src/kernels/ngram.cpp`, `src/kernels/cuda/qsa_prompt_attn.cu`, `qsa_select.cu`, `fused_gr.cu`.
* **Differing / third-party MIT notices embedded**:
  * `src/prefill/moe_f16_common.cuh` and `src/prefill/moe_f16_kq.cu`: full MIT text, **"Copyright (c) 2026 gufo contributors"**, "Adapted from gufo 97b8645f" (RoutedF16GEMMKernel, ROCm/HIP gfx11 kernels).
  * `src/prefill/moe_mmq_tab.cuh`: full MIT text, **"Copyright (c) 2023-2026 The ggml authors"** (mirrors llama.cpp `mmq.cuh` at pin 3cf03257, with 16 weight pointers).
  * `src/prefill/moe_mmq.cu`, `include/strata/prefill/moe_mmq.hpp`: "llama.cpp's MMQ (ggml-cuda, MIT)" referenced; compiled against `third_party/ggml` (licence at `third_party/ggml/LICENSE`).
  * `moe_iq4_w4a8.cu`: "get_int_from_table_16 (llama.cpp, MIT)"; `moe_fused_iq.cu:1288`: barrier "idea and wording of libstrix (asf0/libstrix, MIT)"; `kernels.cu:2471`: idea from halo-box/strix-llama.cpp #213 ("MIT, The ggml authors"; kernel is Strata's).
  * `src/kernels/cuda/iq_kernels.cu`, `dequant_bf16.cu`, `native_*.cu`: carry MIT notices (llama.cpp transcriptions; `docs/DETAILS.md:1400-1404`).
  * Several kernels are credited in comments to contributors (PR numbers, gufo, eddoursul, Vop3p #4006, quantforge-br #5108) without separate licence text.
* All of the above are MIT-compatible; none of the files read is GPL/Apache. (`data/dflash2-tuned/` is Apache-2.0 but not part of the prompt path.)

---------------------------------------------------------------------------------------------------------------------------

## Appendix: key line index

* Loop / layers: pf:3314 `run_impl`, 3479 chunk loop, 4052 layer loop, 4134 half loop, 4230 GDN, 4281 QSA, 4636 MoE, 6181 HC write, 6362 `on_chunk`.
* Expert stream: pf:3713 `stream_all`, 3750-3809 plan, 3914-4028 issuer, 4787-4867 fused path, 4869-5999 MMQ/host-group path, 6000-6028 FP16 path, 6044-6111 walks, 6113-6153 combine.
* Stager: pf:712-905; ring rules pf:229-468; CPU share pf:278-331, 2981-3009.
* Buffers: pf:1127-1312 `Impl`, 1445-1735 sizing, 2047-2202 `carve`, 2891-2953 `bytes_needed_impl`.
* Layer-major: pf:6535-7900; dispatcher pf:7941-8028.
* generate.cpp: options 810-966; `--prefill` parse 3580-3601; NVIDIA chunk raise 3947-3972; GDN defaults 3909-3914; PLE policy 5462-5580; chunk plan 9829-10028; small-card rule 10922-10968; init paths 11259-11391; read windows 17002; refill 17085-17148; lend 17153-17237; read_part 17270; cuts and loop 18034-18140; checkpoints 12654-12743, 2227-2300.
* PLE: pf:3427-3459, 3610-3669, 4068-4126; `src/kernels/ngram.cpp:649-720`; `src/ngram/ple_reader.cpp:431-545`; `include/strata/ngram/ple_reader.hpp`.
* KVA: pf:3326-3354, 3499-3503, 3767-3775, 4059-4067, 4248-4280, 4404-4411, 6480-6517; `docs/KVA.md`.
