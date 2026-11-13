# Ideas from Strata and SlenderEngine that could help our engine

One entry per idea. These are their ideas, to be built in our own way inside our engine - their ideas, not their code. An idea
is worth building when it fits our design and is implemented properly in ours, with our checks. Groups: **(a)** fits our engine
as it is, **(b)** fits with an adjustment to our engine, **(c)** we already have it, **(d)** not for us, or later. Every entry has the field **answers**: *same bits* / *rounding only* (GPU vs CPU,
batch shape, summation order: accepted) / *changed*. **A part with "changed" is never a checklist item**; it sits in group (d) with
"possible later as an opt-in mode".

Fields: what it does; their gain and where it was measured (their numbers, not reproduced); which of our parts it touches; class by
`docs/fit-with-our-core.md` (A helps / B handling needed / C blocks); hardware; licence; the targeted check. Sources: `others-today.md`
and `reading-notes/` (`S:` = Strata main `13eedf55`, `SE:` = SlenderEngine `8af6fd84`). Both repos are MIT. We write our own
implementation; if a piece of their code is ever used directly, its MIT notice stays in the file header ("The ggml authors",
"gufo contributors").

**Prompt-processing parts first**, in the most detail. Before building any prompt idea, one profile of a 32k prompt on our engine
(item Q-P0 in the README) says where our time goes; on Ling a 32k prompt took 34.5 s of which only 1.27 s was SSD wait
(`reports/C12-s4-end.md:144`, this laptop, 175 slots), so our prompt is bound by GPU work and copies, not by the SSD.

## Prompt processing

### P1. Fused hyper-connection kernels (`gr_upmix`, `gr_write_norm_rs`) - group (b)
- **What:** `gr_upmix` runs the rank-320 up projection and the mix (sigmoid gate x row scale x weight, mean of the 4 streams) in one
  kernel, so the T x 10,240 FP32 gate tensor is never written to memory (336 MB at 8,192 tokens); `gr_write_norm_rs` adds the block
  output into the 4 streams and computes the next half's RMSNorm in the same pass (`SE:src/prefill/kernels.cu:174-206, 2618-2862`).
- **Their gain:** 50K prompt 2,183 / 2,182 -> 2,258 / 2,296 tok/s (+4.3 %), SlenderEngine's PC (RTX 4070 Ti), logits byte-identical
  there (`SE:docs/fork/PROMPT-ROOM.md:136-141`).
- **Ours today:** `build_hc_mix` = rms_norm, mul, two or three mat-muls, silu, then `ggml_dsv4_hc_pre_gated` (gate + mean fused) when
  `cparams.fused_dsv4_hc_pre` is set; `build_hc_combine` uses `ggml_dsv4_hc_post` (`src/models/qwen4exp.cpp:348-445`). So the gate
  tensor is written and read; the norm is a separate pass.
- **Adjustment:** a ggml CUDA op (or a graph fusion) "up mat-mul + gated mean" and "combine + next norm", used by the qwen4exp graph in
  prompts and decode; the decode versions are Strata's `fused_gr_down` / `fused_gr_up` (`S:src/kernels/cuda/fused_gr.cu`).
- **Touches:** the VRAM side only (dense compute). **Class A.** **Answers:** rounding only (a different accumulation order than
  cuBLAS; same bits only if the kernel reproduces it). **Hardware:** NVIDIA sm_80+ (tensor-core MMA); a portable fallback stays.
- **Check:** profile of one 8k call (time of the HC nodes) before / after; KLD inside plain llama.cpp's own spread; the decode
  ms per token.

### P2. The prompt room: lend idle VRAM blocks to a long prompt - group (b), measure first
- **What:** during a prompt of >= 4,096 new tokens the output head (497 MiB on their file; 0.49 GiB Q6_K on Grawnch's) and a
  drafter's weights are idle; the prompt's buffers continue into them at the same addresses, and they are rewritten before the first
  token (the head re-read from the model file, ~130-150 ms; a checksum option).
- **Their gain:** 50K +10 % (defaults) to +19.5 % (MTP, 192-slot ring); the gain comes from **one pass fewer** over the experts (chunk
  8,448 -> 10,240) (`SE:docs/fork/PROMPT-ROOM.md:11-19, 33-36`).
- **For us:** our layer-major already reads each expert once per call (cap 32k at 65k context), so the pass saving is smaller; the
  0.5 GB could instead hold more VRAM expert places or a larger call during the prompt. We have no drafter loaded, so only the head.
- **Adjustment:** with U48's lending machinery (`--moe-stream-vram-lend-prompt`, `docs/flags.md:333`): lend the output head's buffer
  as prompt space, reload it from the model file with a direct read at the prompt's end (0.5 GB at 5.5 GB/s: ~0.1 s, sum), verify a
  checksum.
- **Touches:** VRAM layout during prompts (check 3 of the fit rules: the first reply token waits for the reload). **Class B** - handling:
  reload issued before the last layer group, the first token waits on it. **Answers:** same bits (only where weights live); rounding only
  if the call size changes. **Hardware:** any GPU.
- **Check:** head checksum after every reload; prompt time at 32k / 50k; the first token's time.

### P3. Parallel refill of the VRAM places a prompt borrowed - group (c) to check, maybe (b)
- **What:** after a prompt, the borrowed expert-cache slots are refilled by N threads with their own pinned buffers and streams.
- **Their gain:** after a 50K prompt 1,048-1,059 ms -> 405-414 ms; after short prompts 190 -> 70 ms (`SE:docs/fork/PERFORMANCE.md:36`;
  opt-in `STRATA_REFILL_PINNED=4`, their launcher).
- **Ours:** the VRAM pool's tail is refilled at the prompt's end (`--moe-stream-vram-end-fill`, `docs/flags.md:385`) and copy-ahead /
  copy-back exist (`:321`, `:138`). How long our refill takes on this model is not known.
- **Check (first):** the refill time in our start-of-reply lines after a 32k prompt; build the parallel form only if it is long.
  **Answers:** same bits. **Class A.** Any GPU.

### P4. Stream every non-held expert through a VRAM ring, overlapped across layers - group (c)
- **Theirs:** for a chunk of >= 1,024 tokens every expert not in VRAM is copied once per layer per chunk, in id order, routed or not,
  through a ring of 96 / 384 / 1,024 whole-expert slots; an issuer thread keeps copies a ring ahead across layer boundaries
  (`SE:src/prefill/prefill.cpp:3712-3767`, `:430-438`, checked). Their prompt is **one pass per chunk**: 5-7 passes for 50K.
- **Ours:** layer-major (`docs/flags.md:98-99`) reads each expert once per call, not per chunk (2 passes for 50K at a 32k cap), with
  read-ahead, the window over the RAM pools, the wave plan and GPU prompt waves (`:394`, `:402-405`, `:407`). Ours moves fewer bytes;
  whether our copies overlap the GPU work as well as their issuer thread does is what Q-P0's profile must show.
- **Answers:** same bits (ours kept Ling's text in its first step). Nothing to build unless the profile shows idle GPU time between
  waves.

### P5. In-place stager (direct read straight into the pinned buffer) - group (c)
- **Theirs:** +8.5 % at 50K, +13.9 % at 3K, CPU 87 % -> 10 % busy (`SE:docs/fork/PERFORMANCE.md:69`).
- **Ours:** every read already lands directly in its RAM pool slot (`moe_stream_read_range`, `src/llama-moe-stream.cpp:4233-4235`);
  there is no staging copy. Their remaining advantage is a page-locked RAM tier (the GPU copies straight from it); ours cannot be
  page-locked on this laptop (`cudaHostRegister` fails, AGENTS.md T58), so our RAM -> VRAM copies go through the driver's own staging.

### P6. Dense arena - group (c)
- llama.cpp already allocates a backend's weights in one buffer per buffer type; there is no per-matrix `cudaMalloc` to save.

### P7. Bigger prompt chunk (16,384, 32,768) - group (c)
- Ours: the layer-major call cap `--moe-stream-layer-major-batch` (`docs/flags.md:99`); its value is the autotuner's.

### P8. Sparse-attention prompt kernels (INT8 tensor-core attention, 3xTF32 block scores, radix top-k) - group (b), after the profile
- **Theirs:** one CTA per (query, KV head) over the 2,048 selected cells, online softmax, INT8 tensor cores on int8 K/V (default on
  cc >= 8.0); block scores on 3xTF32 tensor cores; a 4-pass radix select (`reading-notes/slender_prefill.md` f). A separate gain for
  each kernel was not found in what was read.
- **Ours:** the indexer "scores with plain ops" because the CUDA lightning indexer takes 32 or 64 heads and this model has 4
  (`src/models/qwen4exp.cpp:480-484`); selection and attention are built from ggml ops.
- **Touches:** VRAM side only. **Class A.** **Answers:** rounding only (3xTF32 is not bitwise; int8 attention reads the same stored
  int8 values). **Hardware:** NVIDIA cc >= 8.0 for the tensor-core forms; others keep the ggml path.
- **Check:** profile share of the attention / indexer nodes at 32k and 128k; KLD inside the spread; md5 of decode unchanged.

### P9. GDN tensor-core chunked scan - group (b), after the profile
- **Theirs:** Strata main sets `STRATA_GDN_TC=1` and a one-pass conv on Linux CUDA: prompt +4.4 % at 4K, +5.5 % at 16K, +5.7 % at 32K;
  KL mean 0.0019 (`reading-notes/strata_prefill.md` d.2).
- **Ours:** upstream's recurrent op (`build_recurrent_attn`, `src/models/qwen4exp.cpp:1232`); its kernel form and speed here are not
  known. **Answers:** rounding only. **Class A.** NVIDIA cc >= 8. **Check:** profile of the GDN nodes in one 8k call; KLD.

### P10. n-gram rows read by the engine, one call ahead - group (b) (= `port-needs.md` C3)
- **Theirs:** one request for all rows of a chunk: grouped by 4 KiB page, sorted, 256 in flight, a 2^20-row cache, the next chunk's rows
  fetched during the current chunk; in decode the rows of each window or draft token are fetched as soon as the token is known.
  The n-gram block of an 8K prompt: 845 ms direct reads / 451 mmap / 132 RAM (Strata, R9700 + IQ3_S).
- **Adjustment:** our OS-layer reads and I/O threads, a bounded row cache, consumed at layer 1; decode one token ahead.
- **Touches:** the SSD queue (check 1-3: tiny reads that must not wait behind expert reads). **Class A**, with the B-style handling
  "these reads go first". **Answers:** same bits. **Hardware:** any (Linux pread / AIO, Windows overlapped reads).
- **Check:** the gather's time per token and per 8k call; md5 unchanged.

### P11. Keep the n-gram table in RAM when there is room - group (c)
- The page cache does this by itself on a machine with spare RAM; an explicit "lock it" option is not useful at 30 GB. Note for the
  autotuner: on a 64 GB+ machine the table can be resident.

### P12. CPU share of experts in small prompt chunks - group (b), measure first
- **Theirs:** in a chunk below 1,024 (SlenderEngine) / 3,072 tokens (Strata), the idle CPU computes the non-held experts few tokens route
  to while the GPU computes the rest: 512-token prompts -28..-34 %, 1,000 tokens -20..-26 % (`S:docs/DETAILS.md:1518`).
- **Ours:** below `--moe-stream-gpu-waves` (512 tokens) the whole call's experts run on the CPU; above it all on the GPU waves. A
  split inside one call does not exist.
- **Touches:** which experts are copied to the GPU per call (check 1). **Class B** - handling: the split decided from the pool
  contents (experts already in RAM and used by few rows go to the CPU). **Answers:** rounding only (the same experts, GPU vs CPU
  arithmetic; their KL 0.004-0.011). **Hardware:** any GPU + CPU.
- **Check:** prompt time at 300 / 600 / 1,000 / 2,000 tokens; KLD.

### P13. Short prompt parts through the decode path - group (c)
- Ours: llama-server's short calls and the prompt's end already take the decode path when every token is an output (AGENTS.md project
  state, 2026-11-01).

### P14. Conversation checkpoints, parking whole chats on the SSD - group (c)
- Ours: `--ctx-checkpoints` (`docs/flags.md:341`), in-call checkpoints, `--cache-disk` (`:121-124`). For this model the indexer cache
  and the n-gram history must be part of a saved state (`port-needs.md` D2).

### P15. SSD keep-alive reads - group (a)
- **What:** one random table page every 100 ms while idle, for up to 60 s, because some SSDs enter a power state after ~250 ms and
  stall the next reads 50-150 ms (WD_BLACK SN7100 on Windows, Strata's note, `reading-notes/strata_main.md` 3.2).
- **Touches:** the OS layer (a read loop when the engine is idle). **Class A.** **Answers:** same bits. **Hardware:** any, mainly
  laptops and Windows. Licence MIT.
- **Check:** time of the first read after a 2 s / 10 s pause, with and without (our drive is a WD SN5000S).

## Decode and memory

### D1. Lend unused K/V memory to the experts - group (b)
- **Theirs:** (1) SlenderEngine `STRATA_KV_LEND`: the page-locked **host** copy of the K/V is sized for the whole context; cells a request
  cannot reach are lent to the RAM expert tier and taken back at the next request: SSD misses -27 %, short answers 37.9 -> 42.2 tok/s
  (`SE:docs/fork/PERFORMANCE.md:67`). (2) Strata `--kv-grow`: the VRAM expert cache gives 2 MiB pieces to the K/V only as the context
  reaches them: R9700 at 196,608 context 6,213 -> 7,968 cache slots, decode +2.3 % (`S:docs/DETAILS.md:1443`).
- **For us:** we hold no host K/V copy, so the RAM form does not apply; the **VRAM form** does: at 262k the q8_0 K/V takes 3.98 GiB
  (sum), most of it unreached in a normal chat; those cells could hold VRAM expert places until the context reaches them.
- **Touches:** the VRAM expert pool's size changes during a session (places given back when the context grows) - **class B**;
  handling: the pool's give-back / take-back logic (U48, X1) decides which places go. **Answers:** same bits. **Hardware:** NVIDIA
  first (CUDA virtual memory), AMD later.
- **Check:** VRAM expert places at 4k / 32k / 128k of context used in a 262k session; ms per token; no failed allocation when the
  context grows.

### D2. K/V in RAM, only part of it in VRAM - group (d) later, or (b)
- **Theirs:** Strata from 64K context: the whole K/V in page-locked RAM, 32,768 cells per attention layer in VRAM; +6 % at 128K, +23 % at
  262K (RTX 5070, Q2_0, `S:docs/DETAILS.md:236`).
- **For us:** frees VRAM for experts at long context, but adds PCIe traffic that competes with expert copies in prompts and needs
  page-locked RAM (which fails on this laptop). **Class B.** **Answers:** same bits. Large; after D1.

### D3. Read-through RAM tier (a miss lands in a RAM place) - group (c)
- Ours: every read lands in a RAM pool slot and eviction is by decayed use (`--moe-stream-policy decay`, `docs/flags.md:52`); plus
  guesses, the waiting list and the prompt rules. Theirs (+5.2..+13.3 %, `SE:FORK.md:118-125`) is the same idea added late to a
  profile-ranked tier.

### D4. Adaptive VRAM expert cache - group (c)
- Ours: X1 `--moe-stream-vram-follow` (every 8 words up to 64 swaps, `docs/flags.md:125-133`); theirs every 4 windows up to 96 (Strata) /
  24 (SlenderEngine) swaps with decay 0.7. Same idea; values are the autotuner's.

### D5. Ranked fill at start, read-back check - group (c)
- Ours: the VRAM list file (`--moe-stream-vram-experts-file`), the usage seed + `--moe-stream-idle-load`, `--moe-stream-verify`.

### D6. Reads handed over as they land; compute ready experts while others read - group (c)
- Ours: per-slot ready flags, `--moe-stream-io-wake chunk`, `--moe-stream-proj-ready` (`docs/flags.md:305`), `--moe-stream-overlap`
  (`:19`). Theirs: `STRATA_DIRECT_BATCH=1` +9 % after 50K, kept reader threads.

### D7. Router-trick prefetch and read-ahead - group (c)
- Theirs: layer l+1's router on layer l's input, top 10, 1-2 layers ahead (Strata); SlenderEngine's read-ahead lost at K=10 and gained
  2.7 % at K=5. Ours: the 1-ahead guess, lookahead 2, the waiting list, guess-inflight, the correction file - more developed. For this
  model the guess input itself is new work (`port-needs.md` B1).

### D8. CPU expert kernels for Q2_0 (AVX-512 VNNI + VBMI, AVX2 bitplane) - group (b)
- **Theirs:** Strata's own Q2_0 kernels for AVX-512 / AVX2 / AVX-VNNI / AVX1 (`S:include/strata/kernels/cpu/expert_layout.hpp:91`, `S:src/kernels/cpu/q2_avx2.cpp`, `q2_avx1.cpp`,
  `reading-notes/strata_main.md` 4.2). No separate gain number found.
- **For us:** a reference for the Q2_0 part of the 4-row AVX-512 kernels (`port-needs.md` E2); Q3_K and Q4_0 have no Strata kernel
  (Strata does not run Q3_K experts at all).
- **Touches:** CPU compute only. **Class A.** **Answers:** same bits if the kernel matches ggml's `vec_dot_q2_0_q8_0` order (not checked
  for Strata's kernel; otherwise rounding only and a new md5 reference). **Hardware:** x86 AVX-512 / AVX2. MIT.
- **Check:** `kernel_bench --check` against ggml's one-row kernel; CPU ms per token.

### D9. CPU pool: a layer's down rows start when that expert's gate / up finish - group (b), with D8
- **Theirs:** `STRATA_POOL_FUSED`: their laptop +5.6 % decode, CPU time per expert -13 %; their desktop -0.2 % (`SE:docs/fork/WORKLOG.md:1183-1197`,
  `LAPTOP-RTX4060.md:35`). Same bits (bitwise test).
- **Ours:** `--moe-stream-cpu-fuse` / `-cpu-x4` do this for MXFP4 one-token calls only (`docs/flags.md:68-70`); extend with the new kernels.
  **Class A.** **Answers:** same bits.

### D10. Fused decode hyper-connection kernels (`fused_gr_down` / `fused_gr_up`) - group (b), with P1
- **Theirs:** previous write + 4-stream RMSNorm + down projection + SiLU + inject in one kernel, up + mix in a second; one launch pair
  for all tokens of a window (`reading-notes/strata_main.md` section 1). No separate gain number found.
- **Ours:** 96 hyper-connection mixes per token built from separate ggml ops (`qwen4exp.cpp:348-445`). **Class A.** **Answers:**
  rounding only. NVIDIA. **Check:** decode profile (launches and time per HC mix).

### D11. Hyper-connection weights in Q8_0 - group (c)
- Grawnch's file stores them in Q8_0 and llama.cpp uses them as stored; Strata's default converts to BF16. Nothing to do.

## Not for us, or later - group (d)

| part | answers | why |
|---|---|---|
| Route tail skip (`STRATA_ROUTE_TAIL_SKIP=7`, Strata default) | **changed** | possible later as an opt-in mode |
| Tail skip for SSD reads only (`STRATA_ROUTE_TAIL_SKIP_FILE`) | **changed** | possible later as an opt-in mode |
| Route toward held experts (`STRATA_ROUTE_PRIOR`, `STRATA_ROUTE_RESIDENT`) | **changed** | possible later as an opt-in mode (Strata withdrew ROUTE_PRIOR after repetition loops) |
| KVA "faster long prompts" (skip late layers' attention and experts) | **changed** | possible later as an opt-in mode |
| Experimental speed projection (control vector) | **changed** | not a speed method for us |
| PCIe share of decode misses (GPU computes some RAM experts) | rounding only | on this laptop the CPU computes an expert in ~0.06 ms, a PCIe copy takes 0.218 ms (`docs/fit-with-our-core.md` section 4); SlenderEngine itself runs faster with it off on its PC. A row for `docs/other-machines.md` (weak CPU, strong link) |
| One CUDA graph per decode window with mapped-memory doorbells | same bits | needs page-locked shared memory that fails on this laptop (check 6); our split graphs stay. Revisit if the OS layer gets page-locked pools on other machines |
| Second-SSD mirror (Windows) | same bits | one SSD only (user); later |
| `--vram-elastic` (give VRAM back to other programs) | rounding only | an app feature; later |
| Drafting (MTP head, DFlash2, prompt lookup) | not an answer change (each draft checked) | later (`others-today.md` section 8). SlenderEngine's DFlash2 drafter is CC BY-NC-ND: never for us |
| Grouped MMQ gather (16 experts per group buffer), FP16 dequant + cuBLAS dense path | rounding only | ggml's `mul_mat_id` MMQ already handles all experts of a call in one launch; revisit only if the profile shows launch-bound expert GEMMs |
| sm_89 kernel table, AMD/Intel kernel tables, IQ3 VBMI decode | - | their own kernels for formats we do not run; the VBMI byte-permute idea may help Q3_K unpacking on Zen 5 (idea only) |
| Low-RAM automatic fallback, file-tier timing probe | same bits | ours sizes pools from free RAM already; the probe matters only for drives slower than direct reads (other machines) |
