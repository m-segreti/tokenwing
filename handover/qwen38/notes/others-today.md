# How Strata and SlenderEngine run Qwen3.8-Flash-Next today

Read (2026-11-13), MIT code, clones in `traces/refs/` (git-ignored):

- **Strata** `main` at `13eedf552417a6a791a7d13c6597e24300ee3a09` (2026-11-12, after tag v0.2.13.1 `628f749`; its AGENTS.md says
  "Version 0.2.13, engine 0.1.59"). Paths below starting `S:` are in `traces/refs/strata/`.
- **Strata v0.2.11** `c3b3f5035506f3e966f1b4b7b188aeaabae92f1d` (SlenderEngine's base), `traces/refs/strata-0.2.11/`.
- **SlenderEngine** `speed-fork` at `8af6fd844fc6737f25d21852e62591563dd14cec` (2026-11-11), `traces/refs/slender/` (a snapshot of that
  commit; the git clone stalled). Paths starting `SE:` are in it. `diff -r` against v0.2.11's `src/` shows 15 changed source files.

Bulk reading was done by four sub-readers; their notes, with `path:line` on every claim, are in `reading-notes/`. The key lines
used below were opened and checked by me; they are marked **(checked)**. Every number is the other project's own measurement on
the machine named; none was reproduced here.

## 1. In one paragraph each

**Strata** is its own engine (not llama.cpp) for this one model family (plus DeepSeek V4.1 and Qwen3.6). The design is "every expert
in RAM": the GPU holds the dense part, the KV cache and a cache of the most-used experts; RAM holds **all** 24,576 experts in one
page-locked arena; the CPU computes the experts the GPU does not hold, in place, at the same time as the GPU; the SSD holds only the
28.8 GB n-gram table (`S:docs/HOW_IT_WORKS.md`). When RAM is too small it now falls back by itself to "the hottest experts by a
fixed profile in RAM, the rest read from the file", through the page cache or direct reads (`reading-notes/strata_main.md` 2.2,
`S:src/program/generate.cpp:7781-7835`). A decode step is one captured CUDA graph per window size; the CPU is signalled through
mapped memory, with no per-layer host sync.

**SlenderEngine** is a speed fork of Strata 0.2.11 for one PC: Windows 11, NVIDIA, the IQ3_S pack (`experts.bin` 46.8 GB), 32 GB RAM,
12 GB card, two NVMe drives. It turns Strata's low-RAM mode into a real three-tier engine - ~2,000 experts in VRAM, a 10 GiB RAM
tier (5,272 experts), the rest read from the SSD every time - and adds read-through caching, lending idle memory to experts, a
second-SSD mirror and a faster prompt path. **Almost all of its own defaults switch on only on Windows with `--pack`**: the
`#if defined(_WIN32)` block of `apply_fork_defaults` (`SE:src/program/generate.cpp:3362-3409`, checked). On Linux it sets only three
values: skip 0, CPU-share limit 1,024, the small-card rule off (`:3359-3361`, checked).

## 2. Decode

- **Strata:** per layer: hyper-connection read (`fused_gr_down` / `fused_gr_up`), the mixer (GDN or sparse attention), the FFN
  hyper-connection read, the router (BF16 GEMV + top-10 kernel: softmax over all 512, renormalised, no bias), then the routed experts
  split three ways - GPU-cached experts on the GPU, a "PCIe share" of missed experts copied to the GPU from pinned RAM, the rest on the
  CPU pool - and one combine kernel (`reading-notes/strata_main.md` section 1, `S:src/core/verify.cpp:1476-2476`). The shared
  expert runs on a side stream. The layer's FFN write is folded into the next layer's read.
- **SlenderEngine:** the same kernels; its changes are in the memory tiers and threads: a missed expert's SSD read lands straight in a
  RAM-tier place (read-through, `STRATA_RAM_THROUGH`), kept reader threads, reads handed over one by one as they land
  (`STRATA_DIRECT_BATCH=1`), an sm_89 table of seven existing kernel switches, the K/V lend (section 4). On their PC 55-60 % of a token's
  experts came from VRAM, 25-30 % from RAM, 7-11 % from the SSD (`SE:docs/fork/WORKLOG.md:610, 703`, checked; measured while expert
  skipping was still on).

## 3. Prompt path (studied in the most detail; the user's main interest)

Both engines share Strata's prompt code; SlenderEngine changes the buffers and the reads around it
(`reading-notes/slender_prefill.md`, `reading-notes/strata_prefill.md`).

1. **Chunks, not layer-major.** A prompt is read in chunks; each chunk goes through all 48 layers before the next. The chunk is the
   largest whose buffers fit in VRAM: Strata up to 8,192 tokens (32,768 on NVIDIA cc >= 8.0), SlenderEngine 16,384 with a RAM
   budget on a 28 GB+ PC, and up to 10,240 when the "prompt room" applies. **Each chunk is one pass over all experts the GPU does
   not hold** - a 50K prompt is 6 passes at 8,448 tokens, 5 at 10,240 (`SE:docs/fork/PROMPT-ROOM.md:33-36`). Their own measurement:
   "each pass takes about 4 seconds here, most of it reading". A layer-major mode exists in Strata (off by default; it reads each
   layer's experts once per super-chunk but changes the summation order; on their PC KL 0.02 against chunk-major at 50K, unexplained,
   so the fork keeps it off - `WORKLOG.md:1092-1175`).
2. **Every non-held expert streams, in id order, routed or not** (chunks of >= 1,024 tokens; checked:
   `SE:src/prefill/prefill.cpp:3712-3767` builds the sequence over layers and experts 0-511, skipping only VRAM-held ones). Each is
   copied once per layer per chunk into a **ring of VRAM slots** (one whole expert blob per slot): 384 slots when nearly every
   streamed expert is in pinned RAM, 96 when a large share goes through host copies (checked, `SE:src/prefill/prefill.cpp:430-438`),
   1,024 with the fused kernels on the Q2_0 pack. An issuer thread keeps the copies a ring-length ahead **across layer boundaries**,
   so layer L+1's first experts arrive during layer L's attention. Unpinned blobs (RAM tier not page-locked, or the SSD) go through
   **stager threads** into pinned host buffers first; the fork reads a file blob unbuffered **straight into the pinned buffer**
   (in-place stager, `STRATA_STAGER_IN_PLACE`, +8.5 % at 50K, +13.9 % at 3K, CPU 87 % -> 10 % busy, `SE:docs/fork/PERFORMANCE.md:69`).
3. **Expert GEMMs:** llama.cpp's MMQ int8 kernels over groups of 16 routed experts (activations quantised once per token to q8_1,
   gathered into group buffers), or Strata's own fused int8 kernels (default only on sm_120 cards with 80+ SMs), or an FP16
   dequant + cuBLAS fallback for formats MMQ lacks (`reading-notes/slender_prefill.md` b.4). **Q3_K and Q4_0 are not in their MMQ
   list** (`SE:src/prefill/moe_mmq.cu:319-347`, checked) (they would take the FP16 fallback) - and Q3_K gate/up has no decode kernel at all (section 6).
4. **VRAM during a prompt ("the loan"):** the prompt buffers are carved out of the **top slots of the VRAM expert cache**; those
   experts are marked "not held" for the prompt and **refilled byte for byte** afterwards from RAM / SSD (the decode cache is not
   re-ranked). After a 50K prompt the refill took 1,048-1,059 ms with one thread, 405-414 ms with `STRATA_REFILL_PINNED=4`
   (4 threads, own pinned buffers; not a default) (`SE:docs/fork/PERFORMANCE.md:36`).
5. **The prompt room (SlenderEngine):** while a long prompt (>= 4,096 new tokens) is read, the drafter's weights (DFlash2 pack 323 MiB
   or the MTP layer's 512 experts 675 MiB), the draft head (81 MiB) and the output head (497 MiB) are idle; the chunk's buffers
   continue into those blocks at the same addresses (CUDA graphs stay valid), and they are written back before the first token -
   the drafter from a RAM copy, the output head re-read from the model file (~130-150 ms). Bigger chunk -> one pass fewer:
   50K +10 % (defaults) to +19.5 % (MTP, 192-slot ring); the first-token KL from the different chunk size 0.0003-0.0022, same top token
   (`SE:docs/fork/PROMPT-ROOM.md:11-19, 114-125`).
6. **Dense arena (fork):** the dense matrices in one VRAM allocation, so the 2 MiB allocation granule is paid once; part of the
   "50K in 6 passes instead of 7" step (1,741 -> 1,879 tok/s), no separate number (`SE:docs/fork/PERFORMANCE.md:48`).
7. **Fused hyper-connection kernels:** `gr_upmix` does the up projection (BF16 tensor-core MMA, FP32 accumulate) and the mix
   (sigmoid gate, x row scale x weight, mean of 4 streams) in one kernel, so the T x 10,240 FP32 gate tensor (336 MB at 8,192 tokens)
   is never written; `gr_write_norm_rs` does the residual write and the next half's RMSNorm in one pass. `gr_upmix` is a fork default on
   Windows NVIDIA (Strata: opt-in on CUDA), chunks of 33+ tokens: 50K +4.3 %, logits byte-identical on their PC
   (`SE:docs/fork/PROMPT-ROOM.md:136-141`; code `SE:src/prefill/prefill.cpp:568-580`, checked).
8. **GDN in prompts:** a token-by-token recurrence kernel that walks the whole chunk in one launch with the 128 x 128 state slices in
   registers; the conv history and the state carry across chunks in device memory. Strata main turns on a tensor-core chunked scan on
   Linux CUDA by default (prompt +4.4 % at 4K, +5.7 % at 32K; KL mean 0.0019) (`reading-notes/strata_prefill.md` d.2).
9. **Sparse attention in prompts:** K/V written int8 per 64 values; indexer keys appended; block scores on 3xTF32 tensor cores and a
   radix top-k select of 2,048 cells per query; attention on **INT8 tensor cores** over the selected cells (default on cc >= 8.0)
   (`reading-notes/slender_prefill.md` f).
10. **n-gram rows in prompts:** one request for all 16 x T rows of a chunk: rows grouped by 4 KiB page, sorted by offset, up to 256 reads
    in flight, a 2^20-row cache; the next chunk's rows are fetched while the current chunk runs layers 1-47; uploaded to the GPU from
    two pinned buffers (`reading-notes/slender_prefill.md` g). Strata switches the table into RAM when RAM >= table + all experts +
    8 GiB (not on a 32 GB PC).
11. **Small prompts:** parts of <= 64 tokens go through the decode windows; chunks under 1,024 tokens (3,072 in Strata) stage only the
    routed experts and give the CPU pool the experts few tokens route to (the "CPU share", section 7).
12. **Conversation cache:** a checkpoint holds the running state only (36 GDN states + indexer tails + n-gram history,
    117,669,888 B); 6 kept in RAM, one every 16,384 tokens plus turn boundaries; the K/V cells stay in place. SlenderEngine adds parking
    whole conversations on the SSD (`--park-dir`).

Where their prompt stands (SlenderEngine's PC, skip 0): **50K prompt 2,064-2,145 tok/s with the fork's defaults, 2,258-2,296 with
its author's launcher** (192-slot ring, 95 % lend) (`SE:docs/fork/PERFORMANCE.md:17-18`, checked). The prompt is dominated by
passes: each pass moves the ~43 GB of experts not in VRAM over PCIe (the fork measured a 24.1 GB/s PCIe probe on that PC,
`WORKLOG.md:837`).

## 4. Expert memory, SSD reading, lending

- **VRAM expert cache:** filled from a fixed ranking (`data/expert-profile.bin`, global over all layers), slot 0 read back and
  compared at start; adapted every 4 windows (up to 96 swaps, decay 0.7; SlenderEngine uses 24 swaps): non-held experts with a decayed
  count >= 2 replace the least-used held ones (`reading-notes/strata_main.md` 2.1).
- **RAM tier (low-RAM mode):** the hottest experts by the same fixed ranking, `--resident-budget-gib N`; Strata can swap a few RAM
  places per window (`STRATA_RAM_ADAPT`); SlenderEngine's read-through puts every missed expert's read into the RAM place of the
  layer's least-used expert not routed in this layer (decode windows only; +5.2..+13.3 % whole runs, reads -16..-23 %, `FORK.md:118-125`).
- **SSD reads:** Linux: `O_DIRECT` + kernel AIO (`io_submit`), up to 256 in flight per thread; Windows: unbuffered overlapped
  `ReadFile`, all requests of a batch in flight before the first wait. No io_uring anywhere. Strata picks direct reads or the page
  cache at start by timing both (`S:src/core/expert_source.cpp:804-1010`). Prefetch: Strata applies **layer l+1's router to layer
  l's MoE input** and warms the next 1-2 layers' experts (the same "router trick" as our guess; no accuracy figure published).
  SlenderEngine's read-ahead from that guess lost 18 % at K=10 (70 % of blobs unused) and gained 2.7 % at K=5, nothing on top of
  read-through (`WORKLOG.md:299, 1205, 1041-1045`).
- **Lending idle memory to experts:** (a) SlenderEngine `STRATA_KV_LEND`: the page-locked **host** copy of the streamed K/V is sized for
  the whole context; the cells a request cannot reach are lent to the RAM tier as read-through places and taken back at the next
  request: SSD misses -27 %, short answers 37.9 -> 42.2 tok/s, answers byte-identical at temperature 0 (`SE:docs/fork/PERFORMANCE.md:67`;
  code `SE:src/program/generate.cpp:12944-12995`, checked: needs one GPU, int8 streamed K/V, read-through, the adaptive RAM tier).
  (b) Strata `--kv-grow` (opt-in): the VRAM expert cache lives in a CUDA virtual-memory range and gives 2 MiB chunks to the K/V only as
  the context reaches them; R9700, 196,608 context: 6,213 -> 7,968 slots, decode 96.3 -> 98.5 tok/s (`S:docs/DETAILS.md:1443`).
  (c) The prompt loan and the prompt room (section 3).
- **KV in RAM (Strata, default from 64K when RAM allows):** the whole K/V in page-locked RAM, only 32,768 cells per attention layer in
  VRAM (`--kv-resident 32768`); +6 % at 128K, +23 % at 262K (RTX 5070, Q2_0, `S:docs/DETAILS.md:236`).
- **Two SSDs (SlenderEngine, Windows only):** each read >= 256 KiB split between two copies of `experts.bin`: +12 % text, +39 % prompt
  read (`FORK.md:135-143`).

## 5. Settings and defaults that matter (Qwen3.8)

Strata's engine defaults are bare; its setup writes `--expert-profile <shipped> --expert-cache auto --prefill auto --spec 4
--spec-min-p 0.5 --mtp`, `--kv int8` above 8K, `--kv-resident 32768` from 64K, the low-RAM flags where needed, and DFlash2 drafting
on CUDA (`S:strata_control/setup_config.py:213-330`, `S:strata_control/drafting.py`). SlenderEngine's `FAST-SETUP.bat` writes
`--spec 4 --spec-min-p 0.5 --resident-budget-gib N --adapt-every 4 --adapt-swaps 24 --adapt-async 1 --kv-resident 32768
--conversation-cache-mib 0`, `--context 262144 --kv int8`, asks "skip experts? y/N" (default no) (`SE:tools/fast_setup.py:294-415`).
The full tables of switches: `reading-notes/strata_main.md` section 5, `reading-notes/slender_docs.md` section 2,
`reading-notes/slender_prefill.md` section j.

## 6. Quant formats: neither engine runs Grawnch's file today (checked)

Strata (and SlenderEngine, same lists) check every layer's expert formats at start and stop if the GPU kernels are missing
(`S:src/program/generate.cpp:4664-4676`, checked). The gate/up list `STRATA_GU_FMTS` has no type 11 (Q3_K) in either build variant
(`S:src/kernels/cuda/iq_kernels.cu:849-851`, checked; same at `SE:src/kernels/cuda/iq_kernels.cu:849-851`), and
`native_expert_supported` requires a gate/up kernel (`S:iq_kernels.cu:5480-5484`, checked). Down Q4_0 and Q2_0 are supported (types 2
and 42 in `STRATA_D_FMTS`). So **Strata and SlenderEngine refuse Grawnch's quant at start** (Q3_K gate/up) - as far as the code
shows; neither was run. Their supported packs are IQ1_M / IQ2_* / IQ3_XXS / IQ3_S / IQ4_XS gate/up, Q2_0 (one ISTA pack), Unsloth UD
sets (K-quants Q4_K / Q5_K / Q6_K gate/up), Q4_0 / Q8_0 and NVFP4. Down K-quants are impossible for this model (640 values per down
row is not a whole 256-value super-block).

## 7. Every part that changes the model's answers

"Default" = state with the engine's own defaults (Strata main / SlenderEngine). Rounding-level differences (GPU vs CPU, batch shape,
summation order) are accepted by the user's rule; they are listed so nothing is hidden, and marked "rounding".

| # | part | Strata main | SlenderEngine | what it does | effect on answers | measured speed effect |
|---|---|---|---|---|---|---|
| 1 | **Route tail skip** `STRATA_ROUTE_TAIL_SKIP=7` | **on** on CUDA (off when every expert is held, >= 90 % cached, batch, peer GPU, NVFP4) (`S:generate.cpp:12862`, checked) | **off** (0) (`SE:generate.cpp:3359`, checked) | a missed expert that every token of a decode window routes at rank 7-9 is neither read nor computed; the others keep their weights (no renormalisation) (`SE:src/core/expert_source.cpp:5919-5950`, checked) | **changed** - Strata: KL 0.003-0.006 vs no skip (`S:docs/DETAILS.md:1520`); SlenderEngine on its PC: KL 0.0807, top-1 agreement 90.7 % (`WORKLOG.md:1260-1273`) | Strata: +10..+24 % decode on 12-16 GB cards (`DETAILS.md:1521`); SlenderEngine: skip 7 45.9 vs skip 0 35.9 tok/s (+28 %, `WORKLOG.md:1247`, checked) |
| 2 | `STRATA_ROUTE_TAIL_SKIP_FILE=R` | - | off | the same skip from rank R, only for experts that would come from the SSD | **changed** (quality not measured) | rank 5: +12 % short, +21 % at 50K (`WORKLOG.md:822`) |
| 3 | `STRATA_ROUTE_PRIOR` (route toward held experts) | off (withdrawn in 0.2.8 after repetition loops and weaker agent results) | off | re-picks the top 10 with a bonus for VRAM-held experts | **changed** | +15 % / +37 % on an RTX 3060 before withdrawal (`DETAILS.md:1100, 1523`) |
| 4 | `STRATA_ROUTE_RESIDENT` (swap tail experts for held ones) | off | off | swaps non-held rank 6-9 experts for held ones within a logit margin | **changed** | no number published |
| 5 | `STRATA_KVA` "faster long prompts" | off (listed in the app) | off | layers 36-47 of a cold prompt (except the last 2,048 tokens) only build their state and K/V; their attention output and experts are skipped | **changed** (KL 0.018-0.020; lost records in long extraction tasks) | +11 % at 14K, +18 % at 30K prompt (`S:docs/KVA.md`) |
| 6 | DeepSeek "faster" mode (5 of 6 experts) | on for <= 16 GB cards, **DeepSeek only** | - | top-k cut | changed | not for Qwen |
| 7 | Experimental speed projection (a control vector) | off | off | removes a direction from the residual | **changed** (KL 0.063) | - |
| 8 | **CPU share of prompt experts** `STRATA_PREFILL_CPU_SHARE` | **on** for chunks < 3,072 tokens (one GPU, CUDA) | **on** for chunks < 1,024 tokens (`SE:generate.cpp:3360`; limit code `SE:src/prefill/prefill.cpp:294-305`, checked) | the idle CPU pool computes some non-held experts few tokens route to (same experts, none dropped) with its own activation quantisation | **rounding** (GPU vs CPU): KL ~0.004 below 1,024 tokens, ~0.011 at 1-3K (`SE:src/prefill/prefill.cpp:2952` message, checked); "run-to-run different answers" at temperature 0 (`WORKLOG.md:475`) | 512 tokens -28..-34 %, 1,000 tokens -20..-26 % prompt time (`DETAILS.md:1518`) |
| 9 | **PCIe share** of decode misses | on (auto-refined) | on unless `--pcie-frac 0` (the fork's recommended setting is 0) | the GPU computes some missed experts from pinned RAM instead of the CPU | **rounding** (first-token KL 0.0013-0.0015, `DETAILS.md:1522`) | RTX 3060 x1.30 at share 0.20; SlenderEngine: `--pcie-frac 0` 7 % faster per step on its PC |
| 10 | **K/V precision** `--kv int8` | engine default fp16; **setup writes int8 above 8K** | int8 always (setup) | 8-bit K/V with a scale per 64 values | **changed (precision)** - the same kind of choice as our q8_0 KV presets; q4_0 / k8v4 opt-in are lossier (q4_0 perplexity +8-12 %) | halves the K/V memory |
| 11 | Small-card prompt rule `STRATA_PF_SMALL` | on (one card, default chunk < 8,192: layer-major + unfused) | **off** (`SE:generate.cpp:3361`) | another summation order | rounding | 9-50 % sooner prompts (Strata's docs) |
| 12 | Prompt kernels: int8 prompt attention, 3xTF32 block scores, GDN tensor-core scan (Linux, main only), K-quant MMQ (cc >= 8), fused int8 experts (sm_120 80+ SMs), chunk size, prompt room | on | on (GDN TC: off; room: on with a drafter) | other arithmetic in prompts | **rounding** (K-quant MMQ first-token KL 0.024-0.035; GDN TC KL 0.0019; chunk size 0.0001-0.0024) | GDN TC +4.4..+5.7 %; room +10..+19.5 %; K-quant MMQ +26..+38 % on an RTX PRO 6000 |
| 13 | `STRATA_HC_Q8` / hyper-connection weights in BF16 | BF16 copies of the file's Q8_0 hyper-connection tensors unless `STRATA_HC_Q8` (on for CUDA with Unsloth UD files) | off on NVIDIA | which copy of the small tensors is used | changed vs the file's own weights (small: KL vs Q8_0 0.0562 -> 0.0526 on one UD file) | the laptop page: +21 % decode with HC_Q8 (`LAPTOP-RTX4060.md:36`) |
| 14 | The adaptive VRAM cache (`--adapt-every`, `--adapt-async`) | on | on | which experts the GPU computes changes between runs | **rounding** ("not reproducible run to run", `DETAILS.md:573`) | - |
| 15 | Drafting (MTP `--spec 4`, DFlash2, prompt lookup) | on (setup) | on | each drafted token is checked by the full model | not an answer change; the window shape changes rounding ("`--spec 4` changes the wording", `FORK.md:157`) | MTP 1.6-1.8x (Strata); SlenderEngine laptop page: MTP off -19..-22 % |
| 16 | The table formats | the shipped n-gram table is IQ4_NL (already a quantisation of the BF16 table: KL 0.0146) | same | - | part of the model file | - |

**Not found in either code base:** a top-k cut for Qwen3.8, expert substitution by default, a "use only held experts" mode, lower
precision for the GDN state (it is F32).

**Which of their published numbers had an answer-changing part on:**

- **SlenderEngine's headline 48.9 tok/s** (`FORK.md:32-44`) and every number before 2026-11-06 were measured **with skip 7 on**
  (`WORKLOG.md:1278`, checked). Its **skip-0 numbers** (2026-11-08/09, `PERFORMANCE.md:12-26`, checked): short answers
  **46.5-49.5 tok/s**, after a 50K prompt **35.5-41.4**, follow-up 34.6-41.8, 50K prompt **2,064-2,145 tok/s (defaults)** /
  2,258-2,296 (author's launcher). These still have: the CPU share for chunks < 1,024 tokens (rounding), int8 K/V (a precision choice,
  like our q8_0), the adaptive cache (rounding), and DFlash2 drafting with 1.6-1.9 tokens accepted per step (exact check). No skip-0
  number **without drafting** is published; their laptop page measured MTP off at -19..-22 % (`LAPTOP-RTX4060.md:47-61`), so a
  no-drafting skip-0 figure would be roughly 46.5-49.5 x 0.8 = **~37-40 tok/s** (my estimate, from their ratio on another machine).
- **Strata's main Qwen tables** (`S:docs/DETAILS.md:45-73`, engine 0.1.26, RTX 5070 + Ryzen 5 7600 + **64 GB** RAM, all experts in RAM)
  predate tail skip and the CPU share (both from 0.1.41/0.1.42), so those two were off; int8 K/V above 4K, the PCIe share and MTP were
  on. Community reports from 0.1.42 on have tail skip on wherever the card misses experts. Strata publishes **no** tok/s with every
  answer-changing part off (only correctness gates, `reading-notes/strata_main.md` 6.3).
- The RTX 4060 laptop page (another owner) names lossy items (HC_Q8, file-read tail skip rank 5) among the changes behind its "tuned"
  column, without saying which were on.

## 8. Drafting and the MTP head - later (nothing planned)

- **What they do:** Strata loads Qwen3.8's own MTP layer (from the BF16 checkpoint, not the GGUF; 512 experts in VRAM, ~0.7-1 GiB) and
  drafts up to 3 tokens a window (`--spec 4 --spec-min-p 0.5`), checks them in one pass over all 48 layers; plus prompt-lookup /
  suffix drafts (up to 12 tokens) and, on CUDA, the DFlash2 block drafter (Strata's tuned pack is Apache-2.0; **SlenderEngine's docs
  say the DFlash2 drafter it uses is CC BY-NC-ND**, `FORK.md:111`, `PERFORMANCE.md:73` - not usable for us). The n-gram rows of every
  drafted token are fetched ahead.
- **Their numbers:** Strata MTP 1.6-1.8x (47-57 -> 82-92 tok/s, 2.4-3.6 tokens per pass, all experts in RAM); SlenderEngine: MTP accept
  79-83 / 70 / 61 % for drafts 1-3, DFlash2 1.6-1.9 tokens per step in the skip-0 runs.
- **Where it would plug in for us:** Grawnch's file has no MTP block, so it needs an MTP GGUF; our tree has the qwen4exp MTP graph
  and draft memory already (`port-needs.md` A8); the needed experts per call become the union over the drafted tokens (check 1 of
  `docs/fit-with-our-core.md`, class B); MTP with our prefetcher together (memory note). Not part of this plan.

## 9. What changed in Strata since our 2026-10-01 study (`docs/others-code-study.md` section 2)

From `reading-notes/strata_main.md` section 8: many packs and three model families; a low-RAM mode (file tier, page cache or direct
reads, profile-ranked RAM copy, automatic fallback, router lookahead); tail skip and the CPU share (default on); drafting with DFlash2;
direct-read n-gram table with a row cache and keep-alive reads; `--kv-grow`, `--vram-elastic`, KV in RAM, the prompt loan; prompt
speed 285-571 (paper) -> 1,300-2,200 (0.1.26) -> up to ~3,500 tok/s on faster cards. Between v0.2.11 and main: admission of
PCIe-fetched experts (opt-in), file-tier defaults off while every expert is in VRAM, GDN tensor-core scan default on Linux CUDA,
an IQ4 W4A8 prompt kernel and a two-pass QSA select (opt-in), fixes.

## 10. Licences

Both repos MIT ("Niko1221 and the Strata contributors"; SlenderEngine adds "Rafid Ahsan"). Vendored code is MIT (ggml, project-maya,
sfnx, antirez-ds4, libstrix, gufo); some kernels carry their own MIT notices ("The ggml authors", "gufo contributors") that must be
kept if any of their code is ever used directly. Data with other licences: Strata's DFlash2 tuned pack Apache-2.0; the speed-projection vector and Gyro weights Qwen
Community License; SlenderEngine's DFlash2 drafter CC BY-NC-ND (stated in its docs; no licence file in the repo).
