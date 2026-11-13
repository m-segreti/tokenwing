# What the port of Qwen3.8-Flash-Next needs on today's engine

Engine read: `llama.cpp/` on branch `moe-lab` at `e2330ec67` (read only). Model facts: the GGUF headers of Grawnch's file
(`model-numbers.md`). Every code claim gives `path:line` in `llama.cpp/` unless it says otherwise. Sizes: **small** = one function or
a list entry and its check; **medium** = a few functions in one or two files; **large** = a new mechanism over several files.
Nothing here was built or run; "to check when running" marks what only a run can show.

Pattern: this follows what the Flint port took (`reports/C17-flint-port.md`, `llama.cpp/docs/adding-a-model.md` section 3).
The big difference to Flint: Flint's architecture had to learn our hooks; Qwen3.8 additionally has a hyper-connection residual
(the guess input is not "residual + norm gain"), a hybrid memory with a third cache (the sparse-attention indexer), a 26.8 GiB
table read per token, a file that is not aligned for direct reads, and expert formats none of our fast CPU paths cover.

## A. Loading the file

### A1. A two-file GGUF
- **Exists:** the engine's experts-file route handles split models (the recorded size is the sum of the shards,
  `src/llama-moe-stream.cpp:1020`, and each projection records its `source_shard`, `:2035-2045`). The n-gram table in shard 2 is mapped lazily per
  shard (the lazy ranges are kept per file index, `src/llama-model-loader.cpp:1121-1123`).
- **Missing:** the one-file TokenWing model format refuses a split GGUF, in the tool (`tokenwing-tools/experts/make_model_file.py:176-183`:
  "merge the shards first") and in the engine (`--moe-stream model`, `src/llama-moe-stream.cpp:1877`). Merging gives one 84 GB file. All routed experts are in shard 1, so the format could keep shard 2 as it
  is. That also matches the user's later plan (weights file + small data file, memory note "model package").
- **Size:** medium (tool + the engine's model-file check accepting a split model whose experts are all in shard 1).
- **Check:** `make_model_file.py`'s own read-back; `--moe-stream-verify` on every streamed layer; plan-only line.
- **Needs:** the files (no run).

### A2. The file is not aligned for our direct reads (the first real blocker)
- **Facts:** `general.alignment` is 32 and no expert tensor starts on a 512- or 4096-byte boundary. Gate / up slices are 704,000 B
  (not a multiple of 4096, but 1,375 x 512); Q2_0 down slices 460,800 B (900 x 512); Q4_0 down 921,600 B (225 x 4096).
- **Today:** the engine needs every slice and offset to be a multiple of 4096: the model-file route refuses at
  `src/llama-moe-stream.cpp:1153-1155` (alignment) and `:1256-1258` (slice), the experts-file route at `:2047-2055`. A pool is
  `ggml_new_tensor_3d(type, ne0, ne1, slots)` with the slice as its stride (`create_pool`, `:3793`, the tensor at `:4029`), and a
  read lands at `w.mem + slot*w.size` (`moe_stream_read_range`, `:4233-4235`), so a slot of 704,000 B is not 4096-aligned either.
  `expert_repack.py` writes gate|up|down back to back without per-projection padding (its own `offset_formula`,
  `tokenwing-tools/experts/expert_repack.py:290-296`), so the experts-file route refuses this model too.
- **Measured here (a tiny probe, `scripts/odirect_align_probe.py`, run 2026-11-13):** on this laptop's ntfs3 partition (NVMe logical
  block 512 B, `/sys/block/nvme0n1/queue/logical_block_size`) O_DIRECT reads with 512-aligned offsets, lengths and buffers all succeed.
  AGENTS.md's "4096-aligned" is therefore stricter than this drive needs.
- **Options:**
  1. **Direct-read alignment from the device**, not the fixed 4096: read the file's direct-I/O alignment (Linux `statx` with
     `STATX_DIOALIGN`, Windows the sector size) through the OS layer, and let the tools accept 512-multiple slices.
     `make_model_file.py` already writes `general.alignment = 4096` (`make_model_file.py:7`), which puts every tensor start on a
     boundary; then every slice of this model is 512-aligned and every pool slot too. Zero-copy reads stay.
  2. **Fallback for a drive that needs 4096 (4Kn drives):** read the aligned window around a slice into a per-thread bounce buffer
     and copy into the slot (one ~2 MB copy per read, CPU memory bandwidth; cost not measured). Without it such a machine could
     not run the model at all, which breaks the "never refuse" rule.
  3. Padded experts file + padded pool stride (the old sidecar route, a second 50 GB file) - not recommended.
  - Asking Grawnch to change types is no help: at 2560 x 640 only Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K and IQ4_NL give
    4096-multiple slices, all bigger than Q3_K; **every** common format gives a 512-multiple (table in `needed-from-grawnch.md`). So
    option 1 covers any quant of this model, option 2 only 4Kn drives.
- **Size:** medium (OS-layer function + alignment as a value in the engine, the tools' checks, the fallback copy).
- **Check:** `--moe-stream-verify` and `--moe-stream-slot-check` on; streamed CPU run gives the same md5 as the plain CPU run; a
  test switch that forces 4096 so the fallback path runs and gives the same md5.
- **Needs:** the files; this is the first item, nothing streams without it.

### A3. Tensor names, `build_moe_ffn`, routing
- **Exists:** `qwen4exp` calls `build_moe_ffn` (`src/models/qwen4exp.cpp:1249-1266`): softmax over 512, top 10, weights normalised
  (`norm_w` true), no selection bias, no groups. The file has the three separate expert tensors (no fused gate-up; header), which
  is what `llama_moe_stream_is_exps` looks for (`src/llama-model.cpp:1688-1697`). Gate and up share type and shape in every layer
  (the VRAM pool's rule). No dense layers: all 48 layers are MoE (`n_layer_dense_lead` 0), so layer 0 is a streamed layer too.
- **Missing:** nothing for plain streaming once A2 and A4 are done.
- **Check:** `--moe-stream-plan-only` prints 48 streamed layers with 2,329,600 / 1,868,800 B experts; the remap op runs.

### A4. The RAM plan counts the 26.8 GiB n-gram table as RAM (a real bug for this model)
- **Today:** the walk of "what this process holds in RAM besides the pools" skips the expert tensors and, for archs on the lazy list,
  `token_embd.weight` only (`src/llama-model.cpp:1946-1980`, the exemption at `:1966-1975`). `per_layer_token_embd.weight` is lazy
  (`src/models/qwen4exp.cpp:216-217`) but would be counted as 26.8 GiB of host weights; the plan side has the same rule
  (`src/llama-model.cpp:3619-3622`). The pools would come out near zero, or the plan would refuse.
- **To add:** exempt every lazily read tensor (the loader knows them: `ml.lazy.has(...)`), in the load walk and in
  `llama_moe_stream_get_plan`.
- **Size:** small. **Check:** plan-only RAM line equals the hand sum (`model-numbers.md` section 4); RSS after load.
- **Needs:** the files.

### A5. The token table read lazily (+0.33 GiB for experts)
- **Exists:** the lazy word table for Ling and Flint (`llama_arch_marks_token_embd_lazy`, `src/llama-model.cpp:891-893`).
- **To add:** `LLM_ARCH_QWEN4EXP` in that list and `TENSOR_READ_LAZY` on `tok_embd` in `qwen4exp.cpp` (the file has a separate
  `output.weight`, so the table is not tied).
- **Size:** small. **Check:** the start-up line "the word table ... is read lazily"; same md5 with and without; RSS 0.33 GiB lower.

### A6. Recording tool refuses the model
- **Today:** `llama-moe-trace` requires `expert_group_count` and `expert_group_used_count` (`tools/moe-trace/moe-trace.cpp:396-400`);
  the file has neither key, so it stops with "the model is not an MoE model".
- **To add:** make the two keys optional (0 = no groups). The engine's own recorder (`--moe-stream-record`) is the other route.
- **Size:** small. **Needed for:** the VRAM list, the usage seed, the layer order and the guess study (B1). **Check:** recorded ids =
  the engine's picks on the same forced text (as T46 did).

### A7. Chat template, thinking, tool calls
- **Facts:** `tokenizer.chat_template` is an empty string in the file. llama.cpp then uses a plain chatml template
  (`common/chat.cpp:776-780`), which knows nothing about thinking or tool calls. The pre-tokenizer is `qwen35`; our tree carries
  Qwen3.5 / Qwen3-Coder templates (`models/templates/`), none for Qwen3.8.
- **To add:** the official Qwen3.8 template (from Qwen's own model repo), given with `--chat-template-file`, or written into the
  file / our data file. Which reasoning and tool-call parser it needs is not verified.
- **Size:** small (+ app side: earlier replies' reasoning, as Flint's B6). **Check:** llama-server: a thinking reply is split into
  reasoning and content; a tool call is parsed N of N.
- **Needs:** the template (Grawnch or Qwen's repo), the model running.

### A8. MTP and images
- The file has no MTP block and no image file. Our tree already has the MTP graph (`qwen4exp.cpp:582-...`) and the draft
  memory filter (`src/llama-model.cpp:4153-4159`). Nothing to do now; see the "later" section of `others-today.md`.

## B. The guess for the next layer

### B1. Qwen3.8 makes no guesses today, and our shared guess code assumes a plain residual
- **Today:** `qwen4exp.cpp` never calls `moe_guess_routers` / `moe_guess_finish` / `moe_ws_layer`, and passes no next router to
  `build_moe_ffn` (`qwen4exp.cpp:1252-1266`), so the engine streams with no guess at all (as Flint before its port). The arch is
  also not in `arch_leads` (`src/llama-moe-stream.cpp:1493`), so `--moe-stream-lookahead` above 1 and `--moe-stream-wordstart`
  fall back with one line.
- **Why it is not a copy of Flint's hook:** the shared code builds the guess input as `RMSNorm(residual + shared) * ffn_norm(L+1)`
  (`src/llama-graph.cpp:2411-2415`), and the far guesses and the recorder use `ffn_norm` of each target layer (`:2426`, `:2461`).
  Qwen3.8 has no `ffn_norm`: a layer's router reads `hc_mix`'s output - a grouped RMSNorm over the 4 streams with an
  [n_embd, 4] gain, a rank-320 down / SiLU / up, a sigmoid gate and the mean of the 4 streams (`qwen4exp.cpp:348-415`) - applied
  to the residual after that layer's own attention (`qwen4exp.cpp:535-546`).
- **What the router of layer L+1 really reads:** `hc_mix_ffn(L+1)( res_A(L+1) )`, where `res_A(L+1)` = the 4 streams after layer
  L+1's attention, which depend on layer L's routed experts. Candidate guess inputs, cheapest first:
  - g1: layer L's own router input with layer L+1's router (no new op; ignores layer L's FFN output and L+1's attention).
  - g2: `res_A(L) + 2*sigmoid(inject_L) * shared_L` (layer L's residual plus its gated shared expert, as `build_hc_combine` adds it,
    `qwen4exp.cpp:417-441`), mixed with layer L+1's **FFN** hyper-connection weights, then router L+1 - the same idea as Ling's
    `--moe-stream-pred-shared` (it waits for the shared expert, which already runs before the routed experts).
  - g3: as g2 but mixed with layer L+1's attention weights (what L+1's attention reads).
  - g2 / g3 cost one extra `hc_mix` per layer per token (two small matrices of 10,240 x 320; sum: ~6.6 M multiply-adds each).
- **To add:** a per-arch guess-input hook in the shared code (the guess input, the far guesses' inputs, the recorder's leads,
  the word-start sources), the qwen4exp calls, the arch in `arch_leads`. Pick g1/g2/g3 by **counting on real recordings** (recall
  of the 10 picks at 1, 2, 3 layers ahead; counting on recordings is allowed, no simulator).
- **Size:** medium. **Depends on:** A6 (recordings), A2/A4 (streaming). **Check:** Flint's gate - the md5 with the guess on equals
  off (a guess changes what is read, never the maths); needed reads per token and SSD wait go down; recall per layer from the
  recorder's lead logits.
- **Needs:** the model running.

### B2. The correction file cannot apply as defined today
- **Today:** the file format defines the corrected guess as `(W_X diag(g_X) + Bt) x0` with `x0 = RMSNorm(residual + shared)` and
  `g_X = blk.X.ffn_norm.weight` (`tokenwing-tools/correction/correction_file.py:1-13`); the engine refuses a layer without an F32
  `ffn_norm` (`src/llama-moe-correction.cpp:135-136`, `:170-172`) and hashes `ffn_norm` into its model check (`:111-145`). The
  training tools carry Ling's shape (512 experts, hidden 2,560, top 8, 8 groups / 4, layers 5-41; `docs/adding-a-model.md:158`).
- **To add:** a second fold for this arch: `x0` = the chosen guess input of B1 (unnormalised or after its own norm), `W' = W_X + Bt`
  (no gain), the model check over the routers plus whatever tensors the guess input uses; the training scripts for 512 experts,
  top 10, no groups, layers 0-47.
- **Size:** medium. **Depends on:** B1 (the input must be fixed before anyone trains). **Check:** the engine's own refusals and
  checks; needed reads per token with the file vs without (5 pairs).
- **Needs:** Grawnch's file, made on the agreed input (`needed-from-grawnch.md`).

### B3. Routing skew: to be measured, not assumed
- The old study (`docs/qwen38-port.md` section 0) quoted upstream PRs saying the routing is near-uniform. SlenderEngine's own
  measurements on this model say a small held set already serves most picks: with ~8 % of experts in VRAM and ~21 % in RAM,
  60 / 29 / 11 % of a token's experts came from VRAM / RAM / SSD (`docs/fork/WORKLOG.md:610`; coding 59-68 / 25-30 / 7-11 %,
  `:703`; measured with expert skipping on, their PC). Our RAM pools would hold ~52 % of the experts on the laptop
  (`model-numbers.md` section 4).
- **Consequence for us:** the usage seed, the VRAM list and decay eviction should work as on Ling; whether whole VRAM layers or a
  VRAM expert pool pays more must be measured (Strata/SlenderEngine keep no whole layers, only a per-expert cache).
- **To check when running:** per-layer use counts from recordings; hit shares VRAM / RAM / SSD per token (our counters).

## C. The n-gram table

### C0. What exists
- The table is created lazy (`qwen4exp.cpp:216-217`); with `--load-mode dio` the file is still mapped when a lazy tensor exists
  (`src/llama-model-loader.cpp:1434`), the lazy tensor is read from the mapping (`:1738`) and never locked (`:1765`); the
  mapping is `POSIX_MADV_RANDOM` (`src/llama-mmap.cpp:519`). Per call, the host hashes 16 row ids per token
  (`qwen4exp.cpp:1327-1386`), asks the kernel to read those pages (`MADV_WILLNEED`, `qwen4exp.cpp:1388-1395` ->
  `src/llama-impl.cpp:22-40` -> `src/llama-mmap.cpp:837-912`) and the graph gathers them with `ggml_get_rows` on the CPU
  (`qwen4exp.cpp:1469`). The gather sits at the start of the graph, before layer 0 (`qwen4exp.cpp:468-473`), although only
  layer 1 uses it.

### C1. The RAM plan bug (A4)

### C2. The table's file cache under the memory cap (to check when running)
- Our "file cache the kernel can free" formula subtracts file pages that are mapped into a process
  (`src/llama-moe-stream.cpp:376-399`). The table is mapped by our own process, so every table page read stays counted as our use
  under a memory cap, and it grows with every new row (up to 16 pages = 64 KiB per token). The kernel can still drop those clean
  pages under the cgroup's pressure; what is unclear is whether our own prompt-time memory checks (FIX 1's give-back, the call cap)
  misread it.
- **Possible fix:** read the table with our own direct reads into a bounded row cache (C3), which removes the mapping; or count
  the table's resident pages as freeable.
- **Size:** small to medium. **Check:** the cap line over a 100k-token session; no give-back caused by table pages.

### C3. Reading the rows ourselves, ahead of time
- **Today:** the kernel reads up to 16 random 4 KiB pages per token when `set_input` hints them, and the CPU gather waits on page
  faults before layer 0 starts. In a prompt that is 16 x T rows per call.
- **To add (same bits):** an engine row reader through the OS layer: direct 4 KiB page reads on our I/O threads, rows on the same
  page read once, sorted by offset, a bounded row cache, issued **one token ahead** in decode (the rows depend only on tokens
  already known) and one call ahead in prompts, consumed at layer 1. Strata does exactly this (`reading-notes/strata_main.md`
  section 3.2; 16 pread threads, 256 in flight, 2^20-row 8-way cache).
- **Their number:** the n-gram block of an 8K prompt 845 ms with direct reads, 451 ms through the mapping, 132 ms with the table
  in RAM (Strata, R9700 + IQ3_S, `src/program/generate.cpp:5480-5490` comment in their tree). Decode cost per token for us: not
  measured.
- **Class:** A by `docs/fit-with-our-core.md`, but it shares the SSD queue with expert reads (a B-style handling: its reads go
  first, they are tiny and needed by layer 1).
- **Size:** medium. **Check:** time of the gather per token and per prompt (profile of one call); same md5.

### C4. One sequence per token
- `set_input` asserts that no token belongs to two sequences (`qwen4exp.cpp:1352-1355`). Fine for `-np 1`; parallel slots that
  share prompt tokens would stop. Note for the server presets.

## D. Attention, GDN, prompts and the server

### D1. Layer-major prompts do not exist for this arch (large)
- **Today:** layer-major is limited to Ling and Flint (`src/llama-context.cpp:131-136`); `llama_memory_hybrid_idx` has no
  `rewind()` / `reapply()` (the default returns false, `src/llama-memory.h:106-110`; only the hybrid and recurrent memories
  implement them, `src/llama-memory-hybrid.h:119-120`, `src/llama-memory-recurrent.h:167-168`).
- **Without it:** a prompt runs ubatch by ubatch; each ubatch of 2048 tokens touches nearly every expert of every layer, so the
  experts not held are read once per ubatch: a 50K prompt at `-ub 2048` is ~25 passes (sum). With layer-major (and its call cap of
  32k at 65k context, `presets/ling.sh` `preset_lmb`) the same prompt is 2 passes.
- **To add:** the layer-range graph builder in `qwen4exp.cpp` (as `flint.cpp:125-250`: the range `[lm_il0, lm_il1)`, the three
  graph kinds, `lm_x` / `lm_hid` / `lm_row0`); `rewind()` / `reapply()` for `llama_memory_hybrid_idx` including the indexer cache
  and the k-pool state per ubatch (`src/llama-memory-hybrid-idx.cpp:333-560`); the n-gram rows per ubatch; the 4-wide residual
  kept between layer groups (4 x 2560 floats per token: 40 KiB per token, 1.3 GiB at 32k tokens - sum - which may have to live in
  RAM on a 12 GB card).
- **Depends on:** A2, A4. **Check:** md5 of a 2-ubatch prompt equals the normal path (one sequence); prompt time at 4k / 16k / 32k /
  50k; SSD wait in the prompt; the reply right after.
- **Needs:** the model running.

### D2. Server snapshots and saved chat states
- **Exists:** the RAM plan counts recurrent rows (`src/llama-model.cpp:1550-1583`, `:1600-1607`, including the n-gram conv state);
  a snapshot is ~117.7 MB (sum, `model-numbers.md`).
- **Missing:** the indexer cache is not counted ("caches of their own (an indexer's keys ...) are not counted",
  `src/llama-model.cpp:1588-1590`): 3,264 B per token (sum), 0.2 GiB at 65k. Whether in-call checkpoints and `--cache-disk`
  save / restore the indexer cache and the n-gram history correctly is **not verified** (there is a qwen4exp note in
  `src/llama-kv-cache.cpp:2472`).
- **Size:** small (plan) + tests. **Check:** plan-only vs measured; save / restore gives the same continuation md5.

### D3. KV cache at 65k and 262k
- Sums (`model-numbers.md`): 16,320 B per token with q8_0 = 1.0 GiB at 65k, 3.98 GiB at 262k. At 262k that takes most of the VRAM
  expert pool (5.4 -> 2.4 GiB). Ideas that help here (lending unreached KV cells to experts; KV in RAM) are in
  `ideas-from-other-engines.md`.

### D4. GDN and the sparse attention in prompts and decode
- Both run through upstream's ops (`build_recurrent_attn`, `qwen4exp.cpp:1232`; the indexer "scores with plain ops" because the
  CUDA lightning indexer takes 32 or 64 heads and QSA has 4, `qwen4exp.cpp:480-484`). Their speed at 32k-262k is unknown until a
  profile of one long call. Strata and SlenderEngine have dedicated kernels for both (`others-today.md`).

## E. The expert formats

| place | Q3_K (gate / up) | Q4_0 (down, 20 layers) | Q2_0 (down, 28 layers) |
|---|---|---|---|
| tools (`expert_index.py`, `expert_repack.py`) | mixed formats handled (per-layer sizes); alignment: A2 | aligned | A2 |
| RAM pools | per-layer slot size (Flint port) | | |
| VRAM pool groups (`vx_setup`) | two groups (Q3_K+Q4_0, Q3_K+Q2_0): the per-place fit runs only with one group, the tail lives in the bigger group (`docs/adding-a-model.md:115`) | | |
| ggml CPU kernel | AVX2 `vec_dot_q3_K` only | AVX2; a repacked 8x8 kernel exists (`ggml/src/ggml-cpu/repack.cpp:4974-4978`) but our pools are not repacked | AVX2 (`ggml/src/ggml-cpu/arch/x86/quants.c:703-760`) |
| our 4-row AVX-512 kernel (`--moe-stream-cpu-x4`) | no (MXFP4 only) | no | no |
| our prompt kernels (`--moe-stream-cpu-iqk`, `ggml/src/ggml-cpu/iqk/iqk_port.cpp:96-107`, `:338-344`) | no - but ik's plain Q3_K GEMM is in the ported file, not wired (`iqk_gemm_kquants.cpp:223, 489, 856`) | no - ik's Q4_0 GEMM is in the ported file, not wired (`iqk_gemm_legacy_quants.cpp:588-1101`) | no |
| CUDA MMQ (prompts, VRAM pass) | yes (`ggml/src/ggml-cuda/mmq.cu:35-36`) | yes (`:16-17`) | yes (`:13-14`) |
| CUDA MMVQ (decode) | yes | yes | yes (`ggml/src/ggml-cuda/mmvq.cu:43, 1381`) |

- **What this means:** the CPU computes most decode experts (those in RAM). Ling's CPU experts cost 16-18 ms per token for 320
  uses of 3.13 MB (AGENTS.md, T25); here it is 430+ uses of ~2 MB on AVX2-only kernels, Q3_K being the slowest to unpack. **CPU
  kernels are a main decode term for this model.**
- **To add:** E1 wire Q3_K and Q4_0 into `--moe-stream-cpu-iqk` (prompt-size calls; small-medium); E2 4-row AVX-512 kernels for
  Q3_K, Q2_0 and Q4_0 with the same output as ggml, the T97 way (large; Flint's item 9 was the same job for Q5_K / IQ4_XS);
  Strata's own AVX-512 Q2_0 kernels (MIT) are a reference (`ideas-from-other-engines.md`).
- **Check:** `kernel_bench --check` (4 rows per call memcmp-equal to 4 one-row calls); CPU-only md5 unchanged; CPU ms per token.
- **CUDA:** MMQ / MMVQ cover all three. Our fused GPU route (`--moe-stream-fused-route`) needs expert groups and does not apply
  (`docs/adding-a-model.md`, Flint's G1); check that upstream's softmax-top-k MoE fusion fires for top 10 of 512.

## F. What only a run can tell

The compute buffer at `-ub 512 / 2048`; the hit shares VRAM / RAM / SSD; per-layer use counts; the recall of g1 / g2 / g3; CPU
ms per token per format; where a 32k prompt's time goes (GPU compute, PCIe, SSD, the n-gram gather, GDN, the indexer); the
table's page cache over a long session; tool-call and thinking behaviour with the official template; the KLD of our streamed run
against plain llama.cpp on the same file (plain llama.cpp can only run it from the page cache on this laptop, slowly; a short
forced text is enough, `docs/adding-a-model.md` section 4).
