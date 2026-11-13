# Qwen3.8-Flash-Next: the recommended command and the first numbers (checklist Q12 + Q13)

Worker report, written as the work goes (2026-11-13). Engine measured: the shared build `llama.cpp/build/bin` = `moe-lab`
`308bc4b42` (not rebuilt, no engine code changed). Scripts: `bench/queue/qwen38/b3_*.sh`, `b3_client.py`; outputs
`traces/qwen38/b3/`. Every model run through `tools/model-lock.sh Qwen-b3` and a memory cgroup. AC on, PPT pl1 41 W.

## 0. Summary
- **Numbers (section 4), this laptop, measured while the PC was in use:** GPU case story reply **32.1 tok/s** (5 runs), 10k
  prompt **259 tok/s**; plain llama.cpp on the same file 3.15 tok/s and ~10 tok/s. 8 GB card: 16.1 / 172 tok/s against 3.0 / <6.
  CPU-only: 6.6-6.8 tok/s and 31-36 tok/s against 0.31-1.4 tok/s (plain could not run the 10k prompt under 11 GiB: killed by
  the cap). SlenderEngine (skip 0, their PC, with drafting): 46.5-49.5 tok/s short answers, 2,064-2,145 tok/s on a 50K prompt.
- `presets/qwen38.sh` written (3 cases, 65k / 262k); it runs as written (chat answered, thinking split). Flags: section 3.1.
- **A bug found:** copy-ahead with two copy sets makes the 10k prompt's answer garbage on this model (section 2). **Fixed on branch
  `qwen-fix` (section 9):** the quantized matmul's row padding read stale bytes of another format after unused tail places; the
  preset now uses the engine's rule (value 1) on this laptop - with the fixed engine only.
- **The perfect-guess run (section 10):** today's guess 6.58 ms of SSD wait per token, perfect guesses 1.83 (3 pairs, 200 slots, all
  layers streamed); the oracle's tally bug (part 1's 2.1 %) found and fixed on the way.
- The perfect-guess run is not usable (section 4.3). The frozen texts match on the measured build (1.1).

## 1. Checks on the measured build

### 1.1 Frozen Qwen texts (`bench/queue/qwen38/b3_gates.sh`, BIN = shared build)
```
gate_cpu_120 rc=0 md5=0a3c44f700c1e2a72ee787d66b72a88d
gate_cpu_120_ub12 rc=0 md5=0a3c44f700c1e2a72ee787d66b72a88d
gate_gpu_150 rc=0 md5=6417ddef0d82258c961f2a7449ddc811
expected: cpu 0a3c44f700c1e2a72ee787d66b72a88d (both), gpu 6417ddef0d82258c961f2a7449ddc811
```
All three match.

### 1.2 Same-output settings, CPU path (`bench/queue/qwen38/b3_cpu_flags.sh`)
The gate's command (120 slots, `--no-repack`, 32 words) with every same-output setting of the CPU-only preset on (overlap,
pred-shared, chunk-prefetch, io-wake, promote, guess-cap, policy decay, prompt-end recency, cpu-skip-zero, cpu-fuse,
cpu-dispatch, plan, drop-stale, needed-early, lookahead 2, guess-combined, proj-ready, wordstart 2, protect-named,
guess-inflight 4, cpu-topk, cpu-mmid-merge, wave-plan, prompt-defer, cpu-weighted-sum, cpu-small-ops, cpu-state-inplace), then
also with the two rounding-level CPU settings (cpu-q8-tiled, cpu-fa):
```
cpu_same_on rc=0 md5=0a3c44f700c1e2a72ee787d66b72a88d
cpu_same_on_round rc=0 md5=0a3c44f700c1e2a72ee787d66b72a88d
gate (all off): 0a3c44f700c1e2a72ee787d66b72a88d
```
All on = all off: one text.

## 2. A bug found: copy-ahead with two copy sets gives garbage on this model

`--moe-stream-copy-ahead 1` picks two copy sets when one set costs at most a quarter of the pool's room, else one set. The set
size changes from run to run (with the live usage table): 232 places (two sets) in one run, 273 / 286 (one set) in the next ones.
With two sets the 10,469-token prompt goes about twice as fast, and its answer is garbage:

| run | copy sets | 10k prompt | 10k answer |
|---|---|---|---|
| `c1_smoke` (value 1) | two (232 places) | 689 tok/s | server error: `vector::_M_range_check: __n (which is 2147483647) >= this->size() (which is 248320)` (the backend sampler's token) |
| `c1_p10k_only`, `c1_r0` (value 1) | one (273 / 286) | 323-324 tok/s, 73-77 GiB read | sane ("Summarise ... five short bullet points") |
| `c1_ca3_chat`, `c1_ca3_nochat` (value 3) | two | 681-696 tok/s | the same server error |
| `c1_ca3_nobs` (value 3, `--backend-sampling` off) | two | 687 tok/s, **3.8 GiB read** | `'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'` |

So with two sets the prompt's expert work is skipped or reads wrong bytes (3.8 GiB read instead of ~75; the logits end as
garbage). The short story prompt (99 tokens) and its reply were sane in every run. Not fixed (this task is commands and runs).
The preset uses one set (`--moe-stream-copy-ahead 2`) in both GPU cases. Flint's preset still has value 1 (two sets on this
laptop) - Flint was measured fine with two sets; the cause on Qwen3.8 is not known (this model's pool has 2 groups of formats).

## 3. The recommended command: `presets/qwen38.sh` (Q12)

`source presets/qwen38.sh; QWEN_CASE=1|2|3 preset_command 65k|262k` prints the llama-server command (Flint's preset style; a
proposal until the user decides). Built from `traces/qwen38/preset/preset_run.sh` (Flint's case-1 flags with this model's data
files), plus `--jinja --chat-template-file traces/qwen38/qwen38-chat-template.jinja`. Changes against Flint's preset:
- **copy-ahead one set (`2`) in both GPU cases** (section 2: two sets give garbage on this model).
- **Whole MoE layers: none named** - the engine's automatic fit decides (65k, case 1: `3 whole MoE layers kept on the GPU (layers
  0, 1, 2), 45 layers streamed` in every run; case 2 and 262k: section 4).
- Left out because they have nothing to act on (evidence below): `--moe-stream-layer-major`, `-window`, `-read-ahead`,
  `-cpu-iqk`, `-cpu-x4`, `-cpu-down-scale`, `-guess-combined`; not added: `-conv-silu`, `-fused-route`, `-correction`.
- Added for the CPU-only case: `--moe-stream-cpu-state-inplace 1` (acts: 36 of 36 linear-attention layers; same text).

### 3.1 Which flags act on this model

"Acts" = a start-up line, an exit line or a per-request counter of a run of this work shows it doing something on Qwen3.8 (run
names in brackets; `c1_r0` = the case-1 preset in llama-server, `cpu_same_on` = the CPU same-output run).

| flag | acts? | evidence |
|---|---|---|
| `--moe-stream-overlap 1`, `-io-wake chunk`, `-promote 1`, `-guess-cap 2`, `-chunk-prefetch 512` | yes (engine-wide, any model) | settings line `io-wake chunk, promote on, guess-cap 2 ...`; read timing lines `guess chunks 16710` (cpu_same_on) |
| `--moe-stream-pred-shared 1` | yes | the guess for hyper-connection models (Part 2); `guess precision [decode]: 76.21 %` (cpu_same_on) |
| `--moe-stream-policy decay`, `-prompt-end recency` | yes (engine-wide) | settings line `evict-halflife 64.0 ... prompt-end recency (halflife 128.0)` |
| `--moe-stream-plan 1`, `-drop-stale 1` | yes | `list [decode] per word: added 53.39, bound 45.59 ... stale 7.80` (c1_smoke) |
| `--moe-stream-needed-early 1` | yes | GPU: `wordstart on the GPU producer ... queued by the split hook 1084`; CPU: `in-graph CPU form ... 47 layers` |
| `--moe-stream-lookahead 2` | yes | `lookahead: 23848 guess calls ... far guess names made 233060` (c1_r0) |
| `--moe-stream-wordstart 2` | yes on the GPU case (whole layers 0-2 are its sources); **0 on CPU-only** | `wordstart: ... layer 2 d1: 542; layer 2 d2: 542; layer 1 d2: 542` (c1_r0); CPU: `wordstart: D 0 (clamped from 2: the layers before the first streamed one are 0)` |
| `--moe-stream-proj-ready 1` | yes | `proj-ready [decode] per word: needed reads split (gate+up, then down) 12.58 ...` (c1_r0) |
| `--moe-stream-protect-named 1` | yes | `protect-named [decode] per word: guess victim choices the rule changed ... 1.65` |
| `--moe-stream-guess-inflight 4` | yes | `in flight at a needed read: 1.14 reads` (cpu_same_on) |
| `--moe-stream-guess-combined 1` | **no** | `guess-combined: 0 streamed layers build their guesses in one step ... 44 layers with 2+ targets keep the separate ops` (each target mixes its own guess input) |
| `--moe-stream-cpu-skip-zero 1` | yes where prompt waves run on the CPU | `prompt-split: 5 prompt ubatches (2 .. 511 rows, CPU waves)` (c1_r0); CPU-only prompts run in CPU waves |
| `--moe-stream-cpu-fuse 1`, `-cpu-dispatch 1` | yes | `fused gate+up steps 1489`, `one-token split ops 1489` (cpu_same_on) |
| `--moe-stream-cpu-mmid-merge 1` | yes | `activation unpacks saved in the tiled MUL_MAT_ID 18156` (c1_r0) |
| `--moe-stream-cpu-topk 1` | only on a machine with no GPU; no counter | same text on = off (1.2) |
| `--moe-stream-cpu-q8-tiled 1`, `-cpu-fa 1` | CPU-only prompts / a q8_0 cache; no counter; the 1.2 check had an f16 cache (cpu-fa had nothing to act on there) | same text in 1.2 |
| `--moe-stream-cpu-weighted-sum 1`, `-cpu-small-ops 1` (CPU-only) | weighted-sum yes; small-ops: settings line only (its grouped pick has no groups here; its CONCAT split has no counter) | `--moe-stream-cpu-weighted-sum: 48 of 48 MoE layers fuse`; `cpu-small-ops on` |
| `--moe-stream-cpu-state-inplace 1` (CPU-only) | yes | `--moe-stream-cpu-state-inplace: 36 of 36 recurrent layers on the CPU device ...: their delta step reads and writes the state in place` |
| `--moe-stream-conv-silu` | **no** | only `bailingmoe3.cpp` uses it (qwen4exp's GDN conv already feeds its SiLU directly) |
| `--moe-stream-cpu-x4` | **no** | the 4-row kernel is for plain MXFP4 tensors only (`docs/flags.md`); this file's experts are Q3_K / Q4_0 / Q2_0 |
| `--moe-stream-cpu-down-scale` | **no** | `the down-input scale step left out in 0 layer builds, kept in 0` (Part 2's preset run) - mistral4 / mellum / flint only |
| `--moe-stream-cpu-iqk 8` | **no** | `experts computed with ik's kernels 0 (0 rows)` (Part 2's preset run) |
| `--moe-stream-layer-major 1`, `-window 1` | **no** | the arch gate `llama_arch_layer_major` (Ling, Flint only); Part 2: `window: 0 calls` |
| `--moe-stream-read-ahead 1` | **no** | by the code (`remap_guess`): only in a call of >= 4096 tokens' remap; without layer-major a remap sees one ubatch (2048) |
| `--moe-stream-wave-plan 1` | yes (multi-wave prompts) | start-up line `--moe-stream-wave-plan 1`; the 10k prompt runs 88 GPU waves per call |
| `--moe-stream-idle-load 2` | yes | `loading: experts into RAM done: 10485 experts ... 10485 of 10485 slots filled` before the first request |
| `--moe-stream-prompt-defer 1` | yes | `176 prompt remap ops left their wait to mul_mat_id` (c1_r0) |
| `--moe-stream-gpu-waves 512` | yes | `tail waves: this call's GPU waves 88` |
| `--moe-stream-gpu-share-q8 1` | by design on any model (q/k/v and the GDN projections read one input); no counter | settings line `gpu-share-q8 on` |
| `--moe-stream-gpu-skip-zero 1`, `-gpu-wave-skip-zero 1` | yes | `gpu-skip-zero on`; `74712 GPU prompt matmuls ... marked to skip their zero-slice pairs` |
| `--moe-stream-async-h2d 1` | yes | `async-h2d (--moe-stream-async-h2d 1): on - CPU -> GPU split inputs ... async` |
| `--moe-stream-prompt-split 1` | yes | `prompt split: 82 layer jobs, 2298 experts wanted on the GPU, 2203 copied` |
| `--moe-stream-vram-free 200MiB`, `-vram-experts-file` | yes | `VRAM experts: 20 per layer x 45 streamed layers + 474 tail places`; the list file read |
| `--moe-stream-vram-follow 1`, `-vram-copyback 1` | yes | `VRAM follow: ... swaps 3242`; `copy-backs 1283` |
| `--moe-stream-vram-parallel 1` | yes | `vram-parallel: 45 of 45 streamed layers had the device pass in their launch` |
| `--moe-stream-vram-end-fill 1` | yes | `end fill 1: 3 fills, 1369 fill jobs` |
| `--moe-stream-copy-ahead 2` | yes | `copy-ahead: on, one copy set; 440 jobs, 60707 experts copied ahead` |
| `--moe-stream-vram-lend-prompt 1` | yes | `7 prompt calls planned their GPU compute buffer into lent VRAM pool places` |
| `--backend-sampling` | yes | `Backend sampler selected token` (debug lines of Part 2's run) |


## 4. The first numbers (Q13)

All through llama-server, started by `bench/queue/qwen38/b3_srv.sh` (the preset's command; plain llama.cpp = the same build
without any `--moe-stream` flag, the ORIGINAL two-file GGUF mapped, `-fit on` with `-ngl` unset on the GPU (it used 10.9 GB of
VRAM), `-ngl 0` for CPU-only; the same `-c 65536`, q8_0 K/V, `-fa on`, `-b / -ub 2048`, template, memory cgroup). Requests
(`b3_client.py`): the story prompt (99 tokens) with 512 forced tokens streamed (per-token times from the arrival of each token:
p99 and the mean of the slowest 1 %), then the 10,469-token prompt `rec/p8k.txt` + 32 tokens. GPU power check OK before / after
every GPU series (`series_gpu.log`, `series_rest.log`, `series_final.log`; lowest 67 % of the limit at 1,800 MHz in the middle
check of the GPU series, the rest 94-99 %). **Measured while the PC was in use** (the user's apps open: free RAM 19.8-24.9 GiB
at the starts, so the RAM slot counts move between runs - each row gives them). One context size: 65k. The live usage table was
updated by every run (`--moe-stream-idle-load 2`, the preset's behaviour).

| case (65k, q8_0 K/V) | engine | slots per layer (RAM; VRAM) | reply tok/s (story, 512) | slowest 1 % / p99 (ms per token) | reply SSD wait ms/token | 10k prompt tok/s | GiB read for it | reply after it tok/s |
|---|---|---|---|---|---|---|---|---|
| 1: this laptop, GPU 12 GB + free RAM | **ours**, 5 runs | 201-236; 21-22 VRAM; whole layers 0-2 | **32.1** (30.1-33.9) | 71.0 / 57.3 | 3.9 | **259** (240-275) | 93 | 18.2 |
| 1 | plain llama.cpp, 2 runs (96 tokens) | - | **3.15** | 721 / 721 | - | **~10** (8,192 tokens in 810 s, stopped at 15 min) | - | - |
| 2: 8 GB free VRAM (`vram_hold 8192`) + 11 GiB cap | **ours**, 3 runs | 96; 22 VRAM; no whole layer | **16.1** (14.9-16.9) | 122 / 108 | 30.6 | **172** (170-174) | 166 | 12.3 |
| 2 | plain llama.cpp, 2 runs (96 tokens) | - | **3.0** | 965 / 965 | - | **< 6** (2,048 tokens in 352 s, no further batch by 900 s) | - | - |
| 3: CPU-only, 11 GiB cap | **ours**, 3 runs (128 tokens) | 56 | **6.6** (6.5-6.8) | 193 / 189 | 11.0 | **30.7** (1 run) | 192 | 5.9 |
| 3 | plain llama.cpp, 2 runs (96 tokens) | - | **0.31** | 3,533 | - | killed by the 11 GiB cap at the 10k prompt's start (`oom-kill`, journal) | - | - |
| 3: CPU-only, free RAM | **ours**, 3 runs (128 tokens) | 161-173 | **6.75** (6.5-6.9) | 194 / 188 | 1.7 | **35.6** (1 run) | 138 | 6.8 |
| 3 | plain llama.cpp, 2 runs (96 tokens) | - | **1.4** (1.1-1.7) | 1,626 | - | **9.4** (8,421 tokens in 900 s, stopped) | - | - |
| theirs: SlenderEngine skip 0, its defaults (their PC: i7-14700KF, RTX 4070 Ti 12 GB, 32 GB DDR5, two 990 PRO; IQ3_S; 262K; int8 K/V; DFlash2 drafting) | - | 10 GiB RAM tier | 46.5-49.5 (short answers); 35.5-41.4 after a 50K prompt | - | - | 2,064-2,145 on a 50K prompt (2,258-2,296 with its author's launcher) | - | - |

Theirs is quoted from `docs/qwen38/README.md` "The bar" (their published numbers, their machine, another quant, with drafting:
their laptop page put "MTP off" at -19..-22 %, i.e. about 37-40 tok/s without drafting - the study's estimate). Not a same-machine
comparison. Ours has no drafting.

What the rows say (plain words):
- Replies: ours ~10x plain llama.cpp's on the GPU case, ~5x on the small card, 21x (11 GiB) and ~5x (free RAM) on CPU-only. The
  10k prompt: ~26x on the GPU case, > 29x on the small card, 3.8x on CPU-only with free RAM (plain: 9.4 tok/s); under 11 GiB plain
  could not run it.
- The reply is compute-bound on CPU-only with free RAM (SSD wait 1.7 ms of ~148 ms per token) and mostly so on the GPU case
  (SSD wait 3.9 of ~31 ms); the small card and the 11 GiB CPU case wait on the SSD more (30.6 and 11.0 ms/token).
- The 10k prompt reads 93-192 GiB (each ubatch of 2,048 reads most experts again): layer-major prompts (Q14) are what changes that.
- Short prompts are slow on every case (the story's 99 tokens at 31-47 tok/s on the GPU, 17-23 on the CPU).
- Per-run spread: reply 30.1-33.9 tok/s on the GPU case with the slot count moving 201-236; slowest 1 % 52.6-98.1 ms.

### 4.1 The n-gram table's file cache over a longer session (case 1)
`c1_session`: the 10k prompt + 400 tokens, then a second request that continues the same chat (23 new tokens, 10,868 reused, 200
tokens; the reply sane). The engine's per-request cap line:
```
moe stream: process memory cap 30159 MiB, peak use 22469 MiB, margin to the cap 7690 MiB     (after request 1)
moe stream: process memory cap 30159 MiB, peak use 22488 MiB, margin to the cap 7671 MiB     (after request 2)
```
The peak grew 19 MiB over the second request; nothing was given back. Under the 11 GiB cap (CPU-only) the peak line reached the cap
after the 10k prompt (`peak use 11264 MiB, margin to the cap 0 MiB`, `c3cap_run1`) without a kill: the peak counts file cache the
kernel can free (the mapped tables' pages among it); the run went on and answered. Not split further.

### 4.2 Placement the engine picked (whole layers: its own fit)
- Case 1, 65k: `3 whole MoE layers kept on the GPU (layers 0, 1, 2), 45 layers streamed` (every run).
- Case 1, 262k (`c1_262k`): the first fit took layers 0-2, then the context did not fit and common loaded the model again with
  fewer: `1 whole MoE layers kept on the GPU (layers 0), 47 layers streamed`; it answered the chat request (`391`, thinking split)
  and a 64-token story at 27.8 tok/s (one run).
- Case 2 (8 GB free), 65k: no whole layer, 22 VRAM experts per layer, 96 RAM slots under 11 GiB. 262k for case 2 not run (the
  lead trimmed the runs).

### 4.3 The perfect-guess run: not usable
The oracle needs a recording of the same streamed layers, so the pair ran with all 48 layers streamed (`--moe-stream-gpu-layers 0`;
with the preset's whole layers the engine refuses: `the oracle trace covers 48 MoE layers, the engine streams 45`) and
`MOE_STREAM_TEST_ORACLE_LIST=1` (the oracle feeds the waiting list; the settings line names the trace).
```
c1g0_base   story512 reply 37.04 tok/s | reply SSD wait 3.72 ms/tok, 34.8 MiB/tok    (242 slots)
c1g0_oracle story512 reply 29.79 tok/s | reply SSD wait 3.81 ms/tok, 25.1 MiB/tok    (258 slots)
```
Both replies leave the recording's text after 495 characters (of ~2,200): the recording (`rec/06_story`) was made with another
build and placement, so after that point the oracle names the wrong experts for every token. The run therefore does not measure
perfect guesses; I stopped there as asked (no new recording, no fix). A usable form needs a recording made with the same command
and build (or a forced text, as `llama-moe-decode-batch --force-text` does for Ling).

## 5. The preset runs as written
`asis`: `CMD=$(source presets/qwen38.sh; preset_command 65k); eval "exec $CMD"` (port 8080), one chat request:
```
chat reasoning_content: 'We need answer user: "What is 17 times 23? Answer briefly." Need compute 17*23 = 391. Brief.'
chat content: '391'
```
Thinking split into `reasoning_content`, the answer right. (Cases 2 and 3 run through the same function with `QWEN_CASE`; every
number of section 4 came from it.)

## 6. Limits and doubts
- One context size measured (65k); 262k only started and answered once (case 1).
- Plain llama.cpp's GPU split is its own `-fit on` choice (not printed at the default verbosity; 10.9 GB of VRAM used); its 10k
  prompt ran 15 minutes before the lead's 5-minute limit arrived (the rate is reported).
- More runs were made than the lead's trimmed plan (it arrived after they had run): 3 instead of 2 CPU-only runs per memory size, 2
  plain runs per case; all are reported, none left out.
- Per-token times come from the arrival of streamed tokens at the client (HTTP included).
- `cpu-topk`, `cpu-q8-tiled`, `cpu-fa`, `cpu-small-ops`' CONCAT part, `gpu-share-q8`: no counter proves them on this model.
- The 10k prompt's speed on the GPU case moved 240-275 tok/s with one copy set; with two sets it is ~690 but wrong (section 2).

## 7. For the lead to decide / follow-ups
1. **The copy-ahead two-set bug** (section 2): fixed on `qwen-fix`, section 9 (not merged).
2. The preset is a proposal (copy-ahead 2, the automatic whole-layer fit, the flags of section 3.1).
3. A usable perfect-guess run needs a recording made with the measured command (4.3).
4. Q14 (layer-major) would cut the 93-192 GiB a 10k prompt reads today; notes in `reports/qwen38-q14.md`.

## 8. Time
About 3 h of work (16:15-18:50 plus the earlier start), most of it model runs.

## 9. The copy-ahead bug: cause and fix

Taken over from a worker whose session was lost about 15 minutes in (its uncommitted test checks kept, see 9.2). Engine: worktree
`llama.cpp-qwen`, branch `qwen-fix` (from `moe-lab` `175feb0f9`). Runs: `bench/queue/qwen38/cafix_run.sh` (the case-1 preset's
flags in llama-completion or the eval-callback tool, memory cgroup, through the lock), outputs `traces/qwen38/cafix/`.

### 9.1 Cause (measured)
**The CUDA quantized expert matmul reads past the end of every expert's last row** - its rows are padded to 512 values (the buffer
type's allocation adds `ggml_row_size(type, 512 - ne0 % 512)` bytes after a tensor for this), so the last row of the expert in place
p reads the first bytes of place p+1. Qwen3.8's down projection has ne0 = 640 (Q4_0): 384 values = 216 bytes past every place. Values
there meet zero activations, so a normal next expert adds 0 - but **a NaN / Inf block scale there gives 0 x NaN = NaN**. The tail's
copy places are shared by both format groups (each with its own slice size) and by both copy sets, so a place the wave does not use
holds stale bytes of another format at another offset, whose "scales" are often NaN. The engine assumed those bytes were a valid
expert (the comment at the tail tensors said so); with two groups that is false.

Evidence, on a 1,795-token prompt (`cafix/p1k9.txt`, one 2048 ubatch): value 3 = `'!!!!...'` (`p1k9_ca3`), value 2 = a sane
start (`p1k9_ca2`: "We need answer user's request: \"Summarise the following design notes in five short bullet points.\" ...").
Then the eval-callback tool asking only for chosen tensors (a local diagnostic patch, not committed; `-c 4096` so that its
all-token logits leave room for the VRAM pool - at 65k the pool was off and the bug did not run):
```
ecb_f3 (ffn_moe_out of every layer): the first non-finite values in layer 3, the first streamed layer:
ECB ffn_moe_out-3 [2560,1795,1] sum -250.55 nonfinite 57 first 33279 (row 12)
ECB ffn_moe_out-4 [2560,1795,1] sum 0 nonfinite 4595200 first 0 (row 0)        (everything after it NaN)
ecb_f4 (layer 3's MoE tensors): waves 0 and 2 clean; wave 1 (set 1): gate / up / swiglu clean, down not
ECB ffn_moe_down-3 [2560,10,1795] sum 79.1662 nonfinite 72 first 12799 (row 4)
ecb_f6 (that down matmul: weight, ids, and the bytes after each bad place):
ECB   weight g1.down type q4_0 ne [640,2560,225] nb1 360 nb2 921600 ...
ECB   place 35: 42 nonfinite; next place 36 used by this wave 0; its first 16 block scales: 2 non-finite (first: -nan -0.0180511 56.3438 4828 8188 1.99902)
ECB   place 199: 55 nonfinite; next place 200 used by this wave 0; its first 16 block scales: 1 non-finite (first: -0.0797729 nan ...)
  ... 14 bad places, every one: the next place unused by the wave, NaN scales in its first row
ECB   places whose first row has a non-finite scale: 87 (of them used by this wave: 0)
```
Every NaN is in output column 2559 (the expert's last row). The early checks agree that nothing else is wrong: with
`MOE_STREAM_TEST_COPY_AHEAD_CHECK=1` (`p2k_nj_chk2`, copy-ahead jobs off) every place a wave names equals its RAM slot after the
gather (2,190 places, 0 differ), the zero places it trusts are zero, the graph's matmul reads the tensor the gather fills (132 of 132)
and the last step's places are unchanged at the next gather (6,540 projections, 0 changed) - and the answer was still `!!!!`.
Forcing every step into one set (`MOE_STREAM_TEST_CA_SET=0` or `=1`, jobs off) gave a non-garbage end (`p2k_set0`, `p2k_set1`);
alternating gave garbage - alternating leaves more unused places next to used ones holding the other group's bytes.
Toggles that did not change the garbage: async-h2d, vram-parallel, prompt-defer, CUDA graphs off, CUDA fusion off (`p2k_*`), and the
lost worker's lend-prompt 0 / gpu-wave-skip-zero 0 / prompt-split 0 (`b_*`).

Why Flint and Ling were not hit: Flint's down ne0 = 512 (no padding read); Ling has one format group (the stale bytes are aligned
MXFP4 blocks of real experts).

### 9.2 The fix (branch `qwen-fix`)
- `4a42b18a8` test checks (kept from the lost worker and extended): `MOE_STREAM_TEST_COPY_AHEAD_CHECK=1` reads back every place a wave
  names, the trusted zero places, the graph's tail tensor and the last step's places; `MOE_STREAM_TEST_CA_NO_JOBS`, `_CA_SET`.
- `6198e2d9a` **the fix**: the tail layout leaves each copy's row padding (the buffer type's allocation size minus the tensor's size,
  216 bytes for Qwen3.8's down; 0 for a format that needs none - then the layout is as before) inside the copy places, and every
  tail-wave gather writes zeros there and into the first bytes of each place after a read one that the wave does not fill (on the
  compute stream, before the matmuls; ~35 small writes per wave). The copy-ahead rule that keeps another format's job away from this
  layer's down parts now counts the padding too.
- `9a7f8c315` **the check against the same kind of mistake anywhere**: every prompt call and every 16th reply call of a streamed
  model reads 256 logits of the last output row; NaN / Inf fails the call (`llama_decode` -3) with one line and drops its sequences'
  memory. Over-read bytes meet zero activations, so they either add exactly 0 or make NaN - this check sees every case of the
  class. Test switches `MOE_STREAM_TEST_NO_ROW_PAD=1` (the writes off), `MOE_STREAM_TEST_OUT_NAN=1` (forced failure),
  `MOE_STREAM_TEST_PAD_CHECK=1` (after each call: held VRAM places followed by an empty place, that place's first bytes
  dequantized and checked).
- `ba543b7e9` without tail waves (other backends, merged gate|up models), a format whose rows need padding takes no pool copies in
  the tail (one start-up line; they stay in the compute buffer, as in llama.cpp), because the gathers that clean the bytes do not run
  there.
- Docs: `docs/flags.md` (three rows), `llama.cpp-qwen/docs/settings.md` (the switches, the output check).

First checks (llama-completion, `cafix_run.sh`, 1,795-token prompt):
```
p1k9_fix_ca3  (two sets, fixed):  "We need answer user's request: \"Summarise the following design notes in five short bullet points.\" Need produce final ..."
              prompt 277.7 tok/s; tail-wave gather ... 7364 copy calls ... (4601 of them zeros for the matmuls' row padding, 654.2 KiB)
p1k9_nopad    (two sets, MOE_STREAM_TEST_NO_ROW_PAD=1): rc 1, no text -
              E decode: the streamed model's output holds non-finite values (256 of 256 logits of the call's last row are NaN / Inf) -
                the expert computation went wrong; this call fails instead of giving a wrong answer
p1k9_padchk   (two sets, MOE_STREAM_TEST_PAD_CHECK=1, 40 words, a 999.6 MiB lend in the prompt): after every call at most 91 held
              places followed by an empty one, 0 of them starting with non-finite values; the text = p1k9_fix_ca3's
```

### 9.3 Checks
**(a) + (b): the 10k prompt (10,469 tokens + 300 words) through the case-1 preset in llama-server** (`bench/queue/qwen38/cafix_a.sh`,
`cafix_chain.sh`; `PRESET_BIN` = `llama.cpp-qwen/build/bin` at `ba543b7e9`; the live usage table deleted before each run; GPU power
check OK before: `106 W of its 108 W limit (98 %)`; the PC in use). Outputs `traces/qwen38/b3/fix_*.{log,json,out}`.

| run | copy-ahead | sets (start-up line) | backend sampling | 10k prompt tok/s | GiB read for it | reply tok/s | the answer's end (`.json`) |
|---|---|---|---|---|---|---|---|
| `fix_ca3_bs` | 3 | two (247 places, 440 MiB) | on | 291.2 | 90.9 | 24.8 | "...- Key experiment: “oracle replay” with perfect prefetch to establish the true achievable speed ceiling on the machine." |
| `fix_ca3_nobs` | 3 | two | off | 295.9 | 89.1 | 24.9 | "...- Plan: benchmark machine, capture real routing traces, oracle replay, build lossless llama.cpp MVP, ..." |
| `fix_ca1_bs` | 1 | **one** (258 places, 459.8 MiB; room 1728.2 MiB: 4 x 459.8 > 1728.2) | on | 281.4 | 83.4 | 23.2 | "...- Plan: benchmark machine, capture real routing traces, oracle replay ceiling, build lossless llama.cpp MVP," |
| `fix_ca1_nobs` | 1 | one (260 places) | off | 280.5 | 82.6 | 24.5 | "...</think> - Goal: run Ling-3.0-flash ... - Core constraint: ..." |
| `fix_ca1s200_bs` | 1, 200 RAM slots pinned | **two** (210 places, 374.3 MiB) | on | 270.9 | 106.5 | 22.7 | "...- Plan: benchmark machine, capture real routing traces, simulate cache/predictor hit rates, run oracle replay, then build a lossless llama.cpp MVP ..." |
| `fix_ca1s200_nobs` | 1, 200 slots | two | off | 277.5 | 106.5 | 22.2 | "...- Key experiment: “oracle replay” with perfect prefetch ..." |
| `fix_ca2_bs` | 2 | one (262 places) | on | 260.8 | 81.8 | 25.1 | "...</think> - Goal: run Ling-3.0-flash (124B total, 5.1B active, ~70 GB) on 12 GB VRAM + 30 GB RAM ..." |

Every answer is the five-bullet summary (thinking first; texts differ at the rounding level between runs: the VRAM sets and slot
counts differ). Two sets now read what the prompt needs (89-91 GiB with free RAM, 106.5 GiB at 200 pinned slots - before the fix
3.8 GiB and garbage), and the prompt is faster than one set in this series (291 / 296 against 261-281 tok/s; one run each, PC in use).
Value 1 picked one set at free RAM this time (the set is 258-260 places when the RAM pools are large) and two sets at 200 slots.
Row-padding zero writes per run: 10,042-12,977 at free RAM, 4,514-4,575 at 200 slots (`copy calls (N of them zeros ...)`).

**(b) value 2 unchanged** (`cafix_v2.sh`: llama-completion, the 1,795-token prompt, 200 slots pinned, 40 words): with the row-padding
writes on, off (`MOE_STREAM_TEST_NO_ROW_PAD=1`) and on again the same text, and the same as value 2 on the build before the fix
(`p1k9_ca2`, run by the lost worker's script before any engine change of mine):
```
3e62e6c781c0668d208c19105ea1832d  v2_pad.txt       (copy calls (4408 of them zeros for the matmuls' row padding, 629.8 KiB))
3e62e6c781c0668d208c19105ea1832d  v2_nopad.txt     (copy calls (0 of them zeros ...))
3e62e6c781c0668d208c19105ea1832d  v2_pad_rep.txt
3e62e6c781c0668d208c19105ea1832d  p1k9_ca2.txt     (before the fix)
```
In the server series above value 2 answered the 10k prompt with the summary (`fix_ca2_bs`).

**(c) Flint with two sets** (`cafix_rest.sh`): Flint's case-1 preset (65k, `--moe-stream-copy-ahead 1`) in llama-server from
`llama.cpp-qwen`, the long prompt with the planted fact (`bench/pod2/long_prompt.txt`, one chat request, 64 tokens):
```
vx_setup: --moe-stream-copy-ahead 1: one set of copy places = 262 places, 540.4 MiB; the pool's room 5260.5 MiB -> two sets
prompt_n 7703 prompt tok/s 1871.7 | reply '4417'
contains 4417: True
llama_moe_stream: tail-wave gather copy 82 GPU waves, ... (0 of them zeros for the matmuls' row padding, 0.0 KiB)   (Flint's rows need none)
```
The short check `traces/qwen38/flint_ab2.sh` (64 slots, 64 words): `5be5e4cb7cbfdfc24384bf8a9f25d915` = the expected md5.

**(d) the frozen Qwen texts** (`traces/qwen38/gate_md5.json`, the same three commands on `llama.cpp-qwen`):
```
gate_cpu_120 rc=0 md5=0a3c44f700c1e2a72ee787d66b72a88d
gate_cpu_120_ub12 rc=0 md5=0a3c44f700c1e2a72ee787d66b72a88d
gate_gpu_150 rc=0 md5=6417ddef0d82258c961f2a7449ddc811
expected: cpu 0a3c44f700c1e2a72ee787d66b72a88d (both), gpu 6417ddef0d82258c961f2a7449ddc811
```

**The preset** (`presets/qwen38.sh`): copy-ahead **1** in case 1 (the engine's own rule - two sets when one set costs at most a
quarter of the VRAM pool's room, else one; as Flint's case 1) and **2** in case 2 (one set on the small card, as Flint's case 2:
more VRAM experts per layer; two sets were not measured there). Why 1 and not 3: both are correct now and 3 was faster on the 10k
prompt in this series (291 / 296 against 281 tok/s for value 1 picking one set - one run each, the PC in use, not a measured gain);
choosing a value by speed waits for the tuner (AGENTS.md), so the preset takes the engine's rule. The comment says the preset needs
the fixed engine: until `qwen-fix` is merged it must run with `PRESET_BIN=.../llama.cpp-qwen/build/bin` (the shared build can still
pick two sets and garble).

**(e) `check.py` on Ling-tiny** (`cafix_chain2.sh`: it built every target of `llama.cpp-qwen` first, under the lock):
```
  OK   the build
  OK   ctest  (100% tests passed, 0 tests failed out of 64)
  OK   test_expert_index.py / test_expert_repack.py
  OK   plain run on the CPU / the settings reached the program / streamed run on the CPU read experts from the SSD  (3788 reads)
  OK   streamed text identical to the plain run / waves (rc 0) / streamed run on the GPU
Everything passed.
```
(Its 3.4 GiB experts file in `traces/qwen38/cafix/checkpy/` was deleted after the run.)

**Task 1 done.** Engine commits on `qwen-fix`: `4a42b18a8`, `6198e2d9a`, `9a7f8c315`, `ba543b7e9`, `9caa241a1` (docs). Not merged.

## 10. The perfect-guess run

### 10.1 How (sound by construction)
- **A new recording** with the fixed build and the pair's own placement and flags (`bench/queue/qwen38/t2_rec.sh`, `t2_flags.sh`):
  `llama-moe-trace` with the case-1 preset's engine flags, all 48 MoE layers streamed (`--moe-stream-gpu-layers 0`), 200 RAM slots
  pinned, -c 65536, the story prompt, 600 greedy words -> `traces/qwen38/rec2/06_story.{bin,json}` (rc 0).
- **A forced text**, so that both arms compute exactly the recording's words (free generation leaves any recording after a few
  hundred words: the GPU / CPU split of the experts differs run to run): the recording's 600 decode ids as text through
  llama-server's `/detokenize`, checked by tokenizing it back the way `llama-moe-decode-batch --force-text` does (`t2_text.sh`):
  `decode words 600, text 2530 chars; tokenized back 600 tokens, equal: True`; the prompt: `prompt without the last newline: 98
  tokens, equal to the recording's 98: True` (`-f` drops the file's last newline in both tools).
- **The pair** (`t2_pair.sh`): `llama-moe-decode-batch -f rec/06_story.txt --force-text rec2/06_story.decode.txt --gen 512
  --no-batch` with the same flags and 200 slots in both arms; the oracle arm adds `--moe-stream-oracle rec2/06_story
  --moe-stream-oracle-lead 48` (one word ahead) and `MOE_STREAM_TEST_ORACLE_LIST=1` (its experts go into the waiting list); the usage
  table deleted before every run; arms interleaved; the GPU power check before and after.

### 10.2 The oracle's tally (part 1's 2.1 %): a real off-by-a-prompt bug, fixed
The first smoke runs (32 words) showed the tally counting 14,890 decode picks where 31 x 48 x 10 = 14,880 exist, and a new per-layer
line (`7869bfb1a`, with `MOE_STREAM_TEST_ORACLE_LIST=1`) put the difference in one layer:
```
smoke2_oracle: oracle list: lead 48; 14890 picks of the decode words compared with the trace, 802 differ (5.386%)
               per streamed layer: 0:0/310 1:1/310 ... 45:22/310 46:19/310 47:290/320
```
Cause: llama.cpp's crop keeps only the logit rows in the **last layer** of a prompt call, so that layer's op sees 1 row and its token
count (`n_tok_all`, the oracle's row index) grew by 1 instead of 98. For every later word the oracle then checked **and prefetched**
layer 47 against a prompt row (290 of 310 picks wrong, i.e. layer 47 got no perfect guesses), and the cut row itself was counted as a
decode word (+10 ids). Part 1's `313 of 14910 (2.099%)` at `-b 9` is the same thing: three prompt pieces, +30 ids, layer 47 = 1/48 =
2.08 % of the picks. Fix (`7869bfb1a`): the first streamed layer's op notes the ubatch's rows and a cut layer counts them (its rows
checked as the ubatch's last ones). After:
```
smoke3_oracle: oracle list: lead 48; 14880 picks of the decode words compared with the trace, 617 differ (4.147%)
               per streamed layer: 0:0/310 1:1/310 ... 45:26/310 46:23/310 47:15/310
```
The 4.1 % left is the recording's and the run's rounding (0-2 % in the first layers, growing to 7-9 % in the last ones: the VRAM
expert sets differ between the recording and each run, so the hidden states differ in the last bits and some router picks flip).
The texts cannot be affected (the count feeds only the oracle and the score recorder).

### 10.3 The pair: today's guess against perfect guesses (story, 512 forced words, 200 slots pinned, 3 pairs interleaved)
`traces/qwen38/oracle/{base,oracle}_{1,2,3}.{txt,log}`, `pair.log`. GPU power check before `OK - GPU at 109 W of its 111 W limit
(99 %) ... 2347 MHz` and after `OK - GPU at 111 W of its 113 W limit (99 %) ... 2302 MHz`; AC 1, PPT pl1 41 W; **the PC in use**
(the user's apps open; the GPU clock per run not recorded). Both arms: 200 RAM slots x 48 layers (18.42 GiB), 57 VRAM experts per
layer, the same forced words (`llama-moe-decode-batch`'s `gen:` line, 511 calls of 1 token).

| pair | arm | SSD wait ms/token | ms/token | needed reads (misses) / token | reads / token | oracle picks differing |
|---|---|---|---|---|---|---|
| 1 | today's guess | 6.72 | 29.67 | 5.47 | 25.61 | - |
| 1 | perfect | 1.70 | 31.25 | 0.74 | 14.29 | 2.56 % (layer 47: 173 / 5110) |
| 2 | today's guess | 6.51 | 35.70 | 5.46 | 25.70 | - |
| 2 | perfect | 1.69 | 31.12 | 0.72 | 14.30 | 2.58 % (155 / 5110) |
| 3 | today's guess | 6.52 | 35.73 | 5.45 | 25.72 | - |
| 3 | perfect | 2.09 | 26.16 | 0.72 | 14.30 | 2.52 % (173 / 5110) |
| **mean** | today's guess | **6.58** | 33.70 | 5.46 | 25.68 | |
| **mean** | perfect | **1.83** | 29.51 | 0.73 | 14.30 | |

**What the guess can still win on this model (this laptop, 200 slots, all layers streamed): ~4.8 ms of SSD wait per token** (6.58 ->
1.83; -5.0 / -4.8 / -4.4 in the three pairs - steady), with about half the reads (25.7 -> 14.3 per token: today's guess reads ~11
experts per token that are not used). ms/token moved +1.6 / -4.6 / -9.6 (mean 33.7 -> 29.5, -12 %) - inside this machine's noise
while in use (the same arm moved 29.7 -> 35.7 between pairs), so the SSD wait is the measure here. Perfect guesses still leave 0.73
needed reads per token and ~1.8 ms of wait: 2.5 % of the picks differ from the recording (rounding between the recording and each
run, 0 % in layer 0 growing to ~3-4 % in the last layers), and some guessed reads land late.

The oracle tally disagreement of part 1 shows up here before the fix (10.2: layer 47, +10 ids per prompt call) and not after it
(245,280 = 511 x 48 x 10 picks compared; layer 47 at the level of its neighbours).

Check after the counter fix: the frozen GPU text again on the final build (`cafix/gate_gpu_150_final`): `6417ddef0d82258c961f2a7449ddc811`
= the expected md5.

### 10.4 Limits, doubts, follow-ups (sections 9 and 10)
- Nothing is merged: `moe-lab` and the shared build still have the bug; `presets/qwen38.sh` (value 1) needs
  `PRESET_BIN=.../llama.cpp-qwen/build/bin` until the merge (its comment says so).
- The same byte read can still meet stale bytes in two places I did not make clean, only checked: (1) a held VRAM place followed
  by an empty place at reply time (after tail waves or a lend, before the refill reaches it) - `MOE_STREAM_TEST_PAD_CHECK` found 0
  bad of at most 91 such places over 40 reply words after a lent prompt; (2) without tail waves the tail now refuses the pool copies
  of a format whose rows need padding (not run: needs a backend or model without tail waves). Both are covered by the output check:
  a NaN fails the call within 16 reply words instead of printing garbage (shown with the writes switched off).
- The output check adds one small device read and wait per prompt call and per 16th reply call; its speed cost was not measured.
- AMD (ROCm) runs the same tail code and the same padding rule (`ggml_backend_buft_get_alloc_size`); not run there. CPU-only runs have
  no tail and are not affected.
- llama.cpp's own compute-buffer weight copies do not zero their row padding (`init_tensor` skips compute buffers) - the same kind of
  read; not ours to change, noted.
- The speed numbers are single runs or 3 pairs on a PC in use; the prompt with two sets (291 / 296 tok/s) against one set (261-281)
  is not a measured gain.
- A local diagnostic patch to `llama-eval-callback` (ask only for chosen tensors, report non-finite values, the bytes after a bad place)
  was used for 9.1 and not committed: `/path/to/scratch/ecb_diag.patch`.
- Engine commits on `qwen-fix` (over `175feb0f9`): `4a42b18a8`, `6198e2d9a`, `9a7f8c315`, `ba543b7e9`, `9caa241a1`, `7869bfb1a`.
- Time: about 1 h 15 min (19:20-20:35), most of it model runs.

## 11. The output check's setting and the reply-time padding hole (branch `qwen-fix2`)
Engine: worktree `llama.cpp-qwen`, branch `qwen-fix2` from `7869bfb1a`, its own build dir (llama-completion / llama-server first, every
target through check.py at the end). Runs under `tools/model-lock.sh` (WHO `Qwen-fix2`), memory cgroups as the existing runners; the PC
in use (MemAvailable ~21-22 GiB); every comparison with the same pinned slot count in both arms (Qwen 150, Flint 100, Ling 134). Outputs
`traces/qwen38/fix2/`, scripts `bench/queue/qwen38/fix2_{srv.sh,client.py,flint.sh,ling.sh}`. The uncommitted eval-callback patch the
task mentions was not in the worktree at the start (`git status` clean); the patch file of section 10.4 is still in the scratchpad.

### 11.1 `--moe-stream-out-check N`
- `common/arg.cpp` (the flag; < 0 clamped to 0 with a warning), `common/common.{h,cpp}` (`moe_stream_out_check = 16`), `include/llama.h` +
  `src/llama-model.cpp` (`llama_model_params::moe_stream_out_check`, default 16; `p1.out_check`), `src/llama-moe-stream.h`
  (`out_check_due`: 0 = never, else every prompt call and every Nth reply call), `src/llama-context.cpp` (`llama_context::decode`: a steady
  clock around the get + synchronize + scan, per kind). Exit line (only when the check ran):
  `output check (--moe-stream-out-check N): F calls failed; prompt calls P checked, ... reply calls R counted, C checked, X ms ... (Y us each; times include the wait for the call's remaining GPU work)`.
  `MOE_STREAM_TEST_OUT_NAN` kept. Docs: `docs/settings.md` (a row in "Measuring and debugging" instead of the "Without a flag" sentence),
  `docs/flags.md` (the output-check row is now the flag's row).
- Checks (Flint, `fix2_flint.sh`: the case-1 preset's flags in llama-completion, -c 8192, the 1,795-token prompt, 64 words, 100 slots):
```
flint_oc0_nan  (--moe-stream-out-check 0, MOE_STREAM_TEST_OUT_NAN=1): rc=0 md5=d6f86a416822 (= flint_after), no output-check line, no failure line
flint_nan      (default 16,        MOE_STREAM_TEST_OUT_NAN=1): rc=1, no text
  E decode: the streamed model's output holds non-finite values (0 of 256 logits of the call's last row are NaN / Inf, MOE_STREAM_TEST_OUT_NAN) - ... this call fails instead of giving a wrong answer
  llama_moe_stream: output check (--moe-stream-out-check 16): 1 calls failed; prompt calls 1 checked, 0.0 ms in the checks (28.0 us each); reply calls 0 counted ...
```
- **Cost** (Ling's GPU command at 134 slots = the lead's "Ling GPU 05" gate through `bench/run.sh`, 512 reply words, 44.23 ms/word; GPU check
  before `OK - GPU at 110 W of its 114 W limit (97 %) ... 2362 MHz`, after `OK - GPU at 103 W of its 107 W limit (97 %) ... 2415 MHz`; AC 1, PPT pl1 41 W):
```
output check (--moe-stream-out-check 16): 0 calls failed; prompt calls 1 checked, 0.0 ms in the checks (34.0 us each); reply calls 512 counted,
32 checked, 25.3 ms in the checks (790.5 us each; times include the wait for the call's remaining GPU work)
```
  **32 checked reply calls, 790.5 us per checked call = 49.4 us per reply word at the default (25.3 ms / 512; 0.11 % of 44.23 ms).** That
  is an upper bound, not the cost: the check's wait includes the rest of the call's GPU work (the output layers after the last CPU split),
  which the next sync point waits for anyway; what the check really adds is at most min(that GPU rest, the CPU work after the check in the
  same call). Prompt calls: 28-35 us (their GPU work is done by then). Flint's case-1 flags: 1285.7 us per checked reply call (3 of 63
  words). Qwen's preset (sections 11.2 runs): 52-53 ms over 54 checked calls per run (before the per-kind split). Since the bound is 0.11 %,
  no N = 0 / N = 16 ms-per-word pair was run.

### 11.2 The reply-time path: the hole is real; closed
**Ling has row padding too** (new finding): its MXFP4 down projection has ne0 = n_ff = 768, so the matmul reads 136 bytes past a place
(start-up line `VRAM pool group 0 (mxfp4 / mxfp4): the matmul reads 0 / 136 bytes past a place`); an E8M0 scale byte 0xFF is NaN. Qwen3.8:
group 0 (q3_K / q4_0) 0 / 216 bytes, group 1 (q3_K / q2_0) 0 / 108. Flint: 0 everywhere (no line). So "pad = 0 for Ling" was wrong; Ling's tail
copies are its own format at the places' alignment (finite), but places lent to a prompt call (`--moe-stream-vram-lend-prompt 1`, not in
Ling's recommended command) hold compute-buffer bytes.

What the device pass reads: in a reply graph (and any graph without tail waves) the whole tensor of each group, places named by the id
table (held places and the zero slices); the matmul on held place p reads the first `pad` bytes of p + 1. How a place can hold other bytes
(read from the code):
- the tail (tail group, places [0, tail_n)): the GPU-wave copies of every format, the gathers, copy-ahead's sets, layer-major tensors in its
  extension - only while vacated (no expert held there), in graphs whose view starts after the tail;
- VRAM lending (`lend_lend` / `lend_vacate`): a prompt call's compute buffer in the front of each group's held places [lend_start, +U);
- layer-major tensors (`lm_lend_place`): the front [p0, p0 + used) with lending on, else the end [end - used, end);
- not other bytes: the follower's swaps (the leaving expert's bytes until the incoming copy lands - both the group's format), the fills and
  the end fill (whole experts), prompt split's borrowed places (experts of the tail group's format), the zero slices.
Three ways a held place meets such bytes, all real by the code:
1. **The fill after a vacate or a lend** (`vf_fill_t`) gives the free places to candidates lowest first; when the READY candidates run out
   (or the mark cap), a filled place q sits before an unfilled q + 1 that still holds the other use's bytes, until a later fill reaches it.
2. **A prompt call without tail waves that lends** (65..511 tokens with the preset: `--moe-stream-gpu-waves 512`, lend-short 64): the tail
   group's lent range starts at p0, the tail's last place p0 - 1 can hold an expert (the tail was refilled) and is in that graph's view
   (the tail group's view is not shifted in such graphs) - its padding read meets the call's live compute buffer **during the call**.
3. Layer-major lending at the end of a group (lending off): the held place before the call's tensors, during the call.

The fix (`src/llama-moe-stream.{h,cpp}`), only for groups whose rows need padding (`vx_pad_init` in `vf_setup`: allocation size minus
nbytes per tensor; a group without padding gets no marks, and every new step returns at once - Flint's path is unchanged):
- **marks**: a place whose memory gets another use is marked when that use starts (`vx_pad_mark`): `tail_ubatch` (every tail graph:
  [0, tail_vac_n)), `lend_vacate` (front lending and layer-major front), `lm_lend_place` (the end range, every call);
- **the first place of a range is never written at its first `pad` bytes** (`vx_pad_skip`): the lent pieces (`lend_pieces`) and the
  layer-major tensors start `pad` bytes later (256-aligned), and the planners count it (`lend_places_for`: 512 + pad per group;
  `lm_lend_plan`: `skip[g]`) - this closes way 2 and 3 (the held place before a range only ever reads bytes that were an expert or zeros);
- **zeros before places are held again** (`vx_pad_clean`, at the start of `vf_fill_t`, the only way a free place gets an expert): a
  device-wide wait (the last graph, copies still landing), then zeros into the first `pad` bytes of every marked free place (both tensors
  where they pad; on the calling context's stream, then a wait for them), before any of the fill's copies is queued (the copies wait for
  the fill's event / device wait after that) - this closes way 1; a switched-in expert clears its place's mark (`vf_switch`). A marked
  place of a pending job is counted and skipped (`places of a job found marked`, must be 0: every use that marks places drains the jobs
  first).
Why that is enough: other bytes are written only into marked places (or, at a range's first place, never into its first `pad` bytes);
when a use starts, every place before a marked one is marked too or is the range's first place's predecessor; a free place gets an
expert only through a fill, which zeroes every marked free place first; so a held place is never followed by a marked place whose first
bytes another use wrote.
- exit line: `VRAM places' row padding: F fills wrote zeros into P places (W writes, T ms with their GPU waits); M places marked now; places of a job found marked J (must be 0)`.
- test switches: `MOE_STREAM_TEST_NO_PLACE_PAD=1` (the rule off: no zeros, ranges written from the first byte), `MOE_STREAM_TEST_PAD_PLANT=1`
  (every marked place gets 0xFF in its first `pad` bytes - not the range's first place - after a device wait: the worst bytes another use
  could leave). Docs: `docs/settings.md` (two rows), `docs/flags.md` (a behaviour row + the switches' row).

**The test** (`fix2_srv.sh`: the case-1 preset `presets/qwen38.sh` 65k in llama-server from `llama.cpp-qwen`, `--moe-stream-vram-follow 1`,
150 slots pinned in every run, `MOE_STREAM_TEST_PAD_CHECK=1`; three turns of one chat: the 10,469-token prompt + 300 words, then
follow-ups of 574 tokens (tail waves) and 372 tokens (no tail waves, lent) + 256 words each):
```
r3_plant_off (PAD_PLANT=1, NO_PLACE_PAD=1): PAD_CHECK lines 822, lines with bad places 14
   7 x "(after a decode call): 2 held places followed by a place without an expert, 1 of those places start with non-finite values in the held one's format"
   7 x "(after a decode call): 1 held places followed by a place without an expert, 1 of those places start with non-finite values ..."
   (reply calls 10-16 of the first reply, after the end fill; calls 7-13 after the 574-token follow-up); 4401 places planted, 0 fills zeroed
r2_plant_on  (PAD_PLANT=1):                 PAD_CHECK lines 822, lines with bad places 0
   VRAM places' row padding: 3 fills wrote zeros into 2039 places (2039 writes, 3.1 ms with their GPU waits); 0 places marked now; places of a job found marked 0 (must be 0); 4388 places planted
```
Both answered sanely in all three turns (no output-check failure in either: in the rule-off run the bad places' held neighbours were not
picked in a checked call, or their NaN did not reach a checked last row). Without plants, copy-ahead 1 / 2 / 3 (11.3) also 0 bad places
in 822 checks each. Way 2 itself was not seen by the check in these runs (turn 3's prompt call: 35-40 held places before an empty one,
0 bad even with the rule off - the lent pieces were not in the tail group, or the tail's last place was empty): it is closed by the code
(the skip), not shown by a run.

### 11.3 Checks
1. **Qwen frozen texts** (`qrun.sh`, the gate commands of `gate_md5.json`, BIN = `llama.cpp-qwen/build/bin`):
```
0a3c44f700c1e2a72ee787d66b72a88d  fix2/gate_cpu_120.txt     (frozen 0a3c44f700c1e2a72ee787d66b72a88d)
6417ddef0d82258c961f2a7449ddc811  fix2/gate_gpu_150.txt     (frozen 6417ddef0d82258c961f2a7449ddc811)
```
2. **The 10k prompt with copy-ahead 1, 2, 3** (three-turn runs above; texts are the ends of the first reply / the third reply):
```
r1_ca1_chk (ca 1, two sets; short follow-ups of 23 / 30 tokens): "...- Plan: benchmark hardware, capture real routing traces, simulate cache/predictor performance, build a lossless llama.cpp MVP, then add prefetching, eviction, and miss fallbacks. - Key experiment: “oracle replay” records"
                                                     turn 3: "...choose one experiment to run first. Need likely prioritize measuring CPU MoE compute rate ..."; 1 fill zeroed 1085 places, 1.5 ms
r_ca2_chk  (ca 2, one set):  "...- Plan: measure hardware, capture real routing traces, simulate cache/predictor gains, build a lossless llama.cpp MVP, then add prefetchers and non-blocking miss handling. - Key experiment: “oracle replay” with perfect prefetch"
                              turn 3: "...I would run the Qwen3.6-35B-A3B sanity benchmark first because it is the cheapest way to confirm the llama.cpp build ..."; 3 fills, 1425 places, 2.2 ms
r_ca3_chk  (ca 3, two sets): "...- Prefetch/eviction: use temporal, layer-ahead router, cross-layer, MTP next-token hints; LFU-aging/probabilistic Belady; non-blocking miss fallbacks with quality checks. - Plan: benchmark machine, capture real routing traces, oracle replay, build"
                              turn 2: "...The target laptop has a 12 GB RTX 5070 Ti, Ryzen AI 9 HX 370, 30 GB RAM, and a PCIe 4.0 SSD ..."; 3 fills, 2039 places, 3.2 ms
prompt 229.5-245.7 tok/s, replies 20.4-24.2 tok/s (PC in use; not a speed comparison); output check 0 failed in every run
```
3. **Flint and Ling, one short GPU text against the build before the change**:
   - Flint, the case-1 preset's flags (`fix2_flint.sh`): **in VRAM follow's real mode the text is not repeatable on the build before the
     change either** - the shared build `llama.cpp/build` (= `7869bfb1a`) gave `d6f86a416822` and `2dc420ba9035` in two runs, the first
     run on `llama.cpp-qwen`'s old binaries `2dc420ba9035`, the new build `d6f86a416822` twice (they part at word ~30, both sane; the
     copy-ahead / tail-wave counters differ from run to run). With `--moe-stream-vram-follow-fixed 1` (the test mode for exact texts),
     interleaved: before `d6f86a416822`, after `d6f86a416822`, before `d6f86a416822`, after `d6f86a416822`.
   - Flint `traces/qwen38/flint_ab2.sh` (64 slots, 64 words, no VRAM pool): `5be5e4cb7cbfdfc24384bf8a9f25d915` = the expected md5.
   - Ling GPU 05 (`fix2_ling.sh` = the lead's `merge_mgpu.sh` step 1 with BIN = `llama.cpp-qwen/build/bin`): `8b339cffb421` = frozen
     `8b339cffb421` (that command makes no marks: `0 fills ... 0 places marked`).
4. **check.py on Ling-tiny** (it built every target of `llama.cpp-qwen` first; under the lock; `fix2/checkpy.log`, its 3.4 GiB experts
   file deleted after the run):
```
  OK   the build
  OK   ctest  (100% tests passed, 0 tests failed out of 64)
  OK   test_expert_index.py / test_expert_repack.py
  OK   plain run on the CPU / the settings reached the program / streamed run on the CPU read experts from the SSD  (3795 reads)
  OK   streamed text identical to the plain run / streaming with a pool smaller than the prompt needs (waves) / streamed run on the GPU
Everything passed.
```

### 11.4 Not covered, doubts
- Way 2 (the tail group's front lending in a prompt call without tail waves, with the tail's last place held) and the layer-major
  lending paths were not caught by a run: closed by the code only (layer-major is not built for qwen4exp; Flint has no padding).
- The plant test emulates the worst bytes at every marked place; the real other uses wrote no NaN that a check saw in these runs
  (r1 / ca2 / ca3 with the rule on: 0 bad; the rule-off run without plants was not made).
- `MOE_STREAM_TEST_PAD_CHECK` looks at the state after each call (a call's own lent bytes are still there then, so way 2 would show),
  not inside a graph.
- The output check's cost is an upper bound from its own timer (11.1); not an N = 0 / N = 16 pair.
- AMD (ROCm) runs the same code (`ggml_backend_cuda_device_sync`, async sets on the HIP stream); not run there.
- Commits: engine `qwen-fix2` = `27ad3cb80` (one commit on `7869bfb1a`; not merged, not pushed); `moe-lab` and the shared build not touched.

### 11.5 Follow-ups
- Ling's MXFP4 down needs 136 bytes of row padding, so the earlier commits that act on "a format whose rows need padding" act on Ling too:
  `6198e2d9a` (the tail layout leaves room for the copies' padding: Ling's tail grew by that room) and `ba543b7e9` (without tail waves the
  tail takes no pool copies of such a format - on Ling that is every copy: a backend or setting without tail waves now keeps Ling's
  GPU-wave copies in the compute buffer). Worth a look by the lead (the frozen texts cannot show it: layout, not numbers).
- Ling with `--moe-stream-vram-lend-prompt 1` (the proposed case-2 command for Flint; not Ling's) was exposed to way 1 / 2 before this
  change (E8M0 0xFF = NaN in lent compute-buffer bytes); now covered by the same rule.

Time: the runs went from the Flint before-run (22:34) to check.py's end (22:58); the reading and the code before them were not timed.
