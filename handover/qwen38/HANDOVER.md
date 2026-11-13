# Qwen3.8-Flash-Next on TokenWing: handover to Grawnch (2026-11-14)

This folder (`handover/qwen38/`) is temporary: it sits on the `dev` branch of `runmyai/tokenwing` for the handover and will be
removed again. The engine code is the rest of this repo, as of commit `4dfc5aa38`. The folder holds what the engine repo itself
does not: the small data files, the test scripts as they were run on the lead's laptop, and the reports. Paths below that start
with `notes/`, `data/` or `scripts/` are inside this folder; everything else is in the repo.

Paths in the scripts and notes are placeholders: `/path/to/project` is the folder that holds the engine checkout and our
`traces/` folder, `/path/to/models` holds the model files. Set your own. Some notes point at internal files of ours that are not
in this folder (`reports/...`, `docs/flags.md`, `AGENTS.md`); the repo's `docs/settings.md` lists every setting.

## 1. Where it stands

Your quant (`Grawnch/Qwen3.8-Flash-Next-3.5bpw-GGUF`, arch `qwen4exp`) runs on the engine with the experts streamed from the
SSD and the n-gram table (file 2) left on the SSD.

- **Model file:** file 1 written again with 4096 alignment, weights only (`tokenwing-tools/experts/make_model_file.py`, section 4);
  file 2 is used as it is, next to it. Expert slices are multiples of 512: read direct at the drive's own alignment, with a
  copy path for drives that need 4096.
- **Correct:** on the CPU the text is identical to plain llama.cpp on the same file. Saved chat states restore. The official
  chat template works (thinking split, tool calls 3 of 3); the file's own template is empty, so it is passed as a file.
- **The guess** (`--moe-stream-pred-shared 1`): the wide residual plus the layer's gated shared expert, mixed with the target
  layer's FFN hyper-connection weights. It names 77 % of the next layer's 10 experts (the layer's own router input: 64 %).
  Lookahead 2 and the word-start guesses work. The text never changes with it. Against no guess, 5 pairs: SSD wait -6.0 / -7.2
  ms/token in every pair; ms/token -6 % on code, a tie on math - its mix / combine are plain ops (~740 extra GPU node runs per
  token) and take part of the gain back.
- **Data files** (`data/`): the VRAM list and the usage seed, made from seven recordings; the chat template.
- **The command:** `scripts/presets/qwen38.sh` (65k and 262k; GPU, an 8 GB card, CPU-only). Seven of Flint's preset flags have
  nothing to act on for this model yet and are left out: layer-major, window, read-ahead, cpu-iqk, cpu-x4, cpu-down-scale,
  guess-combined (`notes/qwen38-b3.md`, section 3.1 has the evidence per flag).
- **What the guess can still win:** a run with perfect guesses (story, 512 forced tokens, 200 RAM slots, all 48 layers
  streamed, 3 pairs) has an SSD wait of 1.83 ms/token against 6.58 with today's guess, with about half the reads (14.3
  against 25.7 experts per token). So about 4.8 ms/token of SSD wait is still open on this laptop (`notes/qwen38-b3.md`,
  section 10; the scripts are in `scripts/oracle/`).

First numbers, lead's laptop (RTX 5070 Ti 12 GB, Ryzen AI 9 HX 370, 30 GB RAM, NVMe 5.5 GB/s), 65k context, q8_0 K/V, measured
while the PC was in use (so a lower bound):

| case | ours: reply tok/s / 10k-prompt tok/s | plain llama.cpp, same file and caps |
|---|---|---|
| GPU, all free RAM (201-236 RAM slots per layer, 5 runs) | 32.1 / 259 | 3.15 / ~10 |
| card held to 8 GB free, 11 GiB RAM cap (3 runs) | 16.1 / 172 | 3.0 / under 6 |
| CPU-only, 11 GiB cap (3 runs) | 6.6 / 30.7 | 0.31 / killed by the cap |
| CPU-only, free RAM (3 runs) | 6.75 / 35.6 | 1.4 / 9.4 |

For scale, SlenderEngine publishes 46.5-49.5 tok/s short answers and 2,064-2,145 tok/s on a 50K prompt (its PC, another quant,
with drafting; about 37-40 tok/s without drafting by its own ratio). **Prompts are where this model is furthest behind.**

Where a reply token's time goes (43.3 ms at 190 RAM slots, no VRAM pool): CPU experts 16.3, waits between CPU and GPU 12.7,
SSD wait 10.9, copies 2.0, rest 1.4. A 10k prompt without layer-major: half the time is SSD wait, because every 2048-token piece
reads most of each layer's experts again (93-192 GiB read for 10k tokens).

## 2. Your items

The three items of "our own speed work" for this model, in order. `notes/qwen38-README.md` (the checklist) has each item's
reason and test; `notes/port-needs.md` names files and lines.

1. **Layer-by-layer prompts for `qwen4exp`** (checklist Q14) - the biggest prompt gain. `--moe-stream-layer-major` processes a
   prompt call layer group by layer group, so each layer's experts are read about once; it exists for `bailingmoe3` and `flint`
   and is switched off for this arch (`llama_arch_layer_major`). Needed: the layer-range graph builder, rewind / reapply for the
   hybrid memory with the sparse-attention indexer cache and the k-pool state, the n-gram rows per ubatch, the four streams
   carried between layer groups. Test: CPU text = the normal path's (md5); prompt times at 10k / 32k / 50k; the reply after.
   Notes so far: `notes/qwen38-q14.md`.
2. **CPU kernels for Q3_K, Q4_0, Q2_0** (Q15, Q16): first wire the existing ik_llama.cpp GEMMs for Q3_K and Q4_0 into
   `--moe-stream-cpu-iqk`, then 4-row AVX-512 kernels with ggml's exact output for the three formats, and the fused gate / up ->
   down step. CPU experts are the largest part of a reply token and all of a CPU-only PC's speed.
3. **A correction file for this model's guess** (Q18): the engine refuses Ling's kind (this model has no `ffn_norm`); it needs
   the fold for the guess input above (`W' = W_X + Bt`), the engine's model check, and the training scripts
   (`tokenwing-tools/correction/` is Ling's version).

### How each of the three was done before, and a way into it here

The repo's own guide `docs/adding-a-model.md` describes what Flint's port changed in the code (section 3) and the correction file
(section 5). Read those two sections first; the notes below add this model's side.

**1. Layer-by-layer prompts**
- *What it is:* a prompt call of several pieces is processed layer by layer over the whole call, instead of piece by piece through
  all layers. Each layer's experts are then read about once per call, not once per piece. When it landed on Ling, a 16k / 32k
  prompt took 78 % less time.
- *How Ling and Flint do it:* `src/llama-context.cpp` runs each layer in two phases - phase 1 per piece (attention, router, shared
  expert), phase 2 one expert graph per streamed layer for the whole call. The model's graph builds only the layers it is asked
  for and reads the `lm_*` parameters (`src/models/flint.cpp`, `src/models/bailingmoe3.cpp`). The memory is applied once and
  replayed for the later layers (`rewind` / `reapply`; Flint's window cache keeps each piece's own cell records,
  `src/llama-kv-cache.cpp`). The switch per architecture is `llama_arch_layer_major`. `notes/qwen38-q14.md` has file and line.
- *What is different here:* the hybrid memory with the sparse-attention indexer cache and the k-pool state has no `rewind` /
  `reapply` yet; the n-gram rows are per piece; four hyper-connection streams must be carried between layer groups, where Ling
  and Flint carry one.
- *A way in:* (1) the layer-range graph builder with whole layers per piece (`MOE_STREAM_TEST_LM_GROUP=G`). It runs the same ops
  in the same order as the normal path, so its CPU text must equal the normal path's exactly: this proves the memory replay before
  any speed work. (2) Then the two-phase form; its text may differ at the rounding level, so check KLD and a sane answer.
  (3) Prompt time and GiB read at 10k / 32k / 50k, and the first reply tokens after the prompt.

**2. CPU kernels for Q3_K, Q4_0, Q2_0**
- *What it is:* the CPU computes most of a reply token's experts (16 of 43 ms here), and for your three formats it uses ggml's
  one-row AVX2 kernels.
- *How Ling and Flint do it:* for replies, a 4-row AVX-512 kernel that gives exactly ggml's output (`ggml_vec_dot_mxfp4_q8_0_x4`
  in `ggml/src/ggml-cpu/arch/x86/quants.c`, switched by `--moe-stream-cpu-x4`), and one fused step for gate + up + SwiGLU
  (`--moe-stream-cpu-fuse`). For prompt-size calls, ik_llama.cpp's matmuls (`ggml/src/ggml-cpu/iqk/`, `--moe-stream-cpu-iqk`),
  which today cover Q4_K, Q5_K, IQ4_XS, IQ3_S and Q8_0.
- *A way in:* (1) small: ik's Q3_K and Q4_0 matmuls are already in the ported files (`iqk_gemm_kquants.cpp`,
  `iqk_gemm_legacy_quants.cpp`); wire them into the format switch in `iqk_port.cpp`. (2) large: 4-row kernels for the three
  formats with ggml's exact output. (3) the fused step for these formats. `notes/port-needs.md` section E has the table of what
  exists per format.
- *Checks:* a kernel test where 4 rows per call must be byte-equal to 4 one-row calls of ggml's kernel
  (`scripts/kernels/kernel_bench.cpp` is our test tool for Ling's format, as a pattern); the CPU frozen text unchanged; CPU time
  per token and CPU-only tok/s before and after.

**3. The correction file**
- *What it is:* a small trained file that sharpens the guess of the next layer's experts: the routers' weights plus a low-rank
  term, applied to the guess input. The text never changes with it; only fewer experts are read late. On Ling it saved
  1.4 ms per token on math at a small RAM size.
- *How Ling does it:* recordings of real runs with the engine's recorder (`--moe-stream-record`), training with
  `tokenwing-tools/correction/` (`correction_fit.py`, `make_correction.py`, `check_correction.py`), loading with
  `--moe-stream-correction FILE`; the engine checks the file against the model (`src/llama-moe-correction.cpp`).
- *What is different here:* this model has no `ffn_norm`, which the file format and the engine's check both use, so the engine
  refuses the file today. The guess input is the hyper-connection mix (`--moe-stream-pred-shared 1`), and the shape is 512
  experts, top 10, no groups, layers 0-47.
- *A way in:* (1) keep the guess input as it is now - it must not change once training starts. (2) A second form in the file
  format and the engine's check for this kind of model (`W' = W_X + Bt`; `notes/port-needs.md` B2). (3) The training scripts for
  this shape. (4) Checks: how often the guess is right with and without the file, on recordings the training did not see
  (`scripts/q10/erec_recall.py`); the frozen texts unchanged; needed reads and SSD wait in 5 pairs.

**Also yours: improvements and speed-ups**, kernels first, for the engine in general or for this model. Two places to start.

What our measurements say:
- A token is about 1,000 small GPU steps; the card's memory bandwidth is barely used, the time goes into the steps' fixed cost
  and into the waits between CPU and GPU (12.7 of 43 ms here; on an AMD card under ROCm far more: `notes/amd-rx9060-speed.md`).
- Fewer, bigger steps: Strata and SlenderEngine fuse the hyper-connection mix into one kernel (one launch pair per layer). The
  same idea, written for our engine, would also remove the guess's ~740 extra node runs per token
  (`notes/ideas-from-other-engines.md` P1 and D10).
- Two ideas the study set aside, worth a second look: one GPU graph per token with the CPU handing results over through shared
  memory (set aside because page-locked memory fails on the lead's laptop), and grouped expert gathering on the GPU (set aside
  "unless a profile shows launch-bound GPU time" - the AMD profile shows that).
- Tools: `GGML_CUDA_PROFILE=1` (GPU time per node kind), `GGML_TRACE`, `--moe-stream-timing 1` in llama-server
  (`timings.moe_stream.time_per_word`: SSD wait / CPU / GPU / hand-off per token); `docs/settings.md` in the repo.

What Strata and SlenderEngine do on this model: `notes/others-today.md` is how they run it today, and
`notes/ideas-from-other-engines.md` lists each idea of theirs with what they measured, how it would fit our engine and what to
check first. `notes/reading-notes/` has our notes on their code, with file and line for every claim. We use their ideas, not their code: an idea is worth building when it fits our design and is implemented properly
in our engine, with our checks.

Not yours for now: the ports to other systems, the autotuner.

## 3. Fixed just before this handover (in the commit above)

1. **Garbage / NaN after a long GPU prompt.** The CUDA / ROCm quantized matmul reads its rows padded to 512 values, so the
   last row of the expert in place p reads the first bytes of place p+1 (this model's down projection, ne0 = 640: 216 bytes
   in one format group, 108 in the other). Left-over bytes of another format there decode to NaN scales, and 0 x NaN = NaN.
   It showed with `--moe-stream-copy-ahead` and two copy sets. The engine now keeps those bytes clean: the tail-wave
   gathers write zeros there, the VRAM pool's places get zeros before experts are held next to them again, and places lent
   to a prompt call leave their first bytes alone. `notes/qwen38-b3.md` sections 9 and 11.
2. **`--moe-stream-out-check N`** (default 16, 0 = off): every prompt call and every Nth reply call reads 256 logits of the
   last row; NaN / Inf fails the call with one line instead of printing a wrong answer. At most 0.05 ms per reply token.
   If you ever see `the streamed model's output holds non-finite values`, a matmul read a wrong byte: that is a bug to report.
3. **The perfect-guess tally** was off by a prompt in the last layer (a layer cut to the output rows); fixed.

Not covered: none of this was run on an AMD card. Two of the three reply-time cases of item 1 are closed by the code only
(no run triggered them, even with the fix switched off). Test switches for all of it: `MOE_STREAM_TEST_PAD_CHECK`,
`_PAD_PLANT`, `_NO_PLACE_PAD`, `_NO_ROW_PAD`, `_OUT_NAN` (`docs/settings.md` in the repo).

## 3b. Traps when testing (not engine bugs)

- `llama-completion` (the test tool, not the server) stops when a prompt piece ends on `<|im_end|>` with a small `-b`. This is
  upstream llama.cpp behaviour; it only matters to test scripts.
- The recording tool (`llama-moe-trace`) at `-b 2048` asks for the logits of every prompt token, 2 GB with this vocabulary:
  record with a small `-b`.
- A run with `--moe-stream-idle-load` saves what it learned as `<model>.usage.json` next to the model, and later runs start
  from it instead of the seed (this is how the engine is meant to work). Delete that file before runs that must be comparable.

## 4. Build, files, run, checks

Build (Linux, NVIDIA, a CUDA 13 toolkit; in a CUDA 12 build `--moe-stream-async-h2d` keeps itself off and says so) - see
`docs/getting-started.md` in the repo:
```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON
cmake --build build -j 16
```
The model file (55 GB more disk, ~10 minutes with its read-back check; file 2 next to file 1):
```bash
python3 tokenwing-tools/experts/make_model_file.py Qwen3.8-Flash-Next-3.5bpw-00001-of-00002.gguf \
    -o Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf
```
Put `data/Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf.usage-seed.json` next to the model; pass
`--moe-stream-vram-experts-file data/qwen38.vram-experts.json` and `--jinja --chat-template-file data/qwen38-chat-template.jinja`.

Run: every command has `--moe-stream model --load-mode dio`, inside a memory cap
(`systemd-run --user --scope -q -p MemoryMax=<RAM - 1 GiB> -p MemorySwapMax=0 -- ...`). **Never open the model with a tool that
loads it whole** (`llama-gguf`, a default mmap load): it does not fit in RAM. The scripts in `scripts/` are as run on the
lead's laptop, with the paths replaced by the placeholders above (`qrun.sh` = one llama-completion run; `presets/qwen38.sh` = the server commands;
`b3/` = the series behind the table above; `q10/pairs5.sh` + `p5_table.py` = a 5-pair measure; `q10/erec_recall.py` = recall
of a guess input from `--moe-stream-record`).

Checks before a PR:
- `python3 tokenwing-tools/testing/check.py --model <a small MoE GGUF>` (Ling-tiny: `noctrex/Ling-3.0-tiny-MXFP4_MOE-GGUF`)
  must end with "Everything passed."
- The frozen Qwen texts: `data/gate_md5.json` has the commands and the md5s from the lead's laptop. **They depend on the CPU's
  instruction set and the build**: on your machine run them once on the unchanged code to get your own reference, then a path
  that should be unchanged must still give it. A new speed path on the CPU must give the normal path's text; on the GPU
  rounding-level differences are accepted (garbage, NaN, a wrong answer are bugs).
- Speed claims: SSD wait and ms/token over 512 tokens, 5 pairs, arms interleaved, same RAM slot count in both arms
  (`--moe-stream-slots N`); say the machine.

## 5. Sending work back

- Branch from `dev`, one piece per pull request against `dev`. The lead reads the whole diff, runs your branch through the
  Qwen checks and the Ling and Flint frozen-text gates on the laptop, and merges when they pass.
- House rules for engine code: every new behaviour is a flag, off by default, with its row in `docs/settings.md`; nothing
  sized or named for one model in shared code (sizes come from the model's hparams); a resource limit never crashes - the
  engine falls back to the slower way with one clear line; vendor-specific GPU code stays behind its vendor check so the other
  vendor's build does not change; no method that changes the model's answers (no expert skipping, top-k cuts, drafting);
  no work labels, dates or private paths in code, comments, messages or commit messages.
