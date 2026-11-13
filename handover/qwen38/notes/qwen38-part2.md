# Qwen3.8-Flash-Next on TokenWing, Part 2 (Q9, Q-P0, Q10, Q11)

Worker report, written item by item. Engine: worktree `llama.cpp-qwen`, branch `qwen` (from `82cf5c5ad`, Part 1's end), build
`llama.cpp-qwen/build`. Model: `traces/qwen38/Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf` (+ link to file 2), untouched.
Chat template: the official one, `traces/qwen38/qwen38-chat-template.jinja` (9,004 B, sha256 `af16f654...6375`; the lead's message).

## State

| item | state | commit |
|---|---|---|
| Q9 recordings and counts | **done** (g1 from the recordings here; g1 / g2 / g3 from the engine's recorder in Q10): **g2 chosen** | (no code) |
| Q-P0 profile | **done** (tables below) | (no code) |
| Q10 the guess for hyper-connection models | **done**: checks passed; 5 pairs: g2's SSD wait -6.0 / -7.2 ms/token in every pair, ms/token -6 % on code, a tie on math | `2ab40851d`, `0ce414d5e`, `4addd3453` (settings page rows, the lead) |
| Q11 VRAM list + usage seed | **done**: both load, the idle load fills every RAM slot from the seed | (no code; data files) |

The 5-pair measure of ms/token with no guess / g1 / g2 is done (the table at the end of Q10, added by the lead: the worker's
session was ended by an app restart at ~13:28 while `traces/qwen38/q10/pairs5.sh` ran detached). **g2 stays the choice.**

## Q9: recordings and counts

**Recordings** (`traces/qwen38/rec/`, no engine code): `llama-moe-trace --dump-router-input` on the GPU (`rec.sh`: `-ngl 99 -c 2048
-t 8 -n 700`, the engine's own whole-layer pick, inside the memory cgroup, through the lock), seven bench prompts
(`bench/prompts/flash/NN.messages.json`, the set the Flint recordings used plus 02) rendered through the official template with
`make_prompts.py` (jinja2; Hugging Face's `{% generation %}` tag removed, it only marks training tokens). The template adds its
default system line ("Reasoning effort is set to xhigh ...") and opens the reply with `<think>\n`. Greedy, stops at end of text:

| recording | prompt | decode | note |
|---|---|---|---|
| 01_casual_chat | 99 | 537 | ended |
| 02_technical_explain | 121 | 700 | |
| 03_python_code | 119 | 700 | |
| 04_cpp_code | 106 | 700 | |
| 05_math_word_problem | 109 | 483 | ended before 512 |
| 06_story | 98 | 700 | |
| 10_json_output | 134 | 314 | ended before 512 |
| **all** | 786 | **4,134** | |

```
0.34.017 main: traced 636 tokens (99 prefill, 537 decode) in 31.77 s, 20.02 t/s       (01)
0.40.320 main: traced 592 tokens (109 prefill, 483 decode) in 38.12 s, 15.53 t/s      (05)
0.31.627 main: traced 448 tokens (134 prefill, 314 decode) in 27.31 s, 16.40 t/s      (10)
```
The first 10_json_output run was killed by the machine's out-of-memory killer at 12:33 (another job loaded a whole model file
into RAM; the lead's message); its partial files were deleted and it was recorded again.

**Counts** (`count_q9.py`, numpy, through the lock; decode tokens only; the routers `blk.N.ffn_gate_inp.weight` read from the
model file). Sanity check first: the router of layer L on the recorded input of layer L gives the recorded picks
(`d=0 w10 0.9999`), so the dumped vector is the router input and the reading is right.

*Spread of expert use per layer* (4,134 decode tokens x 10 picks per layer):

| | min over layers | mean | max |
|---|---|---|---|
| experts used at all (of 512) | 456 | - | 511 |
| share of picks taken by the top 10 % of experts (51) | 0.288 | 0.391 | 0.514 |
| share taken by the top 25 % (128) | - | 0.663 | - |
| experts that cover 50 % / 80 % / 90 % of picks | 52 / 128 / 201 | 79 / 193 / 264 | 118 / 261 / 332 |

Layers 0-2 are the flattest (top 10 % = 0.29-0.31, 80 % needs 244-261 experts), layers 40-47 the most skewed (0.49, 140).
Neither uniform nor a small hot set: about 200 experts per layer cover 80 % of the picks.

*Recall of g1* (the guess input today's shared code uses without `--moe-stream-pred-shared`: layer L's own router input with the
router of layer L+d), share of the 10 real picks of layer L+d inside the guess's top W:

| d (layers ahead) | W 10 (the default width = top-k) | W 16 | W 20 | layers 0-11 (W 10) | 12-35 | 36-47 | worst source layer |
|---|---|---|---|---|---|---|---|
| 1 | **0.651** | 0.782 | 0.824 | 0.615 | 0.659 | 0.676 | 0.287 (layer 0 -> 1) |
| 2 | **0.563** | 0.687 | 0.735 | 0.556 | 0.573 | 0.546 | 0.328 (1 -> 3) |
| 3 | **0.517** | 0.636 | 0.684 | 0.497 | 0.528 | 0.514 | 0.145 (0 -> 3) |

(For comparison only, other models, other inputs: Flash's router trick recall@8 0.757 / 0.672 / 0.611 at d 1 / 2 / 3.)

*Missing experts per layer per decode token* for one LRU pool of S slots per layer (all 48 layers counted, the pool warmed by
the prompt; LRU stands in for the engine's eviction, so this is a count of what a pool of that size cannot hold, not a speed
number), and the share of those misses that g1's 1-ahead guess (W 10) named:

| S slots | misses / layer / token | misses / token (48 layers) | named by g1 (d 1, W 10) |
|---|---|---|---|
| 120 | 1.576 | 75.6 | 0.533 |
| 138 | 1.296 | 62.2 | 0.524 |
| 159 (CPU run) | 1.024 | 49.2 | 0.513 |
| 190 (GPU run) | 0.723 | 34.7 | 0.492 |
| 205 (GPU, today's free RAM) | 0.613 | 29.4 | 0.483 |

At 205 slots layers 0-2 miss 1.5-2.2 per token and layers 44-47 0.3-0.5 (the spread above).

**Order (a deviation from the brief, decided here):** g2 and g3 need the 4-stream residual after a layer's attention, the
layer's FFN scatter weights and its gated shared expert - none of which `llama-moe-trace` records (it records the 10 ids and
the router input). Instead of a second dumping tool plus a numpy copy of `hc_mix`, the g2 / g3 recall is measured with the
engine itself: Q10's hook builds g2 (the `--moe-stream-pred-shared 1` path) and g3 (a test switch), and the engine's recorder
(`--moe-stream-record`) writes the guess logits per lead next to the real picks. g1 is measured both ways (here and there), which
also checks the engine recorder against this count. The table with all three is in Q10.

## Q10: the guess for hyper-connection models

**What changed** (branch `qwen`, commits `2ab40851d` and `0ce414d5e`):
- `src/llama-graph.{h,cpp}` (shared code): `moe_guess_finish_in(model, il, base, guess_in, g)` - a model gives its own base and,
  with `guess_in` (a function `X -> tensor`), its own guess input per target layer X. It is used by the 1-ahead guess (`pred_in`),
  the far guesses (`build_far`), the recorder's leads and the word-start guesses (`build_ws_guess`, through `moe_ws_layer`'s new
  optional argument) instead of "base normed with ffn_norm(X)" (new field `llm_graph_rec_leads::in[d]`). The far / recorder inputs
  are built only where a one-token graph reads them (each is a few ops; unused ones would only fill the graph's context).
  `moe_guess_finish` (Ling, Flint) now calls it with `ffn_inp + shared` and no `guess_in`: the same nodes in the same order.
- `src/models/qwen4exp.cpp` (+ `models.h`): every layer calls `moe_guess_routers` / `moe_guess_finish_in` / `moe_ws_layer` and
  passes the guesses to `build_moe_ffn`. `--moe-stream-pred-shared 0` = **g1** (the layer's own router input, no new op);
  `--moe-stream-pred-shared 1` = **g2**: base = the wide residual after the layer's attention plus the layer's gated shared expert,
  scattered with the layer's FFN weights (`build_hc_combine`), and the guess input for X = `build_hc_mix` with X's FFN
  hyper-connection weights (plain ops, `il -1`: no fused node registered for a tensor the graph may not use); the shared expert is
  then built before the routed experts (new `build_layer_shexp`). Sized from the model: the stream count is `dsv4_hc_mult` from the
  hparams, nothing is sized for Qwen3.8. Test switch `MOE_STREAM_TEST_GUESS_IN_ATTN=1` = **g3** (X's attention weights).
  The routers' softmax gating is now in the hparams (the file has no key; the word-start guesses read it - found by the check below:
  the first word-start run aborted in `build_moe_probs`, "fatal error", gating 0).
- `src/llama-moe-stream.cpp`: `qwen4exp` joins `arch_leads` (`--moe-stream-lookahead` > 1 and `--moe-stream-wordstart` work).
- `docs/flags.md`: a behaviour row and the test switch's row. `docs/other-machines.md`: a row (plain ggml ops, every backend).

**The choice of input (Q9's open part)** - engine recordings (`traces/qwen38/q10/erec_all.sh`: `--moe-stream-record`, all 48
layers streamed, `--moe-stream-slots 150`, 512 forced words, prompts 01 / 03 / 04 / 05 in the official chat format; counted by
`erec_recall.py`, the leads read as `predict/t91_analyze.py` reads them). Share of the 10 real picks inside the guess's top W:

| input | d 1, W 10 | W 16 | W 20 | d 2, W 10 | W 16 | d 3, W 10 | W 16 | d 1 layers 0-11 / 12-35 / 36-47 |
|---|---|---|---|---|---|---|---|---|
| g1 (own router input) | 0.642 | 0.771 | 0.814 | 0.553 | 0.674 | 0.508 | 0.623 | 0.610 / 0.650 / 0.661 |
| **g2 (residual + shared expert, X's FFN mix)** | **0.774** | **0.894** | **0.923** | **0.666** | **0.792** | **0.567** | **0.684** | 0.722 / 0.794 / 0.788 |
| g3 (residual + shared expert, X's attention mix) | 0.572 | 0.707 | 0.759 | 0.510 | 0.636 | 0.471 | 0.593 | 0.580 / 0.597 / 0.506 |

```
g1 words 2044 d1 w10 0.6422 w16 0.7711 w20 0.8143 (L0-11 0.610 L12-35 0.650 L36-47 0.661)
g2 words 2044 d1 w10 0.7742 w16 0.8943 w20 0.9233 (L0-11 0.722 L12-35 0.794 L36-47 0.788)
g3 words 2044 d1 w10 0.5715 w16 0.7067 w20 0.7590 (L0-11 0.580 L12-35 0.597 L36-47 0.506)
```
g2 is the best at every distance and width; g3 is below g1. g1 here (0.642) agrees with Q9's count on the trace recordings
(0.651, other texts): the engine recorder and the offline count agree. **The guess reads g2 (`--moe-stream-pred-shared 1`)**;
`0` stays g1 (the engine's default, as on every model).

**Check: the guess never changes the text** (`q10/q10_checks.sh`, and the word-start runs after the gating fix):
```
0a3c44f700c1e2a72ee787d66b72a88d  cpu_noguess.txt     (CPU gate qwen_cpu_120, --moe-stream-prefetch 0)
0a3c44f700c1e2a72ee787d66b72a88d  cpu_g1.txt          (guess on, g1)
0a3c44f700c1e2a72ee787d66b72a88d  cpu_g2.txt          (g2)
0a3c44f700c1e2a72ee787d66b72a88d  cpu_g2_la2.txt      (g2 + plan + drop-stale + lookahead 2: "far guess names" in the log)
0a3c44f700c1e2a72ee787d66b72a88d  cpu_g2_ws.txt       (g2 + lookahead 2 + wordstart 2, on 0ce414d5e)
```
= the frozen md5 of Part 1 (made on a build with no guess at all). On the GPU, every arm of every comparison gave one text:
the 12 engine recordings (`01 ccf056cc241c`, `03 98285658a27d`, `04 b62e34f96bce`, `05 0c966664697d` for g1 = g2 = g3), the 8
SSD runs below (`3a7710d46b06`, `266f360d3ad4` for all four arms), and the word-start / combined-step runs:
```
a1932e93238ac881195223981c6f051f  ws_ref2.txt     (prefetch 0)
a1932e93238ac881195223981c6f051f  ws_g2.txt       (g2 + lookahead 2 + wordstart 2: "wordstart D 2: words with adds 63"; layer 3's
                                                    needed reads per word 0.556 -> 0.365)
a1932e93238ac881195223981c6f051f  ws_g1comb.txt   (g1 + lookahead 2 + guess-combined 1: "41 streamed layers build their guesses
                                                    in one step"; with g2 the combined step keeps the separate ops - no ffn_norm)
```
Flint (the shared refactor of `moe_guess_finish`), `flint_ab2.sh` (64 slots, -n 64, greedy):
```
5be5e4cb7cbfdfc24384bf8a9f25d915  flint_shared_ps1.txt   (shared build llama.cpp/build, --moe-stream-pred-shared 1)
5be5e4cb7cbfdfc24384bf8a9f25d915  flint_q10_ps1.txt      (this build, --moe-stream-pred-shared 1)
5be5e4cb7cbfdfc24384bf8a9f25d915  flint_q10.txt          (this build, defaults; = Part 1's Flint md5)
```
Ling-tiny `check.py` (`--no-ctest --no-build`, through the lock): `Everything passed.` (plain CPU, settings, streamed CPU 3795
reads, streamed = plain, waves, GPU).

**Check: needed reads and SSD wait per decode token** (`q10/ssd_ab.sh`: the GPU command with whole layers 0,1,2,4 and 190 slots
pinned, 512 forced words, arms interleaved per prompt, one run per arm; `tools/gpu-cap-check.sh` before `OK - GPU at 105 W of its
110 W limit ... 2392 MHz` and after `OK - GPU at 116 W of its 117 W limit ... 2362 MHz`; AC online 1, PPT 41 W). The decode
phase's own counters (`moe stream: decode tokens = 511, misses = ..., read = ..., stall = ...`):

| prompt | arm | needed reads / token | read MiB / token | SSD wait ms / token | ms / token | prompt SSD wait ms / token |
|---|---|---|---|---|---|---|
| 01 chat | no guess (`--moe-stream-prefetch 0`) | 19.83 | 38.5 | 14.71 | 46.00 | 34.2 |
| 01 chat | g1 | 12.33 | 69.0 | 11.91 | 43.68 | 36.0 |
| 01 chat | **g2** | **7.57** | 54.8 | **8.63** | 48.20 | 35.3 |
| 01 chat | g2 + lookahead 2 (+ plan, drop-stale) | 7.13 | 68.2 | 7.59 | 50.17 | 34.9 |
| 05 math | no guess | 42.94 | 83.6 | 25.99 | 60.39 | 32.7 |
| 05 math | g1 | 27.11 | 137.1 | 25.70 | 59.70 | 32.1 |
| 05 math | **g2** | **18.90** | 121.4 | **21.58** | 56.28 | 32.9 |
| 05 math | g2 + lookahead 2 | 18.16 | 141.7 | 19.26 | 66.64 | 31.7 |

Needed reads per token: g2 -62 % (chat) / -56 % (math) against no guess; SSD wait per token -41 % / -17 %. Both checks pass.
(Part 1's "about 51 ms/token" was the whole run, prompt included; these are the decode phase alone, with 190 pinned slots.)

**Not settled - a decision with a real cost:** ms/token does not follow the SSD wait in these single runs: g2 is +2.2 ms/token
on chat and -4.1 on math against no guess, and g2 + lookahead 2 is slower than no guess on both (+4.2 / +6.3). One run per arm
is inside the noise our rules name (paired tok/s swung +-14 % between runs), so these four numbers prove nothing either way. What
g2 adds on the word's serial path is real, though: the shared expert moves before the routed experts, plus one `hc_mix` per
target per layer (two matmuls of 10,240 x 324 / 320 x 10,240 and ~10 small ops; x2 with lookahead 2). Whether g2's lower SSD
wait pays in ms/token needs the 5-pair measure (SSD wait and ms/token over >= 512 words, arms interleaved) - the lead's call;
not run here. The profile (Q-P0) shows the GPU time of these ops.

**The 5-pair measure (added by the lead from the raw logs, `q10/p5_table.py`)** - `q10/pairs5.sh`: 5 rounds x (no guess, g1, g2)
interleaved, math and code prompts, 512 words (`-n 512 --ignore-eos`, greedy), the GPU command of `ssd_ab.sh` (whole layers
0,1,2,4, 190 slots pinned, no VRAM expert pool, `-ub 512`); GPU check before `OK - GPU at 114 W of its 116 W limit ... 2505 MHz`
and after `OK - GPU at 112 W of its 113 W limit ... 2385 MHz`; all 30 runs rc=0. Decode phase, mean of 5 (per-round values in
`p5_table.py`'s output):

| prompt | arm | needed reads / token | SSD wait ms / token | ms / token | against no guess, per round |
|---|---|---|---|---|---|
| 05 math | no guess | 42.95 | 26.72 | 62.17 | |
| 05 math | g1 | 27.11 | 25.27 | 64.88 | SSD wait -1.45 (5 of 5 lower); ms/token +2.70 (0 of 5 lower) |
| 05 math | **g2** | 18.91 | **20.75** | 62.71 | SSD wait **-5.97** (5 of 5); ms/token +0.54 (2 of 5 lower: +3.64 -0.18 +3.67 -5.43 +1.01) |
| 03 code | no guess | 30.69 | 21.13 | 59.83 | |
| 03 code | g1 | 17.87 | 17.72 | 57.29 | SSD wait -3.40 (5 of 5); ms/token -2.54 (5 of 5 lower) |
| 03 code | **g2** | 11.73 | **13.92** | 56.20 | SSD wait **-7.21** (5 of 5); ms/token **-3.63** (4 of 5 lower: -4.50 +3.48 -2.83 -12.61 -1.67) |

g2 against g1, ms/token per round: math -2.16 (4 of 5 lower), code -1.09 (4 of 5 lower).

Reading: g2 lowers the SSD wait in every pair (-6.0 / -7.2 ms/token, -22 % / -34 %); in ms/token it is ahead on code (-6 %) and a
tie on math, and ahead of g1 on both. The SSD-wait saving does not arrive in full in ms/token: the guess's own work on the token's
serial path (the shared expert first, the combine and one mix per target layer in plain ops, ~740 node runs per token) takes part
of it back. **g2 stays the guess for this model; fusing its mix / combine ops is the named next gain** (the fused
hyper-connection kernels of the checklist's copy part cover the same ops). Texts: math one text in all 15 runs; code one text in
13 runs and another in 2 (`p5_g1_..._r5`, `p5_g2_..._r5`): those two started with less free RAM and got 189 slots instead of 190
(`190 slots per layer (--moe-stream-slots) do not fit ... 189`), so the prompt's waves differ - accepted rounding, not the guess
(the same arms gave the common text in rounds 1-4).

## Q-P0: one profile (decode and a 10k-token prompt, GPU)

All runs (`traces/qwen38/qp0/qp0_all.sh`, through the lock): `-ngl 99 -c 16384 -b 2048 -ub 512 -t 8`, whole layers 0,1,2,4 and
190 slots pinned, `--moe-stream-pred-shared 1` (g2) unless named; `tools/gpu-cap-check.sh` before `OK - GPU at 113 W of its 114 W
limit ... 2550 MHz` and after `OK - GPU at 114 W of its 115 W limit ... 2467 MHz`; AC 1, PPT 41 W. The long prompt is
`rec/p8k.txt` (the first 31 KB of `docs/research.md` behind "Summarise ... in five short bullet points", official template):
**10,468 tokens** with this tokenizer (aimed at 8k; 32k not run - this prompt alone took 2.7 min, see below). One run each: a
picture, not a benchmark. No conclusions drawn here beyond the numbers.

**Decode, where a word's time goes** (llama-server `--moe-stream-timing 1`, `qp0/srv_profile.sh`: a 100-token chat prompt, then
255 forced words; `timings.moe_stream.time_per_word`):
```
{"words": 255, "calls": 255, "ms_per_word": 43.268, "ssd_wait_ms": 10.940, "cpu_ms": 16.256, "gpu_ms": 12.674, "handoff_ms": 14.723,
 "rest_ms": 1.351, ... "handoff_copies_ms": 2.002, "handoff_syncs_ms": 12.721}
```
| part (decode) | ms / word | share of 43.27 |
|---|---|---|
| SSD wait (the engine's wait for reads) | 10.94 | 25.3 % |
| CPU work (the host backend's compute minus the SSD wait: the CPU experts of the 44 streamed layers + small CPU ops) | 16.26 | 37.6 % |
| hand-off: waits between CPU and GPU (syncs) | 12.72 | 29.4 % |
| hand-off: copies | 2.00 | 4.6 % |
| rest | 1.35 | 3.1 % |
| (GPU kernels busy, beside the above - overlapping, not added) | (12.67) | (29.3 %) |

Reply counters of that request: 7.80 needed reads / word, 60.9 MiB read / word, RAM hit 98.2 %.

**Decode, GPU node time by kind** (`GGML_CUDA_PROFILE=1`, llama-completion, chat prompt, 127 words, **g1**; CUDA graphs are off in
this mode and a node's time runs from its event to the next one, so launch gaps are inside; grouped by node name with
`qp0/opprof_sum.py`; "unnamed" = nodes without a name, listed by weight type):

| kind (decode, g1) | ms / word | share of GPU node time (17.63 ms / word) |
|---|---|---|
| other small ops (unnamed, views, copies) | 2.29 | 13.0 % |
| linear attention (Gated DeltaNet: scan, conv, its projections that have names) | 2.24 | 12.7 % |
| shared expert (+ its gate) | 2.19 | 12.4 % |
| hyper-connection mix / combine (`hc_*`, fused HC ops) | 1.98 | 11.2 % |
| full attention + sparse indexer (12 layers: Q/K/V/out, indexer score / top-k / mask, flash attention, RoPE) | 1.64 | 9.3 % |
| router (logits, softmax, top-k, weights) | 1.54 | 8.7 % |
| unnamed matmul q6_K (4,572 = 36 per word: one per linear-attention layer) | 1.41 | 8.0 % |
| unnamed matmul q8_0 (12,321 = 97 per word: two per layer + the head - the hyper-connection down / inject matmul) | 1.21 | 6.8 % |
| unnamed matmul bf16 | 1.17 | 6.7 % |
| output head (q6_K, 248k vocabulary) | 0.91 | 5.2 % |
| guess (next layers' routers, g1) | 0.77 | 4.4 % |
| routed experts on the GPU (whole layers 0,1,2,4) | 0.28 | 1.6 % |

The same profile with **g2** (`prof_chat_g2`): 27.0 ms / word of GPU node time against 17.6, with every kind higher, also the ones
g2 does not touch (linear attention 2.24 -> 3.14, output head 0.91 -> 1.25) - so between two runs this mode's times move with
more than the guess; what is solid is the node count: g2 adds **~740 node runs per word** (285,128 -> 378,854 over 127 words, of
which "other small ops" +618 per word, hyper-connection +84, unnamed q8_0 matmuls +42 = the guess's down matmul in each streamed
layer: the guess's unfused `hc_mix` and the base's combine, built with plain ops - see Q10's open point).

**The 10,468-token prompt** (GGML_TRACE, `trace_p8k_g2`: 166.0 s = 63.0 tok/s; the trace summed over the prompt's window by
`qp0/trace_sum.py`; the server run of the same prompt: 164.3 s):

| part (prompt) | s | share of 166.0 s |
|---|---|---|
| CPU splits (the streamed experts' CPU work + the engine's wait for reads inside them) | 97.09 | 58.5 % |
| - of it the remap ops = the engine's wait for reads (SSD wait; the counters say 80.4 s) | 81.03 | 48.8 % |
| - so CPU work | ~16.1 | ~9.7 % |
| GPU splits (host time from issuing a GPU split to the next split) | 68.68 | 41.4 % |
| copies of experts to the GPU for its wave matmuls (`copy experts`, nested in the splits) | 31.15 | 18.8 % |
| synchronous input copies (`copy sync`, 13,788, nested) | 32.60 | 19.6 % |
| waits (noinput 5.36, dep_in 5.31, war 3.64; nested) | 14.31 | 8.6 % |
| the n-gram rows (the graph's first CPU split, `node_0`: token rows + PLE gather, 22 calls) | 0.20 | 0.1 % |
| (GPU pieces busy, union of issued -> done) | (15.18) | (9.1 %) |

SSD side of the prompt (server counters): 208,847 needed reads = 19.9 per token, **406 GB read** (38.8 MiB / token), hit rate
27.8 %, SSD wait 7.68 ms / token. Without layer-major prompts for this arch (Q14) every 512-token ubatch reads most of each
layer's experts again.

GPU node time in the prompt by kind (`GGML_CUDA_PROFILE`, `prof_p8k_g2`, token class 9+, 12.95 s in all):

| kind (prompt) | s | share of GPU node time |
|---|---|---|
| routed experts on the GPU (wave matmuls + whole layers: gate / up q3_K 5.46 + 0.48 s, down q2_0 1.72, q4_0 1.04 + 0.21) | 9.40 | 72.6 % |
| linear attention (Gated DeltaNet scan 0.39 s + its matmuls) | 0.85 | 6.5 % |
| full attention + sparse indexer (flash attention 0.36 s) | 0.65 | 5.0 % |
| hyper-connection mix / combine | 0.46 | 3.6 % |
| other small ops | 0.45 | 3.5 % |
| unnamed matmuls (q6_K 0.33, q8_0 0.24, bf16 0.20) | 0.77 | 5.9 % |
| router | 0.18 | 1.4 % |
| shared expert | 0.14 | 1.1 % |
| guess | 0.05 | 0.4 % |

## Q11: the data files from the recordings

**Files** (separate small files next to the model, nothing appended to the model file), made with the existing tools from the
seven Q9 recordings:
```
$ python3 tokenwing-tools/experts/make_vram_list.py traces/qwen38/qwen38.vram-experts.json rec/01_casual_chat ... rec/10_json_output --top 128
wrote qwen38.vram-experts.json: 48 layers x 128 experts, 4134 decode words of 7 recordings
$ python3 tokenwing-tools/experts/make_usage_seed.py traces/qwen38/Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf.usage-seed.json rec/...
wrote ...usage-seed.json: 48 layers x 512 experts, 4920 tokens, 7 recordings, scaled to 163840 picks per layer
```
- `traces/qwen38/qwen38.vram-experts.json` (60,635 B, "tokenwing vram-experts v1", arch qwen4exp, 128 most-picked experts per layer
  from the reply words; the engine continues a list in expert-id order when the free VRAM holds more). Given with
  `--moe-stream-vram-experts-file`.
- `traces/qwen38/Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf.usage-seed.json` (114,057 B, "tokenwing usage v1"; prompts +
  replies, 4,920 tokens): the name the engine looks for next to the model (`<model>.usage-seed.json`), read while the live table
  `<model>.usage.json` does not exist.

**Check** (`q11/q11_check.sh`, through the lock: the GPU command - whole layers 0,1,2,4, g2, `-c 4096`, chat prompt, 64 words - without
and with `--moe-stream-idle-load 2 --moe-stream-vram-free 200MiB --moe-stream-vram-experts-file traces/qwen38/qwen38.vram-experts.json`):
```
llama_moe_stream: --moe-stream-vram-experts-file '.../qwen38.vram-experts.json': 44 streamed layers list 128 of their 512 experts: their
    lists go on with the rest in expert-id order, so the free VRAM alone sizes the pool
vx_report: VRAM placement: whole MoE layers 0,1,2,4 (4); VRAM experts 25 per layer x 44 layers + 10 zero slices = 2.136 GiB
    (.../qwen38.vram-experts.json); free after the context 0.333 of 11.588 GiB
llama_moe_stream: usage table: the seed '.../Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf.usage-seed.json' (3592304 picks after
    halving); idle load on (empty RAM slots filled after 500 ms without a call); read-ahead off
llama_moe_stream: loading: experts into RAM 0 % (0 of 9944, 0.0 of 20.4 GB, 0.0 s)
...
llama_moe_stream: loading: experts into RAM 100 % (9944 of 9944, 20.4 of 20.4 GB, 4.8 s)
llama_moe_stream: loading: experts into RAM done: 9944 experts read into the RAM pools of 44 streamed layers, 9944 of 9944 slots filled;
    the most-used experts of each layer by the seed '...usage-seed.json'
print_stats: idle load: 1 rounds, 9944 experts read (19410.8 MiB) into empty slots, ... 5935 used by a call, 3363 dropped unused;
    picks of a slot that was not empty 0 (must be 0)
moe stream: VRAM expert hits 40.553/token (18.09% of the expert uses), decode 92.905/token
```
The first request against the run without the files (same command otherwise; one run each):

| | prompt misses / token | prompt hit rate | prompt SSD wait ms / token | prompt eval | decode hit rate | VRAM hits / decode word |
|---|---|---|---|---|---|---|
| without | 83.97 | 0.00 % | 34.02 | 6.00 s / 99 tokens | 98.40 % | - |
| with the seed + VRAM list | 21.47 | 72.44 % | 8.80 | 4.15 s / 99 tokens | 97.31 % | 92.9 (of 440 uses; 25 of 512 experts per layer) |

The texts differ (`d651ece3df3d` / `a1932e93238a`): the VRAM experts are computed by the GPU (rounding; the flag says so) and the
pools differ in the prompt's waves - accepted rounding, both sane.

**Note for the next run:** that run saved the live table `traces/qwen38/Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf.usage.json`
(the engine's normal behaviour with idle load: counted live, halved at each start). From now on the engine reads it instead of
the seed; deleting it brings the seed back.

## Extra: a first run with the preset settings that apply (one pass, no tuning)

`traces/qwen38/preset/preset_run.sh` (llama-server, through the lock; GPU check before `OK - GPU at 113 W of its 117 W limit ...
2512 MHz` and after `OK - GPU at 115 W of its 117 W limit ... 2422 MHz`): Flint's case-1 command from `presets/flint.sh` (65k
context, q8_0 K/V, `-fa on`, `-b 2048 -ub 2048`, `--cache-ram 0 --ctx-checkpoints 2`, its `GGML_CUDA_*` variables, all its
`--moe-stream-*` flags), with two changes:
- `--moe-stream-vram-experts-file model` -> `traces/qwen38/qwen38.vram-experts.json` (Q11): this model file holds the weights only,
  no appended data (Part 1, the user's decision).
- `--moe-stream-gpu-layers-list 43,44,45,46,47,48` (Flint's own layers) left out: the engine's automatic whole-layer fit picked
  layers 0, 1, 2 (no value chosen by me).

Flags kept in the command that have nothing to act on for this model (from the log; the engine switches them off itself):
- `--moe-stream-layer-major 1` and `--moe-stream-window 1`: layer-major prompts are not built for qwen4exp (the arch gate
  `llama_arch_layer_major`; Q14) - `window: 0 calls`.
- `--moe-stream-cpu-iqk 8`: ik's kernels exist for none of this file's expert formats (Q15) - `experts computed with ik's kernels 0`.
- `--moe-stream-cpu-x4 / -cpu-fuse / -cpu-dispatch / -cpu-q8-tiled / -cpu-mmid-merge / -cpu-down-scale`: not checked whether they act on
  Q3_K / Q2_0 / Q4_0 experts (no line names them in this run).
Not in Flint's preset, so not added: Ling's `--moe-stream-cpu-state-inplace` / `--moe-stream-conv-silu` (this model has 36
linear-attention layers like Ling's; for a CPU-only command later), `--moe-stream-correction` (refused: no `ffn_norm`).

Placement and memory: `3 whole MoE layers kept on the GPU (layers 0, 1, 2), 45 layers streamed`; `RAM: 23953 MiB available, 0.50
GiB kept free ... 19.95 GiB of pools = 233 slots per layer x 45 layers`; `VRAM experts 21 per layer x 45 layers + 10 zero slices =
2.539 GiB`; KV 816 + indexer 204 MiB; prompt compute buffer 1063.9 MiB lent from the VRAM pool (`VRAM lending: ... 7 prompt calls`).
The live usage table (Q11's run) filled the RAM pools before the first request.

```
story400 prompt_n 99 prompt 41.4 tok/s predicted_n 400 reply 35.94 tok/s prompt SSD wait 8.37 ms/token reply SSD wait 5.01 ms/token
p10k prompt_n 10469 prompt 289.7 tok/s predicted_n 32 reply 20.03 tok/s prompt SSD wait 1.73 ms/token reply SSD wait 20.64 ms/token
```
| case | prompt | reply | reply SSD wait |
|---|---|---|---|
| story, 400 words after a 99-token prompt | 41.4 tok/s | **35.9 tok/s** (27.8 ms / token) | 5.01 ms / token |
| 10,469-token prompt (the "8k" prompt), then 32 words | **289.7 tok/s** (36.1 s) | 20.0 tok/s | 20.64 ms / token |

Against Q-P0's command (`-ub 512`, 190 slots, no VRAM pool, f16 K/V, 16k context): the same 10k prompt 63.0 tok/s there, 289.7
here; replies 43.3 ms / word there (chat), 27.8 here (story). One pass each, different commands: which flag brings what is not
measured (no tuning runs). The texts are sane (the story's reasoning: "We need answer user's request: write short story about 600
words, lighthouse keeper ..."; the official template's terse reasoning style, the same as in every run of this report).
