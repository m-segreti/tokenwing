# Qwen3.8-Flash-Next on TokenWing: the study and the proposed checklist

Study of 2026-11-13. Reading only: no engine code changed, no model run, nothing built, no GPU used. Engine read: `llama.cpp/`
`moe-lab` at `e2330ec67`. Other engines read: Strata `main` `13eedf55` and v0.2.11 `c3b3f50`, SlenderEngine `speed-fork`
`8af6fd84` (clones in `traces/refs/`, git-ignored). The model: Grawnch's `Grawnch/Qwen3.8-Flash-Next-3.5bpw-GGUF`, read from its
two GGUF headers over HTTP (the download itself was not touched).

## Summary

- **Grawnch's file (header facts):** 48 layers, all MoE, 512 experts top 10, gate / up Q3_K everywhere, down Q4_0 in **20**
  layers and Q2_0 in **28** (the card says 16 / 32). One expert is 2,329,600 or 1,868,800 B; all routed experts 47.17 GiB; the
  fixed part 4.09 GiB; the n-gram table 26.8 GiB in the second file. A decode token touches 989 MB of experts if none were held.
- **What blocks a first streamed run (code changes, not flags):**
  1. The file is not aligned for direct reads (alignment 32; gate / up slices of 704,000 B are not 4096-multiples). This laptop's
     drive accepts 512-byte-aligned direct reads (measured with a tiny probe), and every slice of this model is a 512-multiple, so
     the fix is "read the drive's real alignment" plus a TokenWing file written with 4096 alignment.
  2. Our RAM plan would count the 26.8 GiB n-gram table as RAM and leave no room for the pools.
  3. The one-file format refuses a two-file GGUF.
  4. Neither the guess nor the correction file works as-is: the router reads a 4-stream hyper-connection mix, not "residual + norm
     gain". Qwen3.8 would stream with no guess at all.
- **What the port needs beyond that:** layer-major prompts for this arch (large: the hybrid memory with the sparse-attention indexer
  needs rewind / reapply); CPU kernels for Q3_K / Q4_0 / Q2_0 (ggml has AVX2 only and none of our fast CPU paths cover them, and the
  CPU computes most decode experts); an engine reader for the n-gram rows; the chat template (empty in the file); small fixes
  (recording tool, token table, indexer cache in the RAM plan).
- **Memory on this laptop (sums):** ~5.4 GiB of VRAM for experts at 65k (~59 per layer), ~2.4 GiB at 262k; RAM pools ~24.5 GiB =
  ~266 slots per layer of 512. That holds about 63 % of all experts, roughly twice SlenderEngine's share.
- **Strata and SlenderEngine cannot run this quant** (from their code): neither has a gate / up kernel for Q3_K, and both stop at
  start. Their numbers come from other packs (SlenderEngine: IQ3_S, whose gate / up slices are byte for byte the same size).
- **Their prompt path is faster per token than ours is likely to be.** SlenderEngine reads a 50K prompt at 2,064-2,296 tok/s in
  5-7 passes over the experts (its PC). Our layer-major (built for Ling and Flint, not yet for this model) reads each expert once per
  call (2 passes for 50K), but on Ling our 32k prompt
  ran at ~950 tok/s with only 1.27 s of SSD wait, so our limit is GPU work and copies. Their edge is in kernels: fused
  hyper-connection kernels, sparse-attention kernels on tensor cores, a GDN tensor-core scan, an issuer thread overlapping copies,
  and the prompt room. How much of this applies is unknown until a profile of one long prompt on our engine (item Q-P0). The
  comparison is on different machines, so this is a direction, not a measured gap.
- **Answer-changing tricks:** Strata's defaults skip experts (tail skip at rank 7 on CUDA) and split small prompts onto the CPU.
  SlenderEngine turns the skip off; its "skip 0" numbers are our bar. Every such part is listed in `others-today.md` section 7. None
  is in this checklist.

## The bar (user's rule: their speed with every answer-changing part off)

**SlenderEngine at skip 0** (`traces/refs/slender/docs/fork/PERFORMANCE.md:12-26`, 2026-11-08/09). Machine: i7-14700KF, 32 GB
DDR5-6000, RTX 4070 Ti 12 GB, two Samsung 990 PRO, Windows 11; IQ3_S pack, 262K context, int8 K/V, 10 GiB RAM tier, DFlash2
drafting (1.6-1.9 tokens accepted per step).

| | short answers | after a 50K prompt | reading a 50K prompt |
|---|---|---|---|
| SlenderEngine skip 0, its defaults | 46.5-49.5 tok/s | 35.5-41.4 tok/s | 2,064-2,145 tok/s (2,258-2,296 with its author's launcher) |

- Still on in those runs: the CPU share in prompt chunks under 1,024 tokens (rounding only), int8 K/V (a precision choice like our
  q8_0 presets), and drafting with every draft checked by the model.
- Our first comparisons have no drafting. Their laptop page measured "MTP off" at -19..-22 %. So their skip-0 short answers without
  drafting would be about **37-40 tok/s**. That is my estimate from their ratio, measured on another machine.
- Their PC has two SSDs and a desktop GPU; this laptop has one NVMe (5.5 GB/s) and an RTX 5070 Ti Laptop on PCIe 4.0 x8.

## The pages

| page | what is in it |
|---|---|
| [port-needs.md](port-needs.md) | every place where Qwen3.8 differs from what Ling / Flint needed: what exists (`path:line`), what to add, size, check |
| [model-numbers.md](model-numbers.md) | the file in numbers: per-layer types and bytes, fixed part, bytes per token, memory sums for four machine classes |
| [others-today.md](others-today.md) | how Strata and SlenderEngine work today (decode, prompts, memory, SSD, settings), the full list of answer-changing parts, drafting (later), licences |
| [ideas-from-other-engines.md](ideas-from-other-engines.md) | one entry per idea of theirs: group (a) / (b) / (c) / (d), answers, class A / B / C, gain, hardware, check |
| [needed-from-grawnch.md](needed-from-grawnch.md) | the data files our engine uses and their exact formats, the recordings they need, and what in the quant itself would help |
| `scripts/` | `gguf_header_remote.py` (header over HTTP), `model_numbers.py`, `memory_sums.py`, `odirect_align_probe.py` |
| `data/` | the two headers as JSON (tensor table and metadata) |
| `reading-notes/` | the four sub-readers' notes on Strata and SlenderEngine, with `path:line` on every claim (key lines re-checked where the pages say "checked") |

## The proposed checklist (for the user to approve and reorder)

Sizes: small / medium / large (`port-needs.md` explains). "Model" = needs the model running; "Grawnch" = needs his files. Gates for
every item that touches shared code: Ling's frozen texts and `tools/check.sh --full` unchanged, Flint case-3 05 unchanged.

### Part 1: plain support "as usual", with proper flags

| # | item | why | size | depends on | check | needs |
|---|---|---|---|---|---|---|
| Q1 | RAM plan: leave every lazily read tensor out of "RAM besides the pools" (load walk + plan) | the 26.8 GiB table would be counted as RAM (A4) | small | - | plan-only RAM line = the sums of `model-numbers.md` | header only |
| Q2 | Direct-read alignment from the drive (OS layer: `statx` DIOALIGN / Windows sector size), the tools accept 512-multiple slices, a bounce-copy fallback for 4Kn drives | the file cannot be read directly today (A2); measured: 512 works here | medium | - | `--moe-stream-verify` + slot check; streamed CPU md5 = plain CPU md5; a test switch forcing 4096 runs the fallback with the same md5 | files |
| Q3 | The TokenWing model file for a two-file GGUF: experts in shard 1 streamed, shard 2 (n-gram table) kept as its own file; written with alignment 4096 | the tool and `--moe-stream model` refuse split files (A1); a merged file would be 84 GB | medium | Q2 | the tool's read-back; verify on every streamed layer | files |
| Q4 | Token table read lazily for `qwen4exp` (A5) | +0.33 GiB for experts | small | - | start-up line; md5 with = without | files |
| Q5 | `llama-moe-trace`: expert-group keys optional (A6) | the recorder refuses the file | small | - | recorded ids = the engine's picks | files |
| Q6 | First streamed runs: plan-only, a short CPU md5 against plain llama.cpp (from the page cache), GPU sane text, the compute buffer at `-ub 512 / 2048`, KLD within plain llama.cpp's own spread, frozen texts for later gates | the base of every later check | small | Q1-Q4 | as named | model |
| Q7 | The official chat template (`--chat-template-file`, or written into our file), a llama-server test of thinking and tool calls | the file's template is empty: chatml fallback (A7) | small | Q6 | reasoning split; N of N tool calls parsed | model + template |
| Q8 | The indexer cache in the RAM plan and the whole-layer VRAM check; save / restore tests for snapshots and `--cache-disk` (indexer, n-gram history) (D2) | uncounted ~0.2 GiB at 65k; correctness of saved states | small + tests | Q6 | plan vs measured; restored continuation md5 | model |
| Q9 | Recordings of the five text kinds (prompts in the model's chat format); counts: per-layer use spread, the recall of the guess inputs g1 / g2 / g3 at 1-3 layers ahead | data files and the guess choice | small (runs + counting) | Q5, Q6 | counts on real recordings | model |
| Q10 | The guess hook for hyper-connection models (shared code: guess input, far guesses, recorder leads, word-start sources), `qwen4exp` calls it, the arch joins `arch_leads` (B1) | Qwen3.8 streams with no guess today | medium | Q9 | md5 guess on = off; needed reads / token and SSD wait down | model |
| Q11 | VRAM list and usage seed from the recordings (layer order only if whole layers pay) | VRAM picks, start-up loading | small | Q9 | the files load (start-up lines) | model (or Grawnch's recordings) |
| Q12 | The recommended command on Qwen3.8: same-output settings all on vs all off (same md5), quick on / off for the values chosen on Ling, `presets/qwen38.sh` for 65k and 262k with q8_0 K/V; watch the n-gram table's file cache under the memory cap (C2) | the user's commands | small-medium | Q10, Q11 | md5; 5 pairs vs the "before" numbers; cap line over a long session | model |
| Q13 | The "before" numbers: ours vs plain llama.cpp on the same file, under the same caps (this laptop; and the 8 GB-card and CPU-only cases as for Flint), plus the oracle run (perfect guesses) | where it lands; what the guess can still win | small (runs) | Q12 | 512 tokens, 5 pairs, slowest 1 % + average | model |

### Part 2: our own improvements this model needs

| # | item | why | size | depends on | check | needs |
|---|---|---|---|---|---|---|
| Q-P0 | Profile one 32k prompt and one decode token on our engine (GPU nodes, copies, SSD wait, the n-gram gather, GDN, indexer, hyper-connection ops) - no code | decides which prompt copies pay (Ling: 34.5 s for 32k with 1.27 s SSD wait) | small | Q6 (again after Q14 for the real prompt path) | `GGML_CUDA_PROFILE` / `GGML_TRACE` on one call | model |
| Q14 | Layer-major prompts for `qwen4exp`: layer-range graph builder, `rewind` / `reapply` for the hybrid memory with the indexer cache and k-pool state, n-gram rows per ubatch, the 4-wide residual between layer groups (D1) | without it a 50K prompt is ~25 passes over the experts instead of 2 (sums) | large | Q6 | md5 = normal path; prompt times 4k-50k; SSD wait; the reply after | model |
| Q15 | CPU kernels, step 1: wire ik's existing Q3_K and Q4_0 GEMMs into `--moe-stream-cpu-iqk` (E1) | prompt-size calls on the CPU (CPU-only machines, CPU waves) | small-medium | Q6 | op test vs ggml; KLD; CPU prompt time | files (op tests), model (speed) |
| Q16 | CPU kernels, step 2: 4-row AVX-512 kernels with ggml's output for Q3_K, Q2_0, Q4_0 (Strata's Q2_0 kernels as a reference), and the fused gate / up -> down step for these formats (E2, D8, D9) | the CPU computes most decode experts; ggml has AVX2 only | large | Q6 | `kernel_bench --check` memcmp; CPU md5 unchanged; ms / token | model |
| Q18 | Correction file for hyper-connection models: a second fold (input = the chosen guess input, `W' = W_X + Bt`), the engine's model check, the training scripts for 512 / top 10 / no groups / layers 0-47 (B2) | the trained guess gained on Ling (math -1.37 ms / word at 134 slots) | medium | Q10 | engine checks; needed reads with vs without, 5 pairs | Grawnch (or our training) |

### Part 3: ideas from Strata and SlenderEngine, built our own way (all "same bits" or "rounding only")

| # | item | answers | why (their measured gain, their machine) | size | depends on | check | needs |
|---|---|---|---|---|---|---|---|
| Q19 | Fused hyper-connection kernels for prompts and decode (P1, D10) | rounding only | 50K +4.3 % (SlenderEngine PC, RTX 4070 Ti); removes the 336 MB-per-8k gate tensor | medium | Q-P0 | HC node time; KLD | model |
| Q20 | Sparse-attention prompt kernels: tensor-core block scores, radix top-k, INT8 attention on int8 K/V (P8) | rounding only | no separate number found; our indexer uses plain ops | large | Q-P0 | attention share at 32k / 128k; KLD | model |
| Q21 | GDN tensor-core chunked scan (P9) | rounding only | +4.4..+5.7 % prompts (Strata main, Linux CUDA) | medium | Q-P0 | GDN node time; KLD | model |
| Q22 | Lend unreached K/V cells to the VRAM expert pool (D1, the VRAM form of SlenderEngine's K/V lend and Strata's `--kv-grow`) | same bits | K/V lend: SSD misses -27 %, +11 % short answers (SlenderEngine, its RAM form); kv-grow +2.3 % at 196K (R9700); for us 3.98 GiB of K/V at 262k | large | Q12 | VRAM places vs context used; ms / token; no failed growth | model |
| Q23 | CPU share inside small GPU prompt calls (P12) | rounding only | 512-token prompts -28..-34 % (Strata, several cards) | medium | Q12 | prompt time 300-2,000 tokens; KLD | model |
| Q24 | Prompt room: the output head's VRAM lent to long prompts, reloaded from the file (P2) | same bits | +10..+19.5 % at 50K (SlenderEngine), from one pass fewer; smaller for our layer-major | medium | Q14 | head checksum; prompt time 32k / 50k; first token | model |
| Q25 | Parallel refill of the VRAM places after a prompt (P3) - check our refill time first | same bits | 1,050 -> 410 ms after 50K (SlenderEngine, opt-in) | small | Q14 | refill time line | model |
| Q26 | SSD keep-alive reads while idle (P15) | same bits | avoids 50-150 ms stalls on drives that sleep (Strata's note, WD SN7100 on Windows) | small | - | first read after a 2 s / 10 s pause | model |

### Later, with the reason

| item | why later |
|---|---|
| Drafting with the model's MTP head / DFlash2 / prompt lookup | user's decision (not touched yet); the file has no MTP block; SlenderEngine's DFlash2 drafter is CC BY-NC-ND (never) |
| Images (the model has an image token) | no image file in Grawnch's repo |
| K/V in RAM with only part in VRAM (Strata `--kv-resident`, +23 % at 262K on their RTX 5070) | needs page-locked RAM (fails here) and competes with expert copies; after Q22 |
| Second-SSD mirror | one SSD only (user) |
| Expert skipping, routing toward held experts, approximate late layers (KVA) | answers change - possible later as an opt-in mode |
| PCIe share of decode misses | rounding only, but slower on this laptop's CPU / PCIe balance; a row for other machines |
| One CUDA graph per decode window with mapped-memory doorbells | needs page-locked shared memory (fails here); our split graphs stay |
| Giving VRAM back to other programs (`--vram-elastic`) | an app feature |

## Doubts (contradictions found, written down, not worked around)

1. **The model card vs the header:** 16 / 32 Q4_0 / Q2_0 down layers on the card, 20 / 28 in the header; routers "F32 / BF16" on the card,
   all F32 in the file. The header is what the engine sees; Grawnch should say which is intended.
2. **"Plain support as usual with proper flags" is not flags only:** Q1-Q3 and Q10 are code changes. Without them nothing streams (Q1, Q2)
   or nothing is guessed (Q10).
3. **AGENTS.md says O_DIRECT on this partition needs 4096-aligned buffers, offsets and lengths;** the probe shows 512 works on this drive
   (`scripts/odirect_align_probe.py`). The 4096 rule in the engine is stricter than the drive. Other drives and file systems (Windows
   NTFS, 4Kn drives) must be checked by the OS-layer function, not assumed.
4. **The bar was measured on IQ3_S, the port uses Grawnch's Q3_K / Q4_0 / Q2_0** (the user's decision). Gate / up slices are the same
   704,000 B in both. Their down (IQ4_NL, 921,600 B by the sub-reader's arithmetic) is larger than our Q2_0 layers, so our bytes per
   token are a little smaller. Neither Strata nor SlenderEngine can run Grawnch's file, so no same-file comparison is possible
   without a second download (an IQ3_S pack for them, ~47 GB).
5. **"As good as Strata"**: Strata's main design keeps every expert in RAM (64 GB machines in its tables). On 30 GB its low-RAM mode is the
   comparable part, and SlenderEngine is the SSD-tier engine. The pages compare against SlenderEngine's skip-0 numbers as instructed.
6. **Routing skew:** the old study (`docs/qwen38-port.md`) called the routing near-uniform (from upstream PRs). SlenderEngine's own
   measurements show a small held set serving most picks. The real figure is a Q9 count.
7. **SlenderEngine's records:** the sub-reader found no WORKLOG sections behind some table rows (K/V lend +11 %, in-place stager, dense
   arena) and no run scripts in the repo (`reading-notes/slender_docs.md` section 0). Those numbers are their claims without raw data. I did
   not re-check the absence myself.
8. **Strata version:** studied `main` after v0.2.13.1, as allowed. The prompt said "v0.2.13 or main".

## What could not be verified

- No speed number of Strata or SlenderEngine was reproduced; all are theirs, on their machines.
- "Strata / SlenderEngine refuse Q3_K gate / up" is from their code (the start-up check and the kernel lists, checked line by line), not from a run.
- The indexer cache's row width (2 x 128 values) comes from a code comment (`src/llama-memory-hybrid-idx.cpp:52-55`), not a run.
- Our compute buffer for this model, our refill time, the guess recall, the CPU kernel speeds, the table's page cache under the cap: run items.
- Whether in-call checkpoints and `--cache-disk` handle the indexer cache and the n-gram history correctly (Q8).
- Whether Strata's Q2_0 CPU kernels give the same bits as ggml's.
- The official Qwen3.8 chat template and which reasoning / tool-call parser it needs.
