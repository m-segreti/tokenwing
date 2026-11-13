# Qwen3.8-Flash-Next 3.5bpw: the model and the quant in numbers

Every byte count below comes from the two GGUF headers, read over HTTP range requests from the public repo
`Grawnch/Qwen3.8-Flash-Next-3.5bpw-GGUF` on 2026-11-13 (`scripts/gguf_header_remote.py`; the headers are kept in
`data/header-shard1.json` and `data/header-shard2.json`). The per-layer table and the totals are printed by
`scripts/model_numbers.py`. Check: the tensor bytes of shard 1 (55,037,498,880) plus its data start (11,013,696) equal the
file size the repo lists (55,048,512,576) exactly, so the type sizes used are right.

The memory sums in section 4 are **sums, not measurements**. They come from `scripts/memory_sums.py`; the inputs marked
"estimate" there must be replaced by the first run's start-up lines.

## 1. The two files

| file | bytes | contents |
|---|---|---|
| `Qwen3.8-Flash-Next-3.5bpw-00001-of-00002.gguf` | 55,048,512,576 | 1,223 tensors: all 48 layers, the token table, the output head; 57 metadata keys; header 11,013,682 B |
| `Qwen3.8-Flash-Next-3.5bpw-00002-of-00002.gguf` | 28,800,138,432 | one tensor: `per_layer_token_embd.weight` (the n-gram table), Q4_0, [160, 320,001,536], 28,800,138,240 B = 26.82 GiB, 90 B per row |

Facts from the metadata that matter for us:

- `general.alignment` is **not set, so it is 32**. No expert tensor starts on a 512- or 4096-byte boundary (all 144 checked).
- `qwen4exp.block_count` 48; `full_attention_interval` 4 (12 full-attention layers 3, 7, ..., 47; 36 Gated DeltaNet layers);
  `expert_count` 512, `expert_used_count` 10, `expert_feed_forward_length` 640, `expert_shared_feed_forward_length` 640;
  `embedding_length` 2560; attention 24 heads, 2 KV heads, head 256; indexer 4 heads x 128, `top_k` 2048,
  `compress_ratios` 4 on the full-attention layers; GDN `ssm.state_size` 128, `group_count` 16, `time_step_rank` 48,
  `inner_size` 6144, `conv_kernel` 4; `hyper_connection.count` 4, `low_rank` 320; `ple.layers` [1], `ngram_size` 3,
  `heads_per_ngram` 8; `context_length` 262,144; vocabulary 248,320.
- **No `expert_group_count` / `expert_group_used_count` keys** (our `llama-moe-trace` requires them, see `port-needs.md` A6).
- **`tokenizer.chat_template` is an empty string** (see `port-needs.md` A7).
- **No MTP block** (no nextn tensors, block count 48) and no image-encoder file in the repo.

## 2. Per layer: the routed experts

Gate and up are Q3_K in every layer. Down is Q4_0 in 20 layers and Q2_0 (ggml type 42: 64 values in 18 bytes) in 28.
**The model card says 16 Q4_0 / 32 Q2_0; the header says 20 / 28.** The header wins (doubt recorded in the README).

| layers | gate / up | down | gate slice | up slice | down slice | one expert | one layer (512) |
|---|---|---|---|---|---|---|---|
| 0, 1, 2, 3, 6, 7, 11, 15, 19, 23, 27, 31, 34, 35, 38, 39, 42, 43, 46, 47 (20) | Q3_K | Q4_0 | 704,000 | 704,000 | 921,600 | **2,329,600 B** | 1.111 GiB |
| 4, 5, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 36, 37, 40, 41, 44, 45 (28) | Q3_K | Q2_0 | 704,000 | 704,000 | 460,800 | **1,868,800 B** | 0.891 GiB |

All 12 full-attention layers have Q4_0 down, as the card says; so do layers 0-3, 6, 34, 38, 42 and 46.

Alignment of the slices (this decides how our direct reads work, `port-needs.md` A2):

| slice | bytes | / 4096 | / 512 |
|---|---|---|---|
| gate or up (Q3_K, 640 rows x 1,100 B) | 704,000 | 171.875 (not whole) | 1,375 (whole) |
| down Q4_0 (2,560 rows x 360 B) | 921,600 | 225 (whole) | 1,800 |
| down Q2_0 (2,560 rows x 180 B) | 460,800 | 112.5 (not whole) | 900 (whole) |

Totals:

- **Routed experts: 50,646,220,800 B = 47.17 GiB** (24,576 experts, 2,060,800 B on average).
- **One slot in every layer's pool** (what one more slot per layer costs): 20 x 2,329,600 + 28 x 1,868,800 = **98,918,400 B = 94.3 MiB**.
- **Routed bytes a decode token touches**: 48 layers x 10 = 480 uses x 2.06 MB = **989 MB** if none were held.
- **One full pass over all experts** (what a prompt reads once per pass): 47.17 GiB.

For comparison (from AGENTS.md "Models"): Ling's routed set is 59.8 GiB with 3,133,440 B per expert and 320 uses per token
(1.00 GB); Flint's is 21.47 GiB with 1,915,708 B on average and 282 uses per token.

## 3. The fixed part (everything that is not a routed expert)

| part | types (header) | bytes | GiB |
|---|---|---|---|
| attention, GDN, indexer, norms (48 layers) | Q6_K projections; F32 norms / conv / `ssm_a` / `dt`; BF16 `ssm_alpha` / `ssm_beta` / indexer | 2,256,391,680 | 2.101 |
| hyper-connections (48 x attention + FFN mixers, + the head mixer) | Q8_0 down / up, BF16 inject, F32 norm | 687,267,840 | 0.640 |
| output head `output.weight` | Q6_K [2560, 248,320] | 521,472,000 | 0.486 |
| token table `token_embd.weight` | Q4_0 [2560, 248,320] | 357,580,800 | 0.333 |
| routers `ffn_gate_inp.weight` (48) | **F32** [2560, 512] (5.24 MB each) | 251,658,240 | 0.234 |
| shared experts (48) + their gates | Q8_0; gate `ffn_gate_inp_shexp` F32 [2560] | 251,166,720 | 0.234 |
| layer 1's n-gram block projections | BF16 `ple_key` / `ple_value` / `ple_conv1d`, F32 norms | 65,740,800 | 0.061 |
| **total fixed part** | | **4,391,278,080** | **4.090** |

The model card says the routers are "F32 / BF16"; in this file all 48 routers are F32.

The n-gram table (shard 2) is separate: 26.82 GiB, used only at layer 1, 16 rows of 90 B per token (row ids are a hash of the
token and the two before it, `src/models/qwen4exp.cpp:1299-1398`).

## 4. Memory sums for four machine classes (sums, not measurements)

Inputs (from the header unless marked):

- On the GPU with `-ngl 99`: the fixed part minus the token table (which becomes a mapped table once the arch is on our lazy
  list) = **3.757 GiB**.
- KV cache, q8_0 (34 B per 32 values): 12 full-attention layers x (2 KV heads x 256 for K + the same for V) = **13,056 B per
  token**. The indexer cache (one row per token per full-attention layer, "key | pooled", `src/llama-memory-hybrid-idx.cpp:52-55`)
  at 2 x 128 values = **3,264 B per token** (the row width of 2 x 128 is read from a code comment, not run). Together
  **16,320 B per token**: 65,536 tokens = 1.00 GiB, 262,144 tokens = 3.98 GiB.
- GDN state per sequence: 36 layers x (128 x 128 x 48 + 3 x 10,240) x 4 B = **117,669,888 B** (Strata's docs give the same
  117,669,888 B for one of their checkpoints, `reading-notes/strata_prefill.md` section e). A server snapshot is about this size.
- Estimates: CUDA context ~0.45 GiB; compute buffer at `-ub 2048` **~1.0 GiB (unknown until the first run**: Ling's is 626 MiB,
  this model carries a 4-wide residual of 10,240 floats per token); `--moe-stream-vram-free 200MiB` and
  `--moe-stream-ram-free 0.5GiB` as in the presets; the engine's fixed reserve 1.51 GiB (measured on Ling).

| machine | VRAM left for experts | VRAM experts | RAM pools | slots per layer | held per layer (VRAM + RAM) |
|---|---|---|---|---|---|
| this laptop, 12 GB card + 30 GB RAM, 65k context, apps closed (MemAvailable ~26.5 GiB, estimate) | 5.43 GiB | ~2,830 (59 per layer) | 24.5 GiB | 266 | ~325 of 512 (~63 %) |
| this laptop, 262k context | 2.44 GiB | ~1,270 (27 per layer) | 24.5 GiB | 266 | ~293 (~57 %) |
| 8 GB card + 16 GB RAM (~11 GB usable), 65k | 1.49 GiB | ~780 (16 per layer) | 8.2 GiB | 89 | ~105 (~21 %) |
| 8 GB card + 16 GB RAM, 32k | 1.99 GiB | ~1,035 (22 per layer) | 8.2 GiB | 89 | ~111 (~22 %) |
| CPU-only, 16 GB RAM (~11 GB usable), 65k | - | - | 2.4 GiB (after 5.9 GiB of weights, caches and compute) | 26 | 26 (~5 %) |
| CPU-only, 16 GB RAM, 32k | - | - | 2.9 GiB | 31 | 31 (~6 %) |

What sits where, in all GPU cases: the fixed part (3.76 GiB), the KV and indexer caches, the GDN state and the compute buffer in
VRAM; the VRAM expert pool in what is left; the RAM pools in RAM; the token table (0.33 GiB) and the n-gram table (26.8 GiB)
mapped from the SSD (only the pages read are in RAM, as file cache); every routed expert on the SSD. CPU-only: all of it in RAM
except the two mapped tables and the experts.

Decode SSD bytes per token, by the share of a token's 480 picks that are already in VRAM or RAM (sum at 5.5 GB/s, the drive's
measured O_DIRECT speed in AGENTS.md):

| picks already held | MB per token from the SSD | SSD time per token |
|---|---|---|
| 50 % | 495 | 90 ms |
| 60 % | 396 | 72 ms |
| 80 % | 198 | 36 ms |
| 90 % | 99 | 18 ms |

The share held is **not** the same as the share of slots: the hot experts and the guesses push it far above the slot share.
SlenderEngine measured it on this model (IQ3_S, experts ~2.05 MB): with ~8 % of experts in VRAM and ~21 % in RAM, a token found
55-60 % of its experts in VRAM, 25-30 % in RAM and read only 7-11 % from the SSD (`docs/fork/WORKLOG.md:610, 703`; their PC,
i7-14700KF / RTX 4070 Ti / two 990 PRO; `reading-notes/slender_docs.md` section 5). Those shares were measured before
2026-11-06, while expert skipping at rank 7 was still on (`WORKLOG.md:1278` says every speed before then includes it), so
without skipping the SSD share is somewhat higher. Our laptop would hold about twice that share
(63 %), before guesses. The real figure is known only once the model runs.

Prompt: one pass over every expert is 47.17 GiB; with ~63 % held, one pass reads ~17 GiB from the SSD and moves up to 47 GiB over
PCIe to the GPU (sums). How many passes a prompt takes depends on the prompt path (`port-needs.md` D1).
