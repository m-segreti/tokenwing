# What we need from Grawnch (the maker of the 3.5bpw quant)

Our engine uses four small data files per model besides the weights. How Ling's and Flint's are made and loaded is described in
`llama.cpp/docs/adding-a-model.md` sections 2.4-2.5 and 5; the tools are in `llama.cpp/tokenwing-tools/`. All four can be appended
to a TokenWing model file (`make_model_file.py --correction / --vram-list / --usage-seed / --layer-order`, stored as
`tokenwing.blob.<name>.offset/size`, `make_model_file.py:9-11, 204-230, 366-368`) or kept next to the experts file.

## 1. The recordings everything else is made from

- **Format** (`llama-moe-trace`, `docs/adding-a-model.md:69-73`): `<prefix>.bin` = int32 `[tokens][MoE layers][experts per token]`
  (here [T][48][10]); `<prefix>.json` = the layers, the expert counts and `n_prompt` (how many tokens are prompt).
- **Blocker on our side:** `llama-moe-trace` refuses this file because it has no `expert_group_count` keys (`port-needs.md` A6);
  that is a small fix we make first. The engine's own recorder (`--moe-stream-record DIR`, `docs/flags.md:36`) is the other route.
- **What is needed:** many varied prompts **in the model's own chat format** (thinking on and off), with replies: chat, code, math,
  JSON / tool use, long documents. `tokenwing-tools/testing/prompts/` has twelve example kinds written with `<role>` markers.
- **Who makes them:** we can record here once the model streams (the laptop records at decode speed). If Grawnch already has
  routing dumps from his quantisation or evaluation runs (`benchmarks/qfn_stock_35bpw_eval_summary.json` in his repo is a KLD
  summary, not routing), the ids per token and layer in the format above are enough for files 3-5.

## 2. The guess correction file (`--moe-stream-correction FILE`) - wait for our input definition

- **Format** (`tokenwing-tools/correction/correction_file.py`): magic `TWCORR01`, a JSON header (format 1, the model's identity:
  arch, n_layer, n_expert, n_embd, the covered layers and a sha256 over their router tensors), then one float32 matrix
  `Bt [n_expert, n_embd]` per (target layer X, lead d), 4096-aligned. The corrected guess of layer X made d layers early is
  `(W_X diag(g_X) + Bt) x0`, with `x0 = RMSNorm(residual + shared expert of layer X-d)` and `g_X` = `blk.X.ffn_norm.weight`.
- **Why it cannot be made yet:** Qwen3.8 has no `ffn_norm`; its router reads the output of a 4-stream hyper-connection mix
  (`port-needs.md` B1). The engine refuses a layer without an F32 `ffn_norm` (`src/llama-moe-correction.cpp:135-136`). We must first
  choose the guess input (g1 / g2 / g3 in `port-needs.md` B1, chosen by counting on recordings) and define a second fold for this
  architecture: `x0` = that input vector [2560], `corrected = (W_X + Bt) x0`, the model check over the routers (F32 here) and the
  hyper-connection tensors the input uses.
- **Training data:** pairs (the guess input of layer X-d at a token, the true top-10 of layer X at that token); our recorder writes
  the guess input and the lead logits once B1 exists (`--moe-stream-record` with the input on). Our training scripts carry Ling's
  shape (512 experts, hidden 2,560, top 8, 8 groups / 4, layers 5-41; `docs/adding-a-model.md:158`); for this model: 512 experts,
  hidden 2,560, **top 10, no groups, layers 0-47**.
- **Ask Grawnch:** do not train yet. If he trains his own predictor, we need its exact definition: input vector (which tensor,
  before or after which norm), target layers and leads, dtype, and how the logits combine with the router's.

## 3. The VRAM list (`--moe-stream-vram-experts-file`)

- **Format** `"tokenwing vram-experts v1"` (`tokenwing-tools/experts/make_vram_list.py:21-28`):
  `{"format", "arch": "qwen4exp", "n_expert": 512, "n_expert_used": 10, "source", "counted", "layers": {"0": [[expert, count], ...], ...}}`,
  each layer's experts ranked by how often **replies** picked them (prompt tokens not counted), `--top N` kept per layer.
- **Made by:** `make_vram_list.py OUT.json REC1 REC2 ... --top N` from the recordings. With ~60 VRAM places per layer on the laptop
  (`model-numbers.md` section 4) `--top 128` leaves room for larger cards.

## 4. The usage seed (`--moe-stream-idle-load`)

- **Format** `"tokenwing usage v1"` (`make_usage_seed.py:20-28`): `{"format", "arch", "n_expert", "counted", "layers": {"L": [512 counts]}}`,
  picks over prompts **and** replies, scaled so every layer sums to 10 x 16,384 picks. Named `<experts file without .json>.usage-seed.json`,
  or appended to the model file. The engine keeps its own live table afterwards.

## 5. The layer order (whole-layer VRAM placement)

- **Format** `"tokenwing layer order v1"` (`docs/adding-a-model.md:84-90`): `{"format", "arch": "qwen4exp", "order": [layers, the one a
  token waits on most first]}`. No tool makes it; it comes from per-layer needed reads (our counters in a real run) or per-layer
  pick spread in the recordings. Whether whole VRAM layers pay at all on this model is open (Strata and SlenderEngine keep none,
  `port-needs.md` B3); this file is needed only if they do.

## 6. Things in the quant itself that would help, if he can still change them

| request | why | cost |
|---|---|---|
| **Write the GGUF with `general.alignment` = 4096** (the key exists in GGUF; llama.cpp's writer pads every tensor to it) | today it is 32 and no expert tensor starts on a 512- or 4096-byte boundary, so nothing can be read directly from the file as it is (`port-needs.md` A2). With 4096 every tensor start is aligned, and every slice of this model is a multiple of 512, which this laptop's direct reads accept (measured, `scripts/odirect_align_probe.py`) | a few KB of padding per tensor |
| **Put the chat template into `tokenizer.chat_template`** (it is an empty string now) | llama.cpp falls back to plain chatml without it: no thinking or tool-call format (`port-needs.md` A7) | none |
| **Say which down layers are meant to be Q4_0** | the card says 16 Q4_0 / 32 Q2_0, the header has 20 / 28 (`model-numbers.md` section 2) | none |
| Keep the n-gram table as its own file | it is read 16 rows per token and never streamed like the experts; a separate file fits the planned "weights + small data file" package | none |
| (Only if he wants) gate / up as **IQ3_S** instead of Q3_K | the same bytes (110 B per 256 values, slices of 704,000 B); our prompt kernels (`--moe-stream-cpu-iqk`) cover IQ3_S but not Q3_K (`ggml/src/ggml-cpu/iqk/iqk_port.cpp:96-107`); Strata / SlenderEngine run IQ3_S and refuse Q3_K (`others-today.md` section 6), so speeds would compare on one format. Quality against Q3_K is unknown - his call, not a request | a re-quantisation |
| (Later) an MTP GGUF of the model's draft layer | drafting is a later phase (`others-today.md` section 8) | his time |

Slice sizes at this model's shape (2560 x 640 gate / up, 640 x 2560 down), for any format he considers (arithmetic from ggml's block
sizes): every common format gives a 512-multiple; a 4096-multiple only Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K and IQ4_NL. K-quants and
IQ2 / IQ3 / IQ4_XS cannot be down projections here (640 values per row is not a whole 256-value block).

| format | gate / up slice | down slice |
|---|---|---|
| Q3_K, IQ3_S | 704,000 (512) | - |
| Q2_0 | 460,800 (512) | 460,800 (512) |
| Q4_0, IQ4_NL | 921,600 (4096) | 921,600 (4096) |
| IQ4_XS | 870,400 (512) | - |
| Q4_K | 921,600 (4096) | - |
| Q5_K | 1,126,400 (4096) | - |
| Q6_K | 1,344,000 (512) | - |
| Q8_0 | 1,740,800 (4096) | 1,740,800 (4096) |
| MXFP4 | 870,400 (512) | 870,400 (512) |
| IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS | 422,400 / 473,600 / 524,800 / 627,200 (all 512) | - |

## 7. Optional, useful for checks

- His BF16-teacher evaluation set or logits (his repo's KLD table): a second yardstick next to plain llama.cpp for our KLD checks.
