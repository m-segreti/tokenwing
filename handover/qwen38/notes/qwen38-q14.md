# Qwen3.8-Flash-Next: layer-major prompts (checklist Q14)

Status: **stopped before any code** (the lead changed the task to Q12 + Q13 the same hour). Branch `qwen-lm` in `llama.cpp-qwen`
was created from `moe-lab` = `308bc4b42`; it has no commits of its own. Nothing built, nothing run.

## Q14 notes (not started)

What I read and found, so the work can start from here. Line numbers are `llama.cpp-qwen/` at `308bc4b42`.

### What layer-major does today (Ling, Flint)
- `src/llama-context.cpp:3168-3554`: a call of 2+ ubatches runs each layer as phase 1 per ubatch (attention + router + shared
  expert; `lm_hid` = residual + shared expert, `lm_x` = the normed expert input, `lm_ids` / `lm_w` = picks / weights), then one
  expert graph per streamed layer for the whole call (phase 2: `build_moe_ffn` with `lm_phase == 2` adds each chunk's routed output
  straight into `lm_hid`, `src/llama-graph.cpp:3944-4071`). The memory is applied once per ubatch (first group) and replayed
  (`rewind()` + `reapply()`) for every later layer. Arch gate: `llama_arch_layer_major` (`llama-context.cpp:134`).
- `MOE_STREAM_TEST_LM_GROUP=G` runs whole layers per ubatch in layer-major order: same ops in the same order as the normal path, so
  its CPU text is the normal path's (the exactness test of the memory replay). The phase-split form changes how a call's tokens are
  grouped into waves (the whole call's expert set, not a ubatch's), so its sums differ by rounding on Ling and Flint too (Flint
  proved it with 1 wave per ubatch: KLD 0.000000, `reports/C17-M3.md`). A 10k CPU md5 equal to the normal path can only be asked of
  the G form; the split form gets a KLD / sane-text check.

### What `qwen4exp` needs (`src/models/qwen4exp.cpp`)
1. **Graph builder with a layer range** (as `flint.cpp:87-270`, `bailingmoe3.cpp:233-580`): `[lm_b, lm_e)`, the input of a group
   after the first from `lm_hid`, `t_lm_hidden` written at its rows, the output-alone graph (`lm_b == n_layer`).
   - The residual is 4 streams: `lm_hid` must be `[n_embd * hc, T]` (40 KiB per token, 1.25 GiB at 32k). Today the context makes
     it `[n_embd, T]` (`llama-context.cpp:3323`) and the VRAM-lending size uses `n_embd*4` per token (`:3195`, `rg_call_need`
     `:2610`): both must take the model's real hidden width (hparams, not a qwen constant).
   - Only build `build_inp_ple` (the n-gram rows + their page hints) in a group that holds a PLE layer (`hparams.is_ple`), only
     `build_inp_kpool` in a group that holds a QSA layer, only the token embedding in the group starting at 0. Bailingmoe3 keeps the
     KQ mask out of group graphs that do not use it (`bailingmoe3.cpp:536-542`): check `build_inp_mem_hybrid` does the same here, else
     the mask (n_kv x ubatch) is filled 48 times per ubatch.
   - Phase 1 / 2 with the hyper-connection combine. Normal order: `ffn_out = moe + shexp`, `res = res + ffn_out * 2*sigmoid(inject)`
     (`build_hc_combine`, `:423-451`; `build_layer_ffn`, `:1311-1352`). Plan: phase 1 writes `lm_hid` = the residual after the
     attention combine (4 streams), `lm_x` = the FFN mix output, `lm_ids` / `lm_w`, and the FFN inject `[hc, T]` (16 B/token).
     Phase 2 accumulates the routed output into a separate `[n_embd, T]` call tensor (first wave a copy, then adds - the same order
     as the normal `moe_out`), then per chunk: shared expert recomputed from `lm_x` (same per-row result on the CPU when chunks =
     ubatches), `ffn_out = moe + shexp`, `res = combine(lm_hid, ffn_out, inject)` written back. This needs one generic change in
     `build_moe_ffn`'s phase 2: an optional accumulation target other than `lm_hid`, with the chunk results handed back to the arch
     (a member list), because a view of the target does not depend on the chunk copies. Ling / Flint unchanged when the option is
     unset. Total call tensors ~60 KiB per token (1.9 GiB at 32k): the 32k call cap of the presets may need a lower value for this model
     on a 12 GB card - the lending code already splits a call that does not fit (`llama-context.cpp:3194-3224`) once its per-token
     size is right.
2. **Memory replay** (`src/llama-memory-hybrid-idx.{h,cpp}`): `llama_memory_hybrid_idx_context` has no `rewind()` / `reapply()`.
   - attention and recurrent parts: the base `llama_memory_hybrid_context` ones (`llama-memory-hybrid.cpp:269-280`) work as for Ling;
     the recurrent replay restores the slot search (`llama-memory-recurrent.cpp:1336-1360`) and covers all three rows per layer (conv,
     state, and the PLE conv history `get_p_l`) because they are per-layer rows read through `build_rs`.
   - the indexer cache (`ctx_idx`, a plain `llama_kv_cache_context`): its own `rewind()` / `reapply()` exist (n_kv per ubatch).
   - **the k-pool state**: `kpool_build_state` runs at apply against the memory's pool layout, which grows with every ubatch;
     `set_input_kpool` reads the layout at set_input time (`:729-973`). A replay of ubatch u must see u's layout and u's
     `kpool_state` (is_new, generation, n_pool_real, n_new, n_new_g), else the pool mask / tail / new-pool lists are wrong and the
     graph's shapes change. Plan: at apply (calls of 2+ ubatches) record per ubatch the per-sequence cell and pool counts plus the last
     cell pair, and a copy of the state without `rep_gen`; at reapply build the prefix layout (positions only append within a call; if
     the recorded last pair does not match, return false = the call runs ubatch by ubatch with one line), restore the state, set
     `i_kpool = i_cur`. Override `next()`-mirror `i_cur` in `rewind()`.
   - n-gram rows: `get_prev_tokens` reads the predecessors from the attention cells (`:1412-1413`); after all ubatches of the call are
     applied the cells hold more tokens, but a token's predecessors are earlier positions, so a replay gets the same rows.
3. **Context**: `llama_arch_layer_major` + `qwen4exp`; `lm_layer_major_ok` already refuses MTP, `n_rs_seq > 0` and several sequences;
   the in-call server checkpoints stay Ling-only (`:5832`).

### Checks planned (none run)
- G form (`MOE_STREAM_TEST_LM_GROUP=1`), CPU, a 3-ubatch prompt: md5 = normal path (proves rewind / reapply, k-pool replay, residual
  carry, n-gram rows); with the replay of the k-pool state disabled by a test switch it must differ (the check has teeth).
- Split form, CPU: `MOE_STREAM_TEST_LM_DUMP=1` finite rows; short greedy text after 10k tokens; KLD vs normal on a forced text.
- GPU: sane reply after 10k and 32k prompts; prompt tok/s, GB read, SSD wait, first reply tokens, layer-major off vs on.
- Off path: `traces/qwen38/gate_md5.json` texts; `check.py` on Ling-tiny; `traces/qwen38/flint_ab2.sh`.
