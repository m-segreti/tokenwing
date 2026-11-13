# Qwen3.8-Flash-Next on TokenWing, Part 1 (Q1-Q6, then Q7 / Q8)

Worker report, written item by item. Engine: worktree `llama.cpp-qwen`, branch `qwen` (from moe-lab `528e21a2a`), build
`llama.cpp-qwen/build` (CUDA 13.4, configured fresh; only `llama-completion` and `llama-moe-trace` built). Model files untouched.

## State

| item | state | commit |
|---|---|---|
| Q1 RAM plan leaves lazily read tables out | **done**, checked on the CPU run (see Q1) | `01b126057` |
| Q2 direct reads at the drive's alignment + copy fallback | **done**, checks passed | `4402db8ee` |
| Q3 TokenWing model file for a two-file GGUF | **done**: file written and read back, the engine streams from it, verify on all 48 layers | `f4e63825f` |
| Q4 word table read lazily for qwen4exp | **done**: same text with and without, 0.33 GiB less RAM | `086157fc0` |
| Q5 recorder without expert-group keys | **done** (+ a second recorder fix): recorded ids = the engine's picks, 0 of 14,880 differ | `8833ca4cf` |
| Q6 first streamed runs | **done**: plan-only, CPU, GPU, -ub 512 / 2048, CPU md5 = plain llama.cpp, 3 frozen texts stored | (no code) |
| Q7 chat template | **blocked on the official template file** (not on this machine; downloads not allowed); a stand-in test passed | (no code) |
| Q8 indexer cache in the RAM plan, save / restore | **done**: plan = measured (1133 MiB); 3 restore tests give the same text | `82cf5c5ad` |

Files: `traces/qwen38/Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf` (55,049,234,432 B, complete, read back) and
`traces/qwen38/Qwen3.8-Flash-Next-3.5bpw.tokenwing-00002-of-00002.gguf` (a symbolic link to the model's file 2). Run scripts in
`traces/qwen38/`: `qrun.sh` (streamed run in a memory cgroup), `qtrace.sh` (recorder), `qplain.sh` (plain llama.cpp CPU reference).

Build (in `llama.cpp-qwen`):
```bash
PATH=/usr/local/cuda/bin:$PATH cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 -DGGML_NATIVE=ON
cmake --build build -j 8 --target llama-completion llama-moe-trace
```

Run (CPU only and GPU; both work):
```bash
cd traces/qwen38
../../tools/model-lock.sh Qwen-helper ./qrun.sh cpu1 CUDA_VISIBLE_DEVICES= -- -ngl 0 --no-repack -c 4096 -n 32 -t 8 -f prompt_chat.txt -v
# = llama-completion -m Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf --moe-stream model --load-mode dio -fit off
#   --temp 0 --top-k 1 --no-warmup -no-cnv -st --no-display-prompt ... inside systemd-run MemoryMax=(MemAvailable - 1.5 GiB)
../../tools/model-lock.sh Qwen-helper ./qrun.sh gpu1 -- -ngl 99 -c 4096 -n 64 -b 2048 -ub 512 -t 8 -f prompt_chat.txt -v
```

## Q1: the RAM plan leaves every lazily read table out

**What changed** (`src/llama-model.cpp`, commit `01b126057`): the load-time walk of "what this process holds besides the pools"
(the one that also sums the GPU weights for the VRAM fit) skipped only the word table. A new list next to the word-table list,
`llama_arch_marks_tensor_lazy(arch, name)`, names the other tables an arch's `create_tensor` marks `TENSOR_READ_LAZY`:
`per_layer_token_embd.weight` of `qwen4exp` (the n-gram table) and of `gemma4`. The walk skips such a tensor when the loader's own
rule says it is read lazily (`ml.lazy.add(name, t, nullptr)`: lazy mode, size), and prints one line. Before, the 26.8 GiB table was
counted as RAM on CPU-only runs and as GPU weights with `-ngl` (it sits outside the layers, so it took the output device's bucket).

The plan side (`llama_moe_stream_get_plan`) needed nothing: it reads only the first file's header and counts only the word table
as RAM, so the n-gram table (second file) never entered it.

**Check** (the first CPU-only streamed run, `traces/qwen38/cpu1.log`; the load walk is where the bug was):
```
load_tensors: per_layer_token_embd.weight (27466.0 MiB) is read lazily - mapped from the file, only the rows a token uses are read,
              not counted as memory of ours
load_tensors: non-pool memory: 3.76 GiB of model weights stay in RAM, 1.51 GiB reserved for the KV cache, the compute buffers ...
llama_moe_stream: RAM: 23049 MiB available, 2.50 GiB kept free (--moe-stream-ram-free), 5.27 GiB for the rest of this process,
                  14.65 GiB of pools = 159 slots per layer x 48 layers
load_tensors:          CPU model buffer size =  3846.83 MiB
load_tensors:   CPU_Mapped model buffer size =   341.02 MiB      (the word table, Q4)
load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB      (the n-gram table, file 2)
```
3.76 GiB of weights in RAM = `model-numbers.md`'s fixed part minus the word table (3.757 GiB). The plan-only run
(`traces/qwen38/plan.log`, `-ngl 99 -c 4096`): one slot in every layer = 98,918,400 B (the study's 94.3 MiB), 200 slots from
23,013 MiB available - 2.5 GiB keep-free - 1.52 GiB non-pool, i.e. the study's formula (its 266 slots assumed 26.5 GiB available
and the presets' 0.5 GiB keep-free: (26.5 - 0.5 - 1.51) GiB / 94.3 MiB = 266). Without the change the table would have been 27466 MiB
of "model weights" on the CPU run.

## Q2: direct reads at the drive's own alignment; 512-multiple slices; the copy fallback

**What changed** (commit `4402db8ee`):
- `src/llama-moe-stream.cpp`: OS-layer function `os_direct_io_align(fd, file_size, &how)` next to `os_page_size`: Linux `statx`
  STATX_DIOALIGN when the file system reports it; else one direct read at 512, 1024, 2048 bytes (offset, length and buffer address on
  that value and not on twice it) - the first one accepted; 4096 when none is; other systems 4096. Read once in the constructor
  (before any read) into `moe_stream_io_align`, printed on one line.
- `moe_stream_preadv` (every expert read goes through it: slot reads, block reads, scratch reads): when the offset, a length or a
  buffer is off that alignment, it reads the aligned range around the request into a per-thread buffer and copies the bytes out
  (short read at the end of the file handled). The read-back checks' two raw `pread`s now use it too.
- Slices must be multiples of 512 (`MOE_STREAM_SLICE_ALIGN`, `src/llama-moe-stream.h`) instead of 4096, in the model-file route
  and the experts-file route; tensor starts in a TokenWing model file stay on 4096.
- Test switch `MOE_STREAM_TEST_IO_ALIGN=N` (power of two): N is used as the drive's alignment, so the copy path runs on any drive.
- `tokenwing-tools/experts/make_model_file.py`: accepts 512-multiple slices and says how many are not 4096-multiples; its test has a
  new case (`accepts_sector_multiple_slices`).
- `docs/flags.md`: two rows (the behaviour, the test switch). `docs/other-machines.md`: an OS-layer row for `os_direct_io_align`.

**Found on the way:** this laptop's file systems report no DIOALIGN at all (statx probe in the scratchpad: ntfs3 file, btrfs
`/home`, exFAT T7 all `mask_has_dioalign=0`), so the probe read is what decides here. NVMe logical block size 512.

**Checks:**
```
$ tools/model-lock.sh Qwen-helper python3 tokenwing-tools/testing/check.py --model .../Ling-3.0-tiny-MXFP4_MOE.gguf --build build \
      --no-ctest --no-build -j 8 --work traces/qwen38/check-tiny
  OK   test_expert_index.py
  OK   test_expert_repack.py
  128 experts, top-8: pool of 51 slots per layer, -ub 6
  OK   plain run on the CPU
  OK   the settings reached the program
  OK   streamed run on the CPU read experts from the SSD  (3795 reads)
  OK   streamed text identical to the plain run
  OK   streaming with a pool smaller than the prompt needs (waves)  (rc 0)
  OK   streamed run on the GPU  (CUDA0: NVIDIA GeForce RTX 5070 Ti Laptop GPU (11865 MiB, 11656 MiB free))
Everything passed.
(stream_cpu.log) llama_moe_stream: direct reads of '.../model.experts' need 512-byte alignment (a direct read at that alignment):
                 every expert slice is on it and is read straight into its slot
(stream_cpu.log) verify_slot: read-back check done: 23 experts of 23 streamed layers against the model, 0.034 GB compared
```
(`--no-build`: I built only `llama-completion` myself, as the build rule says; check.py's own step builds every target.)

The copy path, forced (same tiny command as check.py's streamed CPU run):
```
MOE_STREAM_TEST_IO_ALIGN=8192:    ... need 8192-byte alignment (TEST ONLY: ...): every expert slice is on it   (tiny's slices are 8192-multiples)
MOE_STREAM_TEST_IO_ALIGN=1048576: ... need 1048576-byte alignment (TEST ONLY: ...): 69 of 69 expert slices ... are not, those are
                                  read as the aligned range around them into an I/O thread's buffer and copied
                                  moe stream: read 35.44 MiB/token (4.43 GiB total, 3795 reads), stalled 8.350 ms/token
de9f1aba8ac19f2149b43a0f4bdbae78  plain_cpu.txt
de9f1aba8ac19f2149b43a0f4bdbae78  stream_cpu.txt
de9f1aba8ac19f2149b43a0f4bdbae78  stream_cpu_a8192.txt
de9f1aba8ac19f2149b43a0f4bdbae78  stream_cpu_a1m.txt
```

Flint, short GPU run, the shared build (`llama.cpp/build`, moe-lab `528e21a2a`) against this build at `4402db8ee`, same command
(`traces/qwen38/flint_ab.sh`: `--moe-stream model --moe-stream-slots 64 -ngl 99 -c 4096 -n 64 -ub 512`, greedy, `01_casual_chat`):
```
5be5e4cb7cbfdfc24384bf8a9f25d915  flint_shared.txt
5be5e4cb7cbfdfc24384bf8a9f25d915  flint_q2.txt
(flint_q2.log) direct reads of '.../flint-UD-Q4_K_XL.tokenwing.gguf' need 512-byte alignment (a direct read at that alignment):
               every expert slice is on it and is read straight into its slot
(flint_q2.log) verify_slot: read-back check done: 35 experts of 35 streamed layers against the model file (a filled slot, and the
               first, middle and last expert against the checksums in the file), 0.258 GB compared
shared: moe stream: read 140.32 MiB/token (19.46 GiB total, 11413 reads)
q2:     moe stream: read 140.37 MiB/token (19.47 GiB total, 11417 reads)
```
(The 4 extra reads are guess timing between two runs; the text is the same.)

**Left:** Windows (the volume's sector size) and macOS (no alignment rule) versions of the OS function - stubs say 4096, which is
correct with the copy path. One read-back path keeps its own reads: an experts file that records the model's offsets compares
against the model through a separate O_DIRECT file, rounded to 4096 - fine on drives with sectors up to 4096 (a drive needing
more would fail that check, not the run). The model-file route (Qwen3.8's) uses the engine's read.

## Q3: the TokenWing model file for the two-file GGUF

**What changed** (commit `f4e63825f`):
- `src/llama-moe-stream.cpp`: `--moe-stream model` refused every split GGUF. New `model_shard_with_experts()` reads the headers of
  files 2..N; when no file but the first holds a routed-expert tensor the experts stream from the first file and the loader reads the
  other files as usual; otherwise one line names the file with experts and says to merge.
- `tokenwing-tools/experts/make_model_file.py`: the first file of a split model is converted alone; the output must be named
  `OUT-00001-of-0000N.gguf` (the loader finds the other files by that name) and keeps the split keys; the tool puts a link
  `OUT-0000k-of-0000N.gguf` next to it for each other file (symbolic, else hard link; refuses if a different file is there) and checks
  on read-back that the links resolve to the model's files and that the tensor count equals `split.tensors.count`. A split model with
  experts in another file is refused. Two new test cases (`converts_a_split_model`, `refuses_experts_in_another_file`): the tool's
  tests pass 7 / 7.
- **Weights only (the user's decision of today):** the file holds the weights (file 1 laid out with `general.alignment` 4096) plus
  header keys only: `tokenwing.format` and the sample checksums `tokenwing.sums.*` (432 values, read by `--moe-stream-verify`; they
  describe the weights and change only with them). No data is appended: I passed none of `--correction`, `--vram-list`,
  `--usage-seed`, `--layer-order`. Neither the tool nor `--moe-stream model` needs appended data: the runs below have none, and the
  engine looks for each data file by its flag or next to the model. I did not remove the tool's options to append data (Ling and
  Flint's files and presets still use them); that is the later packaging piece.
- `docs/flags.md`: a row for `--moe-stream model` with a split GGUF.

**Check:** the conversion with its read-back (`traces/qwen38/make_model_file.log`, through the lock):
```
.../Qwen3.8-Flash-Next-3.5bpw-00001-of-00002.gguf: arch qwen4exp, 1223 tensors, 48 MoE layers, [512] experts per layer; every expert
    slice is a multiple of 512 bytes (124 of 144 are not multiples of 4096: a drive that needs 4096-byte direct reads reads those through a copy)
split model: file 1 of 2 holds every routed expert and is converted; the other files are used as they are
.../Qwen3.8-Flash-Next-3.5bpw.tokenwing-00002-of-00002.gguf: a symbolic link to /path/to/models/Qwen3.8-Flash-Next-3.5bpw-00002-of-00002.gguf
wrote .../Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf: 55049234432 bytes (source 55048512576), 55037498880 tensor bytes, 446.5 s
check: 57 metadata keys equal, 1223 tensors with the same names, types, shapes and sizes, all 4096-aligned
check: all 55037498880 tensor bytes identical, sha256 over all tensor data e1134ef65de60efa3e40b9d60359c0d60a53166af2538ca4ac2472d33df69ce8
check: checksums of 432 sample slices read back from the output match the metadata
check: done in 151.2 s
check: files 2..2 found next to the output and identical to the model's; 1224 tensors in all, as split.tensors.count says
```
The engine on it (CPU run `cpu1.log`; the same lines in every later run):
```
llama_moe_stream: direct reads of '.../Qwen3.8-Flash-Next-3.5bpw.tokenwing-00001-of-00002.gguf' need 512-byte alignment (a direct read
    at that alignment): every expert slice is on it and is read straight into its slot
verify_slot: read-back check done: 48 experts of 48 streamed layers against the model file (a filled slot, and the first, middle and
    last expert against the checksums in the file), 0.396 GB compared, 978.2 ms in total
```
124 of the 144 slice kinds (every gate / up, and the 28 Q2_0 downs) are 512-multiples and not 4096-multiples: on this drive they are
read straight into the slots (no copy).

## Q4: the word table read lazily for qwen4exp

**What changed** (commit `086157fc0`): `src/models/qwen4exp.cpp` marks an untied `token_embd` `TENSOR_READ_LAZY` (the same test as
Ling / Flint: only with a separate `output.weight`); `llama_arch_marks_token_embd_lazy` lists `LLM_ARCH_QWEN4EXP`, so the load walk
and the plan agree.

**Check** (CPU-only, 32 words, the same command; the "without" run keeps the table in RAM with the existing test switch):
```
cpu1.log:        the word table token_embd.weight (341.0 MiB) is read lazily - mapped from the file, MADV_RANDOM, not counted as RAM of ours
                 non-pool memory: 3.76 GiB of model weights stay in RAM ... 14.65 GiB of pools = 159 slots per layer x 48 layers
cpu1_tokram.log: (MOE_STREAM_TEST_TOKEN_TABLE_RAM=1) non-pool memory: 4.09 GiB of model weights ... 14.37 GiB of pools = 156 slots per layer
0a3c44f700c1e2a72ee787d66b72a88d  cpu1.txt
0a3c44f700c1e2a72ee787d66b72a88d  cpu1_tokram.txt
```

## Q5: the recorder (`llama-moe-trace`) on a model without expert-group keys

**What changed** (commit `8833ca4cf`, `tools/moe-trace/moe-trace.cpp`):
1. `expert_group_count` / `expert_group_used_count` are optional (0 = no groups); before, the tool stopped with "the model is not an
   MoE model".
2. Found on the first run: the tool then stopped with "layers of one graph disagree on the number of tokens". Every arch here narrows
   its last layer to the output rows before that layer's experts (`qwen4exp.cpp:528`, also `bailingmoe3.cpp:431`, `flint.cpp:163`),
   and the tool marked only a piece's last prompt token as an output, so the last MoE layer routed one row. The tool now marks every
   prompt token as an output, so every layer routes every row (cost: n_batch x n_vocab floats of logits per prompt piece, printed;
   7.6 MiB at `-b 8`). Not specific to Qwen: the same stop should happen on Ling / Flint since the upstream crop came in (not run).

**Check so far:** the trace records (`traces/qwen38/trace_cpu.*`, CPU, `-b 8 -ub 8`, 32 words):
```
main: arch = qwen4exp, n_layer = 48, 48 MoE layers, n_expert = 512, k = 10, groups = 0 of 0, n_embd = 2560
main: every prompt token is an output, so every MoE layer routes every row: 7.6 MiB of logits per prompt piece of 8
main: traced 59 tokens (27 prefill, 32 decode) in 9.49 s, 6.22 t/s
main: wrote .../trace_cpu.bin (113280 B = 59 rows x 48 layers x 10 ids x 4 B), ...
decoded (vocab from the file): "\n<think>\nThe user wants a simple answer: the capital of France in one short sentence, plus two
famous landmarks. This is straightforward factual information.\n</think>"
```
**Check: recorded ids = the engine's picks.** T46's way first: the engine with `--moe-stream-oracle trace --moe-stream-oracle-lead 0`.
Two things got in the way, neither in the recorder:
- `llama-completion` stops when a prompt piece ends on an end-of-generation token (`tools/completion/completion.cpp:949`). With
  `-b 8` the third piece ends on prompt token 23 = `<|im_end|>` (decoded from the file's vocabulary), so the run printed
  end-of-text after 16 prompt tokens (`2 prompt calls: 8, 8`). Ling-tiny with `-b 8` runs normally (its tokens fall elsewhere). So the
  comparison runs at `-b 9 -ub 9` (pieces 9 / 9 / 9).
- At `-b 9` the oracle's own tally says `the decode calls alone: 313 of 14910 (2.099%)` differ, while the texts are identical (trace9
  = the engine's text = `cpu1`'s text) and it counts 14,910 decode ids where 31 calls x 48 x 10 = 14,880 exist. So I compared against
  the engine's own recorder instead (`--moe-stream-record rec9`, `truth.i32` = the 10 experts the router chose, per decode word and
  layer), same command:
```
engine record vs trace: decode ids differing 0 of 14880 ; token-layers with a difference 0
per layer [0, 0, 0, ... 0]   (48 layers)
```
Q5 passes: the recorder's ids are exactly the engine's picks on all 31 decode words x 48 layers. (The oracle tally's 313: a follow-up.)

## Q6: first streamed runs

All runs: `traces/qwen38/qrun.sh` (`llama-completion -m ...tokenwing-00001-of-00002.gguf --moe-stream model --load-mode dio -fit off
--temp 0 --top-k 1 --no-warmup -no-cnv -st --no-display-prompt`, inside `systemd-run --user --scope -p MemoryMax=(MemAvailable -
1.5 GiB) -p MemorySwapMax=0`), through `tools/model-lock.sh`, prompt `traces/qwen38/prompt_chat.txt` (a chat-format question, 27
tokens), engine defaults otherwise (keep-free 2.5 GiB). AC power on (`online 1`), PPT limit 41 W (`ppt_pl1_spl`), the GPU check
before the GPU runs: `gpu-cap-check: OK - GPU at 106 W of its 108 W limit (98 %), 2317 MHz, P0`. Another helper ran AMD tests on
this machine between my runs (through the same lock), so MemAvailable moved between 20.5 and 23 GiB and the slot counts with it.

**Plan-only** (`plan.log`, `--moe-stream-plan-only -ngl 99 -c 4096`): see Q1 - 48 streamed layers, 98,918,400 B per slot row,
200 slots, `0 dense without experts, 0 MTP/nextn not streamed`.

**CPU only** (`cpu1`, `CUDA_VISIBLE_DEVICES=`, `-ngl 0 --no-repack -c 4096 -n 32`): sane text,
```
<think>
The user wants a simple answer about France's capital, followed by two famous landmarks. This is straightforward factual information. I'll keep it concise as
prompt eval time =    3346.53 ms /    27 tokens (  123.95 ms per token,     8.07 tokens per second)
       eval time =    5498.60 ms /    31 runs   (  177.37 ms per token,     5.64 tokens per second)
moe stream: read 260.40 MiB/token (14.49 GiB total, 7534 reads), stalled 76.266 ms/token (4.35 s total), 3.59 GB/s achieved
(159 slots per layer x 48 layers; CPU compute buffer 161.87 MiB at -ub 512)
```

**GPU** (`gpu_ub512`, `-ngl 99 -c 4096 -n 64 -b 2048 -ub 512`; the engine picked 4 whole layers by itself): the right answer,
```
<think>
The user wants a simple answer about France's capital, followed by two famous landmarks. This is straightforward factual information. I'll keep it concise as requested.
</think>

The capital of France is Paris. Two famous landmarks there are the Eiffel Tower and the Louvre Museum. [end of text]

llama_moe_stream: 4 whole MoE layers kept on the GPU (layers 0, 1, 2, 4), 44 layers streamed
llama_moe_stream: RAM: 20485 MiB available, 2.50 GiB kept free, 1.52 GiB for the rest of this process, 15.94 GiB of pools = 190 slots per layer x 44 layers
CUDA0 model buffer size = 8171.89 MiB; KV 96.00 + indexer 24.00 MiB; RS 112.57 MiB
sched_reserve:      CUDA0 compute buffer size =   526.18 MiB
sched_reserve:  CUDA_Host compute buffer size =    26.19 MiB
vx_report: VRAM placement: whole MoE layers 0,1,2,4 (4); VRAM experts off; free after the context 2.657 of 11.588 GiB
prompt eval time =    2629.42 ms /    27 tokens (   97.39 ms per token,    10.27 tokens per second)
       eval time =    5185.76 ms /    59 runs   (   87.89 ms per token,    11.38 tokens per second)
moe stream: read 183.25 MiB/token (15.21 GiB total, 8001 reads), stalled 52.166 ms/token (4.43 s total), 3.70 GB/s achieved
```
**`-ub 2048`** (`gpu_ub2048`): `CUDA0 compute buffer size = 826.90 MiB`, `CUDA_Host compute buffer size = 104.57 MiB`; the whole-layer
fit reserved 1.00 GiB for the compute buffer (0.50 MiB x 2048) and kept 3 whole layers (0, 1, 2), 186 slots x 45 layers; same answer
(the reasoning line words it differently: a rounding difference from the other placement); 11.84 tok/s, SSD wait 51.3 ms/token. The
study's estimate was ~1.0 GiB at 2048.

First numbers (not a benchmark: 1 run each, 27-token prompt, -c 4096, the engine's defaults, no guess for this arch yet, no
preset flags): GPU 11.4-11.8 tok/s with ~51-52 ms/token of SSD wait at 186-190 slots + 3-4 whole layers; CPU only 5.6 tok/s with
76 ms/token at 159 slots.

**Same text as plain llama.cpp** (CPU, so the same kernels; `--no-repack` on both; `-ub 12` so the streamed side has no prompt
waves: 12 x 10 = 120 = the pool):
```
plain_cpu_ub12  (qplain.sh: this build without --moe-stream, the ORIGINAL two files, --load-mode mmap, CPU, cgroup cap 19083M)
    add: tensor per_layer_token_embd.weight (size = 27465 MiB) lazy read enabled
    CPU_Mapped model buffer size = 52487.85 MiB / 27465.95 MiB
    prompt eval time = 18167.59 ms / 27 tokens; eval time = 21685.23 ms / 31 runs (699.52 ms per token, 1.43 tokens per second)
stream_cpu_ub12 (streamed, 120 slots, -ub 12)
0a3c44f700c1e2a72ee787d66b72a88d  plain_cpu_ub12.txt
0a3c44f700c1e2a72ee787d66b72a88d  stream_cpu_ub12.txt
```
The plain run reads the model through the page cache under the cap (the model is never whole in RAM; mapped, not loaded): it fits
the brief's condition. It took 1 minute.

**Frozen texts for later gates** (`traces/qwen38/gate_md5.json`, with the full commands; each run twice):

| gate | settings | md5 | repeat |
|---|---|---|---|
| `qwen_cpu_120` | CPU only, `--moe-stream-slots 120`, `-n 32 --ignore-eos`, `-ub 512` | `0a3c44f700c1` | equal |
| `qwen_cpu_120_ub12` | the same at `-ub 12` (no waves) = plain llama.cpp | `0a3c44f700c1` | (= plain) |
| `qwen_gpu_150_allstreamed` | `-ngl 99`, `--moe-stream-gpu-layers 0 --moe-stream-slots 150`, `-n 64 --ignore-eos`, `-ub 512` | `6417ddef0d82` | equal |

A gate must pin the slots and the whole layers: at 150 pinned slots the CPU run did not fit (the engine lowered it to 138 and 132
in two runs: `150 slots per layer (--moe-stream-slots) do not fit ... 132 slots per layer`) and the two texts differed by the
prompt's wave split, an accepted rounding difference; the auto whole-layer placement moves with free VRAM (4 layers at -ub 512, 3 at
2048).

**Not done in Q6:** KLD against plain llama.cpp's own spread (the md5 is equal on the CPU path, so the KLD there is 0; a GPU KLD run
was not made).

## Q8: the indexer cache in the memory plan; save / restore tests

**What changed** (commit `82cf5c5ad`, `src/llama-model.cpp`): new `llama_moe_stream_idx_row(hp, arch, type_k)` = the bytes per token
of the sparse-attention indexer's own cache on one attention layer (`llama_memory_hybrid_idx`: `indexer_head_size` x the k-pool
row's slots, in the K cache type, no V; 0 for every arch but qwen4exp). Added to `llama_moe_stream_layer_ctx_bytes` (the per-layer
cache the whole-layer VRAM check charges to a device) and to the per-token bytes of a server's saved state
(`llama_moe_stream_srv_compute`, the `--cache-disk` restore buffer; it now takes the arch). The old comment said the indexer cache
was not counted. `docs/flags.md`: a behaviour row.

**Check, plan vs measured** (65k context, q8_0 K/V, `-fa on`, `--moe-stream-gpu-layers-list 0,1,2,4`, the existing test switch
`MOE_STREAM_TEST_WL_SAY=1` so the check prints its line; `traces/qwen38/q8_65k_wl.log`):
```
whole MoE layers and the free VRAM (--moe-stream-gpu-layers-list): all fit (TEST ONLY: MOE_STREAM_TEST_WL_SAY), kept 0,1,2,4; CUDA0:
    free 11657 MiB - weights 3847 - context 1133 (KV cache + recurrent state) - compute ~320 - cuBLAS + scratch 108 - VRAM pool minimum 0
    - growth 86 = 6163 MiB for whole layers, 4325 used
llama_kv_cache:      CUDA0 KV buffer size =   816.00 MiB      (attention: 12 layers x 65536 x 1088 B)
llama_memory_recurrent:      CUDA0 RS buffer size =   112.57 MiB
llama_kv_cache:      CUDA0 KV buffer size =   204.00 MiB      (indexer: 12 layers x 65536 x 272 B = the new term)
```
1133 = 816 + 204 + 112.57: the plan equals what the run allocates (it would have said ~929 without the change). At 4096 / f16 the
indexer cache is 24.00 MiB = 512 B per token per layer = 256 values, the study's "2 x 128" (read from a code comment then, now
measured). The auto whole-layer fit (`-1`) uses a fixed reserve (3.40 GiB here) and is not changed by this.

**Check, save / restore** (CPU only, the gate settings: 120 slots, `-c 4096`, greedy, `--ignore-eos`; texts compared as strings):
1. Completion-level state file (`llama_state_save_file` / `load_file` through `--prompt-cache`; `pc_save` / `pc_load`): the second run
   loads the state after the prompt (`session file has exact match for prompt!`, `replayed last token from session`) and continues:
```
0a3c44f700c1e2a72ee787d66b72a88d  pc_save.txt
0a3c44f700c1e2a72ee787d66b72a88d  pc_load.txt
0a3c44f700c1e2a72ee787d66b72a88d  gate_cpu_a.txt
pc_state.bin 118839988 bytes (recurrent state 117.7 MB + 27 tokens of KV and indexer rows)
```
2. llama-server `--cache-disk` (`traces/qwen38/srv_restore.py`, token-array prompts, `-np 1 --cache-ram 0 --ctx-checkpoints 2`): chat c1
   (21-token question -> 24-token reply), chat c2 takes the slot, c1 continues with a follow-up. Against a server that kept c1 in the
   slot:
```
D_disk: other chat prompt_n 14; c1 continued: prompt_n 15, cache_n 44 (restored from chat-....state); reply "<think>\nThe user is
        asking about the capital of Italy, following up on the previous question about France. They want a"
D_slot: prompt_n 15, cache_n 44; the same reply                                                      -> D_equal: true
```
3. llama-server's snapshot when the last reply is dropped (P1 -> R1, then P1 + another question): the server rolls back to its
   snapshot at position 17 and processes 18 tokens; against a server that processed P1 (n_predict 0) and then P1 + the question
   (14 tokens after 21 cached):
```
K_snapshot:     prompt_n 18, cache_n 17; reply "<think>\nThe user wants me to name one river in Germany. This is a simple factual
                question. Germany has many rivers"
K_continuation: prompt_n 14, cache_n 21; the same reply                                              -> K_equal: true
```
So the indexer cache, the recurrent state and the n-gram history (the n-gram rows hash the tokens before, which the cache cells
keep) come back correctly in all three paths on the CPU. Not tested: the same on the GPU, `--cache-disk-checkpoints`, and a restore
after a server restart (the state files were reused within one server process).

## Q7: the chat template

**Blocked:** the model file's `tokenizer.chat_template` is empty, and the official Qwen3.8 template is not on this machine. I looked:
our tree's `models/templates/` (Qwen2.5, Qwen3-0.6B, QwQ, Qwen3-Coder, Qwen3.5-4B), the reference clones in `traces/refs/`, and every
`.jinja` file under `/path/to/disk` and `/home/user`. SlenderEngine's setup lists the model pack's own template as
`chat_template.jinja`, 8,952 B, sha256 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` (`traces/refs/slender/setup.py`);
no local file has that size or hash. Its `serve/chat_template.jinja` (9,993 B) is SlenderEngine's own edited copy, so I did not
treat it as official. The brief allows no downloads. **Needed from the user:** that file (Qwen's or Grawnch's repo), to give with
`--chat-template-file` (it is a small separate file, not written into the model file).

**A stand-in test, to know the rest works** (`traces/qwen38/srv_template.py`, llama-server, GPU, `-c 8192`, `--jinja
--chat-template-file models/templates/Qwen3.5-4B.jinja`, greedy). The Qwen3.5 template uses the same `<think>` tags and the same
tool-call format (`<tool_call><function=...><parameter=...>`) as SlenderEngine's served copy (counted in both files; that copy adds a
`preserve_thinking` switch):
```
thinking: reasoning_content "The user wants me to calculate 17 * 23 and reply with just the number.\n\n17 * 23 = 17 * 20 + 17 * 3 =
          340 + 51 = 391", content "391", finish stop  -> split: true
tool calls (get_weather, 3 requests): Paris -> get_weather {"city": "Paris"}, Tokyo -> {"city": "Tokyo"}, Lima -> {"city": "Lima"};
          finish tool_calls each -> 3 of 3 parsed
```
So with a Qwen3.5-family template the server's reasoning split and tool-call parser work with this model's output. Whether the
official template differs (system prompt, earlier replies' reasoning, `preserve_thinking`) is open until the file is here.

## Final checks on the finished branch (`82cf5c5ad`)

```
$ tools/model-lock.sh Qwen-helper python3 tokenwing-tools/testing/check.py --model .../Ling-3.0-tiny-MXFP4_MOE.gguf --build build \
      --no-ctest --no-build -j 8 --work traces/qwen38/check-tiny
  OK   test_expert_index.py
  OK   test_expert_repack.py
  OK   plain run on the CPU
  OK   the settings reached the program
  OK   streamed run on the CPU read experts from the SSD  (3789 reads)
  OK   streamed text identical to the plain run
  OK   streaming with a pool smaller than the prompt needs (waves)  (rc 0)
  OK   streamed run on the GPU  (CUDA0: NVIDIA GeForce RTX 5070 Ti Laptop GPU (11865 MiB, 11656 MiB free))
Everything passed.
de9f1aba8ac19f2149b43a0f4bdbae78  plain_cpu.txt  =  stream_cpu.txt   (the same md5 as at Q2)
Flint, traces/qwen38/flint_ab.sh, this build:  5be5e4cb7cbfdfc24384bf8a9f25d915 = the shared build's (moe-lab 528e21a2a)
python3 tokenwing-tools/experts/test_make_model_file.py: 7 / 7 PASS
```
Built and run: `llama-completion`, `llama-moe-trace`, `llama-server` only (not all targets; ctest not run). The full frozen-text gates
are the lead's, before a merge.

## Limitations and doubts

- Every number here is one run, a 27-token prompt and short replies, with the engine's defaults (no preset flags, keep-free 2.5 GiB,
  4096 context): a first look, not a benchmark. No guess runs for this arch yet (Q10), so most misses are not read ahead.
- Another helper ran AMD tests through the same lock all day; MemAvailable moved between 20.5 and 23 GiB between my runs, so slot
  counts differ between runs (each run names its count).
- The CPU md5 equality with plain llama.cpp holds on the CPU path with `--no-repack`; on the GPU the text differs by placement
  (accepted rounding), and no GPU KLD was measured.
- The recorder now asks for the logits of every prompt token: n_batch x n_vocab floats per piece (2 GB at `-b 2048` with this
  248k vocabulary). Fine for recordings at `-b` up to a few hundred; a long-prompt recording at `-b 2048` should lower `-b`.
- I did not remove the model-file tool's options for appending data (the user's decision is that our data stays in separate
  files); the Qwen3.8 file has none. Removing them touches Ling's and Flint's files and presets: the later packaging piece.
- Q8's restore tests ran on the CPU within one server process; not run: the GPU, a server restart, `--cache-disk-checkpoints`.

## Follow-ups (not fixed here)

1. `--moe-stream-oracle` tally: on Qwen3.8 at `-b 9` it reports `the decode calls alone: 313 of 14910 (2.099%)` differ and counts
   30 ids more than exist, while the engine's own record matches the trace in all 14,880 decode ids. Probably one call compared
   against a neighbouring row (not investigated).
2. `llama-completion` stops (prints end-of-text) when a prompt piece ends on an end-of-generation token
   (`tools/completion/completion.cpp:949`): `-b 8` and a chat prompt whose `<|im_end|>` lands on a piece's last token. Upstream
   behaviour; it matters to any test script with a small `-b`.
3. The recorder fix (every prompt token an output) should also be needed for Ling and Flint since the upstream crop came in
   (`bailingmoe3.cpp:431`, `flint.cpp:163`); not run on them.
4. The auto whole-layer fit leaves 2.7-3.3 GiB of VRAM unused on this model (`free after the context 2.657 / 3.255 of 11.588 GiB`);
   the presets' VRAM expert pool (`--moe-stream-vram-free`) is what uses it - for Q12.
5. `os_direct_io_align` for Windows and macOS (rows in `docs/other-machines.md`).
6. Qwen3.8 items of Part 1 after Q8 as the checklist has them (Q9 recordings, Q10 the guess hook, Q11-Q13), and C2 (the n-gram
   table's file cache under the memory cap over a long session) - not touched.

## Would this work for other models

Q1-Q3 and Q8 are general (any split GGUF with all experts in file 1; any drive; any lazily read table on the list; the indexer term
is 0 for every other arch). Q4 and the indexer term are qwen4exp-specific by design (lists keyed on the arch, as for Ling / Flint).

## Time

About 2 h (10:15-12:15), much of it in runs waiting for the lock (another helper's AMD tests) and the 55 GB conversion (7.5 min
write + 2.5 min read-back). The first CUDA build of the worktree took ~18 min.

## Q7, added by the lead (2026-11-13): the official chat template
The user allowed the download. `chat_template.jinja` from Qwen's own page (`https://huggingface.co/Qwen/Qwen3.8-Flash-Next`, the
page's current version): 9,004 B, sha256 `af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375` - 52 B longer than the
8,952 B copy SlenderEngine pins (an older version; not fetched). Kept as `traces/qwen38/qwen38-chat-template.jinja` (the download
itself in `traces/qwen38/template-official/`), given with `--jinja --chat-template-file`: a small separate file, not written into
the model file. The same test as the stand-in's (`traces/qwen38/srv_template.py`, llama-server, GPU, `-c 8192`, greedy;
result `traces/qwen38/srv_template_official.json`):
```
thinking split: True | content: '391' | reasoning: 'We need answer user: "What is 17 * 23? Reply with the number only." Need compute 17*23 = ...
tool calls ok: 3 of 3 [('get_weather', {'city': 'Paris'}, True), ('get_weather', {'city': 'Tokyo'}, True), ('get_weather', {'city': 'Lima'}, True)]
```
The server's log has no error line; it says the template supports keeping the reasoning of earlier turns (on by default).
Q7 is done; Part 1 (Q1-Q8) is complete.
