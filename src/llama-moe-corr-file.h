#pragma once

// The reader of the TokenWing correction file (the format is in tokenwing-tools/correction/correction_file.py, written by tokenwing-tools/correction/make_correction.py,
// reference loader tokenwing-tools/correction/correction_file.py). This part knows nothing about the model: it checks everything the file itself can prove
// (size, magic, header hash, format, pair table, payload hash). The caller checks the model's identity (arch, sizes, router hash) from
// the header it gets back. Every refusal is one sentence in the returned string; "" = the file is intact.

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

// The OS layer's file open (one named function per system call): takes a UTF-8 path. Linux / macOS: fopen; Windows:
// the wide-character open (a non-ASCII path works). Defined next to the other OS-layer functions in llama-moe-stream.cpp.
FILE * llama_moe_os_fopen_utf8(const char * utf8_path, const char * mode);

struct llama_moe_corr_pair {
    int32_t  target = 0;     // the layer whose router the guess is made for
    int32_t  lead   = 0;     // layers ahead; the source layer is target - lead
    int32_t  source = 0;
    uint64_t offset = 0;     // absolute byte offset of the float32 [n_expert, n_embd] matrix Bt
    uint64_t nbytes = 0;
};

struct llama_moe_corr_header {
    std::string arch;
    std::string name;
    int64_t     n_layer  = -1;
    int64_t     n_expert = 0;
    int64_t     n_embd   = 0;
    std::vector<int32_t> layers;        // the target layers the router hash covers (ffn_gate_inp.weight, exp_probs_b.bias, and ffn_norm.weight or hc_ffn_norm.weight each)
    std::string router_sha256;          // hex, over those tensors' raw bytes in that order
    uint64_t    payload_start = 0;
    uint64_t    file_size     = 0;
    std::string payload_sha256;
    std::string gate_rule;              // text of the shipping gate (for the start-up line)
    std::vector<llama_moe_corr_pair> pairs;
};

// Opens `path` and checks the file itself. `verify_payload` hashes the whole payload (about 1 s per 300 MB) in a pass of its own; the engine passes false and uses
// llama_moe_corr_file_stream, which reads the payload ONCE (hash and matrices from the same bytes). Returns "" or the reason.
// `base` and `size`: the correction file is the `size` bytes at `base` of `path` (data appended to a TokenWing model file); size 0 = the whole file.
// Every offset the file names (payload start, pairs) is relative to its own start.
std::string llama_moe_corr_file_check(const std::string & path, llama_moe_corr_header & h, bool verify_payload, uint64_t base = 0, uint64_t size = 0);

// Reads the payload ONCE, in file order: hashes every byte and hands each wanted pair's matrix (indices into h.pairs) to `on_pair` the moment it is complete (the data
// are valid during the call only). Returns "" when the payload sha256 matches the header's (the matrices handed over were then the verified bytes), else the reason - the
// caller must discard what it built from the callbacks.
std::string llama_moe_corr_file_stream(const std::string & path, const llama_moe_corr_header & h, const std::vector<size_t> & want,
                                       const std::function<void(size_t, const void *)> & on_pair, uint64_t base = 0);
