#pragma once

// The trained correction of the router-trick guess, in the engine.
// The file (the format is in tokenwing-tools/correction/correction_file.py) holds, per (target layer X, lead d), a matrix Bt [n_expert, n_embd]. At model load, for every pair
// this run's guesses can use, this builds
//     W'(X, d) = W_X * g_X[None, :] + Bt          (one tensor on the buffer type of X's router, so the guess runs where it ran before)
// and a tensor of ones on the buffer type of X's ffn_norm. The guess for (X, d) is then   W' . rms_norm(base)   - the same product as the plain
// guess W_X . (g_X * rms_norm(base)) - with the "gain" being the ones. A pair absent from the file, or any refusal, keeps the plain guess.
// Nothing here changes the real routing of any layer.

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

struct llama_model;
struct llama_moe_stream;

// The memory plan's view (the load-time weight walk and --moe-stream-plan-only): the bytes the file's tensors would take, per TARGET layer (the matrices of its pairs
// plus the ones tensor), for a run whose settings let the guess use them. Reads the header only (no payload). An upper bound: a pair is counted when its target is a
// MoE layer and its lead is within the lookahead, whether or not the engine ends up streaming that layer. Empty = nothing would be loaded (no file, another model,
// unreadable header). The caller applies the settings the load refuses for (pred-shared 0, prefetch 0, the chance table, prefetch lead 2).
struct llama_moe_correction_plan_in {
    std::string path;
    std::string arch;
    int64_t     n_embd = 0, n_expert = 0;
    uint32_t    n_layer_all = 0, n_layer_dense = 0, n_layer_nextn = 0;
    int32_t     lookahead = 1;
};
std::vector<std::pair<int32_t, int64_t>> llama_moe_correction_plan(const llama_moe_correction_plan_in & in);

struct llama_moe_correction {
    llama_moe_correction() = default;
    llama_moe_correction(const llama_moe_correction &) = delete;
    llama_moe_correction & operator=(const llama_moe_correction &) = delete;
    ~llama_moe_correction() { clear(); }

    bool on() const { return !w.empty(); }

    // the corrected router matrix of the guess for layer `target`, `lead` layers ahead (nullptr = the plain guess)
    ggml_tensor * gate(int32_t target, int32_t lead) const {
        const auto it = w.find({ target, lead });
        return it == w.end() ? nullptr : it->second;
    }
    // the ones that replace the target's ffn_norm gain in that guess (nullptr = no corrected pair for this target)
    ggml_tensor * unit(int32_t target) const {
        const auto it = unit_of.find(target);
        return it == unit_of.end() ? nullptr : it->second;
    }

    // Reads the file and builds the tensors. Never throws on a refusal: it prints one line ("NOT used (the plain guess runs): <why>") and
    // leaves everything as it was. Must run after the model's tensors are loaded and before the combined-guess stacks are built.
    void load(const llama_model & model, const llama_moe_stream & ms, const std::string & path);

    size_t bytes() const { return n_bytes; }

private:
    std::map<std::pair<int32_t, int32_t>, ggml_tensor *> w;
    std::map<int32_t, ggml_tensor *>                     unit_of;
    std::vector<ggml_context *>                          ctxs;
    std::vector<ggml_backend_buffer_t>                   bufs;
    size_t                                               n_bytes = 0;

    void clear();
};
