#include "llama-moe-correction.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-stream.h"
#include "llama-moe-corr-file.h"

extern "C" {
#include "hash/sha256/sha256.h"
}

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <set>

std::vector<std::pair<int32_t, int64_t>> llama_moe_correction_plan(const llama_moe_correction_plan_in & in) {
    std::vector<std::pair<int32_t, int64_t>> out;
    if (in.path.empty()) {
        return out;
    }
    llama_moe_corr_header h;
    if (!llama_moe_corr_file_check(in.path, h, /* verify_payload = */ false).empty()) {
        return out;
    }
    if (h.arch != in.arch || h.n_expert != in.n_expert || h.n_embd != in.n_embd) {
        return out;
    }
    std::map<int32_t, int64_t> per;
    for (const llama_moe_corr_pair & p : h.pairs) {
        if (p.target < (int32_t) in.n_layer_dense || p.target >= (int32_t) (in.n_layer_all - in.n_layer_nextn) || p.lead > std::max<int32_t>(1, in.lookahead)) {
            continue;
        }
        per[p.target] += (int64_t) p.nbytes;
    }
    for (auto & [x, b] : per) {
        out.push_back({ x, b + in.n_embd * (int64_t) sizeof(float) });
    }
    return out;
}

void llama_moe_correction::clear() {
    for (ggml_backend_buffer_t b : bufs) {
        if (b) {
            ggml_backend_buffer_free(b);
        }
    }
    for (ggml_context * c : ctxs) {
        if (c) {
            ggml_free(c);
        }
    }
    bufs.clear();
    ctxs.clear();
    w.clear();
    unit_of.clear();
    n_bytes = 0;
}

static std::string hex32(const unsigned char * d) {
    static const char hx[] = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < 32; i++) {
        s += hx[d[i] >> 4];
        s += hx[d[i] & 15];
    }
    return s;
}

// the raw bytes of a tensor (device or host) into the hash, in chunks
static void hash_tensor(sha256_t * st, const ggml_tensor * t, std::vector<unsigned char> & tmp) {
    const size_t n = ggml_nbytes(t);
    const size_t chunk = 1 << 22;
    for (size_t off = 0; off < n; off += chunk) {
        const size_t c = std::min(chunk, n - off);
        tmp.resize(c);
        ggml_backend_tensor_get(t, tmp.data(), off, c);
        sha256_update(st, tmp.data(), c);
    }
}

void llama_moe_correction::load(const llama_model & model, const llama_moe_stream & ms, const std::string & path) {
    clear();
    const auto t0 = std::chrono::steady_clock::now();
    const auto refuse = [&](const std::string & why) {
        clear();
        fprintf(stderr, "llama_moe_correction: NOT used (the plain guess runs): %s [%s]\n", why.c_str(), path.c_str());
    };
    if (!ms.pred_shared) {
        return refuse("the correction is trained on the guess input of --moe-stream-pred-shared 1; this run does not use it");
    }
    if (!ms.prefetch_on()) {
        return refuse("--moe-stream-prefetch 0: no guess is made, nothing to correct");
    }
    if (ms.chance_on() || ms.prefetch_lead >= 2) {
        return refuse("the chance table / --moe-stream-prefetch-lead 2 build their guess differently; the correction does not apply to them");
    }

    llama_moe_corr_header h;
    const std::string bad = llama_moe_corr_file_check(path, h, /* verify_payload = */ false);   // the payload is hashed in the one pass that reads it, below
    if (!bad.empty()) {
        return refuse(bad);
    }
    const auto & hp = model.hparams;
    const std::string arch = model.arch_name();
    if (h.arch != arch || h.n_expert != (int64_t) hp.n_expert || h.n_embd != (int64_t) hp.n_embd || (h.n_layer >= 0 && h.n_layer != (int64_t) (hp.n_layer() + hp.n_layer_nextn))) {
        return refuse(format("another model: the file is for %s (n_layer %" PRId64 ", n_expert %" PRId64 ", n_embd %" PRId64 "), this model is %s (%u, %u, %u)",
                h.arch.c_str(), h.n_layer, h.n_expert, h.n_embd, arch.c_str(), hp.n_layer() + hp.n_layer_nextn, hp.n_expert, hp.n_embd));
    }
    // the router tensors of the covered layers: ffn_gate_inp.weight, exp_probs_b.bias, ffn_norm.weight (F32), in this order
    {
        sha256_t st;
        sha256_init(&st);
        std::vector<unsigned char> tmp;
        for (int32_t X : h.layers) {
            if (X < 0 || (size_t) X >= model.layers.size()) {
                return refuse(format("another model: the file covers layer %d, this model has %zu layers", X, model.layers.size()));
            }
            const llama_layer & L = model.layers[X];
            if (L.ffn_gate_inp == nullptr || L.ffn_gate_inp->buffer == nullptr ||
                (L.ffn_gate_inp->type != GGML_TYPE_F32 && L.ffn_gate_inp->type != GGML_TYPE_BF16 && L.ffn_gate_inp->type != GGML_TYPE_F16)) {
                return refuse(format("another model or quant: layer %d has no F32/BF16/F16 router tensor", X));
            }
            hash_tensor(&st, L.ffn_gate_inp, tmp);
            if (L.ffn_exp_probs_b != nullptr) {
                if (L.ffn_exp_probs_b->type != GGML_TYPE_F32 || L.ffn_exp_probs_b->buffer == nullptr) {
                    return refuse(format("another model or quant: layer %d has non-F32 bias tensor", X));
                }
                hash_tensor(&st, L.ffn_exp_probs_b, tmp);
            } else if (arch == "bailingmoe3") {
                return refuse(format("another model or quant: layer %d has no F32 bias tensor", X));
            }
            if (L.ffn_norm == nullptr || L.ffn_norm->type != GGML_TYPE_F32 || L.ffn_norm->buffer == nullptr) {
                return refuse(format("another model or quant: layer %d has no F32 norm tensor", X));
            }
            hash_tensor(&st, L.ffn_norm, tmp);
        }
        unsigned char dig[SHA256_DIGEST_SIZE];
        sha256_final(&st, dig);
        if (hex32(dig) != h.router_sha256) {
            return refuse(format("another model or quant: the sha256 of the router tensors differs (file %s.., model %s..) - this check covers ONLY blk.X.ffn_gate_inp.weight, "
                    "exp_probs_b.bias and ffn_norm.weight of the %zu layers %d..%d (plus the architecture and sizes); the experts and the other weights are not compared",
                    h.router_sha256.substr(0, 12).c_str(), hex32(dig).substr(0, 12).c_str(), h.layers.size(),
                    h.layers.empty() ? -1 : h.layers.front(), h.layers.empty() ? -1 : h.layers.back()));
        }
    }
    const auto t_checked = std::chrono::steady_clock::now();
    if (ms.rec_on() || ms.pred_input_vec_tensor() != nullptr) {
        fprintf(stderr, "llama_moe_correction: NOTE: --moe-stream-record / --moe-stream-pred-input record the CORRECTED guess (the logits of a corrected pair, "
                "and a guess input without the gain): such a recording is not training data for tokenwing-tools/correction/make_correction.py\n");
    }

    // which pairs does this run's guess use?  target streamed; source streamed or a word-start source; lead within the lookahead
    const int64_t n_embd = hp.n_embd, n_exp = hp.n_expert;
    const size_t  nb = (size_t) n_embd * n_exp * sizeof(float);
    std::vector<const llama_moe_corr_pair *> use;
    size_t skipped_unused = 0, skipped_shape = 0;
    for (const llama_moe_corr_pair & p : h.pairs) {
        const bool used = ms.layer(p.target) != nullptr && (ms.layer(p.source) != nullptr || ms.ws_src(p.source)) &&
                          p.lead <= (int32_t) std::max<uint32_t>(1, ms.lookahead);
        if (!used) {
            skipped_unused++;
            continue;
        }
        const llama_layer & L = model.layers[p.target];
        const bool ok_gate_type = L.ffn_gate_inp != nullptr &&
            (L.ffn_gate_inp->type == GGML_TYPE_F32 || L.ffn_gate_inp->type == GGML_TYPE_BF16 || L.ffn_gate_inp->type == GGML_TYPE_F16);
        if (!ok_gate_type || L.ffn_norm == nullptr || L.ffn_gate_inp->buffer == nullptr || L.ffn_norm->buffer == nullptr ||
            L.ffn_gate_inp->ne[0] != n_embd || L.ffn_gate_inp->ne[1] != n_exp ||
            L.ffn_norm->type != GGML_TYPE_F32 || ggml_nelements(L.ffn_norm) != n_embd) {
            skipped_shape++;
            continue;
        }
        use.push_back(&p);
    }
    if (use.empty()) {
        return refuse("no pair of the file is a guess this run makes");
    }

    // one context + buffer per buffer type of the routers (the matrices), one per buffer type of the norms (the ones)
    std::map<ggml_backend_buffer_type_t, std::vector<const llama_moe_corr_pair *>> by_buft;
    std::map<ggml_backend_buffer_type_t, std::vector<int32_t>>                      units_by_buft;
    {
        std::set<int32_t> seen;
        for (const llama_moe_corr_pair * p : use) {
            const llama_layer & L = model.layers[p->target];
            by_buft[ggml_backend_buffer_get_type(L.ffn_gate_inp->buffer)].push_back(p);
            if (seen.insert(p->target).second) {
                units_by_buft[ggml_backend_buffer_get_type(L.ffn_norm->buffer)].push_back(p->target);
            }
        }
    }
    std::map<ggml_tensor *, const llama_moe_corr_pair *> dst_of;
    std::string buft_names;
    size_t total = 0;
    for (auto & [bt, list] : by_buft) {
        ggml_init_params ip = { ggml_tensor_overhead()*(list.size() + 1) + 256, nullptr, true };
        ggml_context * c = ggml_init(ip);
        if (c == nullptr) {
            return refuse("could not create a ggml context");
        }
        ctxs.push_back(c);
        for (const llama_moe_corr_pair * p : list) {
            ggml_tensor * t = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_embd, n_exp);
            ggml_format_name(t, "moe_corr_w-%d-d%d", p->target, p->lead);
            w[{ p->target, p->lead }] = t;
            dst_of[t] = p;
        }
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(c, bt);
        if (b == nullptr) {
            return refuse(format("could not allocate %.1f MiB on %s (not enough memory there)", (double) (list.size()*nb)/1048576.0, ggml_backend_buft_name(bt)));
        }
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bufs.push_back(b);
        total += ggml_backend_buffer_get_size(b);
        buft_names += format("%s%s %.1f MiB (%zu pairs)", buft_names.empty() ? "" : ", ", ggml_backend_buft_name(bt),
                (double) ggml_backend_buffer_get_size(b)/1048576.0, list.size());
    }
    std::map<ggml_tensor *, std::vector<int32_t>> unit_targets;   // the ones tensor -> the targets that use it
    for (auto & [bt, targets] : units_by_buft) {
        ggml_init_params ip = { ggml_tensor_overhead()*2 + 256, nullptr, true };
        ggml_context * c = ggml_init(ip);
        if (c == nullptr) {
            return refuse("could not create a ggml context");
        }
        ctxs.push_back(c);
        ggml_tensor * one = ggml_new_tensor_1d(c, GGML_TYPE_F32, n_embd);
        ggml_set_name(one, "moe_corr_ones");
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(c, bt);
        if (b == nullptr) {
            return refuse(format("could not allocate the ones tensor on %s", ggml_backend_buft_name(bt)));
        }
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        bufs.push_back(b);
        total += ggml_backend_buffer_get_size(b);
        std::vector<float> ones((size_t) n_embd, 1.0f);
        ggml_backend_tensor_set(one, ones.data(), 0, ones.size() * sizeof(float));
        for (int32_t X : targets) {
            unit_of[X] = one;
        }
    }

    // the fold: W' = W_X * g_X + Bt (float32, one multiply and one add per element), done in the ONE pass that reads and hashes the payload: each wanted
    // pair's bytes are folded and uploaded the moment they are complete; the hash of the whole payload decides at the end, and a mismatch discards everything
    std::vector<float> Wx((size_t) n_embd * n_exp), out(Wx.size()), g((size_t) n_embd);
    std::vector<uint16_t> Wx_half;
    std::map<const llama_moe_corr_pair *, ggml_tensor *> tensor_of;
    for (auto & [t, p] : dst_of) {
        tensor_of[p] = t;
    }
    std::vector<size_t> want;
    for (size_t k = 0; k < h.pairs.size(); k++) {
        if (tensor_of.count(&h.pairs[k])) {
            want.push_back(k);
        }
    }
    const std::string bad_stream = llama_moe_corr_file_stream(path, h, want, [&](size_t k, const void * data) {
        const llama_moe_corr_pair & p = h.pairs[k];
        const llama_layer & L = model.layers[p.target];
        if (L.ffn_gate_inp->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(L.ffn_gate_inp, Wx.data(), 0, nb);
        } else {
            Wx_half.resize(Wx.size());
            ggml_backend_tensor_get(L.ffn_gate_inp, Wx_half.data(), 0, Wx.size() * sizeof(uint16_t));
            if (L.ffn_gate_inp->type == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) Wx_half.data(), Wx.data(), (int64_t) Wx.size());
            } else {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) Wx_half.data(), Wx.data(), (int64_t) Wx.size());
            }
        }
        ggml_backend_tensor_get(L.ffn_norm, g.data(), 0, (size_t) n_embd * sizeof(float));
        const float * Bt = (const float *) data;
        for (int64_t e = 0; e < n_exp; e++) {
            const float * wr = Wx.data() + (size_t) e * n_embd;
            const float * br = Bt + (size_t) e * n_embd;
            float *       o  = out.data() + (size_t) e * n_embd;
            for (int64_t i2 = 0; i2 < n_embd; i2++) {
                const float prod = wr[i2] * g[(size_t) i2];
                o[i2] = prod + br[i2];
            }
        }
        ggml_backend_tensor_set(tensor_of[&p], out.data(), 0, nb);
    });
    if (!bad_stream.empty()) {
        return refuse(bad_stream);
    }
    n_bytes = total;
    const auto t1 = std::chrono::steady_clock::now();
    fprintf(stderr, "llama_moe_correction: %s: %zu of the file's %zu pairs are guesses this run makes (%zu not: layer / lead not streamed here%s), "
            "folded W' = W_X*g_X + Bt on the routers' own buffers: %s, %.1f MiB in all; file checked (header, payload sha256 in the same single read as the fold, router-tensor identity %s) in %.1f s, "
            "read + folded in %.1f s; gate: %s\n", path.c_str(), use.size(), h.pairs.size(), skipped_unused,
            skipped_shape ? format(", %zu with another router type / shape", skipped_shape).c_str() : "", buft_names.c_str(), (double) total/1048576.0,
            h.arch.c_str(), std::chrono::duration<double>(t_checked - t0).count(), std::chrono::duration<double>(t1 - t_checked).count(),
            h.gate_rule.empty() ? "-" : h.gate_rule.c_str());
}
