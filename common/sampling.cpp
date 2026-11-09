#include "sampling.h"

#include "common.h"
#include "fit.h"
#include "log.h"
#include "reasoning-budget.h"

#include "ggml.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <random>
#include <unordered_map>
#include <vector>

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * rbudget;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    // for rejection sampling; independent of the draft, or the target distribution is not preserved
    std::mt19937 rng;

    void reset() {
        prev.clear();

        llama_sampler_reset(chain);
    }

    void set_logits(struct llama_context * ctx, int idx) {
        const float *       sampled_probs  = llama_get_sampled_probs_ith     (ctx, idx);
        const float *       sampled_logits = llama_get_sampled_logits_ith    (ctx, idx);
        const llama_token * sampled_ids    = llama_get_sampled_candidates_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        if (sampled_probs) {
            const uint32_t sampled_probs_count = llama_get_sampled_probs_count_ith(ctx, idx);
            cur.resize(sampled_probs_count);
            for (uint32_t i = 0; i < sampled_probs_count; ++i) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], sampled_probs[i]};
            }
        } else if (sampled_logits) {
            const uint32_t sampled_logits_count = llama_get_sampled_logits_count_ith(ctx, idx);
            cur.resize(sampled_logits_count);
            for (uint32_t i = 0; i < sampled_logits_count; i++) {
                cur[i] = llama_token_data{sampled_ids[i], sampled_logits[i], 0.0f};
            }
        } else {
            const auto * logits = llama_get_logits_ith(ctx, idx);
            GGML_ASSERT(logits != nullptr);
            cur.resize(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
            }
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }

    common_time_meas tm() {
        return common_time_meas(t_total_us, params.no_perf);
    }

    mutable int64_t t_total_us = 0;

    int64_t n_pick_backend = 0;   // words a backend sampler picked (--backend-sampling), and the CPU picked
    int64_t n_pick_cpu     = 0;
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f, adaptive_target = %.3f, adaptive_decay = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau, adaptive_target, adaptive_decay);

    return std::string(result);
}

struct common_sampler * common_sampler_init(
        const struct llama_model * model,
        struct common_params_sampling & params) {
    if (!std::isfinite(params.penalty_repeat) ||
        params.penalty_repeat <= 0.0f ||
        !std::isfinite(1.0f/params.penalty_repeat)) {
        throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
    }
    if (!std::isfinite(params.penalty_freq)) {
        throw std::invalid_argument("penalty_freq must be finite");
    }
    if (!std::isfinite(params.penalty_present)) {
        throw std::invalid_argument("penalty_present must be finite");
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    llama_sampler * grmr = nullptr;
    llama_sampler * rbudget = nullptr;
    llama_sampler * chain = llama_sampler_chain_init(lparams);

    std::vector<llama_sampler *> samplers;

    const std::string & grammar_str = common_grammar_value(params.grammar);
    if (grammar_str.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", grammar_str.c_str());
#else
        throw std::runtime_error("failed to parse grammar: llguidance is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<std::string> trigger_patterns;
        std::vector<llama_token> trigger_tokens;
        for (const auto & trigger : params.grammar_triggers) {
            switch (trigger.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                {
                    const auto & word = trigger.value;
                    trigger_patterns.push_back(regex_escape(word));
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                {
                    trigger_patterns.push_back(trigger.value);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                {
                    const auto & pattern = trigger.value;
                    std::string anchored = "^$";
                    if (!pattern.empty()) {
                        anchored = (pattern.front() != '^' ? "^" : "")
                            + pattern
                            + (pattern.back() != '$' ? "$" : "");
                    }
                    trigger_patterns.push_back(anchored);
                    break;
                }
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                {
                    const auto token = trigger.token;
                    trigger_tokens.push_back(token);
                    break;
                }
                default:
                    GGML_ASSERT(false && "unknown trigger type");
            }
        }

        std::vector<const char *> trigger_patterns_c;
        trigger_patterns_c.reserve(trigger_patterns.size());
        for (const auto & regex : trigger_patterns) {
            trigger_patterns_c.push_back(regex.c_str());
        }

        if (!grammar_str.empty()) {
             if (params.grammar_lazy) {
                 grmr = llama_sampler_init_grammar_lazy_patterns(vocab, grammar_str.c_str(), "root",
                         trigger_patterns_c.data(), trigger_patterns_c.size(),
                         trigger_tokens.data(), trigger_tokens.size());
             } else {
                 grmr = llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root");
             }
        }
    }
    if (!grmr && !grammar_str.empty()) {
        throw std::runtime_error("failed to parse grammar");
    }

    // Compute prefill tokens from the generation prompt
    std::vector<llama_token> prefill_tokens;
    if (!params.generation_prompt.empty()) {
        GGML_ASSERT(vocab != nullptr);
        auto tokens = common_tokenize(vocab, params.generation_prompt, false, true);
        for (size_t i = 0; i < tokens.size(); i++) {
            std::string piece = common_token_to_piece(vocab, tokens[i], true);
            if (i == 0 && std::isspace(piece[0]) && !std::isspace(params.generation_prompt[0])) {
                // Some tokenizers will add a space before the first special token, need to exclude
                continue;
            }
            LOG_DBG("%s: prefill token: %d = %s\n", __func__, tokens[i], piece.c_str());
            prefill_tokens.push_back(tokens[i]);
        }
    }

    // Feed generation prompt tokens to the grammar sampler so it advances past
    // tokens the template already placed in the prompt.
    // Only applies to output-format and tool-call grammars; user-supplied grammars must not be prefilled.
    if (grmr && !params.grammar_lazy && common_grammar_needs_prefill(params.grammar)) {
        try {
            for (const auto & token : prefill_tokens) {
                llama_sampler_accept(grmr, token);
                LOG_DBG("%s: grammar accepted prefill token (%d)\n", __func__, token);
            }
        } catch (std::exception &e) {
            LOG_ERR("%s: error initializing grammar sampler for grammar:\n%s\n\nGeneration prompt:\n'%s'\n", __func__,
                common_grammar_value(params.grammar).c_str(), params.generation_prompt.c_str());
            throw e;
        }
    }

    // reasoning budget sampler (skip when budget is unlimited unless a lazy grammar is active, which needs rbudget for thinking-block suppression)
    if (!params.reasoning_budget_start.empty() && !params.reasoning_budget_end.empty() && (params.grammar_lazy || params.reasoning_budget_tokens >= 0 || params.reasoning_control)) {
        rbudget = common_reasoning_budget_init(
            vocab,
            {params.reasoning_budget_start},
            params.reasoning_budget_end,
            params.reasoning_budget_forced,
            params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens);

        for (const auto & token : prefill_tokens) {
            llama_sampler_accept(rbudget, token);
            LOG_DBG("%s: reasoning-budget accepted prefill token (%d)\n", __func__, token);
        }
    }

    // logit bias: user biases + model suppress tokens (-INFINITY)
    {
        std::vector<llama_logit_bias> merged = params.logit_bias;

        int32_t n_suppress = 0;
        const llama_token * suppress = llama_vocab_get_suppress_tokens(vocab, &n_suppress);
        for (int32_t i = 0; i < n_suppress; ++i) {
            merged.push_back({ suppress[i], -INFINITY });
        }

        if (!merged.empty()) {
            samplers.push_back(llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), merged.size(), merged.data()));
        }
    }

    if (params.mirostat == 0) {

        bool use_adaptive_p = false; // see below

        for (const auto & cnstr : params.samplers) {
            switch (cnstr) {
                case COMMON_SAMPLER_TYPE_DRY:
                    {
                        std::vector<const char *> c_breakers;
                        c_breakers.reserve(params.dry_sequence_breakers.size());
                        for (const auto & str : params.dry_sequence_breakers) {
                            c_breakers.push_back(str.c_str());
                        }
                        samplers.push_back(llama_sampler_init_dry(vocab, params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                    }
                    break;
                case COMMON_SAMPLER_TYPE_TOP_K:
                    samplers.push_back(llama_sampler_init_top_k(params.top_k));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_P:
                    samplers.push_back(llama_sampler_init_top_p(params.top_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                    samplers.push_back(llama_sampler_init_top_n_sigma(params.top_n_sigma));
                    break;
                case COMMON_SAMPLER_TYPE_MIN_P:
                    samplers.push_back(llama_sampler_init_min_p(params.min_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_XTC:
                    samplers.push_back(llama_sampler_init_xtc(params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                    break;
                case COMMON_SAMPLER_TYPE_TYPICAL_P:
                    samplers.push_back(llama_sampler_init_typical(params.typ_p, params.min_keep));
                    break;
                case COMMON_SAMPLER_TYPE_TEMPERATURE:
                    samplers.push_back(llama_sampler_init_temp_ext(params.temp, params.dynatemp_range, params.dynatemp_exponent));
                    break;
                case COMMON_SAMPLER_TYPE_INFILL:
                    samplers.push_back(llama_sampler_init_infill(vocab));
                    break;
                case COMMON_SAMPLER_TYPE_PENALTIES:
                    samplers.push_back(llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                    break;
                case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                    // the `adaptive-p` sampler is like `dist` and `mirostat` in that it selects
                    // a single token, so we will add `dist` at the end of the chain by default,
                    // unless the user specifically included `adaptive-p`. we set this flag here
                    // so we know to add the sampler at the very end.
                    use_adaptive_p = true;
                    break;
                default:
                    GGML_ASSERT(false && "unknown sampler type");
            }
        }
        if (use_adaptive_p) {
            // only if user explicitly included adaptive-p sampler
            samplers.push_back(llama_sampler_init_adaptive_p(params.adaptive_target, params.adaptive_decay, params.seed));
        } else {
            // Keep distribution sampling when callers request probabilities.
            const bool greedy = params.n_probs == 0 && !params.samplers.empty() &&
                ((params.samplers.back() == COMMON_SAMPLER_TYPE_TEMPERATURE && params.temp == 0.0f && params.dynatemp_range == 0.0f) ||
                 (params.samplers.back() == COMMON_SAMPLER_TYPE_TOP_K && params.top_k == 1));
            samplers.push_back(greedy ? llama_sampler_init_greedy() : llama_sampler_init_dist(params.seed));
        }
    } else if (params.mirostat == 1) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        samplers.push_back(llama_sampler_init_temp(params.temp));
        samplers.push_back(llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    for (auto * smpl : samplers) {
        llama_sampler_chain_add(chain, smpl);
    }

    if (grmr && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with grammar, disabling\n", __func__);

        params.backend_sampling = false;
    }

    if (rbudget && params.backend_sampling) {
        LOG_WRN("%s: backend sampling is not compatible with reasoning budget, disabling\n", __func__);

        params.backend_sampling = false;
    }

    auto * result = new common_sampler {
        /* .params  = */ params,
        /* .grmr    = */ grmr,
        /* .rbudget = */ rbudget,
        /* .chain   = */ chain,
        /* .prev    = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur     = */ {},
        /* .cur_p   = */ {},
        // mix it, the chain and the draft are seeded from this one too
        /* .rng     = */ std::mt19937(llama_sampler_get_seed(chain) ^ 0x9e3779b9u),
    };

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    llama_sampler_free(gsmpl->grmr);
    llama_sampler_free(gsmpl->rbudget);
    llama_sampler_free(gsmpl->chain);

    delete gsmpl;
}

static bool grammar_should_apply(struct common_sampler * gsmpl) {
    if (!gsmpl->grmr) {
        return false;
    }
    if (!gsmpl->rbudget) {
        return true;
    }
    if (gsmpl->params.grammar_lazy) {
        // if grammar is lazy, only apply when reasoning budget is not active
        const auto state = common_reasoning_budget_get_state(gsmpl->rbudget);
        return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
    }
    return true;
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool is_generated) {
    if (!gsmpl) {
        return;
    }

    const auto tm = gsmpl->tm();

    // grammar_should_apply() checks the reasoning budget state, so calculate this before we accept
    const auto accept_grammar = is_generated && grammar_should_apply(gsmpl);

    if (gsmpl->rbudget && is_generated) {
        llama_sampler_accept(gsmpl->rbudget, token);

        // if done, replay end sequence which may contain a grammar trigger
        const bool is_done = common_reasoning_budget_get_state(gsmpl->rbudget) == REASONING_BUDGET_DONE;
        if (gsmpl->grmr && !accept_grammar && is_done) {
            const llama_tokens * end_seq = common_reasoning_budget_get_end_match(gsmpl->rbudget);
            if (end_seq) {
                for (const llama_token end_token : *end_seq) {
                    llama_sampler_accept(gsmpl->grmr, end_token);
                }
            }
        }
    }

    if (gsmpl->grmr && accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return;
    }

    gsmpl->reset();
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    return new common_sampler {
        /* .params  = */ gsmpl->params,
        /* .grmr    = */ llama_sampler_clone(gsmpl->grmr),
        /* .rbudget = */ llama_sampler_clone(gsmpl->rbudget),
        /* .chain   = */ llama_sampler_clone(gsmpl->chain),
        /* .prev    = */ gsmpl->prev,
        /* .cur     = */ gsmpl->cur,
        /* .cur_p   = */ gsmpl->cur_p,
        /* .rng     = */ gsmpl->rng,
    };
}

void common_sampler_copy(const common_sampler * src, common_sampler * dst) {
    if (!src || !dst || src == dst) {
        return;
    }

    GGML_ASSERT((src->grmr == nullptr) == (dst->grmr == nullptr));
    GGML_ASSERT((src->rbudget == nullptr) == (dst->rbudget == nullptr));

    llama_sampler_copy(src->grmr,    dst->grmr);
    llama_sampler_copy(src->rbudget, dst->rbudget);
    llama_sampler_copy(src->chain,   dst->chain);

    dst->params     = src->params;
    dst->prev       = src->prev;
    dst->cur        = src->cur;
    dst->cur_p      = src->cur_p;
    dst->cur_p.data = src->cur_p.data ? dst->cur.data() : nullptr; // re-point to dst's buffer
    dst->rng        = src->rng;
    dst->t_total_us = src->t_total_us;
}


// The 90th percentile of the waiting list's add -> bind delay from its 100 us buckets (the last one is open): "-"
// with no bind, ">= 9.9 ms" in the open bucket, else "< X ms" (the bucket's upper edge). Also used by the server.
std::string common_moe_pl_p90(const int64_t * hist, int64_t n) {
    if (n <= 0) {
        return "-";
    }
    int64_t cum = 0;
    for (int i = 0; i < 100; i++) {
        cum += hist[i];
        if (10*cum >= 9*n) {
            return i >= 99 ? std::string(">= 9.9 ms") : string_format("< %.1f ms", (i + 1)*0.1);
        }
    }
    return ">= 9.9 ms";
}

// The decode token's total time (stall + the op's own time + the read window), so the guess op's
// cost can be given as a share of it
static double pc_ms_total(const llama_moe_stream_stats & ms) {
    const double nt = ms.dec.n_tokens > 0 ? ms.dec.n_tokens : 1;
    return (ms.dec.t_stall_us + ms.dec.t_op_us + ms.dec.t_read_us)/1000.0/nt;
}

void common_sampler_pick_counts(const struct common_sampler * gsmpl, int64_t * n_backend, int64_t * n_cpu) {
    *n_backend = gsmpl ? gsmpl->n_pick_backend : 0;
    *n_cpu     = gsmpl ? gsmpl->n_pick_cpu     : 0;
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    const double t_sampling_ms = gsmpl ? 1e-3*gsmpl->t_total_us : 0;

    llama_perf_sampler_data data_smpl;
    llama_perf_context_data data_ctx;

    memset(&data_smpl, 0, sizeof(data_smpl));
    memset(&data_ctx,  0, sizeof(data_ctx));

    if (gsmpl) {
        auto & data = data_smpl;

        data = llama_perf_sampler(gsmpl->chain);

        // note: the sampling time includes the samplers time + extra time spent in common/sampling
        LOG_INF("%s:    sampling time = %10.2f ms\n", __func__, t_sampling_ms);
        LOG_INF("%s:    samplers time = %10.2f ms / %5d tokens\n", __func__, data.t_sample_ms, data.n_sample);
        // Where the words were picked (with --backend-sampling, a backend sampler picks on the GPU)
        LOG_INF("%s:    picked words  = %lld on the GPU (backend sampling), %lld on the CPU\n", __func__,
                (long long) gsmpl->n_pick_backend, (long long) gsmpl->n_pick_cpu);
    }

    if (ctx) {
        auto & data = data_ctx;

        data = llama_perf_context(ctx);

        const double t_end_ms = 1e-3 * ggml_time_us();

        const double t_total_ms = t_end_ms - data.t_start_ms;
        const double t_unacc_ms = t_total_ms - (t_sampling_ms + data.t_p_eval_ms + data.t_eval_ms);
        const double t_unacc_pc = 100.0 * t_unacc_ms /  t_total_ms;

        LOG_INF("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
        LOG_INF("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
        LOG_INF("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
                __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
        LOG_INF("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
        LOG_INF("%s: unaccounted time = %10.2f ms / %5.1f %%      (total - sampling - prompt eval - eval) / (total)\n", __func__, t_unacc_ms, t_unacc_pc);
        LOG_INF("%s:    graphs reused = %10d\n", __func__, data.n_reused);
        LOG_INF("%s:   graph reallocs = %10d    / %5d        (same size) / (changed size)\n", __func__, data.n_reallocs_same, data.n_reallocs_diff);

        {
            llama_moe_stream_stats ms;

            if (llama_moe_stream_get_stats(ctx, &ms)) {
                const double n_token = (double) ms.n_tokens;
                const double n_touch = (double) (ms.n_hit + ms.n_miss);
                // A run where every MoE layer is resident in VRAM streams nothing, so the
                // per-layer figures have no denominator
                const double n_layer = ms.n_layers > 0 ? (double) ms.n_layers : 1.0;

                LOG_INF("%s: moe stream: %d layers x %d slots, %.0f tokens, %d experts used per token per layer\n",
                        __func__, ms.n_layers, ms.n_slots, n_token, ms.n_expert_used);
                LOG_INF("%s: moe stream: distinct (layer, expert) touches %.1f/token (%.1f/token/layer)\n",
                        __func__,
                        n_token > 0 ? ms.n_distinct/n_token : 0.0,
                        n_token > 0 ? ms.n_distinct/n_token/n_layer : 0.0);
                LOG_INF("%s: moe stream: hits %.3f/token, misses %.3f/token, hit rate %.2f%%\n",
                        __func__,
                        n_token > 0 ? ms.n_hit/n_token : 0.0,
                        n_token > 0 ? ms.n_miss/n_token : 0.0,
                        n_touch > 0 ? 100.0*ms.n_hit/n_touch : 0.0);
                // The uses the VRAM expert pool served, neither a hit nor a miss of the RAM pools
                if (ms.n_vram_hit > 0) {
                    const double n_dec = (double) ms.dec.n_tokens;
                    LOG_INF("%s: moe stream: VRAM expert hits %.3f/token (%.2f%% of the expert uses), decode %.3f/token\n",
                            __func__, n_token > 0 ? ms.n_vram_hit/n_token : 0.0,
                            ms.n_expert_uses > 0 ? 100.0*ms.n_vram_hit/ms.n_expert_uses : 0.0,
                            n_dec > 0 ? ms.dec.n_vram_hit/n_dec : 0.0);
                }
                // of the reads the call did not have to wait for, the share that a prefetch paid for
                const double n_avail = (double) (ms.n_hit_pref + ms.n_miss);

                LOG_INF("%s: moe stream: prefetch covers %.3f/token (%.2f%% of the avoided reads, %.3f/token of them still in flight), prefetches %.3f/token, wasted %.3f/token\n",
                        __func__,
                        n_token > 0 ? ms.n_hit_pref/n_token : 0.0,
                        n_avail > 0 ? 100.0*ms.n_hit_pref/n_avail : 0.0,
                        n_token > 0 ? ms.n_pref_late/n_token : 0.0,
                        n_token > 0 ? ms.n_pref/n_token : 0.0,
                        n_token > 0 ? ms.n_pref_waste/n_token : 0.0);
                // What the two-layer lead issued, what the correction cancelled, and the
                // bytes split by who asked for them (a demand miss that a prefetch covered reads
                // nothing of its own, so demand bytes are the identity misses x 3,133,440 B)
                if (ms.n_pref > 0 || ms.dec.n_pref_cancel > 0 || ms.dec.bytes_demand > 0) {
                    const double nt72 = ms.dec.n_tokens > 0 ? ms.dec.n_tokens : 1;
                    // What the drive was already doing when the compute needed a read. Without a
                    // cap this is the number that explains the stall, and with the cap it is what the
                    // cap is worth. The sample count is the histogram's total, not the miss count, so
                    // that a re-issued read (the wait loop) is counted as the sample it is.
                    {
                        int64_t nsamp = 0;
                        std::string bk;
                        char buf[32];
                        for (int b = 0; b < 17; b++) {   // the total first (the % used a running sum)
                            nsamp += ms.dec.n_inflight_hist[b];
                        }
                        for (int b = 0; b < 17; b++) {
                            if (ms.dec.n_inflight_hist[b] <= 0) {
                                continue;
                            }
                            snprintf(buf, sizeof(buf), "%s%d:%.0f%%", bk.empty() ? "" : " ", b*4,
                                     100.0*ms.dec.n_inflight_hist[b]/(nsamp ? nsamp : 1));
                            bk += buf;
                        }
                        // The achieved lead of the guesses - issued -> the target layer asking for the
                    // expert - with the percentiles from the 50 us histogram, and the guess op's own
                    // cost on the graph thread (of which the lock wait is a part).
                    {
                        int64_t nh = 0;
                        for (int b = 0; b < 225; b++) {
                            nh += ms.dec.lead_hist[b];
                        }
                        if (nh > 0) {
                            auto pct = [&](double q) {
                                int64_t want = (int64_t) (q*nh), acc = 0;
                                for (int b = 0; b < 225; b++) {
                                    acc += ms.dec.lead_hist[b];
                                    if (acc >= want) {
                                        return 50.0*b;
                                    }
                                }
                                return 50.0*224;
                            };
                            LOG_INF("%s: moe stream: achieved lead of a guess: %lld samples, "
                                    "p10 %.0f us, p50 %.0f us, p90 %.0f us, mean %.0f us, max %.0f us; "
                                    "landed in time %.3f/token of %.3f/token asked, late %.3f/token\n",
                                    __func__, (long long) nh, pct(0.10), pct(0.50), pct(0.90),
                                    ms.dec.lead_sum_us/(double) nh, (double) ms.dec.lead_max_us,
                                    nt72 ? (ms.dec.n_hit_pref - ms.dec.n_pref_late)/nt72 : 0.0,
                                    nt72 ? ms.dec.n_hit_pref/nt72 : 0.0,
                                    nt72 ? ms.dec.n_pref_late/nt72 : 0.0);
                        }
                        if (ms.dec.n_pred_ops > 0) {
                            LOG_INF("%s: moe stream: guess op on the graph thread: %.3f ms/token over "
                                    "%lld calls, of which %.3f ms/token waiting for the engine lock "
                                    "(%.0f%%), against a decode token of %.2f ms\n",
                                    __func__, ms.dec.t_pred_us/1000.0/nt72, (long long) ms.dec.n_pred_ops,
                                    ms.dec.t_pred_lock_us/1000.0/nt72,
                                    100.0*ms.dec.t_pred_lock_us/(ms.dec.t_pred_us ? ms.dec.t_pred_us : 1),
                                    nt72 ? pc_ms_total(ms) : 0.0);
                        }
                    }
                    LOG_INF("%s: moe stream: in flight at a needed read: %.2f reads, %.2f of them "
                                "guessed (%.0f samples), cap turned away %.3f/token, buckets of 4: %s\n",
                                __func__,
                                nsamp > 0 ? ms.dec.n_inflight_at_demand/(double) nsamp : 0.0,
                                nsamp > 0 ? ms.dec.n_inflight_at_demand_pref/(double) nsamp : 0.0,
                                (double) nsamp,
                                nt72 ? ms.dec.n_pref_capped/nt72 : 0.0, bk.c_str());
                    }
                    LOG_INF("%s: moe stream: prefetch detail: prefetches %.3f/token, covered %.3f/token, "
                            "wasted %.3f/token, correction cancels %.3f/token (%.2f MiB), "
                            "demand %.3f MiB/token, prefetch %.3f MiB/token\n",
                            __func__,
                            nt72 ? ms.dec.n_pref/nt72 : 0.0,
                            nt72 ? ms.dec.n_hit_pref/nt72 : 0.0,
                            nt72 ? ms.dec.n_pref_waste/nt72 : 0.0,
                            nt72 ? ms.dec.n_pref_cancel/nt72 : 0.0,
                            nt72 ? ms.dec.n_pref_cancel_bytes/nt72/1024.0/1024.0 : 0.0,
                            nt72 ? ms.dec.bytes_demand/nt72/1024.0/1024.0 : 0.0,
                            nt72 ? ms.dec.bytes_prefetch/nt72/1024.0/1024.0 : 0.0);
                }
                if (ms.n_wave_ubatches > 0) {
                    LOG_INF("%s: moe stream: %lld ubatch-layer pairs needed more than one wave, %lld wave stages, "
                            "%.3f reads per ubatch and layer for %.3f distinct experts (%.3f per expert)\n",
                            __func__,
                            (long long) ms.n_wave_ubatches, (long long) ms.n_wave_stages,
                            (double) ms.n_wave_reads/ms.n_wave_ubatches,
                            (double) ms.n_wave_experts/ms.n_wave_ubatches,
                            ms.n_wave_experts > 0 ? (double) ms.n_wave_reads/ms.n_wave_experts : 0.0);

                    if (ms.n_wave_reads > ms.n_wave_experts) {
                        LOG_WRN("%s: moe stream: %lld wave reads for %lld distinct experts: an expert was read more "
                                "than once for a ubatch and layer\n",
                                __func__, (long long) ms.n_wave_reads, (long long) ms.n_wave_experts);
                    }
                }

                if (ms.n_oracle_cmp > 0) {
                    const double pc   = 100.0*ms.n_oracle_miss/ms.n_oracle_cmp;
                    const double pc_d = ms.n_or_cmp_dec > 0 ? 100.0*ms.n_or_miss_dec/ms.n_or_cmp_dec : 0.0;

                    LOG_INF("%s: moe stream: oracle compared %lld ids with the trace, %lld differ (%.3f%%); the "
                            "decode calls alone: %lld of %lld (%.3f%%)\n",
                            __func__, (long long) ms.n_oracle_cmp, (long long) ms.n_oracle_miss, pc,
                            (long long) ms.n_or_miss_dec, (long long) ms.n_or_cmp_dec, pc_d);

                    if (pc > 1.0) {
                        LOG_WRN("%s: moe stream: the oracle and the router disagree on %.2f%% of the ids (%.2f%% of "
                                "the decode ids), more than 1%%. The prefill rows of a trace can differ by "
                                "construction (batch-shape effects), so record the trace with the same -ub to get a "
                                "perfect oracle.\n", __func__, pc, pc_d);
                    }
                }

                LOG_INF("%s: moe stream: read %.2f MiB/token (%.2f GiB total, %lld reads), stalled %.3f ms/token (%.2f s total), %.2f GB/s achieved\n",
                        __func__,
                        n_token > 0 ? ms.bytes_read/n_token/1024.0/1024.0 : 0.0,
                        ms.bytes_read/1024.0/1024.0/1024.0,
                        (long long) ms.n_reads,
                        n_token > 0 ? ms.t_stall_us/1000.0/n_token : 0.0,
                        ms.t_stall_us/1e6,
                        ms.t_read_win_us > 0 ? 1e-3*ms.bytes_read/ms.t_read_win_us : 0.0);

                // The same figures per phase. The counters above mix the prompt and the decode
                // tokens in one denominator, which is where the stall per token of a decode run went
                // wrong: the tool divides by its own token count, the engine by prompt + decode.
                // pc.n_tokens is the number of real tokens of that phase (counted once per
                // batch, not once per layer), so the per-token figures below are per token and the
                // per-layer ones are labelled as such. The stall, the bytes and the reads are summed
                // over the layers, which is what a token of the model goes through.
                for (int phase = 0; phase < 2; phase++) {
                    const auto & pc  = phase == 0 ? ms.pre : ms.dec;
                    const double nt  = (double) pc.n_tokens;
                    const double ntouch = (double) (pc.n_hit + pc.n_miss);

                    if (nt <= 0.0) {
                        continue;
                    }

                    LOG_INF("%s: moe stream: %-6s tokens = %.0f, misses = %.3f/token (%.4f/token/layer), hit "
                            "rate = %.2f%%, read = %.2f MiB/token (%.4f/token/layer), %lld reads (%.2f "
                            "chunks each), stall = %.3f ms/token, remap op = %.3f ms/token, "
                            "%lld prefetch cancels, %lld read retries\n",
                            __func__, phase == 0 ? "prompt" : "decode", nt,
                            pc.n_miss/nt, pc.n_miss/nt/n_layer,
                            ntouch > 0.0 ? 100.0*pc.n_hit/ntouch : 0.0,
                            pc.bytes_read/nt/1024.0/1024.0,
                            pc.bytes_read/nt/1024.0/1024.0/n_layer, (long long) pc.n_reads,
                            pc.n_reads > 0 ? (double) pc.n_chunks/pc.n_reads : 0.0,
                            pc.t_stall_us/1000.0/nt, pc.t_op_us/1000.0/nt,
                            (long long) pc.n_cancel_wait, (long long) pc.n_read_retry);
                    if (pc.t_cb_wait_us > 0) {   // the copy-back waits, which the stall above leaves out
                        LOG_INF("%s: moe stream [%s]: waits on a copy-back from the VRAM pool = %.3f ms/token (not SSD wait; the stall "
                                "booked as before, SSD + copy-back = %.3f ms/token)\n", __func__, phase == 0 ? "prompt" : "decode",
                                pc.t_cb_wait_us/1000.0/nt, pc.t_wait_all_us/1000.0/nt);
                    }
                }

                // The picked experts of a decode word that were not in RAM when their layer's remap op started, by
                // cause, with each layer's lateness booked to the cause of its last-landing expert; and how many of the
                // needed ones (first, named-not-read, not-named) this pool had evicted <= 8 / <= 64 words before
                if (ms.dec.n_tokens > 0) {
                    const double nw = (double) ms.dec.n_tokens;
                    LOG_INF("%s: moe stream: causes [decode] per word: late %.2f (%.2f ms), no-guess %.2f (%.2f ms), "
                            "named-not-read %.2f (%.2f ms), not-named %.2f (%.2f ms); read-backs <=8: %.2f (by guess %.2f), "
                            "<=64: %.2f (by guess %.2f); evicted by the prompt: %.2f\n",
                            __func__,
                            ms.dec.n_cause[0]/nw, ms.dec.t_cause_us[0]/1000.0/nw,
                            ms.dec.n_cause[1]/nw, ms.dec.t_cause_us[1]/1000.0/nw,
                            ms.dec.n_cause[2]/nw, ms.dec.t_cause_us[2]/1000.0/nw,
                            ms.dec.n_cause[3]/nw, ms.dec.t_cause_us[3]/1000.0/nw,
                            ms.dec.n_rb8/nw, ms.dec.n_rb8_guess/nw, ms.dec.n_rb64/nw, ms.dec.n_rb64_guess/nw,
                            ms.dec.n_rb_prompt/nw);

                    // The goal number, both paths, and the waiting list (--moe-stream-plan 1)
                    if (ms.dec.n_goal_cand > 0) {
                        LOG_INF("%s: moe stream: goal [decode]: of the 1-ahead guess's top-k, %.2f%% were READY when the "
                                "guess was made (%lld of %lld)\n", __func__, 100.0*ms.dec.n_goal_ready/ms.dec.n_goal_cand,
                                (long long) ms.dec.n_goal_ready, (long long) ms.dec.n_goal_cand);
                    }
                    // The guess reads that their layer used (landed in time or late), with and without the binds at the
                    // pick (reads the router had already asked for, always used)
                    if (ms.dec.n_pref > 0) {
                        const double pk = (double) ms.dec.n_pl_bind_pick;
                        LOG_INF("%s: moe stream: guess precision [decode]: %.2f%% of the guess reads were used (%lld of %lld); "
                                "without the %lld binds at the pick %.2f%%\n", __func__,
                                100.0*ms.dec.n_hit_pref/ms.dec.n_pref, (long long) ms.dec.n_hit_pref, (long long) ms.dec.n_pref,
                                (long long) ms.dec.n_pl_bind_pick,
                                ms.dec.n_pref - pk > 0 ? 100.0*(ms.dec.n_hit_pref - pk)/(ms.dec.n_pref - pk) : 0.0);
                    }
                    // The guess helper (--moe-stream-needed-early 1)
                    if (ms.dec.n_ne_post > 0) {
                        // With lookahead, the columns skipped as too late by distance (a post counts as late if any is)
                        std::string late_d;
                        for (int d = 1; d <= 8; d++) {
                            if (ms.dec.n_ne_late_d[d] > 0) {
                                late_d += string_format("%sd%d %.2f", late_d.empty() ? " (columns by distance: " : ", ", d,
                                        ms.dec.n_ne_late_d[d]/nw);
                            }
                        }
                        late_d += late_d.empty() ? "" : ")";
                        LOG_INF("%s: moe stream: helper [decode] per word: posted %.2f, adds done %.2f, guess too late %.2f%s; "
                                "post -> adds done %.3f ms mean, p90 %s; word-end drain: %lld calls waited, %.3f ms in all\n",
                                __func__, ms.dec.n_ne_post/nw, ms.dec.n_ne_done/nw, ms.dec.n_ne_late/nw, late_d.c_str(),
                                ms.dec.n_ne_done > 0 ? ms.dec.t_ne_delay_us/1000.0/ms.dec.n_ne_done : 0.0,
                                common_moe_pl_p90(ms.dec.ne_delay_hist, ms.dec.n_ne_done).c_str(),
                                (long long) ms.dec.n_ne_drain_calls, ms.dec.t_ne_drain_us/1000.0);
                    }
                    // The GPU producer's "router done" (the CPU splits started from a router slot) and the helper's waits
                    if (ms.n_rd_split > 0 || ms.n_rd_uncovered > 0 || ms.n_ne_ready + ms.n_ne_event > 0 || ms.n_rd_unfit > 0) {
                        LOG_INF("%s: moe stream: router done [decode] per word: CPU splits from the slot %.2f (number already there %.2f, "
                                "after the event %.2f; %.3f ms in the hook), not covered %.2f; helper: number already there %.2f, after "
                                "the event %.2f; layers left unmarked at graph builds %lld\n", __func__, ms.n_rd_split/nw, ms.n_rd_ready/nw,
                                ms.n_rd_event/nw, ms.t_rd_us/1000.0/nw, ms.n_rd_uncovered/nw, ms.n_ne_ready/nw, ms.n_ne_event/nw,
                                (long long) ms.n_rd_unfit);
                    }
                    // VRAM follow (--moe-stream-vram-follow 1), engine-wide counts per reply word
                    if (ms.n_vf_words > 0) {
                        const double xw = (double) ms.n_vf_words;
                        LOG_INF("%s: moe stream: VRAM follow [decode] per word: VRAM picks %.2f, swaps %.3f, copies %.3f, switches %.3f; "
                                "skipped (source loading %.3f, not in RAM %.3f, mark cap %.3f); %lld words, %lld ticks (%lld stopped by "
                                "the hysteresis), %lld switches put off, %lld GPU waits (%.3f ms each), %lld drains; sources RAM %lld / sidecar %lld / after a "
                                "wait %lld; read back %lld; mark mismatches %lld, marks at a prompt %lld\n", __func__,
                                ms.dec.n_vram_hit/nw, ms.n_vf_swaps/xw, ms.n_vf_copies/xw,
                                ms.n_vf_switches/xw, ms.n_vf_skip_loading/xw, ms.n_vf_skip_notram/xw, ms.n_vf_skip_marks/xw,
                                (long long) ms.n_vf_words, (long long) ms.n_vf_ticks, (long long) ms.n_vf_hysteresis_stop,
                                (long long) ms.n_vf_switch_wait, (long long) ms.n_vf_gpu_syncs,
                                ms.n_vf_gpu_syncs > 0 ? ms.t_vf_gpu_wait_us/1000.0/ms.n_vf_gpu_syncs : 0.0, (long long) ms.n_vf_drains,
                                (long long) ms.n_vf_src_ram,
                                (long long) ms.n_vf_src_ssd, (long long) ms.n_vf_src_waited, (long long) ms.n_vf_verified,
                                (long long) ms.n_vf_mark_mismatch, (long long) ms.n_vf_marked_prompt);
                    }
                    // The guess's extra ranks k+1..K (--moe-stream-guess-top)
                    // Every removal reason (they and the binds add up to "added"), and "bound, not picked, not cancelled" =
                    // bound - used - cancelled by drop stale (reads that landed unused, still in flight at the target's start, or
                    // given up with their slot). Counted for every rank above k: from guess-top, or prefetch-m above k.
                    if (ms.dec.n_gx_add > 0) {
                        const auto & r = ms.dec.n_gx_rm;   // [0] picked [1] present [2] no room [3] word end [4] list full [5] aborted [6] stale
                        const double un = (double) (ms.dec.n_gx_bind - ms.dec.n_gx_used - ms.dec.n_gx_stale_cx);
                        LOG_INF("%s: moe stream: extra ranks [decode] (ranks above k, from guess-top or prefetch-m above k) per word: added %.2f, "
                                "bound by a reader %.2f, removed unread: stale %.2f, word end %.2f, no room %.2f, present %.2f, list full %.2f, "
                                "aborted %.2f; picked by the target: READY %.2f, loading %.2f, on the list (bound at the pick %.2f, no room "
                                "%.2f); bound and not picked %.2f (cancelled by drop stale %.2f; bound, not picked, not cancelled %.2f); needed "
                                "reads saved per extra read %.3f\n", __func__,
                                ms.dec.n_gx_add/nw, ms.dec.n_gx_bind/nw, r[6]/nw, r[3]/nw,
                                r[2]/nw, r[1]/nw, r[4]/nw, r[5]/nw,
                                (ms.dec.n_gx_used - ms.dec.n_gx_used_late)/nw, ms.dec.n_gx_used_late/nw, ms.dec.n_gx_bind_pick/nw,
                                r[0]/nw, (ms.dec.n_gx_bind - ms.dec.n_gx_used)/nw, ms.dec.n_gx_stale_cx/nw, un/nw,
                                ms.dec.n_gx_bind > 0 ? (double) (ms.dec.n_gx_used - ms.dec.n_gx_used_late)/ms.dec.n_gx_bind : 0.0);
                    }
                    // Lookahead N > 1 - per distance (the names add up: new + vote + pace + pool), the entries by their
                    // latest distance at the bind, dropped entries and the mid-way rule
                    if (ms.dec.n_la_named[2] > 0) {
                        std::string per_d;
                        for (int d = 1; d <= 8; d++) {
                            if (ms.dec.n_la_named[d] == 0) {
                                continue;
                            }
                            per_d += string_format("%sd%d: named %.2f = new %.2f + vote %.2f + pace %.2f + pool %.2f; bound %.2f, at the "
                                    "pick %.2f, picked by the target %.2f (READY %.2f, loading %.2f)", per_d.empty() ? "" : "; ", d,
                                    ms.dec.n_la_named[d]/nw, ms.dec.n_la_new[d]/nw, ms.dec.n_la_vote[d]/nw, ms.dec.n_la_pace[d]/nw,
                                    ms.dec.n_la_pool[d]/nw, ms.dec.n_la_bind[d]/nw, ms.dec.n_la_bind_pick[d]/nw,
                                    ms.dec.n_la_used[d]/nw, (ms.dec.n_la_used[d] - ms.dec.n_la_used_late[d])/nw,
                                    ms.dec.n_la_used_late[d]/nw);
                        }
                        LOG_INF("%s: moe stream: lookahead [decode] per word (new by the distance that added the entry, bound / picked "
                                "by its latest distance at the bind: only the totals over the distances add up): %s; dropped: became "
                                "%.2f, bound while dropped (its read starts moved back) %.2f, named again %.2f, removed at the target's "
                                "start %.2f; mid-way: started reads with a queued piece moved back %.2f; of all moved-back reads: picked "
                                "by the target while loading %.2f, cancelled at its start %.2f, named again %.2f\n", __func__, per_d.c_str(),
                                ms.dec.n_dr_became/nw, ms.dec.n_dr_bound/nw, ms.dec.n_dr_again/nw, ms.dec.n_dr_removed/nw,
                                ms.dec.n_mw_back/nw, ms.dec.n_mw_promoted/nw, ms.dec.n_mw_cancelled/nw, ms.dec.n_mw_again/nw);
                    }
                    // The in-graph form of CPU layers (printed when a layer took it)
                    if (ms.dec.n_ig_calls > 0) {
                        LOG_INF("%s: moe stream: in-graph [decode] per word: layers %.2f, op A -> op B %.3f ms per layer (%.3f ms per "
                                "word summed: how much earlier the needed reads were queued; of it op A itself %.3f ms per layer, the "
                                "work between A and B %.3f), needed reads queued by op A %.2f\n",
                                __func__, ms.dec.n_ig_calls/nw, ms.dec.t_ig_ab_us/1000.0/ms.dec.n_ig_calls, ms.dec.t_ig_ab_us/1000.0/nw,
                                ms.dec.t_ig_a_us/1000.0/ms.dec.n_ig_calls, (ms.dec.t_ig_ab_us - ms.dec.t_ig_a_us)/1000.0/ms.dec.n_ig_calls,
                                ms.dec.n_ig_reads/nw);
                    }
                    // The per-projection ready flags (printed when a needed read was split)
                    if (ms.dec.n_pr_split > 0 || ms.pre.n_pr_split > 0) {
                        LOG_INF("%s: moe stream: proj-ready [decode] per word: needed reads split (gate+up, then down) %.2f, gate/up "
                                "projections computed before their expert's down landed %.2f, waited-on experts whose down landed after "
                                "their gate+up %.2f (%.3f ms mean gap); [prompt] split reads %lld, early gate/up %lld\n",
                                __func__, ms.dec.n_pr_split/nw, ms.dec.n_pr_early/nw, ms.dec.n_pr_gap/nw,
                                ms.dec.n_pr_gap > 0 ? ms.dec.t_pr_gap_us/1000.0/ms.dec.n_pr_gap : 0.0,
                                (long long) ms.pre.n_pr_split, (long long) ms.pre.n_pr_early);
                    }
                    // Protect named (printed when it kept an expert)
                    if (ms.dec.n_pn_refused > 0) {
                        LOG_INF("%s: moe stream: protect-named [decode] per word: guess victim choices the rule changed (the "
                                "rule-free victim was a named expert) %.2f, of them with no other victim (the entry skipped for no "
                                "room) %.2f; experts kept from eviction %.2f, of them picked by their layer (a read saved) %.2f, not "
                                "picked %.2f\n", __func__,
                                ms.dec.n_pn_refused/nw, ms.dec.n_pn_noroom/nw, ms.dec.n_pn_kept/nw, ms.dec.n_pn_kept_picked/nw,
                                ms.dec.n_pn_kept_unpicked/nw);
                    }
                    // The first streamed layer's needed reads (any setting) and the word-start op (--moe-stream-wordstart D):
                    // the guesses per source layer and distance, and per target (its index among the streamed layers) the op's
                    // entries and the picks a word-start guess named, by state at the target's start (the four parts add up)
                    if (ms.first_streamed >= 0) {
                        std::string ws;
                        if (ms.ws_D > 0) {
                            std::string g, t;
                            for (int j = 1; j <= ms.ws_D && j <= 8; j++) {
                                for (int d = 1; d <= 8; d++) {
                                    if (ms.n_ws_guess[j][d] > 0) {
                                        g += string_format("%slayer %d d%d %.2f", g.empty() ? "" : ", ", ms.first_streamed - j, d,
                                                ms.n_ws_guess[j][d]/nw);
                                    }
                                }
                            }
                            for (int j = 0; j < 8; j++) {
                                const int64_t pk = ms.n_ws_pick[j][0] + ms.n_ws_pick[j][1] + ms.n_ws_pick[j][2] + ms.n_ws_pick[j][3];
                                if (ms.n_ws_add[j] == 0 && pk == 0) {
                                    continue;
                                }
                                t += string_format("%sstreamed #%d: added %.2f, bound %.2f, at the pick %.2f, removed unread %.2f; picks "
                                        "named %.2f = READY %.2f + loading %.2f + on the list %.2f + none %.2f", t.empty() ? "" : "; ", j,
                                        ms.n_ws_add[j]/nw, ms.n_ws_bind[j]/nw, ms.n_ws_bind_pick[j]/nw, ms.n_ws_rm[j]/nw, pk/nw,
                                        ms.n_ws_pick[j][0]/nw, ms.n_ws_pick[j][1]/nw, ms.n_ws_pick[j][2]/nw, ms.n_ws_pick[j][3]/nw);
                            }
                            ws = string_format("; wordstart D %d: words with adds %lld, op runs with none %lld; guesses per word: %s; per "
                                    "target: %s", ms.ws_D, (long long) ms.n_ws_words, (long long) ms.n_ws_skip,
                                    g.empty() ? "none" : g.c_str(), t.empty() ? "none" : t.c_str());
                            // The source slots (needed-early 1): posts, the helper's adds, too late (columns by distance)
                            if (ms.n_ws_post > 0) {
                                std::string ld;
                                for (int d = 1; d <= 8; d++) {
                                    if (ms.n_ws_late_d[d] > 0) {
                                        ld += string_format("%sd%d %.2f", ld.empty() ? " (columns by distance: " : ", ", d, ms.n_ws_late_d[d]/nw);
                                    }
                                }
                                ld += ld.empty() ? "" : ")";
                                ws += string_format("; source slots per word: posted %.2f, adds done %.2f (posts the helper took; at N > 1 also a post whose columns were "
                                        "all too late), posts with a column too late %.2f%s, "
                                        "post -> adds done %.3f ms mean", ms.n_ws_post/nw, ms.n_ws_done/nw, ms.n_ws_late/nw, ld.c_str(),
                                        ms.n_ws_done > 0 ? ms.t_ws_delay_us/1000.0/ms.n_ws_done : 0.0);
                            }
                        }
                        LOG_INF("%s: moe stream: first streamed layer %d [decode]: needed reads per word %.3f%s\n", __func__,
                                ms.first_streamed, ms.n_first_miss/nw, ws.c_str());
                    }
                    // The behind rule / guess-inflight (printed when either acted or counted a turn)
                    if (ms.dec.n_turns > 0 || ms.dec.n_gi_wait > 0) {
                        LOG_INF("%s: moe stream: behind [decode]: %lld of %.0f words behind at least once, %.2f%% of the reader turns "
                                "taken while behind (%lld of %lld); entries skipped while behind %lld: bound later %lld, picked by "
                                "their target %lld, removed at their target's start %lld, other %lld; guess-inflight waits %lld, "
                                "%.3f ms in all\n", __func__, (long long) ms.dec.n_bh_words, nw,
                                ms.dec.n_turns > 0 ? 100.0*ms.dec.n_bh_turns/ms.dec.n_turns : 0.0,
                                (long long) ms.dec.n_bh_turns, (long long) ms.dec.n_turns, (long long) ms.dec.n_bh_skip,
                                (long long) ms.dec.n_bh_skip_bound, (long long) ms.dec.n_bh_skip_picked,
                                (long long) ms.dec.n_bh_skip_stale, (long long) ms.dec.n_bh_skip_other,
                                (long long) ms.dec.n_gi_wait, ms.dec.t_gi_wait_us/1000.0);
                    }
                    if (ms.dec.n_pl_add > 0) {
                        LOG_INF("%s: moe stream: list [decode] per word: added %.2f, bound %.2f (add -> bind, the binds at the pick left "
                                "out: %.3f ms mean, p90 %s), removed without a read: picked %.2f, present %.2f, no room %.2f, word end %.2f, "
                                "list full %.2f, aborted %.2f, stale %.2f; bound at the pick %.2f, bound after its target ran %.2f; "
                                "bound and picked by the target %.2f (%.2f still in flight), bound unused %.2f; stale started, "
                                "cancelled %.2f (%.2f MiB), picked while cancelled %.2f\n",
                                __func__, ms.dec.n_pl_add/nw, ms.dec.n_pl_bind/nw,
                                // The delay leaves out the binds at the pick (the router had already asked for them)
                                ms.dec.n_pl_bind - ms.dec.n_pl_bind_pick > 0 ?
                                    ms.dec.t_pl_delay_us/1000.0/(ms.dec.n_pl_bind - ms.dec.n_pl_bind_pick) : 0.0,
                                common_moe_pl_p90(ms.dec.pl_delay_hist, ms.dec.n_pl_bind - ms.dec.n_pl_bind_pick).c_str(),
                                ms.dec.n_pl_rm[0]/nw, ms.dec.n_pl_rm[1]/nw, ms.dec.n_pl_rm[2]/nw, ms.dec.n_pl_rm[3]/nw,
                                ms.dec.n_pl_rm[4]/nw, ms.dec.n_pl_rm[5]/nw, ms.dec.n_pl_rm[6]/nw,
                                ms.dec.n_pl_bind_pick/nw, ms.dec.n_pl_bind_after/nw,
                                ms.dec.n_pl_used/nw, ms.dec.n_pl_used_late/nw,
                                (ms.dec.n_pl_bind - ms.dec.n_pl_used - ms.dec.n_pl_bind_pick)/nw,
                                ms.dec.n_pl_stale_cancel/nw, ms.dec.n_pl_stale_cancel_bytes/nw/1024.0/1024.0,
                                ms.dec.n_miss_stale/nw);
                    }
                }
            }

            llama_moe_stream_print_stats(ctx);
        }

        common_memory_breakdown_print(ctx);
    }
}

struct llama_sampler * common_sampler_get(const struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return nullptr;
    }

    return gsmpl->chain;
}

llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    llama_synchronize(ctx);

    // start measuring sampling time after the llama_context synchronization in order to not measure any ongoing async operations
    const auto tm = gsmpl->tm();

    llama_token id = LLAMA_TOKEN_NULL;

    auto & grmr  = gsmpl->grmr;
    auto & rbudget = gsmpl->rbudget;
    auto & chain = gsmpl->chain;
    auto & cur_p = gsmpl->cur_p; // initialized by set_logits

    gsmpl->set_logits(ctx, idx);

    // Check if a backend sampler has already sampled a token in which case we
    // return that token id directly.
    {
        id = llama_get_sampled_token_ith(ctx, idx);

        if (id != LLAMA_TOKEN_NULL) {
            LOG_DBG("%s: Backend sampler selected token: '%d'. Will not run any CPU samplers\n", __func__, id);
            gsmpl->n_pick_backend++;

            GGML_ASSERT(!gsmpl->grmr    && "using grammar in combination with backend sampling is not supported");
            GGML_ASSERT(!gsmpl->rbudget && "using reasoning budget in combination with backend sampling is not supported");

            for (size_t i = 0; i < cur_p.size; ++i) {
                if (cur_p.data[i].id == id) {
                    cur_p.selected = i;
                    break;
                }
            }

            return id;
        }
    }

    gsmpl->n_pick_cpu++;

    // apply reasoning budget first
    llama_sampler_apply(rbudget, &cur_p);

    if (grammar_first && grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr, &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    id = cur_p.data[cur_p.selected].id;

    if (grammar_first || !grammar_should_apply(gsmpl)) {
        return id;
    }

    // check if it the sampled token fits the grammar (grammar-based rejection sampling)
    {
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        llama_sampler_apply(grmr, &single_token_data_array);

        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            return id;
        }
    }

    // resampling:
    // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
    gsmpl->set_logits(ctx, idx);

    llama_sampler_apply(rbudget,  &cur_p);

    if (grammar_should_apply(gsmpl)) {
        llama_sampler_apply(grmr,  &cur_p);
    }

    llama_sampler_apply(chain, &cur_p);

    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    id = cur_p.data[cur_p.selected].id;

    return id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        // do not accept draft tokens after an EOG - they are not output but would stay in the context
        // on replay the last token is from the target and can be EOG, so a trailing EOG is still accepted
        if (draft[i] != id || (llama_vocab_is_eog(vocab, id) && i + 1 < draft.size())) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

static float prob_of(const llama_token_data * data, size_t n, llama_token id) {
    for (size_t k = 0; k < n; ++k) {
        if (data[k].id == id) {
            return data[k].p;
        }
    }
    return 0.0f;
}

// Accept a drafted token with probability min(1, p/q), else draw from norm(max(0, p - q)).
// Preserves the target distribution exactly, and accepts more often than matching does when the
// draft samples instead of taking its argmax.
std::vector<llama_token> common_sampler_sample_and_accept_n_rejection(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, const std::vector<std::vector<llama_token_data>> & draft_q, bool grammar_first) {
    GGML_ASSERT(idxs.size()    == draft.size() + 1 && "idxs.size() must be draft.size() + 1");
    GGML_ASSERT(draft_q.size() == draft.size() && "draft_q must have one entry per draft token");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    // draws come from the sampler's own stream, so they stay independent of what was drafted
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    std::vector<llama_token_data> residual;

    std::vector<llama_token_data> cand; // candidate array masked by the grammar, if there is one

    size_t i = 0;
    for (; i < draft.size(); i++) {
        // leaves the target distribution in the candidate array
        const llama_token id_tgt = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        const auto * cur_p = common_sampler_get_candidates(gsmpl, true);
        const auto & q     = draft_q[i];

        const bool masked = !grammar_first && grammar_should_apply(gsmpl);
        if (masked) {
            cand.assign(cur_p->data, cur_p->data + cur_p->size);
            llama_token_data_array arr = { cand.data(), cand.size(), -1, false };
            llama_sampler_apply(gsmpl->grmr, &arr);
        }

        // a candidate the grammar rejects carries no probability, whatever the target thinks
        auto p_raw = [&](size_t k) {
            return masked && cand[k].logit == -INFINITY ? 0.0f : cur_p->data[k].p;
        };

        // masking drops probability mass, so rescale what is left or the residual is over-weighted
        float p_sum = 0.0f;
        if (masked) {
            for (size_t k = 0; k < cur_p->size; ++k) {
                p_sum += p_raw(k);
            }
        }

        const float p_norm = masked && p_sum > 0.0f ? 1.0f/p_sum : 1.0f;

        auto p_of = [&](size_t k) {
            return p_raw(k)*p_norm;
        };

        // q_x is never 0 for a token the draft produced, but guard the divide
        const float q_x = prob_of(q.data(), q.size(), draft[i]);

        float p_x = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            if (cur_p->data[k].id == draft[i]) {
                p_x = p_of(k);
                break;
            }
        }

        if (q_x > 0.0f && (p_x >= q_x || uni(gsmpl->rng) < p_x / q_x)) {
            common_sampler_accept(gsmpl, draft[i], true);
            result.push_back(draft[i]);
            continue;
        }

        // rejected: tokens outside q's support keep all of p
        residual.clear();
        float sum = 0.0f;
        for (size_t k = 0; k < cur_p->size; ++k) {
            const float r = p_of(k) - prob_of(q.data(), q.size(), cur_p->data[k].id);
            if (r > 0.0f) {
                residual.push_back({ cur_p->data[k].id, 0.0f, r });
                sum += r;
            }
        }

        llama_token id = id_tgt;
        if (sum > 0.0f) {
            float u = uni(gsmpl->rng) * sum;
            id = residual.back().id;
            for (const auto & e : residual) {
                u -= e.p;
                if (u <= 0.0f) {
                    id = e.id;
                    break;
                }
            }
        }

        common_sampler_accept(gsmpl, id, true);
        result.push_back(id);

        break;
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

bool common_sampler_reasoning_budget_force(struct common_sampler * gsmpl) {
    if (!gsmpl) {
        return false;
    }

    return common_reasoning_budget_force(gsmpl->rbudget);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl, bool do_sort) {
    const auto tm = gsmpl->tm();

    auto * res = &gsmpl->cur_p;

    if (do_sort && !res->sorted) {
        // remember the selected token before sorting
        const llama_token id = res->data[res->selected].id;

        std::sort(res->data, res->data + res->size, [](const llama_token_data & a, const llama_token_data & b) {
            return a.p > b.p;
        });

        // restore the selected token after sorting
        for (size_t i = 0; i < res->size; ++i) {
            if (res->data[i].id == id) {
                res->selected = i;
                break;
            }
        }

        res->sorted = true;
    }

    return res;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ");
        result += std::string(llama_sampler_name(smpl)) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return 's';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return 'a';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_TOP_N_SIGMA: return "top_n_sigma";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        case COMMON_SAMPLER_TYPE_ADAPTIVE_P:  return "adaptive_p";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names) {
    // sampler names can be written multiple ways; generate aliases from canonical names
    static const auto sampler_name_map = []{
        // canonical sampler name mapping
        std::unordered_map<std::string, common_sampler_type> canonical_name_map {
            { "dry",         COMMON_SAMPLER_TYPE_DRY         },
            { "top_k",       COMMON_SAMPLER_TYPE_TOP_K       },
            { "top_p",       COMMON_SAMPLER_TYPE_TOP_P       },
            { "top_n_sigma", COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
            { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P   },
            { "min_p",       COMMON_SAMPLER_TYPE_MIN_P       },
            { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
            { "xtc",         COMMON_SAMPLER_TYPE_XTC         },
            { "infill",      COMMON_SAMPLER_TYPE_INFILL      },
            { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES   },
            { "adaptive_p",  COMMON_SAMPLER_TYPE_ADAPTIVE_P  }
        };
        std::unordered_map<std::string, common_sampler_type> alias_name_map;
        for (const auto & entry : canonical_name_map) {
            const std::string & canonical = entry.first;
            if (canonical.find('_') == std::string::npos) {
                continue;
            }
            // kebab-case: "top-k", "min-p", etc.
            {
                std::string kebab_case = canonical;
                std::replace(kebab_case.begin(), kebab_case.end(), '_', '-');
                alias_name_map.insert({kebab_case, entry.second});
            }
            // no dash: "topk", "minp", etc.
            {
                std::string no_dash = canonical;
                no_dash.erase(std::remove(no_dash.begin(), no_dash.end(), '_'), no_dash.end());
                alias_name_map.insert({no_dash, entry.second});
            }
        }
        // misc. aliases
        alias_name_map.insert({"nucleus", COMMON_SAMPLER_TYPE_TOP_P});
        alias_name_map.insert({"temp",    COMMON_SAMPLER_TYPE_TEMPERATURE});
        alias_name_map.insert({"typ",     COMMON_SAMPLER_TYPE_TYPICAL_P});
        // include aliases + canonical names in the complete mapping
        alias_name_map.merge(canonical_name_map);
        return alias_name_map;
    }();

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        std::string name_lower = name;
        std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
        auto sampler = sampler_name_map.find(name_lower);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
            continue;
        }
        LOG_WRN("%s: unable to match sampler by name '%s'\n", __func__, name_lower.c_str());
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_N_SIGMA), COMMON_SAMPLER_TYPE_TOP_N_SIGMA },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_ADAPTIVE_P),  COMMON_SAMPLER_TYPE_ADAPTIVE_P },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            LOG_WRN("%s: unable to match sampler by char '%c'\n", __func__, c);
        }
    }

    return samplers;
}

void common_sampling_add_preserved_tokens(common_params_sampling & sampling, const llama_vocab * vocab, const std::vector<std::string> & tokens) {
    GGML_ASSERT(vocab != nullptr);
    for (const auto & t : tokens) {
        auto ids = common_tokenize(vocab, t, false, true);
        if (ids.size() == 1) {
            sampling.preserved_tokens.insert(ids[0]);
        }
    }
}

void common_sampling_add_grammar_triggers(common_params_sampling & sampling, const llama_vocab * vocab, std::vector<common_grammar_trigger> triggers) {
    GGML_ASSERT(vocab != nullptr);
    for (auto & trigger : triggers) {
        if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD) {
            const auto & word = trigger.value;
            auto ids = common_tokenize(vocab, word, false, true);
            if (ids.size() == 1) {
                auto token = ids[0];
                if (std::find(sampling.preserved_tokens.begin(), sampling.preserved_tokens.end(), (llama_token) token) == sampling.preserved_tokens.end()) {
                    throw std::runtime_error("Grammar trigger word should be marked as preserved token: " + word);
                }
                common_grammar_trigger token_trigger;
                token_trigger.type  = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                token_trigger.value = word;
                token_trigger.token = token;
                sampling.grammar_triggers.push_back(std::move(token_trigger));
            } else {
                sampling.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, word});
            }
        } else {
            sampling.grammar_triggers.push_back(std::move(trigger));
        }
    }
    if (sampling.grammar_lazy && sampling.grammar_triggers.empty()) {
        throw std::runtime_error("Error: no triggers set for lazy grammar!");
    }
}
