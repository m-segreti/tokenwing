// moe-decode-batch: what does a decode call of N tokens cost?
//
// The MoE expert streaming engine files its counters per decode call, so the question for speculative or
// batched decoding is how a call of K real, consecutive tokens compares with a call of 1. This tool runs
// in two phases:
//
//   1. "gen":   the prompt is decoded, then --gen tokens are generated greedily one per call, exactly as
//               llama-completion does (the same sampler, the same penalties). Their ids are recorded.
//               These calls are the baseline. If --text is given, the generated text is compared with
//               the file, which must be the text of a stock run of the same prompt. The stock runs
//               use -no-cnv, i.e. the raw prompt with no chat template; pass
//               --chat-template for a tool that is given a templated prompt.
//   2. "batch": the memory is cleared, the prompt is decoded again, and the recorded ids are fed K per
//               call. The tokens are therefore real, consecutive tokens of the same run, with the real
//               expert overlap between them. The token the model predicts for the last position of each
//              call is compared with the recorded one; a mismatch is the price of the batch shape itself
//              (a draft of K tokens has to be verified), so the match rate is a result, not a check.
//
// Per call the tool prints the wall time and the engine's own figures for that call (remap op, I/O stall,
// expert misses, reads). With MOE_STREAM_PROFILE_TOKENS=1 the ggml profiler adds the host/device split of
// each call on stderr, and with GGML_PROFILE=1 its totals.
//
// Phase 1 can save every call's logits (--save-logits FILE: n_vocab floats per forced word) and
// compare against such a file (--kld-base FILE: per word KLD(base || this run) and the same top-1, then a summary line), so two
// engine settings are compared on the same forced words; each gen line also carries the call's VRAM hits and VRAM follow's counters
// (tick, copies landed, switches, the GPU wait before a tick).
//
// usage: moe-decode-batch -m MODEL -f PROMPT [--text STOCK_TEXT] [--gen N] [--tokens-per-call K]
#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include "ggml.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// Upstream #29601 removed common_batch_add / common_batch_clear for llama_batch; the same two helpers, local to this tool
static void tool_batch_clear(llama_batch & batch) {
    batch.n_tokens = 0;
}

static void tool_batch_add(llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
    GGML_ASSERT(batch.seq_id[batch.n_tokens] && "llama_batch size exceeded");
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;
    batch.n_tokens++;
}


static void print_usage(int, char **) {
    printf("usage: moe-decode-batch -m MODEL -f PROMPT [--text STOCK_TEXT] [--gen N] [--tokens-per-call K]\n");
}

static std::string read_file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        LOG_ERR("%s: cannot read '%s'\n", __func__, path.c_str());
        exit(1);
    }
    std::string out;
    char buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        out.append(buf, n);
    }
    fclose(f);
    return out;
}

// one engine counter read, used to take the difference over one call
struct moe_delta {
    double remap_ms = 0.0, stall_ms = 0.0, misses = 0.0, reads = 0.0;
    double vram_hit = 0.0, vf_ticks = 0.0, vf_copies = 0.0, vf_switches = 0.0, vf_wait_ms = 0.0;
    double cb_wait_ms = 0.0;   // waits on a copy-back in flight (not in stall_ms)
};

static moe_delta moe_snapshot(llama_context * ctx, const llama_moe_stream_stats & prev, bool have_prev) {
    moe_delta d;
    llama_moe_stream_stats ms;
    if (!have_prev || !llama_moe_stream_get_stats(ctx, &ms)) {
        return d;
    }
    d.remap_ms = (ms.t_op_us    - prev.t_op_us)/1000.0;
    d.stall_ms = (ms.t_stall_us - prev.t_stall_us)/1000.0;
    d.misses   = (double) ((ms.dec.n_miss  + ms.pre.n_miss)  - (prev.dec.n_miss  + prev.pre.n_miss));
    d.reads    = (double) ((ms.dec.n_reads + ms.pre.n_reads) - (prev.dec.n_reads + prev.pre.n_reads));
    d.vram_hit    = (double) ((ms.dec.n_vram_hit + ms.pre.n_vram_hit) - (prev.dec.n_vram_hit + prev.pre.n_vram_hit));
    d.vf_ticks    = (double) (ms.n_vf_ticks    - prev.n_vf_ticks);
    d.vf_copies   = (double) (ms.n_vf_copies   - prev.n_vf_copies);
    d.vf_switches = (double) (ms.n_vf_switches - prev.n_vf_switches);
    d.vf_wait_ms  = (ms.t_vf_gpu_wait_us - prev.t_vf_gpu_wait_us)/1000.0;
    d.cb_wait_ms  = ((ms.dec.t_cb_wait_us + ms.pre.t_cb_wait_us) - (prev.dec.t_cb_wait_us + prev.pre.t_cb_wait_us))/1000.0;
    return d;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    // this tool's own arguments are taken out before common_params_parse sees them
    std::string path_text;
    std::string path_force;          // decode this text's tokens instead of sampling
    std::string path_save_logits, path_kld_base;   // phase 1's logits saved / compared
    bool no_batch = false;           // skip phase 2
    int n_gen = 256, n_tokens_per_call = 1;
    bool apply_template = false;   // the stock runs use -no-cnv, so the raw prompt is what they feed
    std::vector<char *> args;
    args.push_back(argv[0]);   // common_params_parse expects argv[0]
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--text" && i + 1 < argc) {
            path_text = argv[++i];
        } else if (a == "--gen" && i + 1 < argc) {
            n_gen = std::atoi(argv[++i]);
        } else if (a == "--tokens-per-call" && i + 1 < argc) {
            n_tokens_per_call = std::atoi(argv[++i]);
        } else if (a == "--chat-template") {
            apply_template = true;
        } else if (a == "--force-text" && i + 1 < argc) {
            path_force = argv[++i];
        } else if (a == "--save-logits" && i + 1 < argc) {
            path_save_logits = argv[++i];
        } else if (a == "--kld-base" && i + 1 < argc) {
            path_kld_base = argv[++i];
        } else if (a == "--no-batch") {
            no_batch = true;
        } else {
            args.push_back(argv[i]);
        }
    }
    args.push_back(nullptr);

    if (!common_params_parse((int) args.size() - 1, args.data(), params, LLAMA_EXAMPLE_MOE_TRACE, print_usage)) {
        return 1;
    }

    if (params.prompt.empty()) {
        LOG_ERR("%s: -f PROMPT is required\n", __func__);
        return 1;
    }
    if (n_tokens_per_call < 1 || n_gen < 2) {
        LOG_ERR("%s: --tokens-per-call must be >= 1 and --gen must be >= 2\n", __func__);
        return 1;
    }

    params.warmup     = false;   // a warm-up call would be filed as a decode call too
    params.n_predict  = -1;
    params.n_print    = -1;
    params.n_batch    = std::max(params.n_batch, n_tokens_per_call);
    params.n_ubatch   = std::max(params.n_ubatch, n_tokens_per_call);
    params.n_parallel = 1;
    if (no_batch && params.n_outputs_max == 0) {
        // Phase 1 only (--no-batch) asks for one output per call; size the output buffer for that, as llama-completion does
        // (the buffer lives in VRAM: ~0.67 GiB less, the footprint of the real command - needed to run more VRAM experts in this tool)
        params.n_outputs_max = 1;
    }

    const common_init_result_ptr init = common_init_from_params(params);
    if (!init || !init->context()) {
        LOG_ERR("%s: failed to load the model\n", __func__);
        return 1;
    }
    llama_context * ctx = init->context();

    // llama-completion sends the prompt through the model's chat template, so the token ids of the
    // run this tool compares with only match if this tool does the same
    std::string prompt_text = params.prompt;
    if (apply_template) {
        const common_chat_templates_ptr tmpls = common_chat_templates_init(init->model(), params.chat_template);

        std::vector<common_chat_msg> msgs;
        common_chat_msg user;
        user.role    = "user";
        user.content = params.prompt;
        msgs.push_back(user);

        common_chat_templates_inputs in;
        in.use_jinja             = params.use_jinja;
        in.messages              = msgs;
        in.add_generation_prompt = true;
        in.force_pure_content    = params.force_pure_content_parser;

        prompt_text = common_chat_templates_apply(tmpls.get(), in).prompt.str();   // #30389 made the prompt parts
    }
    LOG_INF("%s: prompt after the chat template: %.80s\n", __func__, prompt_text.c_str());

    // common_params_parse has already read -f into params.prompt
    const auto ids_prompt = common_tokenize(ctx, prompt_text, true, true);
    const int  n_prompt   = (int) ids_prompt.size();
    const int  n_ctx      = llama_n_ctx(ctx);

    LOG_INF("%s: prompt %d tokens, ctx %d, n_batch %d, n_ubatch %d, %d token(s) to generate, %d per batched call\n",
            __func__, n_prompt, n_ctx, params.n_batch, params.n_ubatch, n_gen, n_tokens_per_call);
    LOG_INF("%s: first prompt ids:", __func__);
    for (int i = 0; i < std::min(8, n_prompt); i++) {
        LOG_INF(" %d", ids_prompt[i]);
    }
    LOG_INF("\n");

    if (n_ctx < n_prompt + n_gen + n_tokens_per_call + 8) {
        LOG_ERR("%s: n_ctx %d is too small for %d prompt + %d generated tokens\n", __func__, n_ctx, n_prompt, n_gen);
        return 1;
    }

    llama_batch batch = llama_batch_init(std::max(params.n_batch, n_prompt + 8), 0, n_tokens_per_call);

    // llama.cpp's own helpers, the ones llama-completion uses, so that this tool computes what a
    // llama-completion run of the same prompt computes
    int n_past = 0;
    const auto decode_prompt = [&]() {
        for (int i = 0; i < n_prompt; i += params.n_batch) {
            const int n = std::min(params.n_batch, n_prompt - i);
            if (!common_prompt_batch_decode(ctx, ids_prompt, n, n_past, params.n_batch, std::string_view(), false)) {
                LOG_ERR("%s: failed to decode the prompt\n", __func__);
                exit(1);
            }
        }
    };

    decode_prompt();

    // ---- phase 1: generate greedily, one token per call, as llama-completion does --------------
    // (the new columns follow a '|', so an older parser of these lines, whose trailing
    //  optional numbers are the older columns', matches no line instead of reading these as its own)
    printf("phase call n_tok ms ms_per_tok remap_ms stall_ms misses reads id ok | vram_hit vf_tick vf_copies vf_switches vf_wait_ms\n");

    // The logits of every phase-1 call, saved or compared with a saved run's (same forced words)
    // (also from the environment - MOE_DECODE_SAVE_LOGITS / MOE_DECODE_KLD_BASE - so a harness's plan step, which runs
    //  llama-completion with the same arguments, never sees them)
    if (path_save_logits.empty() && getenv("MOE_DECODE_SAVE_LOGITS")) {
        path_save_logits = getenv("MOE_DECODE_SAVE_LOGITS");
    }
    if (path_kld_base.empty() && getenv("MOE_DECODE_KLD_BASE")) {
        path_kld_base = getenv("MOE_DECODE_KLD_BASE");
    }
    const int n_vocab_l = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    FILE * f_save = nullptr, * f_base = nullptr;
    if (!path_save_logits.empty() && !(f_save = fopen(path_save_logits.c_str(), "wb"))) {
        LOG_ERR("%s: cannot write '%s'\n", __func__, path_save_logits.c_str());
        return 1;
    }
    if (!path_kld_base.empty() && !(f_base = fopen(path_kld_base.c_str(), "rb"))) {
        LOG_ERR("%s: cannot read '%s'\n", __func__, path_kld_base.c_str());
        return 1;
    }
    // (the file says what it holds - a header (magic, n_vocab) and, per word, the forced token fed before the
    //  logits - and a compare needs the same forced words: the token of every word is checked, a different vocabulary refused)
    static const char LG_MAGIC[8] = { 'M', 'D', 'B', 'L', 'G', 'T', '1', 0 };
    if (f_base && path_force.empty()) {
        LOG_ERR("%s: --kld-base needs --force-text (both runs must compute the same words)\n", __func__);
        return 1;
    }
    if (f_save) {
        const int32_t nv = n_vocab_l;
        if (fwrite(LG_MAGIC, 1, 8, f_save) != 8 || fwrite(&nv, sizeof(nv), 1, f_save) != 1) {
            LOG_ERR("%s: writing the logits header failed\n", __func__);
            return 1;
        }
    }
    if (f_base) {
        char mg[8] = {};
        int32_t nv = 0;
        if (fread(mg, 1, 8, f_base) != 8 || memcmp(mg, LG_MAGIC, 8) != 0 || fread(&nv, sizeof(nv), 1, f_base) != 1) {
            LOG_ERR("%s: '%s' is not a logits file of this tool\n", __func__, path_kld_base.c_str());
            return 1;
        }
        if (nv != n_vocab_l) {
            LOG_ERR("%s: the base logits have n_vocab %d, this model %d\n", __func__, nv, n_vocab_l);
            return 1;
        }
    }
    std::vector<float> lg_base(f_base ? n_vocab_l : 0);
    std::vector<double> kld_w;
    int64_t kld_same = 0;

    std::vector<llama_token> ids_gen;
    ids_gen.reserve(n_gen);

    common_sampler * smpl = common_sampler_init(init->model(), params.sampling);

    // --force-text: the same tokens in every arm, so that a change that alters the generated text (e.g. the float order of a
    // prompt) is still measured on the same work; the text is tokenized as a continuation (no BOS, no special tokens)
    std::vector<llama_token> ids_force;
    if (!path_force.empty()) {
        ids_force = common_tokenize(ctx, read_file(path_force), false, false);
        if ((int) ids_force.size() < n_gen) {
            LOG_INF("%s: --force-text has %zu tokens, --gen lowered from %d\n", __func__, ids_force.size(), n_gen);
            n_gen = (int) ids_force.size();
        }
        if (n_gen < 2) {
            LOG_ERR("%s: --force-text has fewer than 2 tokens\n", __func__);
            return 1;
        }
    }

    double t_wall = 0.0, t_remap = 0.0, t_stall = 0.0;
    int64_t n_miss = 0, n_reads = 0;
    int    n_calls = 0;

    for (int i = 0; i < n_gen; i++) {
        // the token of this position, from the state the previous call left (or the forced text's)
        const llama_token id = ids_force.empty() ? common_sampler_sample(smpl, ctx, -1) : ids_force[i];
        if (id < 0) {
            LOG_ERR("%s: sampling failed at token %d\n", __func__, i);
            exit(1);
        }
        common_sampler_accept(smpl, id, true);
        ids_gen.push_back(id);

        if (i + 1 == n_gen) {
            break;    // the last token needs no call of its own
        }

        // the call that continues the sequence: this is the one to measure
        llama_moe_stream_stats ms0 = {};
        const bool stats = llama_moe_stream_get_stats(ctx, &ms0);

        const int64_t t0 = ggml_time_us();
        common_batch one = common_batch_get_one(ctx, { id });
        const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, one.get());
        const int64_t t1 = ggml_time_us();
        if (rc != 0) {
            LOG_ERR("%s: failed to decode token %d (%d)\n", __func__, i, rc);
            exit(1);
        }

        const moe_delta d = moe_snapshot(ctx, ms0, stats);

        n_calls++;
        t_wall  += (t1 - t0)/1000.0;
        t_remap += d.remap_ms;
        t_stall += d.stall_ms;
        n_miss  += (int64_t) d.misses;
        n_reads += (int64_t) d.reads;

        printf("gen %5d %5d %9.3f %11.3f %9.3f %9.3f %7.0f %6.0f %8d %3s | %6.0f %3.0f %4.0f %4.0f %8.3f %8.3f\n",
               n_calls, 1, (t1 - t0)/1000.0, (t1 - t0)/1000.0, d.remap_ms, d.stall_ms, d.misses,
               d.reads, id, "-", d.vram_hit, d.vf_ticks, d.vf_copies, d.vf_switches, d.vf_wait_ms, d.cb_wait_ms);

        if (f_save || f_base) {
            const float * lg = llama_get_logits_ith(ctx, -1);
            if (!lg) {
                LOG_ERR("%s: no logits after call %d\n", __func__, n_calls);
                exit(1);
            }
            const int32_t id_w = id;
            if (f_save && (fwrite(&id_w, sizeof(id_w), 1, f_save) != 1 ||
                           fwrite(lg, sizeof(float), n_vocab_l, f_save) != (size_t) n_vocab_l)) {
                LOG_ERR("%s: writing the logits failed\n", __func__);
                exit(1);
            }
            if (f_base) {
                int32_t id_b = -1;
                if (fread(&id_b, sizeof(id_b), 1, f_base) != 1 ||
                    fread(lg_base.data(), sizeof(float), n_vocab_l, f_base) != (size_t) n_vocab_l) {
                    LOG_ERR("%s: the base logits end at call %d\n", __func__, n_calls);
                    exit(1);
                }
                if (id_b != id_w) {
                    LOG_ERR("%s: word %d: the base run fed token %d, this run %d - not the same words\n", __func__, n_calls, id_b, id_w);
                    exit(1);
                }
                // KLD(base || this run) over the softmax of the full vocabulary, in double
                double mb = lg_base[0], mt = lg[0];
                int ab = 0, at = 0;
                for (int v = 1; v < n_vocab_l; v++) {
                    if (lg_base[v] > mb) { mb = lg_base[v]; ab = v; }
                    if (lg[v]      > mt) { mt = lg[v];      at = v; }
                }
                double zb = 0.0, zt = 0.0;
                for (int v = 0; v < n_vocab_l; v++) {
                    zb += std::exp((double) lg_base[v] - mb);
                    zt += std::exp((double) lg[v]      - mt);
                }
                const double lzb = std::log(zb), lzt = std::log(zt);
                double kl = 0.0;
                for (int v = 0; v < n_vocab_l; v++) {
                    const double lpb = (double) lg_base[v] - mb - lzb;
                    const double lpt = (double) lg[v]      - mt - lzt;
                    kl += std::exp(lpb)*(lpb - lpt);
                }
                kld_w.push_back(kl);
                kld_same += ab == at ? 1 : 0;
                printf("kld %5d %.8f %d\n", n_calls, kl, ab == at ? 1 : 0);
            }
        }
    }

    if (f_save) {
        fclose(f_save);
    }
    if (f_base) {
        if (fgetc(f_base) != EOF) {
            LOG_ERR("%s: the base logits have more words than this run's %zu - not the same run length\n", __func__, kld_w.size());
            exit(1);
        }
        fclose(f_base);
        if (!kld_w.empty()) {
            std::vector<double> s = kld_w;
            std::sort(s.begin(), s.end());
            double sum = 0.0;
            for (const double k : s) {
                sum += k;
            }
            LOG_INF("%s: kld: %zu words, mean KLD %.6f, max %.6f, 99th %.6f, median %.6f, same top-1 %.2f %%\n", __func__, s.size(),
                    sum/s.size(), s.back(), s[(size_t) (0.99*(s.size() - 1))], s[s.size()/2], 100.0*kld_same/s.size());
        }
    }

    const int n_gen_calls = n_calls;
    LOG_INF("%s: gen: %d calls of 1 token, %.3f ms/call, remap op %.3f ms/token, stall %.3f ms/token, "
            "%.2f expert misses/token, %.2f reads/token\n",
            __func__, n_gen_calls, t_wall/n_gen_calls, t_remap/n_gen_calls, t_stall/n_gen_calls,
            (double) n_miss/n_gen_calls, (double) n_reads/n_gen_calls);

    if (!path_text.empty()) {
        const std::string got = common_detokenize(ctx, ids_gen, false);
        const std::string ref = read_file(path_text);
        LOG_INF("%s: gen: %d token(s), the text matches the stock run: %s\n",
                __func__, n_gen, got == ref ? "yes" : "NO");
        if (got != ref) {
            fprintf(stderr, "got: %.200s\nref: %.200s\n", got.c_str(), ref.c_str());
        }
    }

    if (no_batch) {   // --no-batch: phase 1 only (a forced-text comparison of two engine settings)
        llama_moe_stream_print_stats(ctx);   // writes the --moe-stream-record files (only the completion tool did before)
        common_sampler_free(smpl);
        llama_batch_free(batch);
        return 0;
    }

    // ---- phase 2: the same tokens, n_tokens_per_call per call ---------------------------------
    printf("phase call n_tok ms ms_per_tok remap_ms stall_ms misses reads pred ok\n");   // (phase 2's own columns)
    llama_memory_clear(llama_get_memory(ctx), true);
    decode_prompt();

    int    b_calls = 0, b_ok = 0;
    double b_wall = 0.0, b_remap = 0.0, b_stall = 0.0;
    int64_t b_miss = 0, b_reads = 0;

    // the last token has no successor, so it is not part of the batched stream
    const int n_stream = (int) ids_gen.size() - 1;

    for (int i = 0; i < n_stream; i += n_tokens_per_call) {
        const int n = std::min(n_tokens_per_call, n_stream - i);

        tool_batch_clear(batch);
        for (int j = 0; j < n; j++) {
            // every position wants its logits: that is what a draft of n tokens needs to be verified
            tool_batch_add(batch, ids_gen[i + j], n_prompt + i + j, { 0 }, true);
        }

        llama_moe_stream_stats ms0 = {};
        const bool stats = llama_moe_stream_get_stats(ctx, &ms0);

        const int64_t t0 = ggml_time_us();
        const int rc = llama_decode(ctx, batch);
        const int64_t t1 = ggml_time_us();
        if (rc != 0) {
            LOG_ERR("%s: decode of batched call %d failed (%d)\n", __func__, b_calls, rc);
            exit(1);
        }

        const moe_delta d = moe_snapshot(ctx, ms0, stats);

        const float * logits = llama_get_logits_ith(ctx, n - 1);
        int pred = -1;
        if (logits) {
            const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
            pred = 0;
            for (int v = 1; v < n_vocab; v++) {
                if (logits[v] > logits[pred]) {
                    pred = v;
                }
            }
        }
        const bool ok = pred == ids_gen[i + n - 1];
        b_ok += ok;
        b_calls++;
        b_wall  += (t1 - t0)/1000.0;
        b_remap += d.remap_ms;
        b_stall += d.stall_ms;
        b_miss  += (int64_t) d.misses;
        b_reads += (int64_t) d.reads;

        printf("batch %4d %5d %9.3f %11.3f %9.3f %9.3f %7.0f %6.0f %8d %3s\n",
               b_calls, n, (t1 - t0)/1000.0, (t1 - t0)/1000.0/n, d.remap_ms, d.stall_ms, d.misses,
               d.reads, pred, ok ? "yes" : "NO");
    }

    const double n_tok = (double) (b_calls*n_tokens_per_call);

    LOG_INF("%s: batch: %d calls of %d token(s), %.0f tokens: %.3f ms/call, %.3f ms/token, remap op %.3f "
            "ms/token, stall %.3f ms/token, %.2f expert misses/token, %.2f reads/token, %d of %d calls "
            "ended on the token of the reference run\n",
            __func__, b_calls, n_tokens_per_call, n_tok, b_wall/b_calls, b_wall/n_tok, b_remap/n_tok,
            b_stall/n_tok, n_tok > 0 ? (double) b_miss/n_tok : 0.0, n_tok > 0 ? (double) b_reads/n_tok : 0.0,
            b_ok, b_calls);

    common_sampler_free(smpl);
    llama_batch_free(batch);

    return 0;
}
