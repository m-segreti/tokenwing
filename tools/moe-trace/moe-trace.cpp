// Records, for every token that goes through the model, which experts every MoE
// layer selected, and optionally the input vector that the router saw.

#include "arg.h"
#include "common.h"
#include "json.h"
#include "log.h"
#include "llama.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Upstream #29601 removed common_batch_add / common_batch_clear for llama_batch: the same two helpers, local to this tool
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


static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -o <prefix> (-p <text> | -f <file>) [-n N] [-ngl N] [--dump-router-input]\n", argv[0]);
    LOG("\n");
}

// tensor names that build_moe_ffn gives to the tensors we record, see llm_graph_context::cb. The
// streaming engine renames the top_k tensor of a layer it fills from a sidecar, one name per wave.
static const char * const NAME_TOPK  = "ffn_moe_topk-";
static const char * const NAME_TOPK_STREAM = "ffn_moe_topk_stream-";
static const char * const NAME_LOGITS = "ffn_moe_logits-";

struct moe_trace {
    // model parameters, all read from the model
    std::string model_path;
    std::string arch;
    int n_layer         = 0;
    int n_embd          = 0;
    int n_expert        = 0;
    int n_expert_used   = 0;
    int n_expert_groups = 0;
    int n_group_used    = 0;
    int n_dense_lead    = 0;
    std::vector<int> moe_layers;  // model layer index of every traced layer, in layer order

    bool dump_router_input = false;

    size_t max_rows = 0;  // row capacity of the buffers
    size_t n_rows   = 0;  // complete rows, the graph being captured starts here
    bool   in_graph = false;
    int    next_moe = 0;  // index in moe_layers of the layer that has to come next
    int    n_tok_graph = 0;  // tokens in the graph being captured

    std::vector<int> graph_sizes;  // tokens per captured graph

    std::vector<int32_t>  topk;  // [n_rows, n_moe, k]      int32
    std::vector<uint16_t> hin;   // [n_rows, n_moe, n_embd] float16

    llama_context * ctx = nullptr;

    bool        failed = false;
    std::string err;
};

static bool trace_fail(moe_trace & tr, const std::string & msg) {
    if (!tr.failed) {
        tr.failed = true;
        tr.err    = msg;
    }

    return true;  // keep the graph intact, the caller checks tr.failed after the decode
}

// copies n_rows rows of t into dst, one row at a time, so that a row stride that is
// larger than a row (t is a view into a bigger tensor) is handled correctly.
// dst_stride is in bytes
static void read_rows(ggml_tensor * t, void * dst, size_t dst_stride, int n_rows) {
    const size_t row_size = ggml_row_size(t->type, t->ne[0]);

    for (int i = 0; i < n_rows; i++) {
        ggml_backend_tensor_get(t, (char *) dst + i * dst_stride, i * t->nb[1], row_size);
    }
}

// copies n_rows rows of t into dst as float16, one row at a time
// dst_stride is in bytes
static void read_rows_f16(ggml_tensor * t, uint16_t * dst, size_t dst_stride, int n_rows) {
    const int64_t       n        = t->ne[0];
    const size_t        row_size = ggml_row_size(t->type, n);
    std::vector<char>   buf(row_size);

    for (int i = 0; i < n_rows; i++) {
        ggml_backend_tensor_get(t, buf.data(), i * t->nb[1], row_size);

        uint16_t * out = (uint16_t *) ((char *) dst + i * dst_stride);

        switch (t->type) {
            case GGML_TYPE_F32:
                {
                    const float * src = (const float *) buf.data();
                    for (int64_t j = 0; j < n; j++) {
                        out[j] = ggml_fp32_to_fp16(src[j]);
                    }
                } break;
            case GGML_TYPE_F16:
                memcpy(out, buf.data(), n * sizeof(uint16_t));
                break;
            case GGML_TYPE_BF16:
                {
                    const ggml_bf16_t * src = (const ggml_bf16_t *) buf.data();
                    for (int64_t j = 0; j < n; j++) {
                        out[j] = ggml_fp32_to_fp16(ggml_bf16_to_fp32(src[j]));
                    }
                } break;
            default:
                GGML_ABORT("unsupported router input type");
        }
    }
}

static int moe_index(const moe_trace & tr, int il) {
    for (size_t j = 0; j < tr.moe_layers.size(); j++) {
        if (tr.moe_layers[j] == il) {
            return (int) j;
        }
    }

    return -1;
}

// eval callback of the backend scheduler
//
// the scheduler walks the graph in dependency order and calls this once per node,
// first with ask=true and then with ask=false, at which point the data is ready.
// the rows of the graph being captured start at tr.n_rows; a new graph (a new micro
// batch) is recognized when a layer arrives after the last layer of the previous graph.
// MoE layers are always computed in layer order, also across backend splits.
static bool moe_trace_cb(ggml_tensor * t, bool ask, void * ud) {
    moe_trace & tr = *(moe_trace *) ud;

    if (tr.failed) {
        return false;
    }

    const char * name = ggml_get_name(t);

    bool        is_topk = false;
    const char * suffix = nullptr;

    if (strncmp(name, NAME_TOPK, strlen(NAME_TOPK)) == 0) {
        is_topk = true;
        suffix  = name + strlen(NAME_TOPK);
    } else if (strncmp(name, NAME_TOPK_STREAM, strlen(NAME_TOPK_STREAM)) == 0) {
        // the engine records a wave of experts per graph node: take the first one of the layer
        is_topk = true;
        suffix  = name + strlen(NAME_TOPK_STREAM);
    } else if (tr.dump_router_input && strncmp(name, NAME_LOGITS, strlen(NAME_LOGITS)) == 0) {
        suffix = name + strlen(NAME_LOGITS);
    } else {
        return false;
    }

    if (ask) {
        return true;
    }

    const int il = atoi(suffix);
    const int j  = moe_index(tr, il);

    if (j < 0) {
        return trace_fail(tr, "tensor " + std::string(name) + " is not a known MoE layer");
    }

    const int64_t n_tokens = t->ne[1];
    const int     n_moe    = (int) tr.moe_layers.size();

    if (n_tokens < 1) {
        return trace_fail(tr, "tensor " + std::string(name) + " has no tokens");
    }

    // a layer that the streaming engine split into expert waves has one top_k tensor per wave, and
    // the routing is the same in all of them: only the first one of the layer is recorded
    if (is_topk && tr.in_graph && j == tr.next_moe - 1) {
        return true;
    }

    // a graph always starts with the first MoE layer, so this closes the previous one
    if (tr.in_graph && tr.next_moe >= n_moe) {
        if (j != 0) {
            return trace_fail(tr, "MoE layer " + std::to_string(il) + " after the last layer of a graph");
        }

        tr.n_rows       += tr.n_tok_graph;
        tr.graph_sizes.push_back(tr.n_tok_graph);
        tr.next_moe     = 0;
        tr.n_tok_graph  = 0;
    }

    tr.in_graph = true;

    if (j != tr.next_moe) {
        return trace_fail(tr, "MoE layer " + std::to_string(il) + " out of order, expected index " +
                              std::to_string(tr.next_moe) + " of the graph");
    }

    if (tr.n_tok_graph > 0 && tr.n_tok_graph != n_tokens) {
        return trace_fail(tr, "layers of one graph disagree on the number of tokens");
    }

    if (tr.n_rows + n_tokens > tr.max_rows) {
        return trace_fail(tr, "more tokens traced than processed");
    }

    if (is_topk) {
        if (t->ne[0] != tr.n_expert_used || t->type != GGML_TYPE_I32) {
            return trace_fail(tr, "tensor " + std::string(name) + " has an unexpected shape or type");
        }

        // the scheduler has just synchronized the backend that computed t, no extra sync needed
        read_rows(t, tr.topk.data() + (tr.n_rows * n_moe + j) * tr.n_expert_used,
                  n_moe * tr.n_expert_used * sizeof(int32_t), n_tokens);

        tr.n_tok_graph = n_tokens;
        tr.next_moe    = j + 1;

        return true;
    }

    // the router input is src[1] of the router matmul, its shape is [n_embd, n_tokens]
    ggml_tensor * src = t->src[1];

    if (src->ne[0] != tr.n_embd) {
        return trace_fail(tr, "router input of layer " + std::to_string(il) + " has " +
                              std::to_string(src->ne[0]) + " elements, expected " + std::to_string(tr.n_embd));
    }

    // src[1] was produced by an earlier split, its backend may still be busy
    if (src->buffer && !ggml_backend_buffer_is_host(src->buffer)) {
        llama_synchronize(tr.ctx);
    }

    read_rows_f16(src, tr.hin.data() + (tr.n_rows * n_moe + j) * tr.n_embd, n_moe * tr.n_embd * sizeof(uint16_t), n_tokens);

    return true;
}

static bool meta_int(const llama_model * model, const std::string & key, int & out) {
    char buf[64] = { 0 };

    if (llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf)) < 0 || buf[0] == '\0') {
        return false;
    }

    out = atoi(buf);

    return true;
}

static std::string meta_str(const llama_model * model, const std::string & key) {
    char buf[256] = { 0 };

    if (llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf)) < 0) {
        return "";
    }

    return buf;
}

static bool write_file(const std::string & path, const void * data, size_t n_bytes) {
    FILE * f = fopen(path.c_str(), "wb");

    if (!f) {
        LOG_ERR("%s: cannot open '%s' for writing\n", __func__, path.c_str());
        return false;
    }

    const bool ok = n_bytes == 0 || fwrite(data, 1, n_bytes, f) == n_bytes;

    fclose(f);

    if (!ok) {
        LOG_ERR("%s: failed to write '%s'\n", __func__, path.c_str());
    }

    return ok;
}

static std::string shell_quote(const std::string & arg) {
    if (arg.find_first_of(" \t\n\"'\\$`&|;<>()*?[]#~=") == std::string::npos) {
        return arg;
    }

    std::string out = "'";

    for (char c : arg) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }

    return out + "'";
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MOE_TRACE, print_usage)) {
        return 1;
    }

    if (params.out_file.empty()) {
        LOG_ERR("%s: -o <prefix> is required\n", __func__);
        return 1;
    }

    if (params.prompt.empty()) {
        LOG_ERR("%s: no prompt, use -p or -f\n", __func__);
        return 1;
    }

    if (params.n_predict < 0) {
        LOG_ERR("%s: -n must not be negative\n", __func__);
        return 1;
    }

    moe_trace tr;

    if (params.warmup) {
        LOG_WRN("%s: --warmup is ignored, the warmup run would show up in the trace\n", __func__);
    }

    params.warmup            = false;
    params.cb_eval           = moe_trace_cb;
    params.cb_eval_user_data = &tr;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);

    llama_model   * model = init->model();
    llama_context * ctx   = init->context();

    if (!model || !ctx) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    tr.ctx               = ctx;
    tr.model_path        = params.model.path;
    tr.dump_router_input = params.moe_trace_dump_router_input;
    tr.arch              = meta_str(model, "general.architecture");
    tr.n_layer           = llama_model_n_layer(model);
    tr.n_embd            = llama_model_n_embd(model);

    if (tr.arch.empty()) {
        LOG_ERR("%s: failed to read general.architecture from the model\n", __func__);
        return 1;
    }

    if (llama_model_n_layer_nextn(model) > 0) {
        LOG_WRN("%s: the model has %d extra nextn layers, only the main model is traced\n",
                __func__, llama_model_n_layer_nextn(model));
    }

    const std::string prefix = tr.arch + ".";

    if (!meta_int(model, prefix + "expert_count", tr.n_expert) ||
        !meta_int(model, prefix + "expert_used_count", tr.n_expert_used)) {
        LOG_ERR("%s: the model is not an MoE model, expert metadata is missing\n", __func__);
        return 1;
    }

    // expert groups are optional: a model without the keys routes over all its experts (0 = no groups)
    if (!meta_int(model, prefix + "expert_group_count", tr.n_expert_groups) ||
        !meta_int(model, prefix + "expert_group_used_count", tr.n_group_used)) {
        tr.n_expert_groups = 0;
        tr.n_group_used    = 0;
    }

    // a model without leading dense blocks is all MoE
    if (!meta_int(model, prefix + "leading_dense_block_count", tr.n_dense_lead)) {
        tr.n_dense_lead = 0;
    }

    if (tr.n_expert <= 0 || tr.n_expert_used <= 0 || tr.n_expert_used > tr.n_expert) {
        LOG_ERR("%s: expert metadata is not usable\n", __func__);
        return 1;
    }

    for (int il = tr.n_dense_lead; il < tr.n_layer; il++) {
        tr.moe_layers.push_back(il);
    }

    const int n_moe = (int) tr.moe_layers.size();

    if (n_moe < 1) {
        LOG_ERR("%s: the model has no MoE layer\n", __func__);
        return 1;
    }

    const std::vector<llama_token> tokens = common_tokenize(vocab, params.prompt, true, true);

    if (tokens.empty()) {
        LOG_ERR("%s: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    const int n_prompt = (int) tokens.size();
    const int n_ctx    = llama_n_ctx(ctx);
    const int n_batch  = llama_n_batch(ctx);
    const int n_ubatch = llama_n_ubatch(ctx);

    if (n_prompt + params.n_predict > n_ctx) {
        LOG_ERR("%s: prompt (%d) + -n (%d) does not fit into the context (%d), raise -c\n",
                __func__, n_prompt, params.n_predict, n_ctx);
        return 1;
    }

    LOG_INF("%s: arch = %s, n_layer = %d, %d MoE layers, n_expert = %d, k = %d, groups = %d of %d, n_embd = %d\n",
            __func__, tr.arch.c_str(), tr.n_layer, n_moe, tr.n_expert,
            tr.n_expert_used, tr.n_group_used, tr.n_expert_groups, tr.n_embd);
    LOG_INF("%s: prompt = %d tokens, -n %d, n_ctx = %d, n_batch = %d, n_ubatch = %d\n",
            __func__, n_prompt, params.n_predict, n_ctx, n_batch, n_ubatch);

    tr.max_rows = n_prompt + params.n_predict;
    tr.topk.resize(tr.max_rows * n_moe * tr.n_expert_used);

    if (tr.dump_router_input) {
        tr.hin.resize(tr.max_rows * n_moe * tr.n_embd);
    }

    // token, position and phase of every row of the trace
    std::vector<llama_token> trace_ids;
    std::vector<llama_pos>   trace_pos;
    std::vector<std::string> trace_phase;

    if (params.sampling.user_sampling_config != 0) {
        LOG_WRN("%s: sampling arguments are ignored, the tool always takes the argmax of the logits\n", __func__);
    }

    // greedy: the token with the largest logit, the same tie break as llama_sampler_greedy_apply
    // (first index wins). --ignore-eos takes the EOG tokens out of the running.
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<uint8_t> skip_eog(n_vocab, 0);

    if (params.sampling.ignore_eos) {
        for (llama_token id = 0; id < n_vocab; id++) {
            skip_eog[id] = llama_vocab_is_eog(vocab, id) ? 1 : 0;
        }
    }

    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

    const auto t_start = std::chrono::high_resolution_clock::now();

    int n_decode = 0;

    // Every prompt token is an output: a model narrows its last layer to the output rows before that layer's experts (the
    // inp_out_ids crop), so with only the last token as an output the last MoE layer would route one row of each piece and the
    // trace would have no ids for the others. The price is the logits of every prompt token of a piece (n_batch x n_vocab floats)
    LOG_INF("%s: every prompt token is an output, so every MoE layer routes every row: %.1f MiB of logits per prompt piece of %d\n",
            __func__, (double) std::min(n_batch, n_prompt)*n_vocab*sizeof(float)/1048576.0, std::min(n_batch, n_prompt));

    // the prompt, in pieces of n_batch so that the positions stay under our control
    for (int off = 0; off < n_prompt && !tr.failed; off += n_batch) {
        const int n = std::min(n_batch, n_prompt - off);

        tool_batch_clear(batch);

        for (int i = 0; i < n; i++) {
            tool_batch_add(batch, tokens[off + i], off + i, { 0 }, true);
        }

        batch.n_tokens = n;

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: failed to decode the prompt\n", __func__);
            return 1;
        }

        for (int i = 0; i < n; i++) {
            trace_ids.push_back(tokens[off + i]);
            trace_pos.push_back(off + i);
            trace_phase.push_back("prefill");
        }
    }

    for (int i = 0; i < params.n_predict && !tr.failed; i++) {
        const float * logits = llama_get_logits_ith(ctx, -1);

        if (!logits) {
            LOG_ERR("%s: no logits for the last token\n", __func__);
            return 1;
        }

        llama_token id    = -1;
        float        best = 0.0f;

        for (llama_token t = 0; t < n_vocab; t++) {
            if (skip_eog[t]) {
                continue;
            }

            if (id < 0 || logits[t] > best) {
                best = logits[t];
                id   = t;
            }
        }

        if (id < 0) {
            LOG_ERR("%s: every token is an end of generation token\n", __func__);
            return 1;
        }

        if (llama_vocab_is_eog(vocab, id)) {
            LOG_INF("%s: end of generation after %d tokens\n", __func__, i);
            break;
        }

        const llama_pos pos = n_prompt + i;

        tool_batch_clear(batch);

        tool_batch_add(batch, id, pos, { 0 }, true);

        batch.n_tokens = 1;

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: failed to decode token %d\n", __func__, i);
            return 1;
        }

        trace_ids.push_back(id);
        trace_pos.push_back(pos);
        trace_phase.push_back("decode");

        n_decode++;
    }

    const auto t_end = std::chrono::high_resolution_clock::now();

    const double t_s = 1e-9 * std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();

    if (tr.failed) {
        LOG_ERR("%s: trace failed: %s\n", __func__, tr.err.c_str());
        return 1;
    }

    tr.n_rows += tr.n_tok_graph;
    tr.graph_sizes.push_back(tr.n_tok_graph);

    if (tr.n_rows != trace_ids.size()) {
        LOG_ERR("%s: traced %zu rows but processed %zu tokens\n", __func__, tr.n_rows, trace_ids.size());
        return 1;
    }

    int n_prefill_graphs = 0;

    for (int n : tr.graph_sizes) {
        if (n > 1) {
            n_prefill_graphs++;
        }
    }

    LOG_INF("%s: traced %zu tokens (%d prefill, %d decode) in %.2f s, %.2f t/s\n",
            __func__, tr.n_rows, n_prompt, n_decode, t_s, tr.n_rows / t_s);
    LOG_INF("%s: %zu graphs, %d of them with more than one token (n_ubatch = %d)\n",
            __func__, tr.graph_sizes.size(), n_prefill_graphs, n_ubatch);

    const std::string path_bin  = params.out_file + ".bin";
    const std::string path_hin  = params.out_file + ".hin.bin";
    const std::string path_json = params.out_file + ".json";

    if (!write_file(path_bin, tr.topk.data(), tr.n_rows * n_moe * tr.n_expert_used * sizeof(int32_t))) {
        return 1;
    }

    if (tr.dump_router_input &&
        !write_file(path_hin, tr.hin.data(), tr.n_rows * n_moe * tr.n_embd * sizeof(uint16_t))) {
        return 1;
    }


    {
        std::string cmdline;

        for (int i = 0; i < argc; i++) {
            cmdline += (i == 0 ? "" : " ");
            cmdline += shell_quote(argv[i]);
        }

        common_json root = common_json::object();

        root["model"]             = tr.model_path;
        root["arch"]              = tr.arch;
        root["n_layer"]           = tr.n_layer;
        root["moe_layers"]        = tr.moe_layers;
        root["n_expert"]          = tr.n_expert;
        root["k"]                 = tr.n_expert_used;
        root["n_embd"]            = tr.n_embd;
        root["n_expert_groups"]   = tr.n_expert_groups;
        root["n_group_used"]      = tr.n_group_used;
        root["n_tokens"]          = (long long) tr.n_rows;
        root["n_prompt"]          = n_prompt;
        root["n_decode"]          = n_decode;
        root["sampling"]          = "greedy-argmax";
        root["dump_router_input"] = tr.dump_router_input;
        root["token_ids"]         = trace_ids;
        root["positions"]         = trace_pos;
        root["phases"]            = trace_phase;
        root["cmdline"]           = cmdline;
        root["topk_file"]         = path_bin;
        root["router_input_file"] = tr.dump_router_input ? path_hin : "";
        root["trace_time_s"]      = t_s;

        const std::string json_text = root.dump(2);

        if (!write_file(path_json, json_text.data(), json_text.size())) {
            return 1;
        }
    }

    LOG_INF("%s: wrote %s, %s and %s\n", __func__, path_bin.c_str(), path_hin.c_str(), path_json.c_str());

    llama_batch_free(batch);

    // the model and the context are owned by init

    llama_backend_free();

    return 0;
}
