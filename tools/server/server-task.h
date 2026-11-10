#pragma once

#include "common.h"
#include "llama.h"

#include <string>
#include <unordered_set>
#include <list>
#include <map>
#include <mutex>

// TODO: prevent including the whole server-common.h as we only use server_tokens
#include "server-common.h"


enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_CONTROL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_GET,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_GET_LORA,
    SERVER_TASK_TYPE_SET_LORA,
    SERVER_TASK_TYPE_CACHE_DISK_ERASE,   // POST /cache-disk/erase (the task's conv_id)
    SERVER_TASK_TYPE_CACHE_DISK_CLEAR,   // POST /cache-disk/clear
    SERVER_TASK_TYPE_CACHE_DISK_FORK,    // POST /cache-disk/fork (conv_id_from -> conv_id)
};

// TODO: change this to more generic "response_format" to replace the "format_response_*" in server-common
enum task_response_type {
    TASK_RESPONSE_TYPE_NONE, // llama.cpp native format
    TASK_RESPONSE_TYPE_OAI_CHAT,
    TASK_RESPONSE_TYPE_OAI_CMPL,
    TASK_RESPONSE_TYPE_OAI_RESP,
    TASK_RESPONSE_TYPE_OAI_ASR, // transcriptions API
    TASK_RESPONSE_TYPE_OAI_EMBD,
    TASK_RESPONSE_TYPE_ANTHROPIC,
};

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};

struct task_params {
    bool stream          = false;
    bool include_usage   = false;
    bool cache_prompt    = true; // remember the prompt to avoid reprocessing all prompt
    bool cache_disk_save = true; // with --cache-disk: false = the request leaves the chat's files on the disk as they are
    bool return_tokens   = false;
    bool return_progress = false;

    int32_t sse_ping_interval = 30; // seconds between SSE comment pings while the stream stays silent, -1 disables

    int32_t n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict
    int32_t n_indent  =  0; // minimum line indentation for the generated text in number of whitespace characters
    int32_t n_cmpl    =  1; // number of completions to generate from this prompt

    int32_t n_cache_reuse = 0; // min chunk size to attempt reusing from the cache via KV shifting (0 = disabled)

    int64_t t_max_prompt_ms  = -1; // TODO: implement
    int64_t t_max_predict_ms = -1; // if positive, limit the generation phase to this time limit

    std::map<int, float> lora; // mapping adapter ID -> scale

    std::vector<std::string> antiprompt;
    std::vector<std::string> response_fields;

    bool timings_per_token   = false;
    bool post_sampling_probs = false;

    struct common_params_sampling sampling;
    struct common_params_speculative speculative;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;

    // realtime control (SERVER_TASK_TYPE_CONTROL)
    std::string        control_action;
    std::string        control_cmpl_id;

    // reported in generation_settings, parsing itself is owned by the chat session
    common_chat_format      chat_format      = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE;

    // message spans for checkpointing
    common_chat_msg_spans message_spans;

    // Embeddings
    int32_t embd_normalize = 2; // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)

    json format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const;
    json to_json(bool only_metrics = false) const;
};

// struct for tracking the state of a task (e.g., for streaming)
struct task_result_state {
    // tracking diffs for partial tool calls
    std::vector<common_chat_msg_diff> diffs;
    common_chat_session chat_session; // owns all parsing for this generation
    common_chat_msg chat_msg;
    std::vector<std::string> generated_tool_call_ids;
    std::unordered_set<size_t> sent_tool_call_names;

    // for OpenAI Responses and Anthropic streaming API:
    // track output item / content block state across chunks
    bool thinking_block_started = false;
    bool text_block_started = false;

    // for OpenAI Responses streaming API
    bool oai_resp_created = false;
    const std::string oai_resp_id;
    const std::string oai_resp_reasoning_id;
    const std::string oai_resp_message_id;
    std::string oai_resp_fc_id; // function call ID for current args delta

    task_result_state(common_chat_session session = {});

    // parse partial tool calls and update the internal state
    common_chat_msg update_chat_msg(
        const common_chat_input & added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls = false);
};

struct server_task {
    int id = -1; // to be filled by server_queue

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // used when there are multiple prompts (batch request)

    // used by SERVER_TASK_TYPE_CANCEL
    int id_target = -1;
    int id_slot   = -1;

    // used by parallel sampling (multiple completions from same prompt)
    int id_parent  = -1;
    // temporary store of child tasks for scheduling
    // note: accessing to elements is invalid after the task is moved to server_slot
    std::vector<server_task> child_tasks;

    // used by SERVER_TASK_TYPE_INFERENCE
    task_params   params;
    server_tokens tokens;

    // The request's X-Conversation-Id header ("" = none), the key of the chat's state in --cache-disk
    std::string conv_id;

    // SERVER_TASK_TYPE_CACHE_DISK_FORK only: the chat whose files are copied to conv_id
    std::string conv_id_from;

    // only used by CLI, this allow tokenizing CLI inputs on server side
    // we need this because mtmd_context and vocab are not accessible outside of server_context
    bool                    cli = false;
    std::string             cli_prompt;
    std::vector<raw_buffer> cli_files;

    server_task_type type;

    // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE
    struct slot_action {
        int id_slot;
        std::string filename;
        std::string filepath;
    };
    slot_action slot_action;

    // used by SERVER_TASK_TYPE_METRICS
    bool metrics_reset_bucket = false;

    // used by SERVER_TASK_TYPE_SET_LORA
    std::map<int, float> set_lora; // mapping adapter ID -> scale

    server_task() = default;

    server_task(server_task_type type) : type(type) {}

    int32_t n_tokens() const {
        return tokens.size();
    }

    bool need_embd() const {
        switch (type) {
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                return true;
            default:
                return false;
        }
    }

    bool need_logits() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                return true;
            default:
                return false;
        }
    }

    bool need_sampling() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
                return true;
            default:
                return false;
        }
    }

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<server_task> & tasks) {
        std::unordered_set<int> ids(tasks.size());
        for (size_t i = 0; i < tasks.size(); i++) {
            ids.insert(tasks[i].id);
            for (auto & child : tasks[i].child_tasks) {
                ids.insert(child.id);
            }
        }
        return ids;
    }

    void apply_chat_session(const common_chat_session & session) {
        if (!session.has_template()) {
            return;
        }

        session.apply_sampling(params.sampling);
        params.chat_format = session.format();
        params.antiprompt.insert(params.antiprompt.end(), session.additional_stops().begin(), session.additional_stops().end());
        params.message_spans = tokens.find_message_spans(session.message_delimiters());
    }

    void add_child(int id_parent, int id_child) {
        server_task copy;

        copy.id        = id_child;
        copy.id_parent = id_parent;
        copy.params    = params;
        copy.type      = type;
        copy.tokens    = tokens.clone();
        copy.id_slot   = -1; // child tasks cannot specify slot

        // use different sampling seed for each child
        // note: https://github.com/ggml-org/llama.cpp/pull/18700#discussion_r2675115723
        if (copy.params.sampling.seed != LLAMA_DEFAULT_SEED) {
            copy.params.sampling.seed += (uint32_t)child_tasks.size() + 1;
        }

        child_tasks.push_back(std::move(copy));
    }

    bool is_parent() const {
        return child_tasks.size() > 0;
    }

    bool is_child() const {
        return id_parent != -1;
    }

    // embedding and rerank tasks always process their whole prompt, their state is never reused
    bool is_stateless() const {
        return type == SERVER_TASK_TYPE_EMBEDDING || type == SERVER_TASK_TYPE_RERANK;
    }
};

struct result_prompt_progress {
    int32_t total = 0;
    int32_t cache = 0;
    int32_t processed = 0;
    int64_t time_ms = 0;

    json to_json() const;
};

struct server_task_result {
    int id           = -1;
    int id_slot      = -1;

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // to be used for batched tasks

    virtual bool is_error() {
        // only used by server_task_result_error
        return false;
    }
    virtual bool is_stop() {
        // only used by server_task_result_cmpl_*
        return true;
    }
    virtual void update(task_result_state &) {
        // only used by server_task_result_cmpl_*
    }
    virtual json to_json() = 0;
    virtual ~server_task_result() = default;
    virtual server_task_result * clone() const {
        GGML_ABORT("not implemented for this task type");
    }
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

struct completion_token_output {
    llama_token tok;
    float prob;
    std::string text_to_send;
    struct prob_info {
        llama_token tok;
        std::string txt;
        float prob;
    };
    std::vector<prob_info> probs;

    json to_json(bool post_sampling_probs) const;

    static json probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs);

    static float logarithm(float x);

    static std::vector<unsigned char> str_to_bytes(const std::string & str);

};

struct server_task_result_cmpl_final : server_task_result {
    common_chat_input content;
    llama_tokens tokens;

    bool stream;
    bool include_usage;
    server_slot_stats stats;
    std::string prompt;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;
    int32_t n_tokens_cached;
    bool has_new_line;
    std::string stopping_word;
    stop_type stop = STOP_TYPE_NONE;

    bool post_sampling_probs;
    std::vector<completion_token_output> probs_output;
    std::vector<std::string>  response_fields;

    task_params generation_params;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    common_chat_msg    oaicompat_msg; // to be populated by update()

    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // for OpenAI Responses API
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;

    virtual bool is_stop() override {
        return true; // in stream mode, final responses are considered stop
    }

    virtual json to_json() override;

    virtual void update(task_result_state & state) override {
        is_updated = true;
        oaicompat_msg = state.update_chat_msg(content, false, oaicompat_msg_diffs);

        oai_resp_id = state.oai_resp_id;
        oai_resp_reasoning_id = state.oai_resp_reasoning_id;
        oai_resp_message_id = state.oai_resp_message_id;
    }

    json to_json_non_oaicompat();

    json usage_json_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_chat_stream();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_resp_stream();

    json to_json_oaicompat_asr();

    json to_json_anthropic();

    json to_json_anthropic_stream();
};

struct server_task_result_cmpl_partial : server_task_result {
    common_chat_input content;
    llama_tokens      tokens;

    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;

    bool post_sampling_probs;
    bool is_progress = false;
    bool is_begin = false; // whether to send 200 status to HTTP client (begin of SSE stream)
                           // ref: https://github.com/ggml-org/llama.cpp/pull/23884
    completion_token_output prob_output;
    server_slot_stats stats;
    result_prompt_progress progress;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // Streaming state copied from task_result_state for this chunk
    bool thinking_block_started = false;
    bool text_block_started     = false;

    // for OpenAI Responses API
    bool oai_resp_created = false;
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;
    std::string oai_resp_fc_id;

    // for Anthropic API: track if any reasoning content has been generated
    bool anthropic_has_reasoning = false;

    virtual bool is_stop() override {
        return false; // in stream mode, partial responses are not considered stop
    }

    virtual void update(task_result_state & state) override;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_asr();

    json to_json_anthropic();
};

struct server_task_result_embd : server_task_result {
    std::vector<std::vector<float>> embedding;

    int32_t n_tokens;

    // response formatting
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();
};

struct server_task_result_rerank : server_task_result {
    float score = -1e6;

    int32_t n_tokens;

    virtual json to_json() override;
};

struct server_task_result_error : server_task_result {
    error_type err_type = ERROR_TYPE_SERVER;
    std::string err_msg;

    // for ERROR_TYPE_EXCEED_CONTEXT_SIZE
    int32_t n_prompt_tokens = 0;
    int32_t n_ctx           = 0;

    virtual bool is_error() override {
        return true;
    }

    virtual json to_json() override;
};

// used by /metrics API
struct server_task_result_metrics : server_task_result {
    // these are immediate stats, not accumulated (server_metrics is cumulative)
    int n_processing_slots = 0;
    int n_tasks_deferred = 0;

    server_metrics metrics;

    virtual json to_json() override;

    struct metric_item {
        std::string name;
        std::string description;
        double value; // prometheus values are always float64
    };
    std::string to_metrics();
};

// used by /slots API
struct server_task_result_slots : server_task_result {
    int n_idle_slots = 0;

    // while we can also use std::vector<server_slot> this requires copying the slot object which can be quite messy
    // therefore, we use json to temporarily store the slot.to_json() result
    json slots_data = json::array();

    virtual json to_json() override;
};

struct server_task_result_slot_save_load : server_task_result {
    std::string filename;
    bool is_save; // true = save, false = load

    size_t n_tokens;
    size_t n_bytes;
    double t_ms;

    virtual json to_json() override;
};

struct server_task_result_slot_erase : server_task_result {
    size_t n_erased;

    virtual json to_json() override;
};

struct server_task_result_control : server_task_result {
    bool        success = false;
    std::string message; // optional detail when success is false

    virtual json to_json() override {
        json out = json { { "success", success } };
        if (!message.empty()) {
            out["message"] = message;
        }
        return out;
    }
};

struct server_task_result_get_lora : server_task_result {
    struct lora {
        common_adapter_lora_info info;
        std::string  alora_invocation_string;
        llama_tokens alora_invocation_tokens;
    };
    std::vector<lora> loras;

    virtual json to_json() override;
};

struct server_task_result_apply_lora : server_task_result {
    virtual json to_json() override;
};

struct server_prompt {
    server_tokens tokens;

    std::list<common_prompt_checkpoint> checkpoints;

    void clear() {
        tokens.clear();
        checkpoints.clear();
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            checkpoints,
        };
    }
};

struct server_prompt_data {
    std::vector<uint8_t> main;
    std::vector<uint8_t> drft;

    size_t size() const {
        return main.size() + drft.size();
    }
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    std::string conv_id;   // the chat (X-Conversation-Id, with --cache-disk) whose state this is, "" = none

    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt.size();
        }

        return res;
    }
};

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens) {
        this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
    }

    std::list<server_prompt_cache_state> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    size_t size() const;

    size_t n_tokens() const;

    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    // conv_id_out: set to the restored state's chat id when a state is restored
    bool load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot,
              std::string * conv_id_out = nullptr);

    // The common prefix with tokens_new of the state load() would restore, -1 = none (load keeps the slot's content)
    int peek(const server_prompt & prompt, const server_tokens & tokens_new) const;

    void update();

private:
    std::list<server_prompt_cache_state>::const_iterator find_best(const server_prompt & prompt, const server_tokens & tokens_new) const;
};

// The chats whose files --cache-disk deletes last when it is over its size limit (POST /cache-disk/pins). Kept in memory only,
// as the hashes the entries carry; set by an HTTP thread, read by the thread that saves.
struct server_cache_disk_pins {
    // replaces the whole set; returns how many different chats it holds
    size_t set(const std::vector<std::string> & chat_ids);
    bool   has(uint64_t chat_hash) const;

private:
    mutable std::mutex mtx;
    std::unordered_set<uint64_t> hashes;
};

// --cache-disk: the chat states on disk. One file per saved state: chat-<hash of the chat id>.state or
// prefix-<hash of the tokens>.state (always a hash, never text from a request), owner-only (0600), written to a temp file and
// renamed. A file holds a header (magic, format version, the model's fingerprint, the settings that change a state's bytes or
// meaning, the chat id's hash, the LoRA set's hash, the write time, the token count, the total size), then the tokens, the
// checkpoints and the main / draft state - the same server_prompt_cache_state the RAM tier keeps, so a restore is the RAM tier's
// restore - then a checksum over all of that. The index keeps each entry's header and tokens (4 B per token), never the state
// bytes. Everything runs on the server's task thread, so saves, restores, erase and clear are ordered.
struct server_cache_disk {
    // what a state's bytes and meaning depend on; a file whose values differ is refused
    struct config {
        uint64_t model_bytes = 0;   // the sum of the model files' sizes (0 = unknown: a size could not be read)
        uint64_t gguf_hash   = 0;   // a hash of the first model file's GGUF header (all bytes before its tensor data)
        uint64_t draft_bytes = 0;   // the same two for a draft model file (0 when there is none, e.g. MTP)
        uint64_t draft_hash  = 0;
        uint64_t ctrl_hash   = 0;   // the control vectors (file + strength) and their layer range; 0 = none
        uint64_t ovr_hash    = 0;   // the model overrides that change a state: RoPE / YaRN settings, --override-kv
        uint32_t n_ctx       = 0;
        int32_t  type_k      = -1;
        int32_t  type_v      = -1;
        int32_t  type_k_dft  = -1;  // the draft context's KV types (-1 = no draft context)
        int32_t  type_v_dft  = -1;
        uint32_t n_seq       = 0;
        uint32_t has_draft   = 0;
        int32_t  flash_attn  = 0;   // as resolved by the context (0 / 1), not "auto"
        int32_t  flash_attn_dft = -1;  // the draft context's, as resolved (-1 = no draft context)
        int32_t  n_ckpt_max  = 0;   // --ctx-checkpoints
        uint32_t state_ver   = 0;   // llama.cpp's LLAMA_STATE_SEQ_VERSION
        uint32_t kv_unified  = 0;   // these three change an SWA model's partial state (the SWA window's size)
        uint32_t swa_full    = 0;
        uint32_t n_ubatch    = 0;
        // format 6 (our own fix of upstream #28498): the K / V rotation matrix sizes of the target and the draft
        // context's attention caches (0 = none, -1 = unknown memory kind); a format-5 file has -2 = "the default rule"
        int32_t  rot_k = -1, rot_v = -1, rot_k_dft = -1, rot_v_dft = -1;
        // not written: this server's sizes under the default rule (what a format-5 file was saved with: no build of ours
        // set LLAMA_ATTN_ROT_DISABLE), compared with a format-5 file's -2
        int32_t  rot_k_def = -1, rot_v_def = -1, rot_k_dft_def = -1, rot_v_dft_def = -1;

        bool operator==(const config & o) const;
        std::string diff(const config & o) const;   // the first field that differs, for the refusal log
    };

    struct entry {
        std::string name;           // file name in the directory
        bool        is_chat   = false;
        bool        is_snap   = false;  // a chat's "start of message" snapshot (snap-<chat>-<n_tokens>.state)
        int64_t     snap_n_tokens = 0;  // the snapshot's position, its memory range and a hash of the tokens before it
        int32_t     snap_pos_min  = 0;
        int32_t     snap_pos_max  = 0;
        uint64_t    snap_prefix   = 0;
        uint64_t    chat_hash = 0;  // the chat id's FNV-1a (also the file name's)
        uint64_t    chat_check = 0; // a second, independent hash of the chat id (cd_mix), checked at the lookup and restore
        uint64_t    lora_hash = 0;  // the slot's LoRA set (adapter + scale) when it was saved
        uint64_t    file_size = 0;
        int64_t     t_write   = 0;  // the header's write time (us since the epoch); the file's mtime when the header is unreadable
        bool        usable    = false;
        std::string why;            // why it is not usable
        llama_tokens tokens;
    };

    // the entry a lookup chose; restore() reads it
    struct pick {
        size_t       i   = SIZE_MAX;
        size_t       lcp = 0;       // its common prefix with the request
        const char * how = "";
        bool ok() const { return i != SIZE_MAX; }
    };

    enum restore_result {
        RESTORE_DONE,               // the slot holds the entry's state
        RESTORE_REFUSED,            // the file was refused before the slot was touched: the slot keeps its content
        RESTORE_FAILED,             // the slot's memory was touched and the restore failed: the caller clears the slot
    };

    std::string dir;
    uint64_t    limit_bytes = 0;
    int32_t     max_snaps   = 0;   // --cache-disk-checkpoints, "start of message" snapshots kept per chat, 0 = off
    int32_t     n_swa       = 0;   // the server's SWA window (0 with --swa-full or no SWA), for find()'s snapshot rule
    config      cfg;
    std::vector<entry> index;
    bool        warned_media = false;
    const server_cache_disk_pins * pins = nullptr;   // the pinned chats; null = none

    // checks that dir can be written, reads the headers of the entries already there; false = refuse --cache-disk
    bool init(const std::string & dir, double limit_gb, const config & cfg);

    // after a reply: saves the slot's state (chat id: its chat file, replaced; else a prefix file with the RAM tier's rules);
    // true = the file was written
    bool save(const server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot,
              const std::string & conv_id, uint64_t lora_hash, int64_t t_reply_end_us);

    // the entry that serves the request better than the slot's own content: by chat id its chat file (when it shares more of
    // the request than the slot and >= 25 % of its own tokens), else (unless chat_only) the best prefix file by the RAM tier's
    // rule (f_keep >= 0.25, better f_keep and f_sim)
    pick find(const server_prompt & slot_prompt, const server_tokens & tokens_new, const std::string & conv_id,
              uint64_t lora_hash, bool chat_only);

    restore_result restore(const pick & pk, server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt,
                           llama_context * ctx_dft, int32_t id_slot, const std::string & conv_id);

    // POST /cache-disk/erase: deletes the chat's file; returns its size (0 when there was none), -1 when it could not be deleted
    int64_t erase_chat(const std::string & conv_id, const char * why = "erased by POST /cache-disk/erase");   // why: for the log

    // POST /cache-disk/clear: deletes every file of this tier in dir (an entry name AND the magic); returns {files, bytes}
    std::pair<size_t, uint64_t> clear();

    // POST /cache-disk/fork: every usable entry of chat `from` (its state, then its snapshots, newest first, up to
    // --cache-disk-checkpoints) is copied as an entry of chat `to` - its own file, written now, counted in the size limit,
    // after the entries of `to` it replaces. A file holds its chat id in the header and the snapshot's name is checked against
    // it, so a link to the same file would not be this chat's entry: the copy gets a new header and the rest of the bytes as
    // they are. `from` is never changed, and the size limit never deletes its entries or the copies.
    struct fork_result {
        bool     found     = false;   // `from` has usable entries; false = nothing was changed
        bool     state     = false;   // the chat's saved state was copied
        size_t   snapshots = 0;       // snapshots copied
        size_t   skipped   = 0;       // entries not copied: over the snapshot count, no room under the size limit, unreadable
        uint64_t bytes     = 0;       // the copied entries' size
        uint64_t replaced_bytes = 0;  // the files of `to` that were deleted first
        bool     failed    = false;   // found, but nothing could be copied (`to` is unchanged)
        bool     replace_failed = false;  // the files of `to` could not all be deleted: the copy was not made

        bool ok() const { return found && !failed; }
        const char * error_text() const {
            return !found ? "POST /cache-disk/fork: no saved state on disk for the chat in \"from\" (never saved, erased, or deleted by the size limit)" :
                   replace_failed ? "the saved files of the new chat could not all be deleted (see the server log); nothing was copied" :
                   "nothing could be copied: the entries do not fit under --cache-disk-gb or cannot be read (see the server log)";
        }
        json to_json() const {
            return { { "state_copied", state }, { "snapshots", snapshots }, { "file_bytes", bytes }, { "skipped", skipped },
                     { "replaced_bytes", replaced_bytes } };
        }
    };
    fork_result fork_chat(const std::string & from, const std::string & to);

    // A chat's "start of message" snapshots on the disk. write_snapshot: after a reply, the snapshot made at the
    // last user message's start (its own file, the oldest of the chat beyond max_snaps deleted). find_snapshot: the newest one
    // of the chat that the server could go back to (pos_max <= pos_next, pos_min < pos_min_thold or 0) and newer than
    // newer_than, whose tokens before it are the request's; read_snapshot: into a checkpoint (checked as a state file);
    // drop_snapshots_after: the ones past where the chat now differs; snapshot_positions: the chat's positions on the disk
    // write_snapshot: true = on the disk now; *fresh = this call wrote it (false: it was there already, its age is refreshed)
    bool    write_snapshot(const common_prompt_checkpoint & c, const server_tokens & tokens, const std::string & conv_id, uint64_t lora_hash,
                           bool * fresh = nullptr);
    bool    remove_snapshot(const std::string & conv_id, int64_t n_tokens, const char * reason);
    bool    has_snapshot(const std::string & conv_id, int64_t n_tokens, const server_tokens & tokens) const;   // for the log line
    void    refuse_snapshot(size_t i, const char * why);   // it failed to load into the slot
    size_t  find_snapshot(const std::string & conv_id, uint64_t lora_hash, int64_t newer_than, int32_t pos_next,
                          int32_t pos_min_thold, const server_tokens & tokens) const;   // SIZE_MAX = none
    bool    read_snapshot(size_t i, const std::string & conv_id, common_prompt_checkpoint & out);
    bool    snapshot_where(size_t i, int64_t & n_tokens, int32_t & pos_min, int32_t & pos_max) const;   // from the index
    size_t  drop_snapshots_after(const std::string & conv_id, int64_t n_tokens, const server_tokens & tokens);
    std::vector<int64_t> snapshot_positions(const std::string & conv_id, const server_tokens * tokens = nullptr) const;   // With
                                                                                       // tokens, only the ones made for that history

    static uint64_t fnv1a(const void * data, size_t n, uint64_t h = 1469598103934665603ull);
    static uint64_t prefix_hash(const server_tokens & tokens, int64_t n);   // over tokens [0, n)
    static uint64_t chat_check_of(const std::string & conv_id);   // the second hash (another function, another seed)
    static uint64_t lora_hash_of(const std::vector<common_adapter_lora_info> & loras);
    static std::string chat_name(const std::string & conv_id);
    static std::string prefix_name(const llama_tokens & tokens);
    static std::string snap_name(const std::string & conv_id, int64_t n_tokens);

private:
    uint64_t total_bytes(const std::string & except) const;
    size_t   remove_snaps_of(uint64_t chat_hash, int64_t newer_than, const char * reason, uint64_t * bytes = nullptr);
    bool     remove_entry(size_t i, const char * reason);   // false = the file is still there (the entry stays); a file without
                                                             // this tier's magic is never deleted, only dropped from the index
    bool     read_header(const std::string & path, entry & e, bool with_tokens) const;
    bool     is_pinned(const entry & e) const;   // a state or snapshot of a pinned chat
    bool     evicts_before(const entry & a, const entry & b) const;   // the size limit's order, see save()
};

struct server_task_result_cache_disk : server_task_result {
    json data;
    virtual json to_json() override { return data; }
    virtual server_task_result * clone() const override {
        return new server_task_result_cache_disk(*this);
    }
};

// used exclusively by router mode
struct server_task_result_router : server_task_result {
    json data;
    virtual json to_json() override { return data; }
    virtual server_task_result * clone() const override {
        return new server_task_result_router(*this);
    }
};
