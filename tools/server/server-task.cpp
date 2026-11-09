#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"

#include <sstream>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <random>
#ifndef _WIN32
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#ifdef _WIN32
#include <process.h>   // _getpid
#endif

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"adaptive_target",           sampling.adaptive_target},
            {"adaptive_decay",            sampling.adaptive_decay},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_format)},
            {"reasoning_format",          common_reasoning_format_name(reasoning_format)},
            {"reasoning_in_content",      stream && reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY},
            {"generation_prompt",         sampling.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"adaptive_target",           sampling.adaptive_target},
        {"adaptive_decay",            sampling.adaptive_decay},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_format)},
        {"reasoning_format",          common_reasoning_format_name(reasoning_format)},
        {"reasoning_in_content",      stream && reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY},
        {"generation_prompt",         sampling.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
task_result_state::task_result_state(common_chat_session session)
    : chat_session(std::move(session))
    , chat_msg(chat_session.msg())
    , oai_resp_id("resp_" + random_string())
    , oai_resp_reasoning_id("rs_" + random_string())
    , oai_resp_message_id("msg_" + random_string()) {
}

common_chat_msg task_result_state::update_chat_msg(
        const common_chat_input & added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    auto msg_prv_copy = chat_msg;
    auto new_msg = is_partial ? chat_session.feed(added) : chat_session.finish(added);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // the JSON library converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content.text},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             stats.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content.text},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (stats.is_set()) {
        deltas.back()["timings"] = stats.to_json();
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;
    size_t output_index = 0;

    if (oaicompat_msg.reasoning_content != "") {
        const size_t item_output_index = output_index++;
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"output_index", item_output_index},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        const size_t item_output_index = output_index++;
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"output_index", item_output_index},
                {"content_index", 0},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"output_index", item_output_index},
                {"content_index", 0},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"output_index", item_output_index},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const size_t item_output_index = output_index++;
        const json output_item = {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"output_index", item_output_index},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    if (stats.is_set()) {
        server_sent_events.back().at("data")["timings"] = stats.to_json();
    }

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    if (is_begin) {
        return; // begin marker only flushes headers, skip parsing
    }
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !state.oai_resp_created && (is_progress || n_decoded == 1)) {
        state.oai_resp_created = true;
    }

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    if (is_begin) {
        return nullptr; // simply signal to HTTP handler to send the headers and status code
    }
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content.text},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content.text},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (stats.is_set()) {
            last_json["timings"] = stats.to_json();
        }
        if (is_progress) {
            last_json["prompt_progress"] = progress.to_json();
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (!oai_resp_created) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    } else if (is_progress) {
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"output_index", 0},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"output_index", 0},
                    {"content_index", 0},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"output_index", anthropic_has_reasoning ? 1 : 0},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"output_index", anthropic_has_reasoning ? 1 : 0},
                        {"content_index", 0},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"output_index", anthropic_has_reasoning ? 1 : 0},
                    {"content_index", 0},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"output_index", (anthropic_has_reasoning ? 1 : 0) + (text_block_started ? 1 : 0) + diff.tool_call_index},
                    {"item", json {
                        {"id",        "fc_" + diff.tool_call_delta.id},
                        {"arguments", ""},
                        {"call_id",   "call_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                    {"output_index", (anthropic_has_reasoning ? 1 : 0) + (text_block_started ? 1 : 0) + diff.tool_call_index},
                }},
            });
        }
    }

    if (!events.empty()) {
        json & data = events.back().at("data");
        if (stats.is_set()) {
            data["timings"] = stats.to_json();
        }
        if (is_progress) {
            data["prompt_progress"] = progress.to_json();
        }
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content.text},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_slots::to_json() {
    return slots_data;
}

json server_task_result_metrics::to_json() {
    // not used, /metrics renders prometheus text via to_metrics()
    return json{};
}

// metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
std::string server_task_result_metrics::to_metrics() {
    const std::vector<metric_item> counters = {
        {
            "prompt_tokens_total",
            "Number of prompt tokens processed, excluding cached tokens",
            (double) metrics.prompt.count
        }, {
            "prompt_tokens_cached_total",
            "Number of prompt tokens reused from the cache",
            (double) metrics.n_prompt_cached
        }, {
            "prompt_seconds_total",
            "Total time spent processing prompts",
            metrics.prompt.time / 1.e6
        }, {
            "tokens_predicted_total",
            "Number of generation tokens processed",
            (double) metrics.predict.count
        }, {
            "tokens_predicted_seconds_total",
            "Total time spent generating tokens",
            metrics.predict.time / 1.e6
        }, {
            "n_decode_total",
            "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding",
            (double) metrics.n_decode
        }, {
            "n_tokens_max",
            "Largest observed sequence length (prompt + generation)",
            (double) metrics.n_tokens_max
        }, {
            "spec_decode_num_draft_tokens_total",
            "Speculative: Total draft tokens generated",
            (double) metrics.n_draft_tokens
        }, {
            "spec_decode_num_accepted_tokens_total",
            "Speculative: Total draft tokens accepted by the target model",
            (double) metrics.n_draft_accepted
        }, {
            "spec_decode_num_drafts_total",
            "Speculative: Total speculative decoding verification steps",
            (double) metrics.n_draft_verif_steps
        },
    };

    const std::vector<metric_item> gauges = {
        {
            "prompt_tokens_seconds",
            "Average prompt throughput in tokens/s",
            metrics.prompt_bucket.n_per_second()
        }, {
            "predicted_tokens_seconds",
            "Average generation throughput in tokens/s",
            metrics.predict_bucket.n_per_second()
        }, {
            "requests_processing",
            "Number of requests processing",
            (double) n_processing_slots
        }, {
            "requests_deferred",
            "Number of requests deferred",
            (double) n_tasks_deferred
        }, {
            "n_busy_slots_per_decode",
            "Average number of busy slots per llama_decode() call",
            (double) metrics.n_busy_slots / std::max((double) metrics.n_decode, 1.0)
        },
    };

    std::stringstream prometheus;

    auto add_items = [&prometheus](const char * type, const std::vector<metric_item> & items) {
        for (const auto & item : items) {
            prometheus << "# HELP llamacpp:" << item.name << " " << item.description << "\n"
                       << "# TYPE llamacpp:" << item.name << " " << type             << "\n"
                       << "llamacpp:"        << item.name << " " << item.value       << "\n";
        }
    };

    add_items("counter", counters);
    add_items("gauge",   gauges);

    // labeled counter: one time series per draft position
    if (!metrics.n_accepted_per_pos.empty()) {
        prometheus << "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
                      " Accepted tokens per draft position\n"
                   << "# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter\n";
        for (size_t i = 0; i < metrics.n_accepted_per_pos.size(); i++) {
            prometheus << "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\""
                       << i << "\"} " << metrics.n_accepted_per_pos[i] << "\n";
        }
    }

    return prometheus.str();
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache
//
size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

server_prompt_cache_state * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt.size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->prompt.tokens.size()) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it = states.erase(it);
        } else {
            ++it;
        }
    }

    if (limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the limit
        while (!states.empty() && size() + state_size_new > limit_size) {
            SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                    states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    std::vector<uint8_t> state_data_tgt;
    std::vector<uint8_t> state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back({
        /*.prompt =*/ {
            /*.tokens      =*/ prompt.tokens.clone(),
            /*.checkpoints =*/ prompt.checkpoints,
        },
        /*.data   =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
        },
        /*.conv_id =*/ {},
    });

    return &states.back();
}

std::list<server_prompt_cache_state>::const_iterator server_prompt_cache::find_best(const server_prompt & prompt, const server_tokens & tokens_new) const {
    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
        }
    }

    if (it_best != states.end()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);
    }

    return it_best;
}

int server_prompt_cache::peek(const server_prompt & prompt, const server_tokens & tokens_new) const {
    const auto it = find_best(prompt, tokens_new);
    return it == states.end() ? -1 : (int) it->prompt.tokens.get_common_prefix(tokens_new);
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot,
                               std::string * conv_id_out) {
    // find_best is const: an empty erase turns its const_iterator into an iterator (erases nothing)
    const auto it_found = find_best(prompt, tokens_new);
    auto it_best = states.erase(it_found, it_found);

    if (it_best != states.end()) {
        {
            auto & data = it_best->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\n", size);

                return false;
            }

            data.clear();
            data.shrink_to_fit();
        }

        {
            auto & data = it_best->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\n", size);

                    return false;
                }

                data.clear();
                data.shrink_to_fit();
            }
        }

        prompt = std::move(it_best->prompt);
        if (conv_id_out) {
            *conv_id_out = it_best->conv_id;
        }

        states.erase(it_best);
    }

    return true;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        while (!states.empty() && size() > limit_size) {
            SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n", states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB, chat '%s'\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.size() / (1024.0 * 1024.0),
                state.conv_id.c_str());   // the chat id (with --cache-disk), for checking erase / clear
    }
}

//
// server_cache_disk (--cache-disk)
//

namespace {

constexpr char     CD_MAGIC[8]   = { 'T', 'K', 'W', 'I', 'N', 'G', 'C', 'D' };
constexpr char     CD_END[8]     = { 'T', 'K', 'W', 'I', 'N', 'G', 'E', 'N' };
constexpr uint32_t CD_VERSION    = 6;      // 2: + draft / control-vector / checkpoint / LoRA fields, the body checksum;
                                           // 3: + the second chat-id hash, the model overrides, the draft's flash attention;
                                           // 4: + kind 2 (a chat's "start of message" snapshot) and, in a chat /
                                           //    prefix file, the list of the chat's snapshot positions kept in their own files;
                                           // 5: + kv_unified, swa_full, n_ubatch at the header's end
                                           // 6: + the K / V rotation sizes of the target and draft caches;
                                           //    a format-5 file is still read: its rotation is the default rule's
constexpr uint32_t CD_VERSION_MIN = 5;     // the oldest format this server reads

static bool cd_version_ok(uint32_t v) {
    return v >= CD_VERSION_MIN && v <= CD_VERSION;
}
constexpr int64_t  CD_TMP_AGE_US = 3600ll*1000000;   // a .tmp file of a writer that is still alive and older than this is stale

// the fixed part of a file, in this order
struct cd_header {
    uint32_t version   = 0;
    uint32_t kind      = 0;   // 0 = prefix, 1 = chat
    server_cache_disk::config cfg;
    uint64_t chat_hash = 0;
    uint64_t chat_check = 0;
    uint64_t lora_hash = 0;
    int64_t  t_write   = 0;
    uint64_t n_tokens  = 0;
    uint64_t total     = 0;   // the whole file's size
};

inline uint64_t cd_rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

// the body checksum: a 64-bit multiply-rotate hash over 16 bytes per step in two lanes (several GB/s); not cryptographic -
// it finds damage, the header's fingerprint finds other models / settings
uint64_t cd_mix(uint64_t h, const void * data, size_t n) {
    constexpr uint64_t K1 = 0x9E3779B97F4A7C15ull, K2 = 0xC2B2AE3D27D4EB4Full;
    const uint8_t * b = (const uint8_t *) data;
    uint64_t a = h ^ (n * K2);
    uint64_t c = h + K1;
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint64_t w0, w1;
        memcpy(&w0, b + i, 8);
        memcpy(&w1, b + i + 8, 8);
        a = cd_rotl(a ^ (w0 * K2), 31) * K1;
        c = cd_rotl(c ^ (w1 * K2), 29) * K1;
    }
    for (; i < n; i++) {
        a = (a ^ b[i]) * 0x100000001B3ull;
    }
    uint64_t r = a ^ cd_rotl(c, 17);
    r ^= r >> 33; r *= K2; r ^= r >> 29;
    return r;
}

// writes (f) or only counts (f == nullptr); while `hashing`, every piece also goes into the checksum
struct cd_writer {
    FILE *   f       = nullptr;
    bool     ok      = true;
    uint64_t n       = 0;
    bool     hashing = false;
    uint64_t hash    = 0;
    int64_t  t_hash_us = 0;
    void bytes(const void * p, size_t len) {
        if (hashing && len > 0) {
            const int64_t t0 = ggml_time_us();
            hash = cd_mix(hash, p, len);
            t_hash_us += ggml_time_us() - t0;
        }
        if (f) {
            ok = ok && (len == 0 || fwrite(p, 1, len, f) == len);
        }
        n += len;
    }
    template <typename T> void put(const T & v) { bytes(&v, sizeof(T)); }
    void blob(const std::vector<uint8_t> & v) { put<uint64_t>(v.size()); bytes(v.data(), v.size()); }
};

// every length is checked against the bytes left in the file before anything is allocated
struct cd_reader {
    FILE *   f       = nullptr;
    uint64_t fsize   = 0;
    bool     ok      = true;
    uint64_t pos     = 0;
    bool     hashing = false;
    uint64_t hash    = 0;
    int64_t  t_hash_us = 0;
    uint64_t left() const { return pos <= fsize ? fsize - pos : 0; }
    void bytes(void * p, uint64_t len) {
        if (!ok || len > left()) { ok = false; return; }
        ok = len == 0 || fread(p, 1, len, f) == len;
        pos += len;
        if (ok && hashing && len > 0) {
            const int64_t t0 = ggml_time_us();
            hash = cd_mix(hash, p, len);
            t_hash_us += ggml_time_us() - t0;
        }
    }
    template <typename T> void get(T & v) { bytes(&v, sizeof(T)); }
    void blob(std::vector<uint8_t> & v) {
        uint64_t len = 0;
        get(len);
        if (!ok || len > left()) { ok = false; return; }
        v.resize(len);
        bytes(v.data(), len);
    }
};

void cd_put_header(cd_writer & w, const cd_header & h) {
    w.bytes(CD_MAGIC, sizeof(CD_MAGIC));
    w.put(h.version); w.put(h.kind);
    w.put(h.cfg.model_bytes); w.put(h.cfg.gguf_hash); w.put(h.cfg.draft_bytes); w.put(h.cfg.draft_hash); w.put(h.cfg.ctrl_hash);
    w.put(h.cfg.ovr_hash);
    w.put(h.cfg.n_ctx); w.put(h.cfg.type_k); w.put(h.cfg.type_v); w.put(h.cfg.type_k_dft); w.put(h.cfg.type_v_dft);
    w.put(h.cfg.n_seq); w.put(h.cfg.has_draft); w.put(h.cfg.flash_attn); w.put(h.cfg.flash_attn_dft); w.put(h.cfg.n_ckpt_max);
    w.put(h.cfg.state_ver);
    w.put(h.chat_hash); w.put(h.chat_check); w.put(h.lora_hash); w.put(h.t_write); w.put(h.n_tokens); w.put(h.total);
    w.put(h.cfg.kv_unified); w.put(h.cfg.swa_full); w.put(h.cfg.n_ubatch);   // format 5
    w.put(h.cfg.rot_k); w.put(h.cfg.rot_v); w.put(h.cfg.rot_k_dft); w.put(h.cfg.rot_v_dft);   // format 6
}

// false = not a file of this tier (bad magic or a truncated header); the version is checked by the caller
bool cd_get_header(cd_reader & r, cd_header & h) {
    char magic[8];
    r.bytes(magic, sizeof(magic));
    if (!r.ok || memcmp(magic, CD_MAGIC, sizeof(magic)) != 0) {
        return false;
    }
    r.get(h.version);
    if (!r.ok || !cd_version_ok(h.version)) {
        return r.ok;   // another format: the caller refuses it by its version, without reading fields it does not know
    }
    r.get(h.kind);
    r.get(h.cfg.model_bytes); r.get(h.cfg.gguf_hash); r.get(h.cfg.draft_bytes); r.get(h.cfg.draft_hash); r.get(h.cfg.ctrl_hash);
    r.get(h.cfg.ovr_hash);
    r.get(h.cfg.n_ctx); r.get(h.cfg.type_k); r.get(h.cfg.type_v); r.get(h.cfg.type_k_dft); r.get(h.cfg.type_v_dft);
    r.get(h.cfg.n_seq); r.get(h.cfg.has_draft); r.get(h.cfg.flash_attn); r.get(h.cfg.flash_attn_dft); r.get(h.cfg.n_ckpt_max);
    r.get(h.cfg.state_ver);
    r.get(h.chat_hash); r.get(h.chat_check); r.get(h.lora_hash); r.get(h.t_write); r.get(h.n_tokens); r.get(h.total);
    r.get(h.cfg.kv_unified); r.get(h.cfg.swa_full); r.get(h.cfg.n_ubatch);   // format 5
    if (h.version >= 6) {
        r.get(h.cfg.rot_k); r.get(h.cfg.rot_v); r.get(h.cfg.rot_k_dft); r.get(h.cfg.rot_v_dft);
    } else {
        h.cfg.rot_k = h.cfg.rot_v = h.cfg.rot_k_dft = h.cfg.rot_v_dft = -2;   // format 5: saved under the default rule
    }
    return r.ok;
}

// the smallest checkpoint record (n_tokens, id_task, pos_min, pos_max, three empty blobs), for bounding a count
constexpr uint64_t CD_CKPT_MIN_BYTES = 8 + 4 + 4 + 4 + 8 + 8 + 8;

int64_t cd_now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string cd_hex(uint64_t v) {
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) v);
    return buf;
}

bool cd_is_entry_name(const std::string & n) {
    const auto ok_hex = [&](size_t from) {
        if (n.size() != from + 16 + 6 || n.compare(from + 16, 6, ".state") != 0) {
            return false;
        }
        for (size_t i = from; i < from + 16; i++) {
            if (!isxdigit((unsigned char) n[i])) {
                return false;
            }
        }
        return true;
    };
    // snap-<16 hex chat>-<16 hex n_tokens>.state
    const auto ok_snap = [&]() {
        if (n.size() != 5 + 16 + 1 + 16 + 6 || n.rfind("snap-", 0) != 0 || n[21] != '-' || n.compare(38, 6, ".state") != 0) {
            return false;
        }
        for (size_t i = 5; i < 38; i++) {
            if (i != 21 && !isxdigit((unsigned char) n[i])) {
                return false;
            }
        }
        return true;
    };
    return (n.rfind("chat-", 0) == 0 && ok_hex(5)) || (n.rfind("prefix-", 0) == 0 && ok_hex(7)) || ok_snap();
}

// A temp file is "<entry>.<pid>.<nonce>.tmp" - this process's id and a random value drawn once - so the
// unfinished write a dead writer left can never block a new save (the old "<entry>.tmp" did, until it was an hour old)
const std::string & cd_tmp_suffix() {
    static const std::string suffix = [] {
        uint64_t nonce = 0;
        try {
            std::random_device rd;
            nonce = ((uint64_t) rd() << 32) ^ (uint64_t) rd();
        } catch (const std::exception &) {
            nonce = (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
        }
#ifndef _WIN32
        const long long pid = (long long) getpid();
#else
        const long long pid = (long long) _getpid();   // a real id, so the name parses; on Windows
                                                                    // cd_pid_alive() cannot tell, so only the age rule removes it
#endif
        char buf[64];
        snprintf(buf, sizeof(buf), ".%lld.%016llx.tmp", pid, (unsigned long long) nonce);
        return std::string(buf);
    }();
    return suffix;
}

// a temp file of this tier: "<entry>.<pid>.<16 hex>.tmp" (pid > 0) or the old "<entry>.tmp" (pid = 0); false = not one
bool cd_parse_tmp(const std::string & name, long long & pid) {
    if (name.size() <= 4 || name.compare(name.size() - 4, 4, ".tmp") != 0) {
        return false;
    }
    const std::string base = name.substr(0, name.size() - 4);
    if (cd_is_entry_name(base)) {
        pid = 0;
        return true;
    }
    const size_t dot_nonce = base.rfind('.');
    if (dot_nonce == std::string::npos || base.size() - dot_nonce - 1 != 16 ||
        base.find_first_not_of("0123456789abcdef", dot_nonce + 1) != std::string::npos) {   // 16 hex digits
        return false;
    }
    const size_t dot_pid = base.rfind('.', dot_nonce - 1);
    if (dot_pid == std::string::npos || !cd_is_entry_name(base.substr(0, dot_pid))) {
        return false;
    }
    const std::string spid = base.substr(dot_pid + 1, dot_nonce - dot_pid - 1);
    if (spid.empty() || spid.find_first_not_of("0123456789") != std::string::npos) {
        return false;
    }
    pid = atoll(spid.c_str());
    return pid > 0;
}

// whether a process with that id exists (EPERM: it exists, another user's)
bool cd_pid_alive(long long pid) {
#ifndef _WIN32
    return pid > 0 && (kill((pid_t) pid, 0) == 0 || errno == EPERM);
#else
    GGML_UNUSED(pid);
    return true;
#endif
}

// three answers: this tier's file / readable but not this tier's / cannot be read (EACCES, EMFILE, ...: never "not ours")
enum cd_magic_answer { CD_MAGIC_YES, CD_MAGIC_NO, CD_MAGIC_UNREADABLE };

cd_magic_answer cd_magic(const std::string & path) {
    char magic[8];
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return CD_MAGIC_UNREADABLE;
    }
    const size_t n = fread(magic, 1, sizeof(magic), f);
    const bool err = ferror(f) != 0;
    fclose(f);
    if (err) {
        return CD_MAGIC_UNREADABLE;
    }
    return n == sizeof(magic) && memcmp(magic, CD_MAGIC, sizeof(magic)) == 0 ? CD_MAGIC_YES : CD_MAGIC_NO;
}

// the common prefix of a saved token list and a request (media positions are LLAMA_TOKEN_NULL there, never in a saved list)
size_t cd_lcp(const llama_tokens & a, const server_tokens & b) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) {
            return i;
        }
    }
    return n;
}

size_t cd_lcp(const llama_tokens & a, const llama_tokens & b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        i++;
    }
    return i;
}

// /proc/meminfo's Dirty and Writeback in kB (-1 when not available)
void cd_meminfo(int64_t & dirty_kb, int64_t & wb_kb) {
    dirty_kb = -1;
    wb_kb    = -1;
#ifdef __linux__
    if (FILE * f = fopen("/proc/meminfo", "r")) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            long long v = 0;
            if (sscanf(line, "Dirty: %lld kB", &v) == 1) {
                dirty_kb = v;
            } else if (sscanf(line, "Writeback: %lld kB", &v) == 1) {
                wb_kb = v;
            }
        }
        fclose(f);
    }
#endif
}

// a directory's own entry is flushed too, so a rename survives a power cut
void cd_fsync_dir(const std::string & dir) {
#ifndef _WIN32
    const int fd = open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd >= 0) {
        fsync(fd);
        close(fd);
    }
#else
    GGML_UNUSED(dir);
#endif
}

// a new file, owner-only (0600), never through a link: O_EXCL | O_NOFOLLOW. Something already at that name: a symlink (or
// anything that is not a regular file) is removed - unlink never touches a link's target - and the create is tried once more;
// a regular file there is another writer's (or a stale write): the create fails (EEXIST) and the caller logs it
FILE * cd_create(const std::string & path) {
#ifndef _WIN32
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0 && errno == EEXIST) {
        struct stat st;
        if (lstat(path.c_str(), &st) == 0 && !S_ISREG(st.st_mode)) {
            SRV_WRN("cache-disk: '%s' existed and was not a regular file (a link?); removed without following it\n", path.c_str());
            if (unlink(path.c_str()) == 0) {
                fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
            }
        } else {
            errno = EEXIST;
        }
    }
    if (fd < 0) {
        return nullptr;
    }
    FILE * f = fdopen(fd, "wb");
    if (!f) {
        const int err = errno;
        close(fd);
        errno = err;
    }
    return f;
#else
    return fopen(path.c_str(), "wb");
#endif
}

} // namespace

bool server_cache_disk::config::operator==(const config & o) const {
    return diff(o).empty();
}

std::string server_cache_disk::config::diff(const config & o) const {
    if (model_bytes != o.model_bytes) return string_format("model files %llu B, this server %llu B", (unsigned long long) o.model_bytes, (unsigned long long) model_bytes);
    if (gguf_hash   != o.gguf_hash)   return "another GGUF header (another model file)";
    if (draft_bytes != o.draft_bytes || draft_hash != o.draft_hash) return "another draft model";
    if (ctrl_hash   != o.ctrl_hash)   return "other control vectors";
    if (ovr_hash    != o.ovr_hash)    return "other model overrides (RoPE / YaRN / --override-kv)";
    if (n_ctx       != o.n_ctx)       return string_format("n_ctx %u, this server %u", o.n_ctx, n_ctx);
    if (type_k      != o.type_k)      return string_format("K cache type %d, this server %d", o.type_k, type_k);
    if (type_v      != o.type_v)      return string_format("V cache type %d, this server %d", o.type_v, type_v);
    if (type_k_dft  != o.type_k_dft)  return string_format("draft K cache type %d, this server %d", o.type_k_dft, type_k_dft);
    if (type_v_dft  != o.type_v_dft)  return string_format("draft V cache type %d, this server %d", o.type_v_dft, type_v_dft);
    if (n_seq       != o.n_seq)       return string_format("n_seq %u, this server %u", o.n_seq, n_seq);
    if (has_draft   != o.has_draft)   return string_format("draft context %u, this server %u", o.has_draft, has_draft);
    if (flash_attn  != o.flash_attn)  return string_format("flash attention %d, this server %d", o.flash_attn, flash_attn);
    if (flash_attn_dft != o.flash_attn_dft) return string_format("draft flash attention %d, this server %d", o.flash_attn_dft, flash_attn_dft);
    if (n_ckpt_max  != o.n_ckpt_max)  return string_format("ctx-checkpoints %d, this server %d", o.n_ckpt_max, n_ckpt_max);
    if (state_ver   != o.state_ver)   return string_format("state format %u, this server %u", o.state_ver, state_ver);
    if (kv_unified  != o.kv_unified)  return string_format("kv-unified %u, this server %u", o.kv_unified, kv_unified);
    if (swa_full    != o.swa_full)    return string_format("swa-full %u, this server %u", o.swa_full, swa_full);
    if (n_ubatch    != o.n_ubatch)    return string_format("n_ubatch %u, this server %u", o.n_ubatch, n_ubatch);
    // A state restored under another K / V rotation would silently attend with wrongly rotated keys
    // -2 = a format-5 file, saved under the default rule: it stands for the other side's default-rule size (either side may be
    // the file: read_header compares cfg.diff(file), the lookups file == cfg)
    const auto rot_diff = [](const char * what, int32_t a, int32_t a_def, int32_t b, int32_t b_def) -> std::string {
        const int32_t va = a == -2 ? b_def : a;
        const int32_t vb = b == -2 ? a_def : b;
        return va == vb ? "" : string_format("%s rotation %d, the other %d", what, vb, va);
    };
    for (const std::string & d : { rot_diff("K", rot_k, rot_k_def, o.rot_k, o.rot_k_def), rot_diff("V", rot_v, rot_v_def, o.rot_v, o.rot_v_def),
                                   rot_diff("draft K", rot_k_dft, rot_k_dft_def, o.rot_k_dft, o.rot_k_dft_def),
                                   rot_diff("draft V", rot_v_dft, rot_v_dft_def, o.rot_v_dft, o.rot_v_dft_def) }) {
        if (!d.empty()) return d;
    }
    return "";
}

uint64_t server_cache_disk::fnv1a(const void * data, size_t n, uint64_t h) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

uint64_t server_cache_disk::chat_check_of(const std::string & conv_id) {
    return cd_mix(0x243F6A8885A308D3ull, conv_id.data(), conv_id.size());
}

uint64_t server_cache_disk::lora_hash_of(const std::vector<common_adapter_lora_info> & loras) {
    uint64_t h = 0;
    for (const auto & l : loras) {
        if (l.scale == 0.0f) {
            continue;   // an adapter at scale 0 does not change the state
        }
        if (h == 0) {
            h = 1469598103934665603ull;
        }
        h = fnv1a(l.path.data(), l.path.size(), h);
        h = fnv1a(&l.scale, sizeof(l.scale), h);
    }
    return h;
}

std::string server_cache_disk::chat_name(const std::string & conv_id) {
    return "chat-" + cd_hex(fnv1a(conv_id.data(), conv_id.size())) + ".state";
}

std::string server_cache_disk::snap_name(const std::string & conv_id, int64_t n_tokens) {
    return "snap-" + cd_hex(fnv1a(conv_id.data(), conv_id.size())) + "-" + cd_hex((uint64_t) n_tokens) + ".state";
}

uint64_t server_cache_disk::prefix_hash(const server_tokens & tokens, int64_t n) {
    uint64_t h = 1469598103934665603ull;
    for (int64_t i = 0; i < n && i < (int64_t) tokens.size(); i++) {
        const llama_token t = tokens[i];
        h = fnv1a(&t, sizeof(t), h);
    }
    return h;
}

std::string server_cache_disk::prefix_name(const llama_tokens & tokens) {
    return "prefix-" + cd_hex(fnv1a(tokens.data(), tokens.size()*sizeof(llama_token))) + ".state";
}

bool server_cache_disk::read_header(const std::string & path, entry & e, bool with_tokens) const {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        e.why = "cannot open";
        return false;
    }
    cd_reader r { f, e.file_size };
    cd_header h;
    bool ok = false;
    try {
        if (!cd_get_header(r, h)) {
            e.why = "not a --cache-disk file (bad magic or a truncated header)";
        } else if (!cd_version_ok(h.version)) {
            e.why = string_format("file format %u, this server reads %u..%u", h.version, CD_VERSION_MIN, CD_VERSION);
        } else {
            e.t_write    = h.t_write;
            e.chat_hash  = h.chat_hash;
            e.chat_check = h.chat_check;
            e.lora_hash  = h.lora_hash;
            if (h.total != e.file_size) {
                e.why = string_format("truncated or damaged: %llu B on disk, the header says %llu B",
                        (unsigned long long) e.file_size, (unsigned long long) h.total);
            } else if (h.kind != (e.is_snap ? 2u : e.is_chat ? 1u : 0u)) {
                e.why = "its kind does not match its name";
            } else if (!(h.cfg == cfg)) {
                e.why = "saved under other settings: " + cfg.diff(h.cfg);
            } else if (e.is_snap) {
                // The snapshot's position, memory range and prefix hash open its (hashed) body
                int64_t sn = 0; int32_t pmin = 0, pmax = 0; uint64_t ph = 0;
                r.get(sn); r.get(pmin); r.get(pmax); r.get(ph);
                ok = r.ok && sn == (int64_t) h.n_tokens;
                // the name is snap_name(chat, n_tokens): a file renamed to another chat or position is refused
                if (ok && e.name.size() == 44 && e.name != "snap-" + cd_hex(h.chat_hash) + "-" + cd_hex((uint64_t) sn) + ".state") {
                    ok = false;
                    e.why = "its name does not match its chat or position";
                } else if (ok) {
                    e.snap_n_tokens = sn; e.snap_pos_min = pmin; e.snap_pos_max = pmax; e.snap_prefix = ph;
                } else if (e.why.empty()) {
                    e.why = "a damaged snapshot header";
                }
            } else if (with_tokens) {
                if (h.n_tokens > r.left() / sizeof(llama_token) || (sizeof(size_t) < 8 && h.n_tokens > SIZE_MAX / sizeof(llama_token))) {
                    e.why = "damaged token count";
                } else {
                    e.tokens.resize((size_t) h.n_tokens);
                    r.bytes(e.tokens.data(), h.n_tokens*sizeof(llama_token));
                    ok = r.ok;
                    if (!ok) {
                        e.why = "truncated tokens";
                    }
                }
            } else {
                ok = true;
            }
        }
    } catch (const std::exception & ex) {
        ok = false;
        e.why = std::string("cannot be read: ") + ex.what();
    }
    fclose(f);
    return ok;
}

bool server_cache_disk::init(const std::string & dir_in, double limit_gb, const config & cfg_in) {
    namespace fs = std::filesystem;
    dir         = dir_in;
    cfg         = cfg_in;
    limit_bytes = (uint64_t) (limit_gb * 1e9);   // limit_gb is checked by the argument parser (finite, > 0, <= 1e6)

    std::error_code ec;
    if (fs::create_directories(dir, ec)) {
        // created by this server: owner-only, as the files are
        fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace, ec);
        ec.clear();
    }
    if (!fs::is_directory(dir, ec)) {
        SRV_ERR("cache-disk: '%s' is not a directory that can be created (%s)\n", dir.c_str(), ec ? ec.message().c_str() : "not a directory");
        return false;
    }

    // the directory must take a new file: one owner-only probe file, written and deleted
    {
        const std::string probe = (fs::path(dir) / string_format(".cache-disk-probe-%lld", (long long) cd_now_us())).string();
        FILE * f = cd_create(probe);
        const int err = errno;
        bool ok = f != nullptr;
        if (f) {
            ok = fwrite("x", 1, 1, f) == 1;
            ok = (fclose(f) == 0) && ok;
            fs::remove(probe, ec);
        }
        if (!ok) {
            SRV_ERR("cache-disk: cannot write in '%s' (%s)\n", dir.c_str(), strerror(f ? EIO : err));
            return false;
        }
    }

    const int64_t now = cd_now_us();
    fs::directory_iterator it(dir, ec);
    for (; !ec && it != fs::directory_iterator(); it.increment(ec)) {
        const auto & de = *it;
        std::error_code ec2;
        if (!de.is_regular_file(ec2)) {
            continue;
        }
        const std::string name = de.path().filename().string();
        int64_t mtime_us = 0;
        {
            const auto ft = fs::last_write_time(de.path(), ec2);
            if (!ec2) {
                // the file clock's epoch is not the system clock's: convert through "now" on both clocks
                mtime_us = now - std::chrono::duration_cast<std::chrono::microseconds>(fs::file_time_type::clock::now() - ft).count();
            }
        }
        long long tmp_pid = 0;
        if (cd_parse_tmp(name, tmp_pid)) {
            // an unfinished write: removed when its writer is gone; a live writer's (another server's write in
            // progress in the same directory) only when it is over an hour old; the old "<entry>.tmp" form by age only
            const bool dead  = tmp_pid > 0 && !cd_pid_alive(tmp_pid);
            const bool stale = mtime_us > 0 && now - mtime_us > CD_TMP_AGE_US;
            if (dead || stale) {
                fs::remove(de.path(), ec2);
                SRV_WRN("cache-disk: removed the unfinished write %s (%s)%s\n", name.c_str(),
                        dead ? "its writer is gone" : "over an hour old", ec2 ? (" - failed: " + ec2.message()).c_str() : "");
            }
            continue;
        }
        if (!cd_is_entry_name(name)) {
            continue;   // not ours: never touched
        }
        const cd_magic_answer mg = cd_magic(de.path().string());
        if (mg == CD_MAGIC_NO) {
            SRV_INF("cache-disk: ignored %s: an entry name without this tier's magic (not ours: never counted, never deleted)\n", name.c_str());
            continue;
        }
        if (mg == CD_MAGIC_UNREADABLE) {
            SRV_WRN("cache-disk: ignored %s for now: it cannot be read (%s)\n", name.c_str(), strerror(errno));
            continue;
        }
        entry e;
        e.name      = name;
        e.is_chat   = name.rfind("chat-", 0) == 0;
        e.is_snap   = name.rfind("snap-", 0) == 0;
        e.file_size = de.file_size(ec2);
        if (ec2) {
            e.file_size = 0;
        }
        e.t_write   = mtime_us;
        e.usable    = !ec2 && read_header(de.path().string(), e, true);
        if (e.usable && e.is_snap && mtime_us > e.t_write) {
            e.t_write = mtime_us;   // a snapshot used again has its file time refreshed, not its header
        }
        if (!e.usable) {
            if (ec2 && e.why.empty()) {
                e.why = "size unknown: " + ec2.message();
            }
            SRV_WRN("cache-disk: refused %s: %s (left in place; it leaves by the size limit)\n", name.c_str(), e.why.c_str());
        }
        index.push_back(std::move(e));
    }
    if (ec) {
        SRV_WRN("cache-disk: listing '%s' stopped early: %s\n", dir.c_str(), ec.message().c_str());
    }

    SRV_INF("cache-disk: %s, limit %.3f GB, %zu entries (%zu usable), %.3f GB on disk\n", dir.c_str(), limit_bytes/1e9,
            index.size(), (size_t) std::count_if(index.begin(), index.end(), [](const entry & e) { return e.usable; }),
            total_bytes("")/1e9);
    return true;
}

uint64_t server_cache_disk::total_bytes(const std::string & except) const {
    uint64_t s = 0;
    for (const auto & e : index) {
        if (e.name != except) {
            s += e.file_size;
        }
    }
    return s;
}

bool server_cache_disk::remove_entry(size_t i, const char * reason) {
    std::error_code ec;
    const auto path = std::filesystem::path(dir) / index[i].name;
    if (std::filesystem::exists(path, ec)) {
        const cd_magic_answer mg = cd_magic(path.string());
        if (mg == CD_MAGIC_UNREADABLE) {
            // cannot tell whether it is ours: a failed delete (the entry stays)
            SRV_WRN("cache-disk: could not delete %s (%s): it cannot be read (%s) - it stays in the index\n", index[i].name.c_str(), reason, strerror(errno));
            return false;
        }
        if (mg == CD_MAGIC_NO) {
            // replaced by a file that is not this tier's since it was indexed: never deleted, only dropped from the index
            SRV_WRN("cache-disk: %s is no longer this tier's file (no magic); dropped from the index, not deleted (%s)\n", index[i].name.c_str(), reason);
            index.erase(index.begin() + i);
            return true;
        }
    }
    std::filesystem::remove(path, ec);   // a file that is already gone counts as deleted
    if (ec) {
        SRV_WRN("cache-disk: could not delete %s (%s): %s - it stays in the index\n", index[i].name.c_str(), reason, ec.message().c_str());
        return false;
    }
    SRV_INF("cache-disk: deleted %s (%.1f MB, %s)\n", index[i].name.c_str(), index[i].file_size/1e6, reason);
    index.erase(index.begin() + i);
    return true;
}

bool server_cache_disk::save(const server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot,
                             const std::string & conv_id, uint64_t lora_hash, int64_t t_reply_end_us) {
    namespace fs = std::filesystem;
    if (prompt.tokens.size() == 0) {
        return false;
    }
    // the tokens through operator[] (get_tokens() asserts once a multimodal projector is loaded); a media position is
    // LLAMA_TOKEN_NULL and such a prompt is not saved
    llama_tokens toks(prompt.tokens.size());
    for (size_t i = 0; i < toks.size(); i++) {
        toks[i] = prompt.tokens[i];
        if (toks[i] == LLAMA_TOKEN_NULL) {
            if (!warned_media) {
                SRV_INF("%s", "cache-disk: not saved: the prompt holds media (only text states are kept on disk; this line is logged once)\n");
                warned_media = true;
            }
            return false;
        }
    }

    const bool        is_chat = !conv_id.empty();
    const std::string name    = is_chat ? chat_name(conv_id) : prefix_name(toks);

    // the RAM tier's rules for prompts without a chat id (server_prompt_cache::alloc): not saved when a saved prefix entry
    // already contains it fully; prefix entries it contains fully are deleted after the write
    std::vector<std::string> contained;
    if (!is_chat) {
        for (const auto & e : index) {
            if (!e.usable || e.is_chat || e.is_snap || e.lora_hash != lora_hash) {
                continue;
            }
            const size_t lcp = cd_lcp(e.tokens, toks);
            if (lcp == toks.size()) {
                SRV_INF("cache-disk: not saved: %zu tokens are contained in %s\n", toks.size(), e.name.c_str());
                return false;
            }
            if (lcp == e.tokens.size() && e.name != name) {
                contained.push_back(e.name);
            }
        }
    }

    const size_t size_tgt = llama_state_seq_get_size_ext(ctx_tgt, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
    const size_t size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
    // 0 is llama.cpp's answer to an internal error: an empty state would restore "successfully" over the slot's old KV
    if (size_tgt == 0 || (ctx_dft && size_dft == 0)) {
        SRV_ERR("cache-disk: not saved: llama.cpp reported an empty %s state (an error while sizing it)\n", size_tgt == 0 ? "main" : "draft");
        return false;
    }

    cd_header h;
    h.version   = CD_VERSION;
    h.kind      = is_chat ? 1 : 0;
    h.cfg       = cfg;
    h.chat_hash  = is_chat ? fnv1a(conv_id.data(), conv_id.size()) : 0;
    h.chat_check = is_chat ? chat_check_of(conv_id) : 0;
    h.lora_hash  = lora_hash;
    h.n_tokens  = toks.size();

    uint64_t n_header = 0;
    {
        cd_writer cw;   // counts only
        cd_put_header(cw, h);
        n_header = cw.n;
    }
    // A chat's snapshots that are in their own files are not stored again here; their positions are listed
    // Only the ones made for this history (the prefix hash), and the size-limit loop below never deletes them
    const std::vector<int64_t> disk_pos = is_chat && max_snaps > 0 ? snapshot_positions(conv_id, &prompt.tokens) : std::vector<int64_t>();
    std::vector<const common_prompt_checkpoint *> ckpts;
    for (const auto & c : prompt.checkpoints) {
        if (std::find(disk_pos.begin(), disk_pos.end(), c.n_tokens) == disk_pos.end()) {
            ckpts.push_back(&c);
        }
    }
    uint64_t total = n_header + toks.size()*sizeof(llama_token) + 4;
    for (const auto * c : ckpts) {
        total += 8 + 4 + 4 + 4 + 8 + c->data_tgt.size() + 8 + c->data_dft.size() + 8 + c->data_spec.size();
    }
    total += 4 + 8*disk_pos.size();
    total += 8 + size_tgt + 8 + size_dft + 8 + sizeof(CD_END);
    h.total = total;

    if (total > limit_bytes) {
        SRV_WRN("cache-disk: not saved: the state is %.1f MB, over the %.3f GB limit\n", total/1e6, limit_bytes/1e9);
        return false;
    }

    // The save's parts, timed (us) - the state copy out of llama.cpp, the checksum, the write (fwrite + fflush, without
    // the checksum), the file's fsync, the close, making room, the rename, the directory's fsync - and the kernel's dirty /
    // writeback memory before and after, for the slow saves (dirty-page throttling or the file system's writeback?)
    int64_t mi_dirty0 = 0, mi_wb0 = 0;
    cd_meminfo(mi_dirty0, mi_wb0);
    const int64_t t_copy0 = ggml_time_us();
    server_prompt_data data;
    try {
        data.main.resize(size_tgt);
        data.drft.resize(size_dft);
    } catch (const std::exception &) {
        SRV_ERR("cache-disk: not saved: cannot allocate %.1f MB for the state\n", (size_tgt + size_dft)/1e6);
        return false;
    }
    if (llama_state_seq_get_data_ext(ctx_tgt, data.main.data(), size_tgt, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE) != size_tgt ||
        (ctx_dft && llama_state_seq_get_data_ext(ctx_dft, data.drft.data(), size_dft, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE) != size_dft)) {
        SRV_ERR("%s", "cache-disk: not saved: the state could not be read from the context\n");
        return false;
    }
    const int64_t t_copy = ggml_time_us() - t_copy0;
    int64_t t_open = 0, t_write = 0, t_fsync = 0, t_close = 0, t_evict = 0, t_rename = 0, t_dirsync = 0;

    const std::string path = (fs::path(dir) / name).string();
    const std::string tmp  = path + cd_tmp_suffix();

    // 1. the temp file (owner-only), complete and on the disk, before anything in the directory is deleted
    h.t_write = cd_now_us();
    int64_t tp = ggml_time_us();
    FILE * f = cd_create(tmp);
    const int err_open = errno;   // before the timing calls
    t_open = ggml_time_us() - tp;
    tp = ggml_time_us();
    cd_writer w { f, f != nullptr };
    bool size_mismatch = false;
    int err_write = 0;   // the errno of the first step that failed, taken right after it
    int64_t mi_dirty1 = 0, mi_wb1 = 0;
    // F10's parts, logged for every save that got to the write - a failed one too
    const auto log_parts = [&](const char * outcome) {
        cd_meminfo(mi_dirty1, mi_wb1);
        SRV_INF("cache-disk: save parts %s (%s): %.1f MB: state copy %.1f, checksum %.1f, open %.1f, write %.1f, fsync %.1f, close %.1f, "
                "room %.1f, rename %.1f, dir fsync %.1f ms; Dirty %.1f -> %.1f MB, Writeback %.1f -> %.1f MB\n",
                name.c_str(), outcome, total/1e6, t_copy/1000.0, w.t_hash_us/1000.0, t_open/1000.0, t_write/1000.0, t_fsync/1000.0,
                t_close/1000.0, t_evict/1000.0, t_rename/1000.0, t_dirsync/1000.0,
                mi_dirty0/1024.0, mi_dirty1/1024.0, mi_wb0/1024.0, mi_wb1/1024.0);
    };
    if (f) {
        cd_put_header(w, h);
        w.hashing = true;
        w.bytes(toks.data(), toks.size()*sizeof(llama_token));
        w.put<uint32_t>((uint32_t) ckpts.size());
        for (const auto * c : ckpts) {
            w.put<int64_t>(c->n_tokens);
            w.put<int32_t>(c->id_task);
            w.put<int32_t>(c->pos_min);
            w.put<int32_t>(c->pos_max);
            w.blob(c->data_tgt);
            w.blob(c->data_dft);
            w.blob(c->data_spec);
        }
        w.put<uint32_t>((uint32_t) disk_pos.size());
        for (const int64_t pos : disk_pos) {
            w.put<int64_t>(pos);
        }
        w.blob(data.main);
        w.blob(data.drft);
        w.hashing = false;
        w.put<uint64_t>(w.hash);
        w.bytes(CD_END, sizeof(CD_END));
        if (!w.ok && err_write == 0) {
            err_write = errno;   // an fwrite failed
        }
        size_mismatch = w.ok && w.n != total;
        w.ok = w.ok && !size_mismatch;
        if (w.ok && fflush(f) != 0) {
            err_write = errno;
            w.ok = false;
        }
        t_write = ggml_time_us() - tp - w.t_hash_us;
        tp = ggml_time_us();
#ifndef _WIN32
        if (w.ok && fsync(fileno(f)) != 0) {
            err_write = errno;
            w.ok = false;
        }
#endif
        t_fsync = ggml_time_us() - tp;
        tp = ggml_time_us();
        if (fclose(f) != 0 && w.ok) {
            err_write = errno;
            w.ok = false;
        }
        t_close = ggml_time_us() - tp;
    }
    std::error_code ec;
    if (!w.ok) {
        if (f) {
            fs::remove(tmp, ec);   // only a temp file this save created (one found at that name is another writer's)
            log_parts("write failed");
        }
        SRV_ERR("cache-disk: the write of %s failed (%s); nothing was deleted, the reply is unchanged\n", name.c_str(),
                size_mismatch ? "internal error: the bytes written differ from the computed size" : strerror(!f ? err_open : err_write));
        return false;
    }

    // 2. room under the limit: the entries written longest ago go first, whatever their kind (not the file this save
    //    replaces, not the prefix entries it contains - they go after the rename); an entry that cannot be deleted stays
    const auto counted = [&]() {
        uint64_t s = 0;
        for (const auto & e : index) {
            if (e.name != name && std::find(contained.begin(), contained.end(), e.name) == contained.end()) {
                s += e.file_size;
            }
        }
        return s;
    };
    tp = ggml_time_us();
    std::vector<std::string> undeletable;
    while (counted() + total > limit_bytes) {
        size_t oldest = SIZE_MAX;
        for (size_t i = 0; i < index.size(); i++) {
            const auto & e = index[i];
            if (e.name == name || std::find(contained.begin(), contained.end(), e.name) != contained.end() ||
                std::find(undeletable.begin(), undeletable.end(), e.name) != undeletable.end() ||
                (is_chat && e.is_snap && e.chat_hash == h.chat_hash)) {   // This chat's own snapshots (listed, not stored)
                continue;
            }
            if (oldest == SIZE_MAX || e.t_write < index[oldest].t_write) {
                oldest = i;
            }
        }
        if (oldest == SIZE_MAX) {
            break;
        }
        const std::string oname = index[oldest].name;
        const bool     ochat = index[oldest].is_chat;
        const uint64_t ohash = index[oldest].chat_hash;
        if (!remove_entry(oldest, "size limit, written longest ago")) {
            undeletable.push_back(oname);
        } else if (ochat && oname != name) {
            remove_snaps_of(ohash, -1, "its chat left by the size limit");   // a chat's snapshots go with it
        }
    }
    if (counted() + total > limit_bytes) {
        fs::remove(tmp, ec);
        log_parts("no room");
        SRV_WRN("cache-disk: not saved: %.1f MB do not fit under the %.3f GB limit (entries that cannot be deleted)\n", total/1e6, limit_bytes/1e9);
        return false;
    }

    t_evict = ggml_time_us() - tp;

    // 3. the rename makes it visible (replacing a chat's earlier file), then the directory entry is flushed
    tp = ggml_time_us();
    fs::rename(tmp, path, ec);
    t_rename = ggml_time_us() - tp;
    if (ec) {
        const std::string msg = ec.message();
        fs::remove(tmp, ec);
        log_parts("rename failed");
        SRV_ERR("cache-disk: the rename of %s failed (%s); the temp file is deleted, the reply is unchanged\n", name.c_str(), msg.c_str());
        return false;
    }
    tp = ggml_time_us();
    cd_fsync_dir(dir);
    t_dirsync = ggml_time_us() - tp;

    for (const auto & cname : contained) {
        for (size_t i = 0; i < index.size(); i++) {
            if (index[i].name == cname) {
                remove_entry(i, "contained in the new state");
                break;
            }
        }
    }

    // the index: replace this name's entry (a chat's earlier state), or add it
    entry e;
    e.name      = name;
    e.is_chat   = is_chat;
    e.chat_hash  = h.chat_hash;
    e.chat_check = h.chat_check;
    e.lora_hash  = lora_hash;
    e.file_size = total;
    e.t_write   = h.t_write;
    e.usable    = true;
    e.tokens    = std::move(toks);
    const size_t n_tok = e.tokens.size();
    auto it = std::find_if(index.begin(), index.end(), [&](const entry & x) { return x.name == name; });
    if (it != index.end()) {
        *it = std::move(e);
    } else {
        index.push_back(std::move(e));
    }

    SRV_INF("cache-disk: saved %s: %zu tokens, %zu checkpoints (+%zu in snapshot files), %.1f MB in %.1f ms (reply end -> rename; checksum %.1f ms); %zu entries, %.3f of %.3f GB\n",
            name.c_str(), n_tok, ckpts.size(), disk_pos.size(), total/1e6, (ggml_time_us() - t_reply_end_us)/1000.0,
            w.t_hash_us/1000.0, index.size(), total_bytes("")/1e9, limit_bytes/1e9);
    log_parts("saved");
    return true;
}

server_cache_disk::pick server_cache_disk::find(const server_prompt & slot_prompt, const server_tokens & tokens_new,
                                                const std::string & conv_id, uint64_t lora_hash, bool chat_only) {
    pick pk;
    if (tokens_new.size() == 0) {
        return pk;
    }

    // the slot's own content is the base, as in server_prompt_cache::load
    const size_t lcp_base = slot_prompt.tokens.get_common_prefix(tokens_new);
    float f_keep_best = slot_prompt.tokens.size() > 0 ? float(lcp_base) / slot_prompt.tokens.size() : -1.0f;
    float f_sim_best  = float(lcp_base) / tokens_new.size();

    // 1. by chat id: its own entry, when it shares more of the request than the slot holds and >= 25 % of its own tokens
    //    (the RAM tier's f_keep floor: a chat edited near its start does not read a large state for a few tokens)
    if (!conv_id.empty()) {
        const std::string cname = chat_name(conv_id);
        for (size_t i = 0; i < index.size(); i++) {
            const auto & e = index[i];
            if (e.name != cname) {
                continue;
            }
            if (!e.usable) {
                SRV_WRN("cache-disk: refused %s: %s (left in place; the chat re-reads)\n", cname.c_str(), e.why.c_str());
            } else if (e.chat_hash != fnv1a(conv_id.data(), conv_id.size()) || e.chat_check != chat_check_of(conv_id)) {
                SRV_WRN("cache-disk: refused %s: it holds another chat id's state (hash collision; left in place)\n", cname.c_str());
            } else if (e.lora_hash != lora_hash) {
                SRV_INF("cache-disk: not used %s: saved with another LoRA set\n", cname.c_str());
            } else {
                const size_t lcp = cd_lcp(e.tokens, tokens_new);
                // Below 25 %, also when one of the chat's disk snapshots lies inside the shared part (an edited older
                // message): a snapshot holds only the state that cannot be cut back, the chat file brings the rest up to it
                const bool snap = lcp > lcp_base && 4*lcp < e.tokens.size() && max_snaps > 0 && lcp >= 2 &&
                        find_snapshot(conv_id, lora_hash, 0, (int32_t) std::min<size_t>(lcp, tokens_new.size() - 1) - 1,
                              std::max<int32_t>(0, (int32_t) lcp - n_swa), tokens_new) != SIZE_MAX;   // The server's own threshold
                if (lcp > lcp_base && (4*lcp >= e.tokens.size() || snap)) {
                    pk.i   = i;
                    pk.lcp = lcp;
                    pk.how = snap ? "chat id, for a disk snapshot inside the shared part" : "chat id";
                    return pk;
                }
            }
            break;
        }
    }

    if (chat_only) {
        return pk;
    }

    // 2. by longest beginning: the RAM tier's choice rule over the prefix entries (f_keep >= 0.25, better f_keep and f_sim)
    for (size_t i = 0; i < index.size(); i++) {
        const auto & e = index[i];
        if (!e.usable || e.is_chat || e.is_snap || e.tokens.empty() || e.lora_hash != lora_hash) {
            continue;
        }
        const size_t lcp = cd_lcp(e.tokens, tokens_new);
        const float f_keep_cur = float(lcp) / e.tokens.size();
        const float f_sim_cur  = float(lcp) / tokens_new.size();
        if (f_keep_cur < 0.25f) {
            continue;
        }
        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;
            pk.i   = i;
            pk.lcp = lcp;
            pk.how = "longest beginning";
        }
    }
    return pk;
}

server_cache_disk::restore_result server_cache_disk::restore(const pick & pk, server_prompt & prompt, const server_tokens & tokens_new,
                                                             llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot,
                                                             const std::string & conv_id) {
    const int64_t t0 = ggml_time_us();
    entry & e = index[pk.i];
    const std::string path = (std::filesystem::path(dir) / e.name).string();

    // read and check the whole entry before touching the slot, so a bad file is refused without harm; every length is
    // bounded by the bytes left in the file and by --cache-disk-gb, and an allocation failure is a refusal, never a crash.
    // The main state (the largest piece next to the checkpoints) is read into memory that is not zero-filled first; the
    // checkpoints are std::vector<uint8_t> (the server's own type), which a resize zero-fills
    server_prompt                  p_new;
    std::unique_ptr<uint8_t[]>     main_buf;
    uint64_t                       main_len = 0;
    std::vector<uint8_t>           drft;
    llama_tokens                   toks;
    bool ok        = false;
    bool transient = false;   // cannot open / replaced between the size and the open: tried again at the next lookup
    std::string why;
    int64_t t_hash_us = 0;
    uint32_t n_disk_listed = 0;
    FILE * f = nullptr;
    try {
        std::error_code ec;
        const uint64_t fsize = std::filesystem::file_size(path, ec);
        f = ec ? nullptr : fopen(path.c_str(), "rb");
        if (!f) {
            why       = "cannot open";
            transient = true;
        } else if (fsize > limit_bytes) {
            why = string_format("%.1f MB, larger than the %.3f GB limit", fsize/1e6, limit_bytes/1e9);
        } else {
            cd_reader r { f, fsize };
            cd_header h;
            const bool hdr = cd_get_header(r, h);
            if (hdr && cd_version_ok(h.version) && h.total != fsize) {
                why       = "its size changed since it was indexed (replaced or being written)";
                transient = true;
            } else if (!hdr || !cd_version_ok(h.version) || !(h.cfg == cfg) || h.kind != (e.is_chat ? 1u : 0u) ||
                       h.lora_hash != e.lora_hash ||
                       (e.is_chat && (h.chat_hash != fnv1a(conv_id.data(), conv_id.size()) || h.chat_check != chat_check_of(conv_id)))) {
                why = "the header no longer matches (changed on disk, another chat or other settings)";
            } else if (h.n_tokens > r.left() / sizeof(llama_token) || (sizeof(size_t) < 8 && h.n_tokens > SIZE_MAX / sizeof(llama_token))) {
                why = "damaged token count";
            } else {
                // The header passed - the slot's own snapshots go now, before the file's buffers are
                // allocated (so they and the file's snapshots / main state are never in RAM together, the engine's plan counts
                // max(snapshots, restore buffer)); a header refusal above keeps them
                if (!prompt.checkpoints.empty()) {
                    size_t n_b = 0;
                    for (const auto & c : prompt.checkpoints) {
                        n_b += c.size();
                    }
                    SRV_INF("cache-disk: slot %d: freed the slot's %zu snapshot(s) (%.1f MiB) before reading %s\n",
                            id_slot, prompt.checkpoints.size(), n_b/1024.0/1024.0, e.name.c_str());
                    prompt.checkpoints.clear();
                }
                r.hashing = true;
                toks.resize((size_t) h.n_tokens);
                r.bytes(toks.data(), h.n_tokens*sizeof(llama_token));
                uint32_t n_ckpt = 0;
                r.get(n_ckpt);
                if (r.ok && n_ckpt > r.left() / CD_CKPT_MIN_BYTES) {
                    r.ok = false;
                }
                for (uint32_t i = 0; r.ok && i < n_ckpt; i++) {
                    common_prompt_checkpoint c;
                    int32_t id_task = 0, pos_min = 0, pos_max = 0;
                    r.get(c.n_tokens); r.get(id_task); r.get(pos_min); r.get(pos_max);
                    c.id_task = -1;   // the task that made it is gone (as a file restore: no task owns it)
                    c.pos_min = pos_min; c.pos_max = pos_max;
                    r.blob(c.data_tgt); r.blob(c.data_dft); r.blob(c.data_spec);
                    p_new.checkpoints.push_back(std::move(c));
                }
                // The chat's snapshot positions kept in their own files (the index has them; listed here)
                uint32_t n_disk = 0;
                r.get(n_disk);
                if (r.ok && n_disk > r.left() / 8) {
                    r.ok = false;
                }
                for (uint32_t i = 0; r.ok && i < n_disk; i++) {
                    int64_t pos = 0;
                    r.get(pos);
                    n_disk_listed++;
                }
                r.get(main_len);
                if (r.ok && (main_len > r.left() || (sizeof(size_t) < 8 && main_len > SIZE_MAX))) {
                    r.ok = false;
                }
                if (r.ok) {
                    main_buf.reset(new uint8_t[(size_t) main_len]);   // default-initialised: no zero fill
                    r.bytes(main_buf.get(), main_len);
                }
                r.blob(drft);
                r.hashing = false;
                t_hash_us = r.t_hash_us;
                uint64_t sum = 0;
                r.get(sum);
                char end[8];
                r.bytes(end, sizeof(end));
                if (!r.ok || r.pos != fsize || memcmp(end, CD_END, sizeof(end)) != 0) {
                    why = "truncated or damaged contents";
                } else if (sum != r.hash) {
                    why = "checksum mismatch (the file was damaged or changed)";
                } else if (main_len == 0) {
                    why = "an empty main state";   // set_data(.., 0) would "succeed" over the slot's old KV
                } else if (!drft.empty() && !ctx_dft) {
                    why = "it holds a draft state but this server has no draft context";
                } else if (drft.empty() && ctx_dft) {
                    why = "this server has a draft context but the file holds no draft state";
                } else {
                    ok = true;
                }
            }
        }
    } catch (const std::exception & ex) {
        ok  = false;
        why = std::string("cannot be read: ") + ex.what();
    }
    if (f) {
        fclose(f);
    }
    if (!ok) {
        if (!transient) {
            e.usable = false;
            e.why    = why;
        }
        SRV_WRN("cache-disk: refused %s: %s (left in place; %sthe request re-reads)\n", e.name.c_str(), why.c_str(),
                transient ? "its index entry is checked again; " : "");
        if (transient) {
            // a file that is gone leaves the index; one that changed (e.g. replaced) is read again from its header, so a
            // stale entry cannot keep winning the lookup
            std::error_code ec2;
            const std::string ename = e.name;
            const size_t ei = pk.i;
            if (!std::filesystem::exists(path, ec2) && !ec2) {
                SRV_INF("cache-disk: %s is gone; dropped from the index\n", ename.c_str());
                index.erase(index.begin() + ei);
            } else {
                entry fresh;
                fresh.name      = ename;
                fresh.is_chat   = e.is_chat;
                fresh.file_size = std::filesystem::file_size(path, ec2);
                if (ec2) {
                    fresh.file_size = 0;
                }
                fresh.t_write   = e.t_write;
                fresh.usable    = !ec2 && read_header(path, fresh, true);
                if (!fresh.usable) {
                    SRV_WRN("cache-disk: %s: %s (left in place; it leaves by the size limit)\n", ename.c_str(), fresh.why.c_str());
                }
                index[ei] = std::move(fresh);
            }
        }
        return RESTORE_REFUSED;
    }

    // the RAM tier's restore (server_prompt_cache::load): the state into the slot's sequence, then the prompt
    try {
        if (llama_state_seq_set_data_ext(ctx_tgt, main_buf.get(), (size_t) main_len, id_slot, 0) != main_len) {
            SRV_ERR("cache-disk: failed to restore %s (%llu B) into the slot; the slot is cleared, the request re-reads\n", e.name.c_str(), (unsigned long long) main_len);
            return RESTORE_FAILED;
        }
        main_buf.reset();
        if (!drft.empty() &&
            llama_state_seq_set_data_ext(ctx_dft, drft.data(), drft.size(), id_slot, 0) != drft.size()) {
            SRV_ERR("cache-disk: failed to restore the draft state of %s; the slot is cleared, the request re-reads\n", e.name.c_str());
            return RESTORE_FAILED;
        }
        const size_t n_tok = toks.size();
        p_new.tokens = server_tokens(toks, tokens_new.has_mtmd);   // keeps the multimodal flag, as the RAM tier's clone() does
        prompt = std::move(p_new);

        SRV_INF("cache-disk: restored %s by %s: %zu tokens (%zu shared with the request of %zu), %zu checkpoints (+%u in snapshot files), %.1f MB in %.1f ms (checksum %.1f ms)\n",
                e.name.c_str(), pk.how, n_tok, pk.lcp, tokens_new.size(), prompt.checkpoints.size(), n_disk_listed, e.file_size/1e6,
                (ggml_time_us() - t0)/1000.0, t_hash_us/1000.0);
    } catch (const std::exception & ex) {
        SRV_ERR("cache-disk: failed to restore %s (%s); the slot is cleared, the request re-reads\n", e.name.c_str(), ex.what());
        return RESTORE_FAILED;
    }
    return RESTORE_DONE;
}

int64_t server_cache_disk::erase_chat(const std::string & conv_id) {
    namespace fs = std::filesystem;
    const std::string cname = chat_name(conv_id);
    // The chat's snapshot files first (counted in the bytes deleted)
    uint64_t snap_bytes = 0;
    remove_snaps_of(fnv1a(conv_id.data(), conv_id.size()), -1, "erased by POST /cache-disk/erase", &snap_bytes);
    for (size_t i = 0; i < index.size(); i++) {
        if (index[i].name == cname) {
            const int64_t size = (int64_t) index[i].file_size;
            return remove_entry(i, "erased by POST /cache-disk/erase") ? size + (int64_t) snap_bytes : -1;
        }
    }
    if (snap_bytes > 0) {
        return (int64_t) snap_bytes;   // the chat file itself was not there
    }
    // not in the index (e.g. written by another server sharing the directory): deleted when it carries this tier's magic
    const std::string path = (fs::path(dir) / cname).string();
    std::error_code ec;
    const cd_magic_answer mg = fs::is_regular_file(path, ec) ? cd_magic(path) : CD_MAGIC_NO;
    if (mg == CD_MAGIC_UNREADABLE) {
        SRV_WRN("cache-disk: could not delete %s: it cannot be read (%s)\n", cname.c_str(), strerror(errno));
        return -1;
    }
    if (mg == CD_MAGIC_YES) {
        uint64_t size = fs::file_size(path, ec);
        if (ec) {
            size = 0;   // unknown; the delete below decides the answer
        }
        fs::remove(path, ec);
        if (ec) {
            SRV_WRN("cache-disk: could not delete %s: %s\n", cname.c_str(), ec.message().c_str());
            return -1;
        }
        SRV_INF("cache-disk: deleted %s (%.1f MB, erased by POST /cache-disk/erase; not in this server's index)\n", cname.c_str(), size/1e6);
        return (int64_t) size;
    }
    return 0;
}

std::pair<size_t, uint64_t> server_cache_disk::clear() {
    namespace fs = std::filesystem;
    size_t   n_files = 0;
    uint64_t n_bytes = 0;
    std::error_code ec;
    fs::directory_iterator it(dir, ec);
    for (; !ec && it != fs::directory_iterator(); it.increment(ec)) {
        std::error_code ec2;
        if (!it->is_regular_file(ec2)) {
            continue;
        }
        const std::string name = it->path().filename().string();
        if (!cd_is_entry_name(name)) {
            continue;   // never another file
        }
        const cd_magic_answer mg = cd_magic(it->path().string());
        if (mg == CD_MAGIC_UNREADABLE) {
            SRV_WRN("cache-disk: clear: could not delete %s: it cannot be read (%s)\n", name.c_str(), strerror(errno));
            continue;
        }
        if (mg == CD_MAGIC_NO) {
            continue;   // never another file
        }
        uint64_t size = it->file_size(ec2);
        if (ec2) {
            size = 0;   // unknown: not counted
        }
        fs::remove(it->path(), ec2);
        if (ec2) {
            SRV_WRN("cache-disk: clear: could not delete %s: %s\n", name.c_str(), ec2.message().c_str());
            continue;
        }
        n_files++;
        n_bytes += size;
        index.erase(std::remove_if(index.begin(), index.end(), [&](const entry & e) { return e.name == name; }), index.end());
    }
    if (ec) {
        SRV_WRN("cache-disk: clear: listing '%s' stopped early: %s\n", dir.c_str(), ec.message().c_str());
    }
    // entries whose files are gone (deleted by someone else) leave the index too
    index.erase(std::remove_if(index.begin(), index.end(), [&](const entry & e) {
        std::error_code ec3;
        return !fs::exists(fs::path(dir) / e.name, ec3);
    }), index.end());
    SRV_INF("cache-disk: cleared: %zu files, %.1f MB deleted; %zu entries left in the index\n", n_files, n_bytes/1e6, index.size());
    return { n_files, n_bytes };
}

//
// A chat's "start of message" snapshots in their own files
//

size_t server_cache_disk::remove_snaps_of(uint64_t chat_hash, int64_t newer_than, const char * reason, uint64_t * bytes) {
    namespace fs = std::filesystem;
    size_t n = 0;
    for (size_t i = 0; i < index.size();) {
        const auto & e = index[i];
        if (e.is_snap && e.chat_hash == chat_hash && e.snap_n_tokens > newer_than) {
            const uint64_t sz = e.file_size;
            if (remove_entry(i, reason)) {
                n++;
                if (bytes) {
                    *bytes += sz;
                }
                continue;
            }
        }
        i++;
    }
    // files of that chat another server wrote (not in this index): the name and the magic decide
    if (newer_than < 0) {
        const std::string pfx = "snap-" + cd_hex(chat_hash) + "-";
        std::error_code ec;
        fs::directory_iterator it(dir, ec);
        for (; !ec && it != fs::directory_iterator(); it.increment(ec)) {
            const std::string fname = it->path().filename().string();
            std::error_code ec2;
            if (fname.rfind(pfx, 0) == 0 && cd_is_entry_name(fname) && it->is_regular_file(ec2) && cd_magic(it->path().string()) == CD_MAGIC_YES) {
                const uint64_t sz = it->file_size(ec2);
                fs::remove(it->path(), ec2);
                if (!ec2) {
                    n++;
                    if (bytes) {
                        *bytes += sz;
                    }
                    SRV_INF("cache-disk: deleted %s (%s; not in this server's index)\n", fname.c_str(), reason);
                }
            }
        }
    }
    return n;
}

std::vector<int64_t> server_cache_disk::snapshot_positions(const std::string & conv_id, const server_tokens * tokens) const {
    std::vector<int64_t> pos;
    const uint64_t h = fnv1a(conv_id.data(), conv_id.size());
    for (const auto & e : index) {
        if (e.is_snap && e.usable && e.chat_hash == h &&
            (!tokens || (e.snap_n_tokens <= (int64_t) tokens->size() && prefix_hash(*tokens, e.snap_n_tokens) == e.snap_prefix))) {
            pos.push_back(e.snap_n_tokens);
        }
    }
    std::sort(pos.begin(), pos.end());
    return pos;
}

bool server_cache_disk::write_snapshot(const common_prompt_checkpoint & c, const server_tokens & tokens, const std::string & conv_id,
                                       uint64_t lora_hash, bool * fresh) {
    namespace fs = std::filesystem;
    if (fresh) {
        *fresh = false;
    }
    if (max_snaps <= 0 || conv_id.empty()) {
        return false;
    }
    for (int64_t i = 0; i < c.n_tokens && i < (int64_t) tokens.size(); i++) {
        if (tokens[i] == LLAMA_TOKEN_NULL) {
            return false;   // media before it: only text states are kept on disk
        }
    }
    const int64_t t0 = ggml_time_us();
    const std::string name   = snap_name(conv_id, c.n_tokens);
    const uint64_t    prefix = prefix_hash(tokens, c.n_tokens);
    for (auto & e : index) {
        if (e.name == name && e.usable && e.snap_prefix == prefix && e.lora_hash == lora_hash) {
            // already on the disk (e.g. the same message answered again); another history's file is replaced.
            // used again, so its age for the size limit starts again (the index, and the file's time for the next start-up)
            e.t_write = cd_now_us();
            std::error_code ec;
            fs::last_write_time(fs::path(dir) / name, fs::file_time_type::clock::now(), ec);
            return true;
        }
    }

    cd_header h;
    h.version    = CD_VERSION;
    h.kind       = 2;
    h.cfg        = cfg;
    h.chat_hash  = fnv1a(conv_id.data(), conv_id.size());
    h.chat_check = chat_check_of(conv_id);
    h.lora_hash  = lora_hash;
    h.n_tokens   = (uint64_t) c.n_tokens;
    uint64_t n_header = 0;
    {
        cd_writer cw;
        cd_put_header(cw, h);
        n_header = cw.n;
    }
    const uint64_t total  = n_header + 8 + 4 + 4 + 8 + 8 + c.data_tgt.size() + 8 + c.data_dft.size() + 8 + c.data_spec.size() + 8 + sizeof(CD_END);
    h.total = total;
    if (total > limit_bytes) {
        SRV_WRN("cache-disk: snapshot not written: %.1f MB, over the %.3f GB limit\n", total/1e6, limit_bytes/1e9);
        return false;
    }

    const std::string path = (fs::path(dir) / name).string();
    const std::string tmp  = path + cd_tmp_suffix();
    h.t_write = cd_now_us();
    FILE * f = cd_create(tmp);
    const int err_open = errno;
    cd_writer w { f, f != nullptr };
    int err_write = 0;
    if (f) {
        cd_put_header(w, h);
        w.hashing = true;
        w.put<int64_t>(c.n_tokens);
        w.put<int32_t>(c.pos_min);
        w.put<int32_t>(c.pos_max);
        w.put<uint64_t>(prefix);
        w.blob(c.data_tgt);
        w.blob(c.data_dft);
        w.blob(c.data_spec);
        w.hashing = false;
        w.put<uint64_t>(w.hash);
        w.bytes(CD_END, sizeof(CD_END));
        if (!w.ok && err_write == 0) {
            err_write = errno;
        }
        w.ok = w.ok && w.n == total;
        if (w.ok && fflush(f) != 0) {
            err_write = errno;
            w.ok = false;
        }
#ifndef _WIN32
        if (w.ok && fsync(fileno(f)) != 0) {
            err_write = errno;
            w.ok = false;
        }
#endif
        if (fclose(f) != 0 && w.ok) {
            err_write = errno;
            w.ok = false;
        }
    }
    std::error_code ec;
    if (!w.ok) {
        if (f) {
            fs::remove(tmp, ec);
        }
        SRV_ERR("cache-disk: the snapshot %s was not written (%s)\n", name.c_str(), strerror(!f ? err_open : err_write));
        return false;
    }
    // room under the limit, as a state file: the entries written longest ago (not this chat's file, not this name)
    const std::string cname = chat_name(conv_id);
    std::vector<std::string> undeletable;
    while (total_bytes(name) + total > limit_bytes) {
        size_t oldest = SIZE_MAX;
        for (size_t i = 0; i < index.size(); i++) {
            const auto & e = index[i];
            if (e.name == name || e.name == cname || std::find(undeletable.begin(), undeletable.end(), e.name) != undeletable.end()) {
                continue;
            }
            if (oldest == SIZE_MAX || e.t_write < index[oldest].t_write) {
                oldest = i;
            }
        }
        if (oldest == SIZE_MAX) {
            break;
        }
        const std::string oname = index[oldest].name;
        const bool     ochat = index[oldest].is_chat;
        const uint64_t ohash = index[oldest].chat_hash;
        if (!remove_entry(oldest, "size limit, written longest ago")) {
            undeletable.push_back(oname);
        } else if (ochat) {
            remove_snaps_of(ohash, -1, "its chat left by the size limit");
        }
    }
    if (total_bytes(name) + total > limit_bytes) {
        fs::remove(tmp, ec);
        SRV_WRN("cache-disk: snapshot not written: %.1f MB do not fit under the %.3f GB limit\n", total/1e6, limit_bytes/1e9);
        return false;
    }
    fs::rename(tmp, path, ec);
    if (ec) {
        const std::string msg = ec.message();
        fs::remove(tmp, ec);
        SRV_ERR("cache-disk: the rename of the snapshot %s failed (%s)\n", name.c_str(), msg.c_str());
        return false;
    }
    cd_fsync_dir(dir);
    // the rename replaced a file of the same name (another history of this chat at the same position)
    index.erase(std::remove_if(index.begin(), index.end(), [&](const entry & x) { return x.name == name; }), index.end());

    entry e;
    e.name          = name;
    e.is_snap       = true;
    e.chat_hash     = h.chat_hash;
    e.chat_check    = h.chat_check;
    e.lora_hash     = lora_hash;
    e.file_size     = total;
    e.t_write       = h.t_write;
    e.usable        = true;
    e.snap_n_tokens = c.n_tokens;
    e.snap_pos_min  = c.pos_min;
    e.snap_pos_max  = c.pos_max;
    e.snap_prefix   = prefix;
    index.push_back(std::move(e));

    // the per-chat cap: the chat's oldest snapshots (the smallest positions) beyond max_snaps go
    size_t n_dropped = 0;
    for (;;) {
        size_t n = 0, oldest = SIZE_MAX;
        for (size_t i = 0; i < index.size(); i++) {
            const auto & x = index[i];
            if (x.is_snap && x.chat_hash == h.chat_hash) {
                n++;
                if (oldest == SIZE_MAX || x.snap_n_tokens < index[oldest].snap_n_tokens) {
                    oldest = i;
                }
            }
        }
        if ((int32_t) n <= max_snaps || oldest == SIZE_MAX || !remove_entry(oldest, "the chat's oldest snapshot beyond --cache-disk-checkpoints")) {
            break;
        }
        n_dropped++;
    }
    if (fresh) {
        *fresh = true;
    }
    SRV_INF("cache-disk: snapshot written %s: the start of the message at %lld tokens (pos %d..%d), %.1f MB in %.1f ms; the chat has %zu on the disk%s\n",
            name.c_str(), (long long) c.n_tokens, c.pos_min, c.pos_max, total/1e6, (ggml_time_us() - t0)/1000.0,
            snapshot_positions(conv_id).size(), n_dropped > 0 ? " (the oldest dropped by the cap)" : "");
    return true;
}

size_t server_cache_disk::find_snapshot(const std::string & conv_id, uint64_t lora_hash, int64_t newer_than, int32_t pos_next,
                                        int32_t pos_min_thold, const server_tokens & tokens) const {
    const uint64_t h  = fnv1a(conv_id.data(), conv_id.size());
    const uint64_t hc = chat_check_of(conv_id);
    size_t best = SIZE_MAX;
    for (size_t i = 0; i < index.size(); i++) {
        const auto & e = index[i];
        if (!e.is_snap || !e.usable || e.chat_hash != h || e.chat_check != hc || e.lora_hash != lora_hash) {
            continue;
        }
        if (e.snap_n_tokens <= newer_than || e.snap_pos_max > pos_next || !(e.snap_pos_min < pos_min_thold || e.snap_pos_min == 0)) {
            continue;   // the server's own rule for a checkpoint it can go back to (server-context.cpp)
        }
        if (e.snap_n_tokens > (int64_t) tokens.size() || prefix_hash(tokens, e.snap_n_tokens) != e.snap_prefix) {
            continue;   // made for another history of this chat
        }
        if (best == SIZE_MAX || e.snap_n_tokens > index[best].snap_n_tokens) {
            best = i;
        }
    }
    return best;
}

bool server_cache_disk::snapshot_where(size_t i, int64_t & n_tokens, int32_t & pos_min, int32_t & pos_max) const {
    if (i >= index.size() || !index[i].is_snap) {
        return false;
    }
    n_tokens = index[i].snap_n_tokens;
    pos_min  = index[i].snap_pos_min;
    pos_max  = index[i].snap_pos_max;
    return true;
}

bool server_cache_disk::read_snapshot(size_t i, const std::string & conv_id, common_prompt_checkpoint & out) {
    const int64_t t0 = ggml_time_us();
    entry & e = index[i];
    const std::string path = (std::filesystem::path(dir) / e.name).string();
    bool ok = false;
    std::string why;
    FILE * f = nullptr;
    int64_t t_hash_us = 0;
    try {
        std::error_code ec;
        const uint64_t fsize = std::filesystem::file_size(path, ec);
        f = ec ? nullptr : fopen(path.c_str(), "rb");
        if (!f) {
            why = "cannot open";
        } else if (fsize > limit_bytes) {
            why = "larger than the size limit";
        } else {
            cd_reader r { f, fsize };
            cd_header h;
            const bool hdr = cd_get_header(r, h);
            if (!hdr || !cd_version_ok(h.version) || !(h.cfg == cfg) || h.kind != 2u || h.total != fsize || h.lora_hash != e.lora_hash ||
                h.chat_hash != fnv1a(conv_id.data(), conv_id.size()) || h.chat_check != chat_check_of(conv_id)) {
                why = "the header no longer matches (changed on disk, truncated, another chat or other settings)";
            } else {
                r.hashing = true;
                common_prompt_checkpoint c;
                int32_t pmin = 0, pmax = 0;
                uint64_t ph = 0;
                r.get(c.n_tokens); r.get(pmin); r.get(pmax); r.get(ph);
                c.id_task = -1;
                c.pos_min = pmin; c.pos_max = pmax;
                r.blob(c.data_tgt); r.blob(c.data_dft); r.blob(c.data_spec);
                r.hashing = false;
                t_hash_us = r.t_hash_us;
                uint64_t sum = 0;
                r.get(sum);
                char end[8];
                r.bytes(end, sizeof(end));
                if (!r.ok || r.pos != fsize || memcmp(end, CD_END, sizeof(end)) != 0) {
                    why = "truncated or damaged contents";
                } else if (sum != r.hash) {
                    why = "checksum mismatch (the file was damaged or changed)";
                } else if (c.n_tokens != e.snap_n_tokens || ph != e.snap_prefix || c.data_tgt.empty()) {
                    why = "its contents do not match its index entry";
                } else {
                    out = std::move(c);
                    ok  = true;
                }
            }
        }
    } catch (const std::exception & ex) {
        ok  = false;
        why = std::string("cannot be read: ") + ex.what();
    }
    if (f) {
        fclose(f);
    }
    if (!ok) {
        e.usable = false;
        e.why    = why;
        SRV_WRN("cache-disk: refused snapshot %s: %s (left in place; the request re-reads)\n", e.name.c_str(), why.c_str());
        return false;
    }
    SRV_INF("cache-disk: snapshot read %s: %lld tokens (pos %d..%d), %.1f MB in %.1f ms (checksum %.1f ms)\n", e.name.c_str(),
            (long long) out.n_tokens, out.pos_min, out.pos_max, e.file_size/1e6, (ggml_time_us() - t0)/1000.0, t_hash_us/1000.0);
    return true;
}

size_t server_cache_disk::drop_snapshots_after(const std::string & conv_id, int64_t n_tokens, const server_tokens & tokens) {
    // the chat's snapshots whose tokens before them are no longer the request's: another history, never usable again.
    // At any position, also at or before n_tokens (a snapshot there with another prefix is another history too)
    GGML_UNUSED(n_tokens);
    const uint64_t h = fnv1a(conv_id.data(), conv_id.size());
    size_t n = 0;
    for (size_t i = 0; i < index.size();) {
        const auto & e = index[i];
        if (e.is_snap && e.chat_hash == h &&
            (e.snap_n_tokens > (int64_t) tokens.size() || prefix_hash(tokens, e.snap_n_tokens) != e.snap_prefix) &&
            remove_entry(i, "made for a history the chat no longer has")) {
            n++;
            continue;
        }
        i++;
    }
    return n;
}

// A snapshot this reply wrote, when the chat file's save that should list it did not happen (no orphan)
bool server_cache_disk::remove_snapshot(const std::string & conv_id, int64_t n_tokens, const char * reason) {
    const std::string name = snap_name(conv_id, n_tokens);
    for (size_t i = 0; i < index.size(); i++) {
        if (index[i].name == name) {
            return remove_entry(i, reason);
        }
    }
    return false;
}

// Whether the chat's snapshot at that position, made for this history, is on the disk
bool server_cache_disk::has_snapshot(const std::string & conv_id, int64_t n_tokens, const server_tokens & tokens) const {
    const auto pos = snapshot_positions(conv_id, &tokens);
    return std::find(pos.begin(), pos.end(), n_tokens) != pos.end();
}

// A snapshot whose state llama.cpp did not accept (a size mismatch): never offered again (left in place)
void server_cache_disk::refuse_snapshot(size_t i, const char * why) {
    if (i >= index.size()) {
        return;
    }
    index[i].usable = false;
    index[i].why    = why;
    SRV_WRN("cache-disk: refused snapshot %s: %s (left in place; the request re-reads)\n", index[i].name.c_str(), why);
}
