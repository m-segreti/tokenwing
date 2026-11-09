#include "llama-moe-corr-file.h"

#include "hash/hash.h"

extern "C" {
#include "hash/sha256/sha256.h"
}

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>

using json = nlohmann::json;

#ifdef _WIN32
#define CORR_FSEEK(f, o, w) _fseeki64(f, (long long) (o), w)
#define CORR_FTELL(f)       ((uint64_t) _ftelli64(f))
#else
#define CORR_FSEEK(f, o, w) fseeko(f, (off_t) (o), w)
#define CORR_FTELL(f)       ((uint64_t) ftello(f))
#endif

static const char     MAGIC[8] = { 'T', 'W', 'C', 'O', 'R', 'R', '0', '1' };
static const int      FORMAT   = 1;
static const uint64_t ALIGN    = 4096;

static std::string hex_of(const unsigned char * d, size_t n) {
    static const char hx[] = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; i++) {
        s += hx[d[i] >> 4];
        s += hx[d[i] & 15];
    }
    return s;
}

static uint64_t le64(const unsigned char * p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) {
        v = (v << 8) | p[i];
    }
    return v;
}

#if defined(__GNUC__)
static std::string fmt(const char * f, ...) __attribute__((format(printf, 1, 2)));
#endif
static std::string fmt(const char * f, ...) {
    char b[512];
    va_list ap;
    va_start(ap, f);
    vsnprintf(b, sizeof(b), f, ap);
    va_end(ap);
    return b;
}

std::string llama_moe_corr_file_check(const std::string & path, llama_moe_corr_header & h, bool verify_payload) {
    FILE * f = llama_moe_os_fopen_utf8(path.c_str(), "rb");
    if (f == nullptr) {
        return fmt("cannot open %s", path.c_str());
    }
    struct closer { FILE * f; ~closer() { fclose(f); } } closer_{ f };
    if (CORR_FSEEK(f, 0, SEEK_END) != 0) {
        return "cannot seek in the file";
    }
    const uint64_t sz = CORR_FTELL(f);
    if (sz < 64) {
        return fmt("file is %" PRIu64 " bytes: shorter than the 64-byte preamble", sz);
    }
    unsigned char pre[64];
    CORR_FSEEK(f, 0, SEEK_SET);
    if (fread(pre, 1, 64, f) != 64) {
        return "short read of the preamble";
    }
    if (memcmp(pre, MAGIC, 8) != 0) {
        return "bad magic (not a TokenWing correction file, or another format version)";
    }
    const uint64_t jl = le64(pre + 8), ps = le64(pre + 16), fs = le64(pre + 24);
    if (fs != sz) {
        return fmt("file is %" PRIu64 " bytes but the header says %" PRIu64 " (truncated or extended)", sz, fs);
    }
    if (jl > ps || ps > sz || ps % ALIGN != 0 || 64 + jl > ps) {
        return fmt("header sizes inconsistent (json %" PRIu64 ", payload_start %" PRIu64 ", file %" PRIu64 ")", jl, ps, sz);
    }
    std::string js(jl, '\0');
    if (fread(&js[0], 1, jl, f) != jl) {
        return "short read of the header";
    }
    unsigned char dig[SHA256_DIGEST_SIZE];
    sha256_hash(dig, (const unsigned char *) js.data(), js.size());
    if (memcmp(dig, pre + 32, 32) != 0) {
        return "header hash mismatch (header bytes changed)";
    }
    json j;
    try {
        j = json::parse(js);
    } catch (const std::exception & e) {
        return fmt("header is not valid JSON: %s", e.what());
    }
    try {
        if (j.at("format").get<int>() != FORMAT) {
            return fmt("format %d, this loader reads %d", j.at("format").get<int>(), FORMAT);
        }
        const json & m = j.at("model");
        h = llama_moe_corr_header();
        h.arch           = m.at("arch").get<std::string>();
        h.name           = m.value("name", std::string());
        h.n_layer        = m.contains("n_layer") && m.at("n_layer").is_number() ? m.at("n_layer").get<int64_t>() : -1;
        h.n_expert       = m.at("n_expert").get<int64_t>();
        h.n_embd         = m.at("n_embd").get<int64_t>();
        h.layers         = m.at("layers").get<std::vector<int32_t>>();
        h.router_sha256  = m.at("router_sha256").get<std::string>();
        h.payload_start  = j.at("payload_start").get<uint64_t>();
        h.file_size      = j.at("file_size").get<uint64_t>();
        h.payload_sha256 = j.at("payload_sha256").get<std::string>();
        if (j.contains("gate") && j.at("gate").contains("rule")) {
            h.gate_rule = j.at("gate").at("rule").get<std::string>();
        }
        if (h.payload_start != ps || h.file_size != sz) {
            return "header JSON and preamble disagree on the payload start / file size";
        }
        const uint64_t nb = (uint64_t) h.n_expert * (uint64_t) h.n_embd * 4;
        std::vector<std::pair<uint64_t, uint64_t>> spans;
        for (const json & p : j.at("pairs")) {
            llama_moe_corr_pair q;
            const int64_t t64 = p.at("target").get<int64_t>(), l64 = p.at("lead").get<int64_t>(), s64 = p.at("source").get<int64_t>();
            const int64_t o64 = p.at("offset").get<int64_t>(), n64 = p.at("nbytes").get<int64_t>();   // signed: a negative value is refused below
            if (t64 < 0 || t64 > 1000000 || l64 < 1 || l64 > 64 || s64 < 0 || s64 > 1000000 || o64 < 0 || n64 < 0) {
                return fmt("pair (%" PRId64 ", %" PRId64 "): a field is out of range", t64, l64);
            }
            q.target = (int32_t) t64;
            q.lead   = (int32_t) l64;
            q.source = (int32_t) s64;
            q.offset = (uint64_t) o64;
            q.nbytes = (uint64_t) n64;
            for (const llama_moe_corr_pair & o : h.pairs) {
                if (o.target == q.target && o.lead == q.lead) {
                    return fmt("pair (%d, %d) is listed twice", q.target, q.lead);
                }
            }
            const std::vector<int64_t> shape = p.at("shape").get<std::vector<int64_t>>();
            if (shape.size() != 2 || shape[0] != h.n_expert || shape[1] != h.n_embd || q.nbytes != nb || p.at("dtype").get<std::string>() != "f32") {
                return fmt("pair (%d, %d): shape / size / type do not fit the model's sizes", q.target, q.lead);
            }
            if (q.offset < ps || q.offset % ALIGN != 0 || q.nbytes > sz || q.offset > sz - q.nbytes) {
                return fmt("pair (%d, %d): offset %" PRIu64 " outside the payload", q.target, q.lead, q.offset);
            }
            bool in_layers = false;
            for (int32_t l : h.layers) {
                in_layers = in_layers || l == q.target;
            }
            if (q.source != q.target - q.lead || !in_layers) {
                return fmt("pair (%d, %d): source layer / target layer not valid", q.target, q.lead);
            }
            spans.push_back({ q.offset, q.offset + q.nbytes });
            h.pairs.push_back(q);
        }
        std::sort(spans.begin(), spans.end());
        for (size_t i = 1; i < spans.size(); i++) {
            if (spans[i - 1].second > spans[i].first) {
                return "two pairs overlap in the file";
            }
        }
    } catch (const std::exception & e) {
        return fmt("header is missing a field or has a wrong one: %s", e.what());
    }
    if (verify_payload) {
        sha256_t st;
        sha256_init(&st);
        std::vector<unsigned char> buf(1 << 22);
        CORR_FSEEK(f, ps, SEEK_SET);
        uint64_t left = sz - ps;
        while (left > 0) {
            const size_t n = (size_t) std::min<uint64_t>(left, buf.size());
            if (fread(buf.data(), 1, n, f) != n) {
                return "short read of the payload";
            }
            sha256_update(&st, buf.data(), n);
            left -= n;
        }
        sha256_final(&st, dig);
        if (hex_of(dig, 32) != h.payload_sha256) {
            return "payload hash mismatch (matrix bytes changed)";
        }
    }
    return "";
}

std::string llama_moe_corr_file_stream(const std::string & path, const llama_moe_corr_header & h, const std::vector<size_t> & want,
                                       const std::function<void(size_t, const void *)> & on_pair) {
    FILE * f = llama_moe_os_fopen_utf8(path.c_str(), "rb");
    if (f == nullptr) {
        return fmt("cannot open %s", path.c_str());
    }
    struct closer { FILE * f; ~closer() { fclose(f); } } closer_{ f };
    std::vector<size_t> order = want;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return h.pairs[a].offset < h.pairs[b].offset; });
    sha256_t st;
    sha256_init(&st);
    std::vector<unsigned char> buf(1 << 22), pbuf;
    if (CORR_FSEEK(f, h.payload_start, SEEK_SET) != 0) {
        return "cannot seek to the payload";
    }
    uint64_t pos = h.payload_start;
    size_t   wi  = 0;
    bool     filling = false;
    while (pos < h.file_size) {
        const size_t n = (size_t) std::min<uint64_t>(h.file_size - pos, buf.size());
        if (fread(buf.data(), 1, n, f) != n) {
            return "short read of the payload";
        }
        sha256_update(&st, buf.data(), n);
        while (wi < order.size()) {
            const llama_moe_corr_pair & p = h.pairs[order[wi]];
            if (p.offset >= pos + n) {
                break;                                     // starts after this chunk
            }
            const uint64_t lo = std::max<uint64_t>(p.offset, pos), hi = std::min<uint64_t>(p.offset + p.nbytes, pos + n);
            if (lo < hi) {
                if (!filling) {
                    pbuf.resize(p.nbytes);
                    filling = true;
                }
                memcpy(pbuf.data() + (lo - p.offset), buf.data() + (lo - pos), (size_t) (hi - lo));
            }
            if (p.offset + p.nbytes <= pos + n) {
                on_pair(order[wi], pbuf.data());           // complete
                wi++;
                filling = false;
            } else {
                break;                                     // continues in the next chunk
            }
        }
        pos += n;
    }
    unsigned char dig[SHA256_DIGEST_SIZE];
    sha256_final(&st, dig);
    if (hex_of(dig, 32) != h.payload_sha256) {
        return "payload hash mismatch (matrix bytes changed)";
    }
    return "";
}
