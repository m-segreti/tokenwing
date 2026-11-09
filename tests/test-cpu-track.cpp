// The CPU track: the CPU-only changes give exactly the bits of the code they replace.
//   1. ggml_cpu_gated_delta_net_inplace (--moe-stream-cpu-state-inplace) against GGML_OP_GATED_DELTA_NET (K = 1) on the
//      GET_ROWS copy of the state + the copy of the new state into the cache: the attention output and the cache, bit for bit
//      (KDA and scalar gates, 1 and 3 tokens, q / k with fewer heads, the read row = / != the write row, 1 / 3 / 8 threads)
//   2. GGML_CPU_TRACK_PICK: MOE_GROUP_TOPK on / off, random scores and scores with equal values at and around the top k
//   3. GGML_CPU_TRACK_CAT:  CONCAT on / off (the conv state's shape: a view + a transposed column)
//   4. GGML_CPU_TRACK_ROWS: GET_ROWS on / off (one long row, two rows, a short row)
// Every case also runs with the switch on against itself twice (no state left behind). Exit code 0 = all equal.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

static int g_fail = 0;

static void fill(ggml_tensor * t, std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    float * p = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); i++) {
        p[i] = d(rng);
    }
}

static bool same(const void * a, const void * b, size_t n, const char * what) {
    if (memcmp(a, b, n) != 0) {
        fprintf(stderr, "  MISMATCH: %s\n", what);
        g_fail++;
        return false;
    }
    return true;
}

static void gdn_cb(ggml_tensor * dst, int ith, int nth, void * ud) {
    ggml_cpu_gated_delta_net_inplace(dst, ith, nth, (int) (intptr_t) ud);
}

// 1. the in-place delta step
static void test_gdn(int S_v, int H, int Hqk, int n_tokens, bool kda, int read_row, int write_row, int n_threads, int seed) {
    const int n_rows = 4;
    const int64_t D = (int64_t) S_v*S_v*H;
    ggml_init_params ip = { (size_t) 64*1024*1024 + (size_t) D*n_rows*4*4, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    std::mt19937 rng(seed);

    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, Hqk, n_tokens, 1);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, Hqk, n_tokens, 1);
    ggml_tensor * v    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, 1);
    ggml_tensor * g    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kda ? S_v : 1, H, n_tokens, 1);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H, n_tokens, 1);
    ggml_tensor * cache_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_rows);   // the old path
    ggml_tensor * cache_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_rows);   // the in-place path
    ggml_tensor * ids  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    fill(q, rng, -1, 1); fill(k, rng, -0.2f, 0.2f); fill(v, rng, -1, 1); fill(g, rng, -0.5f, 0.0f); fill(beta, rng, 0, 1);
    fill(cache_a, rng, -1, 1);
    memcpy(cache_b->data, cache_a->data, ggml_nbytes(cache_a));
    ((int32_t *) ids->data)[0] = read_row;

    // old path: GET_ROWS copy -> reshape -> GATED_DELTA_NET (K = 1) -> CPY of the new state into the cache row
    ggml_tensor * s   = ggml_reshape_4d(ctx, ggml_get_rows(ctx, cache_a, ids), S_v, S_v, H, 1);
    ggml_tensor * out = ggml_gated_delta_net(ctx, q, k, v, g, beta, s, 1);
    ggml_tensor * att = ggml_view_4d(ctx, out, S_v, H, n_tokens, 1, S_v*4, S_v*H*4, S_v*H*n_tokens*4, 0);
    ggml_tensor * att_c = ggml_cont(ctx, att);
    ggml_tensor * ns  = ggml_view_2d(ctx, out, D, 1, D*4, (size_t) S_v*H*n_tokens*4);
    ggml_tensor * cp  = ggml_cpy(ctx, ns, ggml_view_2d(ctx, cache_a, D, 1, cache_a->nb[1], (size_t) write_row*cache_a->nb[1]));
    // the in-place op
    ggml_tensor * args[8] = { q, k, v, g, beta, cache_b, ids,
                              ggml_view_2d(ctx, cache_b, D, 1, cache_b->nb[1], (size_t) write_row*cache_b->nb[1]) };
    // MOE_STREAM_TEST_STATE_SKIP set: the kernel's failure test (the old state put back) - every case must then mismatch
    void * ud = (void *) (intptr_t) (getenv("MOE_STREAM_TEST_STATE_SKIP") ? 1 : 0);
    ggml_tensor * att_b = ggml_custom_4d(ctx, GGML_TYPE_F32, S_v, H, n_tokens, 1, args, 8, gdn_cb, GGML_N_TASKS_MAX, ud);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, att_c);
    ggml_build_forward_expand(gf, cp);
    ggml_build_forward_expand(gf, att_b);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    char what[160];
    snprintf(what, sizeof(what), "gdn S_v %d H %d Hqk %d tokens %d %s rows %d->%d threads %d: attention", S_v, H, Hqk, n_tokens,
             kda ? "kda" : "gdn", read_row, write_row, n_threads);
    same(att_c->data, att_b->data, ggml_nbytes(att_b), what);
    snprintf(what, sizeof(what), "gdn S_v %d H %d Hqk %d tokens %d %s rows %d->%d threads %d: cache", S_v, H, Hqk, n_tokens,
             kda ? "kda" : "gdn", read_row, write_row, n_threads);
    same(cache_a->data, cache_b->data, ggml_nbytes(cache_a), what);
    ggml_free(ctx);
}

// one op computed with the track bits off and on (and on twice)
template <typename F>
static void off_on(const char * what, int bits, int n_threads, F build) {
    std::vector<uint8_t> res[3];
    for (int run = 0; run < 3; run++) {
        ggml_cpu_set_track(run == 0 ? 0 : bits);
        ggml_init_params ip = { (size_t) 64*1024*1024, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * t = build(ctx);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, t);
        ggml_graph_compute_with_ctx(ctx, gf, n_threads);
        res[run].assign((uint8_t *) t->data, (uint8_t *) t->data + ggml_nbytes(t));
        ggml_free(ctx);
    }
    ggml_cpu_set_track(0);
    char w[200];
    snprintf(w, sizeof(w), "%s (threads %d): off vs on", what, n_threads);
    if (same(res[0].data(), res[1].data(), res[0].size(), w)) {
        snprintf(w, sizeof(w), "%s (threads %d): on vs on", what, n_threads);
        same(res[1].data(), res[2].data(), res[1].size(), w);
    }
}

int main() {
    ggml_cpu_init();
    int n_cases = 0;

    // 1.
    for (int nt : { 1, 3, 8 }) {
        for (int kda = 0; kda < 2; kda++) {
            for (int tok : { 1, 3 }) {
                test_gdn(128, 4, 4, tok, kda, 2, 2, nt, 100 + nt*10 + kda*3 + tok); n_cases++;
                test_gdn(128, 4, 2, tok, kda, 3, 1, nt, 200 + nt*10 + kda*3 + tok); n_cases++;
                test_gdn(64, 6, 6, tok, kda, 0, 0, nt, 300 + nt*10 + kda*3 + tok);  n_cases++;
            }
        }
    }

    // 2. the grouped pick: 512 experts, 8 groups, 4 used, k_group 2, k 8 (Ling's routing), and 128 / 8 / 4 (tiny)
    for (int nt : { 1, 8 }) {
        for (int c = 0; c < 8; c++) {
            for (int ne : { 512, 128 }) {
                off_on(c < 2 ? "pick random" : c < 4 ? "pick ties at the top" : c < 6 ? "pick ties at k" : "pick NaN / inf / -0",
                       GGML_CPU_TRACK_PICK, nt,
                       [&](ggml_context * ctx) {
                    std::mt19937 rng(1000 + c*7 + ne);
                    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne, 3);
                    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne);
                    fill(a, rng, 0, 1); fill(b, rng, -0.01f, 0.01f);
                    float * pa = (float *) a->data;
                    float * pb = (float *) b->data;
                    if (c >= 6) {   // c 6: a NaN and a +inf among the scores; c 7: -inf on one expert, -0 and +0 on others
                        for (int r = 0; r < 3; r++) {
                            float * row = pa + r*ne;
                            if (c == 6) { row[5 + r] = NAN; row[ne/2 + r] = INFINITY; }
                            else        { row[7 + r] = -INFINITY; row[11 + r] = -0.0f; row[13 + r] = 0.0f; pb[11 + r] = 0.0f; pb[13 + r] = 0.0f; }
                        }
                    } else if (c >= 2) {   // equal scores: c 2-3 the best score on 3 more experts, c 4-5 the 8th best on 1 more
                        for (int r = 0; r < 3; r++) {
                            float * row = pa + r*ne;
                            std::vector<int> order(ne);
                            for (int e = 0; e < ne; e++) { order[e] = e; }
                            std::sort(order.begin(), order.end(), [&](int x, int y) { return row[x] + pb[x] > row[y] + pb[y]; });
                            const int src   = c < 4 ? order[0] : order[7];
                            const int n_dup = c < 4 ? 3 : 1;
                            for (int d = 1; d <= n_dup; d++) {
                                const int e = order[(c < 4 ? 0 : 7) + d*9 % (ne - 8) + 1];
                                row[e] = row[src];   // the same a and b: the same s = a + b
                                pb[e]  = pb[src];
                            }
                        }
                    }
                    return ggml_moe_group_topk(ctx, a, b, 8, 4, 2, 8);
                });
                n_cases++;
            }
        }
    }

    // 3. CONCAT of the conv state's shape: a [3, 4096] view + a transposed [1, 4096] column, and a 3-D case
    for (int nt : { 1, 3, 8 }) {
        off_on("concat conv state", GGML_CPU_TRACK_CAT, nt, [&](ggml_context * ctx) {
            std::mt19937 rng(7);
            ggml_tensor * st = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3*4096 + 5, 1);
            ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 1);
            fill(st, rng, -1, 1); fill(x, rng, -1, 1);
            ggml_tensor * sv = ggml_view_2d(ctx, st, 3, 4096, 3*4, 5*4);
            return ggml_concat(ctx, sv, ggml_transpose(ctx, x), 0);
        });
        off_on("concat 3-D", GGML_CPU_TRACK_CAT, nt, [&](ggml_context * ctx) {
            std::mt19937 rng(8);
            ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 33, 2);
            ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 5, 2);
            fill(a, rng, -1, 1); fill(b, rng, -1, 1);
            return ggml_concat(ctx, a, b, 1);
        });
        n_cases += 2;
    }

    // 4. GET_ROWS: one 2 MB row (the recurrent state), two rows, and a row too short to split
    for (int nt : { 1, 3, 8 }) {
        for (int shape = 0; shape < 3; shape++) {
            off_on(shape == 0 ? "get_rows 1 long row" : shape == 1 ? "get_rows 2 rows" : "get_rows short row",
                   GGML_CPU_TRACK_ROWS, nt, [&](ggml_context * ctx) {
                std::mt19937 rng(9 + shape);
                const int nc = shape == 2 ? 40 : 524288 + 17;
                ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, 4);
                ggml_tensor * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, shape == 1 ? 2 : 1);
                fill(src, rng, -1, 1);
                ((int32_t *) ids->data)[0] = 2;
                if (shape == 1) { ((int32_t *) ids->data)[1] = 0; }
                return ggml_get_rows(ctx, src, ids);
            });
            n_cases++;
        }
    }

    printf("test-cpu-track: %d cases, %d mismatches\n", n_cases, g_fail);
    return g_fail == 0 ? 0 : 1;
}
