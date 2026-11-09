// The CUDA backend's shared q8_1 copy of matvec inputs (ggml_backend_cuda_set_share_q8_1) must be dropped
// when a node writes into the shared input's bytes. Graph: y1 = A*x, ys = B*copy(x) (the value before the write), a write
// into x (in place / into an overlapping view / none), y2 = B*x (reads x after the write, through the same tensor). The
// same graph runs on two CUDA backends, one with the share off and one with it on (separate backends: a captured CUDA
// graph keeps the setting it was captured with), three input sets each (the later ones replay a captured graph). The
// results must be bit-equal; for a write case y2 must also differ from ys, otherwise the case could not catch anything.
// With GGML_CUDA_SHARE_Q8_1_TEST_NO_DROP=1 (test only) the share is wrong on purpose and this test must FAIL.
// Not part of ctest (it needs a CUDA device); run: build/bin/test-q8-1-share
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static std::vector<float> rnd(size_t n, std::mt19937 & g) {
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (float & x : v) {
        x = d(g);
    }
    return v;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("CUDA");
    if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) {
        printf("SKIP: no CUDA backend\n");
        return 0;
    }
    auto * set_share = (void (*)(bool)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_set_share_q8_1");
    if (set_share == nullptr) {
        printf("FAIL: the CUDA backend has no ggml_backend_cuda_set_share_q8_1\n");
        return 1;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_t be_off = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t be_on  = ggml_backend_dev_init(dev, nullptr);

    const int64_t K = 512, M = 256;
    const char * cases[] = { "none", "inplace", "view" };
    int n_fail = 0, n_check = 0;

    for (const char * cs : cases) {
        for (int64_t N : { 1, 4 }) {
            ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * A  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, M);
            ggml_tensor * B  = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, M);
            ggml_tensor * x0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
            ggml_tensor * z  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K/2, N);
            ggml_tensor * x  = ggml_scale(ctx, x0, 1.0f);         // the shared input: a node of the graph
            ggml_tensor * y1 = ggml_mul_mat(ctx, A, x);            // quantizes x
            ggml_tensor * xc = ggml_scale(ctx, x, 1.0f);           // x before the write
            ggml_tensor * ys = ggml_mul_mat(ctx, B, xc);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, y1);
            ggml_build_forward_expand(gf, ys);
            const bool write = strcmp(cs, "none") != 0;
            if (strcmp(cs, "inplace") == 0) {
                ggml_build_forward_expand(gf, ggml_scale_inplace(ctx, x, 3.0f));       // writes all of x
            } else if (strcmp(cs, "view") == 0) {
                ggml_tensor * half = ggml_view_2d(ctx, x, K/2, N, x->nb[1], (K/2)*ggml_element_size(x));
                ggml_build_forward_expand(gf, ggml_cpy(ctx, z, half));                  // writes the second half of each row
            }
            ggml_tensor * y2 = ggml_mul_mat(ctx, B, x);            // same tensor x: shares y1's copy unless it was dropped
            ggml_build_forward_expand(gf, y2);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be_off);

            std::mt19937 g(1234);
            std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q8_0, K) * M);
            for (ggml_tensor * w : { A, B }) {
                std::vector<float> f = rnd(K*M, g);
                ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), q.data(), 0, M, K, nullptr);
                ggml_backend_tensor_set(w, q.data(), 0, q.size());
            }
            for (int it = 0; it < 3; it++) {
                std::vector<float> xi = rnd(K*N, g), zi = rnd(K/2*N, g);
                std::vector<float> r[2][3];
                for (int on = 0; on < 2; on++) {
                    ggml_backend_tensor_set(x0, xi.data(), 0, ggml_nbytes(x0));
                    ggml_backend_tensor_set(z,  zi.data(), 0, ggml_nbytes(z));
                    set_share(on == 1);
                    ggml_backend_t be = on ? be_on : be_off;
                    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
                        printf("FAIL: graph compute\n");
                        return 1;
                    }
                    ggml_backend_synchronize(be);
                    int k = 0;
                    for (ggml_tensor * t : { y1, ys, y2 }) {
                        r[on][k].resize(ggml_nelements(t));
                        ggml_backend_tensor_get(t, r[on][k].data(), 0, ggml_nbytes(t));
                        k++;
                    }
                }
                const bool eq = r[0][0] == r[1][0] && r[0][1] == r[1][1] && r[0][2] == r[1][2];
                const bool live = !write || r[0][2] != r[0][1];   // the write really changes y2
                const bool ok = eq && live;
                n_check++;
                n_fail += !ok;
                double maxd = 0;
                for (size_t i = 0; i < r[0][2].size(); i++) {
                    maxd = std::max(maxd, (double) std::fabs(r[0][2][i] - r[1][2][i]));
                }
                printf("%-8s N=%lld input %d: share off vs on %s (y2 max |diff| %.4g)%s -> %s\n", cs, (long long) N, it,
                       eq ? "bit-equal" : "DIFFER", maxd, write ? (live ? ", the write changes y2" : ", the write does NOT change y2") : "",
                       ok ? "ok" : "FAIL");
            }
            set_share(false);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
    }
    ggml_backend_free(be_off);
    ggml_backend_free(be_on);
    printf("%s: %d of %d checks failed\n", n_fail ? "FAIL" : "PASS", n_fail, n_check);
    return n_fail ? 1 : 0;
}
