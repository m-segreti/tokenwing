// T97 chunk 1: ONE combined MXFP4 decode-kernel benchmark on real Ling-3.0-flash experts (the joint spec of
// docs/study-ik_llama.cpp.md 14.1 and docs/study-ktransformers.md 14 M2). Extends tools/repack_bench.cpp (T56).
//
// One "word" = 37 streamed MoE layers, each: gate and up (8 experts x [768 rows x 2560]), SwiGLU, down (8 x [2560 x 768]),
// weighted sum over the 8 - the CPU expert part of one decode token. The experts come from a pool of real expert slices
// (default 3 layers x 128 experts = 1.2 GB per layout, far over the 24 MB L3), 8 random ones per layer, the same plan for
// every arm. Two levels:
//   level 1 = the inner kernel under ONE shared driver (ggml's decode chunking: gate/up rows split evenly over the
//             threads, down in 64-row chunks with stealing, a barrier after each op), our own spinning pool;
//   level 2 = each project's own op dispatch: a ggml graph per layer (mul_mat_id x3 + swiglu + weighting + sum) on a
//             persistent ggml threadpool, as the engine calls it (common attaches one; ggml_backend_graph_compute per
//             CPU split). Ours: A (vec_dot), F (the 4-row kernel through mul_mat_id), C (CPU_REPACK forward_mul_mat_id).
//             ik build: B and D through ik's fused moe_up_gate + mul_mat_id (ik's default -fmoe).
// Arms:
//   A  ours today: ggml_vec_dot_mxfp4_q8_0 (AVX2, 1 row per call), Q8_0 activations
//   B  ik plain MXFP4 (iqk_mul_mat, Q8_2_X4 activations)                        [ik build]
//   C  mainline mxfp4_8x8 (ggml_gemv_mxfp4_8x8_q8_0 / CPU_REPACK), Q8_0
//   D  ik MXFP4_R8 (8 rows interleaved, iqk), Q8_2_X4                          [ik build]
//   E  ktransformers' natural 4-row AVX-512 BF16 GEMV (values + separate E8M0 scales, BF16 activations, prefetch)
//   F0 ours, 4 rows per call with AVX-512 VNNI (ggml_vec_dot_mxfp4_q8_0_x4, t97 branch), Q8_0, no prefetch
//   F1 the same with a software prefetch 38 blocks (~646 B) ahead (crossing into the next rows; rows are back to back)
//   F2 the same with a prefetch of the same block of the next 4-row group
//   Fy0, Fy1, Fy2  the same 4-row kernel with 256-bit registers (same bits), prefetch modes as F0-F2
//   G  (level 2 only) today's kernel + the fused up+gate step (ggml_cpu_set_moe_fuse, T97 chunk 2)
//   H  (level 2 only) the Fy2 kernel (4-row ymm, next-group prefetch) + the fused up+gate step
//   R  reference for the rounding check only (not timed): weights dequantized to fp32, fp32 activations, double sums
// T100 arms (llama.cpp-t100: ik_llama.cpp's MXFP4_R8 layout + one-token kernel ported into our ggml-cpu; the pools are
// converted to R8 with ggml_mxfp4_to_r8, and at level 2 registered with ggml_cpu_mxfp4_r8_register):
//   Q  the R8 kernel with our Q8_0 activations (ggml_gemv_mxfp4_r8_q8_0)
//   K  the R8 kernel with ik's Q8_2_X4 activations (ggml_gemv_mxfp4_r8_q8_2x4 = ik's mul_mat_mxfp4_r8_q8_2_avx2<1>)
//   QG, KG (level 2 only) the same + the fused up+gate step (ggml_cpu_set_moe_fuse, as G / H)
//   HD, QGD, KGD (level 2 only, T100 chunk 1b) H / QG / KG + ik's expert split for the one-token mul_mat_id
//            (ggml_cpu_set_moe_dispatch: one chunk queue over all experts, nr0/nth rows per chunk)
// --profile ARMS: per arm, the time of each stage's prefix graph (up/gate, down, weighting, sum) -> each stage's cost
//
// build for T100: the same with llama.cpp-t100 in place of llama.cpp-t97 (traces/t100/kernel_bench); for llama.cpp-t100d
// (the expert split only, no R8) add -DKB_NO_R8
// build, ours (from the MoE_Lab root):
//   g++ -O3 -march=native -std=c++17 -I llama.cpp-t97/ggml/include tools/kernel_bench.cpp -o traces/p4/t97/kernel_bench \
//       -L llama.cpp-t97/build/bin -lggml-base -lggml-cpu -Wl,-rpath,$PWD/llama.cpp-t97/build/bin -lpthread
// build for C18 (Flint, --kq / --check-kq), from the MoE_Lab root:
//   g++ -O3 -march=native -std=c++17 -DKB_NO_R8 -I llama.cpp-c18/ggml/include tools/kernel_bench.cpp -o traces/c18/kernel_bench \
//       -L llama.cpp-c18/build/bin -lggml-base -lggml-cpu -Wl,-rpath,$PWD/llama.cpp-c18/build/bin -lpthread
// build, ik (arms B, D):
//   IK=/path/to/ik_llama.cpp
//   g++ -O3 -march=native -std=c++17 -DKB_IK -I $IK/ggml/include -I $IK/ggml/src -I $IK/ggml/src/iqk tools/kernel_bench.cpp \
//       -o traces/p4/t97/kernel_bench_ik -L traces/p4/t97/ik-build/ggml/src -lggml -Wl,-rpath,$PWD/traces/p4/t97/ik-build/ggml/src -lpthread
//
// Part of arm E is adapted from ktransformers (kt-kernel/operators/amx/fp4-moe.hpp, GemmKernel224MXFP4SmallKGroup:
// mxfp4_to_bf16_32_natural, expand_e8_scales, fp4_mat_vec_kgroup_natural_impl), Copyright (c) 2024 by KVCache.AI,
// All Rights Reserved, Apache License 2.0 (https://github.com/kvcache-ai/ktransformers/blob/main/LICENSE).
// Changes: GGUF nibble order (lo = elements 0-15, hi = 16-31) is decoded directly, so the activation needs no permutation.
// Arms B and D call ik_llama.cpp's own code (MIT, Copyright (c) 2023-2024 The ggml authors, Copyright (c) 2024-2026
// Iwan Kawrakow) through its library; nothing of it is copied here.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifndef KB_IK
#include "ggml-cpu.h"
#else
#include "iqk_mul_mat.h"
extern "C" void iqk_repack_tensor(struct ggml_tensor * tensor);
#endif

#include <immintrin.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifndef KB_IK
extern "C" {
void quantize_row_q8_0(const float * x, void * y, int64_t k);
void ggml_vec_dot_mxfp4_q8_0(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_mxfp4_q8_0_x4(int n, float * s, const void * vx, size_t bx, const void * vy, int width, int pf);
void ggml_gemv_mxfp4_8x8_q8_0(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc);
#ifndef KB_NO_R8
// T100 (llama.cpp-t100 / branch t100-r8 ggml-cpu)
void ggml_mxfp4_to_r8(const void * src, void * dst, int64_t nrows, int64_t n_per_row);
size_t ggml_mxfp4_q8_2_x4_row_size(int64_t k);
void ggml_quantize_row_q8_2_x4(const float * x, void * vy, int64_t k);
void ggml_gemv_mxfp4_r8_q8_2x4(int n, float * s, const void * vx, const void * vy, int nr);
void ggml_gemv_mxfp4_r8_q8_0(int n, float * s, const void * vx, const void * vy, int nr);
#endif
}
#ifdef KB_NO_R8
// T100 chunk 2: a build without the R8 work (llama.cpp-t100d: only the expert split). The R8 arms and check 3 are
// unavailable; these stubs stop loudly if anything reaches them.
static void kb_no_r8() { fprintf(stderr, "this kernel_bench was built with -DKB_NO_R8: no R8 arms\n"); exit(2); }
static void ggml_mxfp4_to_r8(const void *, void *, int64_t, int64_t) { kb_no_r8(); }
static size_t ggml_mxfp4_q8_2_x4_row_size(int64_t k) { return (size_t) (k / 32) * 36; }   // only sizes the unused arm table
static void ggml_quantize_row_q8_2_x4(const float *, void *, int64_t) { kb_no_r8(); }
static void ggml_gemv_mxfp4_r8_q8_2x4(int, float *, const void *, const void *, int) { kb_no_r8(); }
static void ggml_gemv_mxfp4_r8_q8_0(int, float *, const void *, const void *, int) { kb_no_r8(); }
static int  ggml_cpu_mxfp4_r8_register(const void *, size_t) { kb_no_r8(); return 0; }
static void ggml_cpu_mxfp4_r8_unregister(const void *) { kb_no_r8(); }
static void ggml_cpu_set_mxfp4_r8_act(int act) { if (act) kb_no_r8(); }
#endif
static bool is_dispatch_arm(const std::string & name) { return name == "HD" || name == "QGD" || name == "KGD"; }
static bool is_r8_arm(const std::string & name) { return !name.empty() && (name[0] == 'Q' || name[0] == 'K'); }
#endif

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- shapes (Ling-3.0-flash) ----
static const int     N_EMBD = 2560, N_FF = 768, N_USED = 8, N_LAYERS = 37;
static const size_t  SLICE  = 1044480;            // one projection of one expert, MXFP4: 768 x 2560 / 32 x 17
static const size_t  BLOCK  = 3 * SLICE;          // gate|up|down = the sidecar's block (3,133,440 B, 4096-aligned)
static const size_t  WORD_BYTES = (size_t) N_LAYERS * N_USED * BLOCK;   // expert bytes one word reads

static void * amalloc(size_t n) {
    void * p = nullptr;
    if (posix_memalign(&p, 4096, (n + 4095) & ~(size_t) 4095) != 0) { fprintf(stderr, "alloc %zu failed\n", n); exit(1); }
    return p;
}

// ---- the expert pools, one per projection: n_exp slices back to back (the engine's per-projection pool shape) ----
struct Pools {
    int n_exp = 0;
    uint8_t * p[3] = { nullptr, nullptr, nullptr };    // gate, up, down; plain GGUF MXFP4 bytes
};

static const int  PROJ_K[3]    = { N_EMBD, N_EMBD, N_FF };    // inner dimension
static const int  PROJ_ROWS[3] = { N_FF,   N_FF,   N_EMBD };  // output rows

static Pools load_pools(const char * sidecar, const std::vector<size_t> & offs, int per_layer) {
    Pools P;
    P.n_exp = (int) offs.size() * per_layer;
    for (int j = 0; j < 3; j++) P.p[j] = (uint8_t *) amalloc((size_t) P.n_exp * SLICE);
    int fd = open(sidecar, O_RDONLY | O_DIRECT);
    if (fd < 0) { fprintf(stderr, "cannot open %s with O_DIRECT\n", sidecar); exit(1); }
    uint8_t * buf = (uint8_t *) amalloc(BLOCK);
    int e = 0;
    for (size_t l = 0; l < offs.size(); l++) {
        for (int i = 0; i < per_layer; i++, e++) {
            const size_t off = offs[l] + (size_t) i * BLOCK;
            if (pread(fd, buf, BLOCK, (off_t) off) != (ssize_t) BLOCK) { fprintf(stderr, "short read at %zu\n", off); exit(1); }
            for (int j = 0; j < 3; j++) memcpy(P.p[j] + (size_t) e * SLICE, buf + (size_t) j * SLICE, SLICE);
        }
    }
    free(buf);
    close(fd);
    return P;
}

// ---- the word plan: 8 distinct experts + weights per layer, seeded, the same for every arm and both builds ----
struct Plan {
    int words;
    std::vector<int32_t> ids;     // [words][37][8]
    std::vector<float>   w;       // [words][37][8]
    std::vector<float>   x;       // [2560], the layer input (fixed)
};

static Plan make_plan(int words, int n_exp, uint32_t seed) {
    Plan pl;
    pl.words = words;
    pl.ids.resize((size_t) words * N_LAYERS * N_USED);
    pl.w.resize(pl.ids.size());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uw(0.05f, 1.0f);
    for (size_t t = 0; t < (size_t) words * N_LAYERS; t++) {
        int32_t * id = &pl.ids[t * N_USED];
        float   * w  = &pl.w[t * N_USED];
        float sum = 0;
        for (int k = 0; k < N_USED; k++) {
            int c;
            do { c = (int) (rng() % (uint32_t) n_exp); } while (std::find(id, id + k, c) != id + k);
            id[k] = c;
            w[k]  = uw(rng);
            sum  += w[k];
        }
        for (int k = 0; k < N_USED; k++) w[k] /= sum;
    }
    pl.x.resize(N_EMBD);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & v : pl.x) v = nd(rng);
    return pl;
}

// =====================================================================================================================
// level 1: the shared driver
// =====================================================================================================================
struct SpinPool {
    int n;
    std::vector<std::thread> th;
    std::atomic<int> gen{0}, done{0}, quit{0};
    std::atomic<int> bar_cnt{0}, bar_gen{0};
    std::function<void(int)> job;

    explicit SpinPool(int n_) : n(n_) {
        for (int i = 1; i < n; i++) th.emplace_back([this, i] { worker(i); });
    }
    ~SpinPool() {
        quit.store(1);
        for (auto & t : th) t.join();
    }
    void worker(int i) {
        int seen = 0;
        while (true) {
            while (gen.load(std::memory_order_acquire) == seen) {
                if (quit.load(std::memory_order_relaxed)) return;
                _mm_pause();
            }
            seen++;
            job(i);
            done.fetch_add(1, std::memory_order_release);
        }
    }
    void run(const std::function<void(int)> & f) {
        job = f;
        done.store(0, std::memory_order_relaxed);
        gen.fetch_add(1, std::memory_order_release);
        job(0);
        while (done.load(std::memory_order_acquire) < n - 1) _mm_pause();
    }
    void barrier() {
        const int g = bar_gen.load(std::memory_order_acquire);
        if (bar_cnt.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
            bar_cnt.store(0, std::memory_order_relaxed);
            bar_gen.fetch_add(1, std::memory_order_release);
        } else {
            while (bar_gen.load(std::memory_order_acquire) == g) _mm_pause();
        }
    }
};

// one arm at level 1: how it stores the pools, quantizes an activation, and computes rows [r0, r1) of one slice
struct ArmL1 {
    std::string name;
    int    align = 1;                       // row ranges must be multiples of this
    size_t qbytes[3] = {0, 0, 0};           // bytes of a quantized activation for K = 2560 / 768
    std::function<void(const float * x, int K, void * q)> quant;
    std::function<void(int proj, int e, int r0, int r1, const void * q, float * out)> rows;
};

static int64_t align_up(int64_t v, int a) { return (v + a - 1) / a * a; }

// ggml's decode chunking (ggml_compute_forward_mul_mat_id, nr1 = 1): chunk 64 rows, and if fewer than 4 chunks per
// thread, one chunk per thread; chunk boundaries aligned to the arm's row multiple
struct Chunks { int64_t nchunk, dr; };
static Chunks ggml_chunks(int64_t nr0, int nth, int align) {
    int64_t nchunk = (nr0 + 63) / 64;
    if (nchunk < nth * 4) nchunk = nth;
    int64_t dr = align_up((nr0 + nchunk - 1) / nchunk, align);
    nchunk = (nr0 + dr - 1) / dr;
    return { nchunk, dr };
}

struct L1Run {
    const ArmL1 & arm;
    SpinPool & pool;
    // scratch
    std::vector<uint8_t> qx, qact;          // quantized input, quantized down inputs (8)
    std::vector<float>   g, u, a, d, out;   // gate/up/act [8][768], down [8][2560], out [2560]
    std::atomic<int64_t> ctr[N_USED];

    // quantized activations get generous room and a guard pattern behind each: ik's Q8_2_X4 quantizer writes more than
    // ggml_row_size() reports for one row (a heap overflow in the first run), so the slot size is checked, not trusted
    static const size_t QPAD = 8192;
    size_t qstride[3];
    L1Run(const ArmL1 & a_, SpinPool & p_) : arm(a_), pool(p_) {
        for (int j = 0; j < 3; j++) qstride[j] = (arm.qbytes[j] + QPAD + 63) & ~(size_t) 63;
        qx.assign(qstride[0], 0xA5);
        qact.assign((size_t) N_USED * qstride[2], 0xA5);
        g.resize(N_USED * N_FF); u.resize(N_USED * N_FF); a.resize(N_USED * N_FF);
        d.resize((size_t) N_USED * N_EMBD); out.resize(N_EMBD);
    }

    void layer(const float * x, const int32_t * ids, const float * w) {
        const int nth = pool.n;
        pool.run([&](int ith) {
            if (ith == 0) arm.quant(x, N_EMBD, qx.data());
            pool.barrier();
            // gate, then up (two ops, a barrier after each, as in the graph)
            for (int proj = 0; proj < 2; proj++) {
                float * o = proj == 0 ? g.data() : u.data();
                const Chunks c = ggml_chunks(N_FF, nth, arm.align);
                for (int k = 0; k < N_USED; k++) {
                    for (int64_t ch = ith; ch < c.nchunk; ch += nth) {   // nchunk == nth here: static split
                        const int64_t r0 = ch * c.dr, r1 = std::min<int64_t>(r0 + c.dr, N_FF);
                        if (r0 < r1) arm.rows(proj, ids[k], (int) r0, (int) r1, qx.data(), o + (size_t) k * N_FF);
                    }
                }
                pool.barrier();
            }
            if (ith == 0) {
                for (int i = 0; i < N_USED * N_FF; i++) a[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
                for (int k = 0; k < N_USED; k++) {
                    arm.quant(a.data() + (size_t) k * N_FF, N_FF, qact.data() + (size_t) k * qstride[2]);
                    ctr[k].store(nth, std::memory_order_relaxed);
                }
            }
            pool.barrier();
            // down: per expert, 64-row chunks, the first nth by thread index, the rest stolen
            {
                const Chunks c = ggml_chunks(N_EMBD, nth, arm.align);
                for (int k = 0; k < N_USED; k++) {
                    int64_t ch = ith;
                    while (ch < c.nchunk) {
                        const int64_t r0 = ch * c.dr, r1 = std::min<int64_t>(r0 + c.dr, N_EMBD);
                        if (r0 < r1) arm.rows(2, ids[k], (int) r0, (int) r1, qact.data() + (size_t) k * qstride[2],
                                              d.data() + (size_t) k * N_EMBD);
                        if (nth >= c.nchunk) break;
                        ch = ctr[k].fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
            pool.barrier();
            if (ith == 0) {
                // the guard: the last 4096 bytes of every slot must still hold the pattern
                auto guard_ok = [](const uint8_t * p, size_t n) { for (size_t i = n - 4096; i < n; i++) if (p[i] != 0xA5) return false; return true; };
                bool ok = guard_ok(qx.data(), qstride[0]);
                for (int k = 0; k < N_USED; k++) ok = ok && guard_ok(qact.data() + (size_t) k * qstride[2], qstride[2]);
                if (!ok) { fprintf(stderr, "arm %s: the activation quantizer overran its %zu+%zu bytes\n", arm.name.c_str(), arm.qbytes[0], QPAD); exit(3); }
                for (int i = 0; i < N_EMBD; i++) {
                    float s = 0;
                    for (int k = 0; k < N_USED; k++) s += w[k] * d[(size_t) k * N_EMBD + i];
                    out[i] = s;
                }
            }
        });
    }
};

#ifndef KB_IK
// ---------------------------------------------------------------------------------------------------------------------
// arm E: ktransformers' natural-order BF16 GEMV (see the licence note at the top)
// ---------------------------------------------------------------------------------------------------------------------
namespace kt {
alignas(32) static const uint16_t fp4_bf16[16] = { 0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0,
                                                   0x8000, 0xBF00, 0xBF80, 0xBFC0, 0xC000, 0xC040, 0xC080, 0xC0C0 };

// 16 packed bytes (one GGUF block's qs) -> 32 BF16 values in element order (lo nibbles = 0..15, hi = 16..31)
static inline __m512i decode32(__m128i packed) {
    const __m128i lo_mask = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(packed, lo_mask);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask);
    const __m512i idx = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_cvtepu8_epi16(lo)), _mm256_cvtepu8_epi16(hi), 1);
    const __m512i lut = _mm512_castsi256_si512(_mm256_load_si256((const __m256i *) fp4_bf16));
    return _mm512_permutexvar_epi16(idx, lut);
}

static inline void expand_e8(const uint8_t * src, float * dst, int count) {
    int g = 0;
    for (; g + 16 <= count; g += 16) {
        const __m512i codes = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (src + g)));
        __m512i bits = _mm512_slli_epi32(codes, 23);
        const __mmask16 z = _mm512_cmpeq_epi32_mask(codes, _mm512_setzero_si512());
        bits = _mm512_mask_mov_epi32(bits, z, _mm512_set1_epi32(0x00400000));
        _mm512_store_ps(dst + g, _mm512_castsi512_ps(bits));
    }
    for (; g < count; g++) {
        const uint32_t b = src[g] == 0 ? 0x00400000u : (uint32_t) src[g] << 23;
        memcpy(dst + g, &b, 4);
    }
}

// a slice in kt layout: values [rows][nb][16] then scales [rows][nb]
static void gemv(int K, int r0, int r1, const uint8_t * slice, int rows_total, const uint16_t * act, float * out) {
    const int nb = K / 32;
    const uint8_t * vals = slice;
    const uint8_t * scl  = slice + (size_t) rows_total * nb * 16;
    auto a = [act](int gi) { return (__m512bh) _mm512_loadu_si512((const void *) (act + (size_t) gi * 32)); };
    alignas(64) float sc[4][N_EMBD / 32];
    const int PF = 40;
    int r = r0;
    for (; r + 4 <= r1; r += 4) {
        const __m128i * w0 = (const __m128i *) (vals + (size_t) (r + 0) * nb * 16);
        const __m128i * w1 = (const __m128i *) (vals + (size_t) (r + 1) * nb * 16);
        const __m128i * w2 = (const __m128i *) (vals + (size_t) (r + 2) * nb * 16);
        const __m128i * w3 = (const __m128i *) (vals + (size_t) (r + 3) * nb * 16);
        for (int j = 0; j < 4; j++) expand_e8(scl + (size_t) (r + j) * nb, sc[j], nb);
        __m512 c0 = _mm512_setzero_ps(), c1 = _mm512_setzero_ps(), c2 = _mm512_setzero_ps(), c3 = _mm512_setzero_ps();
        for (int gi = 0; gi < nb; gi++) {
            if (gi % 4 == 0 && gi + PF < nb) {
                _mm_prefetch((const char *) (w0 + gi + PF), _MM_HINT_T0);
                _mm_prefetch((const char *) (w1 + gi + PF), _MM_HINT_T0);
                _mm_prefetch((const char *) (w2 + gi + PF), _MM_HINT_T0);
                _mm_prefetch((const char *) (w3 + gi + PF), _MM_HINT_T0);
            }
            const __m512bh av = a(gi);
            c0 = _mm512_fmadd_ps(_mm512_set1_ps(sc[0][gi]), _mm512_dpbf16_ps(_mm512_setzero_ps(), av, (__m512bh) decode32(_mm_loadu_si128(w0 + gi))), c0);
            c1 = _mm512_fmadd_ps(_mm512_set1_ps(sc[1][gi]), _mm512_dpbf16_ps(_mm512_setzero_ps(), av, (__m512bh) decode32(_mm_loadu_si128(w1 + gi))), c1);
            c2 = _mm512_fmadd_ps(_mm512_set1_ps(sc[2][gi]), _mm512_dpbf16_ps(_mm512_setzero_ps(), av, (__m512bh) decode32(_mm_loadu_si128(w2 + gi))), c2);
            c3 = _mm512_fmadd_ps(_mm512_set1_ps(sc[3][gi]), _mm512_dpbf16_ps(_mm512_setzero_ps(), av, (__m512bh) decode32(_mm_loadu_si128(w3 + gi))), c3);
        }
        out[r + 0] = _mm512_reduce_add_ps(c0);
        out[r + 1] = _mm512_reduce_add_ps(c1);
        out[r + 2] = _mm512_reduce_add_ps(c2);
        out[r + 3] = _mm512_reduce_add_ps(c3);
    }
    for (; r < r1; r++) {
        const __m128i * w0 = (const __m128i *) (vals + (size_t) r * nb * 16);
        expand_e8(scl + (size_t) r * nb, sc[0], nb);
        __m512 c0 = _mm512_setzero_ps();
        for (int gi = 0; gi < nb; gi++) {
            c0 = _mm512_fmadd_ps(_mm512_set1_ps(sc[0][gi]), _mm512_dpbf16_ps(_mm512_setzero_ps(), a(gi), (__m512bh) decode32(_mm_loadu_si128(w0 + gi))), c0);
        }
        out[r] = _mm512_reduce_add_ps(c0);
    }
}

static void quant_bf16(const float * x, int K, void * q) {
    uint16_t * o = (uint16_t *) q;
    for (int i = 0; i < K; i += 32) {
        const __m512bh b = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + i + 16), _mm512_loadu_ps(x + i));
        _mm512_storeu_si512((void *) (o + i), (__m512i) b);
    }
}

// plain GGUF block_mxfp4 {e, qs[16]} rows -> values + scales
static void to_kt(const uint8_t * plain, uint8_t * dst, int rows, int K) {
    const int nb = K / 32;
    uint8_t * vals = dst;
    uint8_t * scl  = dst + (size_t) rows * nb * 16;
    for (int r = 0; r < rows; r++) {
        for (int b = 0; b < nb; b++) {
            const uint8_t * blk = plain + ((size_t) r * nb + b) * 17;
            scl[(size_t) r * nb + b] = blk[0];
            memcpy(vals + ((size_t) r * nb + b) * 16, blk + 1, 16);
        }
    }
}
} // namespace kt
#endif

// =====================================================================================================================
// level 2: a ggml graph per layer, as the engine runs its CPU split
// =====================================================================================================================
struct L2Graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * gf  = nullptr;
    ggml_tensor  * x = nullptr, * ids = nullptr, * w = nullptr, * out = nullptr;
    ggml_gallocr_t galloc = nullptr;
    std::vector<ggml_backend_buffer_t> bufs;
    void * small = nullptr;
    std::vector<std::pair<std::string, ggml_tensor *>> stages;   // T100 --profile: the graph's op boundaries, in order

    // pools[j] = data of projection j (n_exp slices); repack = put the pools in CPU_REPACK buffers (arm C)
    L2Graph(ggml_backend_t backend, uint8_t * const pools[3], int n_exp, ggml_type type, bool repack) {
        ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead() + (1 << 16), nullptr, true };
        ctx = ggml_init(ip);
        ggml_tensor * P[3];
        for (int j = 0; j < 3; j++) {
            P[j] = ggml_new_tensor_3d(ctx, type, PROJ_K[j], PROJ_ROWS[j], n_exp);
            const size_t nbytes = (size_t) n_exp * SLICE;
#ifndef KB_IK
            ggml_backend_buffer_t b = repack ? ggml_backend_cpu_repack_buffer_from_ptr(pools[j], nbytes)
                                             : ggml_backend_cpu_buffer_from_ptr(pools[j], nbytes);
#else
            (void) repack;
            ggml_backend_buffer_t b = ggml_backend_cpu_buffer_from_ptr(pools[j], nbytes);
#endif
            P[j]->buffer = b;
            P[j]->data   = pools[j];
            ggml_backend_buffer_init_tensor(b, P[j]);
            bufs.push_back(b);
        }
        x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, 1, 1);
        ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, 1);
        w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, N_USED, 1);
        const size_t sb = 64 * 1024;
        small = amalloc(sb);
        ggml_backend_buffer_t sbuf = ggml_backend_cpu_buffer_from_ptr(small, sb);
        bufs.push_back(sbuf);
        uint8_t * sp = (uint8_t *) small;
        for (ggml_tensor * t : { x, ids, w }) {
            t->buffer = sbuf;
            t->data   = sp;
            sp += (ggml_nbytes(t) + 63) & ~(size_t) 63;
        }
#ifndef KB_IK
        ggml_tensor * gate = ggml_mul_mat_id(ctx, P[0], x, ids);         // [768, 8, 1]
        ggml_tensor * up   = ggml_mul_mat_id(ctx, P[1], x, ids);
        ggml_tensor * act  = ggml_swiglu_split(ctx, gate, up);
        stages.push_back({ "gate", gate });
        stages.push_back({ "up", up });
        stages.push_back({ "swiglu", act });
#else
        ggml_tensor * act  = ggml_moe_up_gate(ctx, P[1], P[0], x, ids, GGML_UNARY_OP_SILU);
        stages.push_back({ "up_gate", act });
#endif
        ggml_tensor * down = ggml_mul_mat_id(ctx, P[2], act, ids);        // [2560, 8, 1]
        ggml_tensor * wd   = ggml_mul(ctx, down, w);
        stages.push_back({ "down", down });
        stages.push_back({ "weight", wd });
        // the sum over the 8 experts as build_moe_ffn does it: views + adds
        ggml_tensor * acc = ggml_view_2d(ctx, wd, N_EMBD, 1, wd->nb[2], 0);
        for (int k = 1; k < N_USED; k++) {
            acc = ggml_add(ctx, acc, ggml_view_2d(ctx, wd, N_EMBD, 1, wd->nb[2], (size_t) k * wd->nb[1]));
        }
        out = acc;
        stages.push_back({ "sum(7 adds)", out });
        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) { fprintf(stderr, "gallocr failed\n"); exit(1); }
    }
    ~L2Graph() {
        ggml_gallocr_free(galloc);
        for (auto b : bufs) ggml_backend_buffer_free(b);
        free(small);
        ggml_free(ctx);
    }
    void layer(ggml_backend_t backend, const float * xin, const int32_t * id, const float * wt, float * res, ggml_cgraph * g = nullptr) {
        memcpy(x->data, xin, sizeof(float) * N_EMBD);
        memcpy(ids->data, id, sizeof(int32_t) * N_USED);
        memcpy(w->data, wt, sizeof(float) * N_USED);
        ggml_backend_graph_compute(backend, g ? g : gf);
        if (res) memcpy(res, out->data, sizeof(float) * N_EMBD);
    }
};

// =====================================================================================================================
// the run
// =====================================================================================================================
struct Opts {
    std::string sidecar, out_dir = ".";
    std::vector<size_t> offs;
    int per_layer = 128, words = 30, reps = 5, seed = 97;
    std::vector<int> threads = { 8, 12 };
    std::string arms;          // comma list, e.g. "A1,C1,E1,F01,F11,A2,C2,F02,F12" (arm + level)
    std::string compare;       // compare mode: a comma list of arm tags against the first
    std::string profile;       // T100: profile mode: level-2 arm tags, each timed stage by stage (graph prefixes)
    // C18: the Flint kernel mode (kq::bench)
    std::string kq, kq_type = "q5_K", kq_arms = "A,Fy0,Fy2,F0,F2";
    std::vector<int> kq_layers;
    int kq_word_layers = 47;
};

static std::vector<std::string> split(const std::string & s, char c) {
    std::vector<std::string> v;
    size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find(c, i);
        if (j == std::string::npos) j = s.size();
        if (j > i) v.push_back(s.substr(i, j - i));
        i = j + 1;
    }
    return v;
}

static void write_f32(const std::string & path, const std::vector<float> & v) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(1); }
    fwrite(v.data(), sizeof(float), v.size(), f);
    fclose(f);
}

static std::vector<float> read_f32(const std::string & path) {
    std::vector<float> v;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return v;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize((size_t) n / sizeof(float));
    if (fread(v.data(), sizeof(float), v.size(), f) != v.size()) v.clear();
    fclose(f);
    return v;
}

static int do_compare(const Opts & o) {
    const auto tags = split(o.compare, ',');
    const auto ref = read_f32(o.out_dir + "/out_" + tags[0] + ".f32");
    if (ref.empty()) { fprintf(stderr, "no reference output %s\n", tags[0].c_str()); return 1; }
    printf("rounding against %s (word 0, %d layers x %d outputs):\n", tags[0].c_str(), N_LAYERS, N_EMBD);
    printf("%-6s %10s %12s %14s %14s\n", "arm", "bit-equal", "max |diff|", "max row rel", "mean row rel");
    for (size_t t = 1; t < tags.size(); t++) {
        const auto v = read_f32(o.out_dir + "/out_" + tags[t] + ".f32");
        if (v.size() != ref.size()) { printf("%-6s (missing)\n", tags[t].c_str()); continue; }
        size_t eq = 0;
        double maxd = 0, maxrel = 0, sumrel = 0;
        for (int l = 0; l < N_LAYERS; l++) {
            double dn = 0, rn = 0;
            for (int i = 0; i < N_EMBD; i++) {
                const size_t k = (size_t) l * N_EMBD + i;
                if (memcmp(&v[k], &ref[k], 4) == 0) eq++;
                const double d = (double) v[k] - ref[k];
                maxd = std::max(maxd, std::fabs(d));
                dn += d * d;
                rn += (double) ref[k] * ref[k];
            }
            const double rel = std::sqrt(dn / std::max(rn, 1e-30));
            maxrel = std::max(maxrel, rel);
            sumrel += rel;
        }
        printf("%-6s %9.3f%% %12.4g %14.4g %14.4g\n", tags[t].c_str(), 100.0 * eq / ref.size(), maxd, maxrel, sumrel / N_LAYERS);
    }
    return 0;
}


#ifndef KB_IK
// =====================================================================================================================
// --check (T97 review): bit identity of the 4-row kernel and the fused step on synthetic MXFP4 experts, with odd and even
// block counts per row, every kernel variant and prefetch mode, both fused patterns, at -t 1..8. No sidecar needed.
// =====================================================================================================================
static int do_check() {
    int fails = 0;
    std::mt19937 rng(97);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // ---- 1. kernel level: 4 rows per call against 4 calls of ggml_vec_dot_mxfp4_q8_0, memcmp ----
    {
        const int Ks[] = { 32, 64, 96, 800, 768, 1312, 2560, 2592 };   // 1, 2, 3, 25, 24, 41, 80, 81 blocks
        int n_cmp = 0, n_bad = 0;
        for (int K : Ks) {
            const int nb = K / 32, rows = 64;
            std::vector<float> w((size_t) rows * K), x(K);
            for (auto & v : w) v = nd(rng);
            for (auto & v : x) v = nd(rng);
            std::vector<uint8_t> wq((size_t) rows * nb * 17), xq((size_t) nb * 34);
            ggml_quantize_chunk(GGML_TYPE_MXFP4, w.data(), wq.data(), 0, rows, K, nullptr);
            quantize_row_q8_0(x.data(), xq.data(), K);
            std::vector<float> ref(rows);
            for (int r = 0; r < rows; r++) ggml_vec_dot_mxfp4_q8_0(K, &ref[r], 0, wq.data() + (size_t) r * nb * 17, 0, xq.data(), 0, 1);
            for (int width : { 256, 512 }) for (int pf : { 0, 38, -1 }) {
                std::vector<float> got(rows);
                for (int r = 0; r + 3 < rows; r += 4) ggml_vec_dot_mxfp4_q8_0_x4(K, &got[r], wq.data() + (size_t) r * nb * 17, (size_t) nb * 17, xq.data(), width, pf);
                n_cmp++;
                if (memcmp(got.data(), ref.data(), sizeof(float) * rows) != 0) { n_bad++; printf("  KERNEL MISMATCH K=%d (%d blocks) width %d pf %d\n", K, nb, width, pf); }
            }
        }
        printf("check 1 (kernel, 4 rows vs 1 row, K of 1/2/3/25/24/41/80/81 blocks x width 256/512 x prefetch none/38/next): %d of %d bit-equal\n", n_cmp - n_bad, n_cmp);
        fails += n_bad;
    }

    // ---- 2. op level: a MoE layer through ggml, every switch at -t 1..8, against the reference (all off, -t 8) ----
    struct Shape { int K, F, E; const char * what; };
    const Shape shapes[] = { { 2592, 800, 16, "n_embd 2592 (81 blocks) x n_ff 800 (25 blocks)" },
                             { 64, 14336, 12, "n_ff 14336 (the fused step must decline at -t 1..3: chunk rows > 4096)" } };
    ggml_backend_t be = ggml_backend_cpu_init();
    for (const Shape & sh : shapes) {
        const int K = sh.K, F = sh.F, E = sh.E;
        const size_t sz_gu = (size_t) F * (K / 32) * 17, sz_d = (size_t) K * (F / 32) * 17;
        // aligned: ggml_backend_cpu_buffer_from_ptr wants aligned memory
        struct ABuf { uint8_t * p; explicit ABuf(size_t n) : p((uint8_t *) amalloc(n)) {} ~ABuf() { free(p); } uint8_t * data() { return p; } };
        ABuf pg((size_t) E * sz_gu), pu((size_t) E * sz_gu), pd((size_t) E * sz_d);
        {
            std::vector<float> w((size_t) F * K);
            for (int e = 0; e < E; e++) {
                for (auto & v : w) v = nd(rng) * 0.05f;
                ggml_quantize_chunk(GGML_TYPE_MXFP4, w.data(), pg.data() + e * sz_gu, 0, F, K, nullptr);
                for (auto & v : w) v = nd(rng) * 0.05f;
                ggml_quantize_chunk(GGML_TYPE_MXFP4, w.data(), pu.data() + e * sz_gu, 0, F, K, nullptr);
                for (auto & v : w) v = nd(rng) * 0.05f;
                ggml_quantize_chunk(GGML_TYPE_MXFP4, w.data(), pd.data() + e * sz_d, 0, K, F, nullptr);
            }
        }
        std::vector<float> x(K); for (auto & v : x) v = nd(rng);
        std::vector<int32_t> ids(8); { std::vector<int> p(E); for (int i = 0; i < E; i++) p[i] = i; std::shuffle(p.begin(), p.end(), rng); for (int k = 0; k < 8; k++) ids[k] = p[k]; }

        for (int clamped = 0; clamped < 2; clamped++) {
            auto run = [&](int kernel, int pf, int fuse, int nth, int64_t * fused, int disp = 0) {
                ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead() + (1 << 16), nullptr, true };
                ggml_context * ctx = ggml_init(ip);
                ggml_tensor * Pg = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, K, F, E);
                ggml_tensor * Pu = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, K, F, E);
                ggml_tensor * Pd = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, F, K, E);
                std::vector<ggml_backend_buffer_t> bufs;
                for (auto pr : { std::make_pair(Pg, (void *) pg.data()), std::make_pair(Pu, (void *) pu.data()), std::make_pair(Pd, (void *) pd.data()) }) {
                    ggml_backend_buffer_t b = ggml_backend_cpu_buffer_from_ptr(pr.second, ggml_nbytes(pr.first));
                    pr.first->buffer = b; pr.first->data = pr.second; ggml_backend_buffer_init_tensor(b, pr.first); bufs.push_back(b);
                }
                ggml_tensor * tx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, 1);
                ggml_tensor * ti = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, 1);
                ABuf small(65536);
                ggml_backend_buffer_t sb = ggml_backend_cpu_buffer_from_ptr(small.data(), 65536); bufs.push_back(sb);
                tx->buffer = sb; tx->data = small.data(); ti->buffer = sb; ti->data = small.data() + 32768;
                memcpy(tx->data, x.data(), sizeof(float) * K); memcpy(ti->data, ids.data(), sizeof(int32_t) * 8);
                ggml_tensor * act;
                if (!clamped) {
                    ggml_tensor * g = ggml_mul_mat_id(ctx, Pg, tx, ti);
                    ggml_tensor * u = ggml_mul_mat_id(ctx, Pu, tx, ti);
                    act = ggml_swiglu_split(ctx, g, u);
                } else {   // Ling's clamped layers: silu(gate) -> clamp(-inf, L), clamp(up, -L, L), mul
                    const float L = 0.02f;   // small, so the clamps really cut
                    ggml_tensor * g = ggml_mul_mat_id(ctx, Pg, tx, ti);
                    ggml_tensor * ga = ggml_clamp(ctx, ggml_silu(ctx, g), -INFINITY, L);
                    ggml_tensor * u = ggml_clamp(ctx, ggml_mul_mat_id(ctx, Pu, tx, ti), -L, L);
                    act = ggml_mul(ctx, ga, u);
                }
                ggml_tensor * out = ggml_mul_mat_id(ctx, Pd, act, ti);
                ggml_cgraph * gf = ggml_new_graph(ctx);
                ggml_build_forward_expand(gf, out);
                ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
                if (!ggml_gallocr_alloc_graph(ga, gf)) { fprintf(stderr, "gallocr failed\n"); exit(1); }
                ggml_cpu_set_mxfp4_kernel(kernel, pf);
                ggml_cpu_set_moe_fuse(fuse);
                ggml_cpu_set_moe_dispatch(disp);
                const int64_t d0 = ggml_cpu_moe_dispatch_count();
                ggml_threadpool_params tpp = ggml_threadpool_params_default(nth);
                ggml_threadpool_t tp = ggml_threadpool_new(&tpp);
                ggml_backend_cpu_set_threadpool(be, tp);
                ggml_backend_cpu_set_n_threads(be, nth);
                const int64_t f0 = ggml_cpu_moe_fuse_count();
                ggml_backend_graph_compute(be, gf);
                *fused = ggml_cpu_moe_fuse_count() - f0;
                if (disp && ggml_cpu_moe_dispatch_count() == d0) { printf("  DISPATCH did not run\n"); *fused = -1; }
                std::vector<float> res((size_t) ggml_nelements(out));
                memcpy(res.data(), out->data, sizeof(float) * res.size());
                ggml_backend_cpu_set_threadpool(be, nullptr);
                ggml_threadpool_free(tp);
                ggml_cpu_set_mxfp4_kernel(0, 0); ggml_cpu_set_moe_fuse(0); ggml_cpu_set_moe_dispatch(0);
                ggml_gallocr_free(ga);
                for (auto b : bufs) ggml_backend_buffer_free(b);
                ggml_free(ctx);
                return res;
            };
            int64_t fz = 0;
            const std::vector<float> ref = run(0, 0, 0, 8, &fz);
            int n_cmp = 0, n_bad = 0, n_fused_runs = 0, n_declined = 0;
            const int cfg[][3] = { { 0, 0, 0 }, { 1, -1, 0 }, { 2, -1, 0 }, { 1, 0, 0 }, { 2, 38, 0 },
                                   { 0, 0, 1 }, { 1, -1, 1 }, { 2, -1, 1 }, { 1, 38, 1 }, { 2, 0, 1 } };
            for (int nth = 1; nth <= 8; nth++) for (const auto & c : cfg) {
                int64_t fused = 0;
                const std::vector<float> got = run(c[0], c[1], c[2], nth, &fused);
                n_cmp++;
                const bool same = memcmp(got.data(), ref.data(), sizeof(float) * ref.size()) == 0;
                if (!same) { n_bad++; printf("  OP MISMATCH %s %s: kernel %d pf %d fuse %d -t %d\n", sh.what, clamped ? "clamped" : "swiglu", c[0], c[1], c[2], nth); }
                // the fused step must run exactly when asked and allowed: rows per chunk pad(F/nth, 16) <= 4096
                const bool allowed = c[2] && (((F + nth - 1) / nth + 15) / 16 * 16 <= 4096);
                if (allowed != (fused > 0)) { n_bad++; printf("  FUSE COUNT %s: fuse %d -t %d ran %lld fused steps\n", sh.what, c[2], nth, (long long) fused); }
                n_fused_runs += fused > 0;
                n_declined += c[2] && !allowed;
            }
            printf("check 2 (op, %s, %s): %d of %d runs bit-equal to the reference (all off, -t 8), fused in %d runs, declined (fallback) in %d\n",
                   sh.what, clamped ? "clamped silu/clamp/mul" : "swiglu", n_cmp - n_bad, n_cmp, n_fused_runs, n_declined);
            fails += n_bad;
            // T100 1b: ik's expert split (ggml_cpu_set_moe_dispatch) with every kernel / prefetch / fuse setting, -t 1..12
            {
                int nc = 0, nbad = 0;
                for (int nth = 1; nth <= 12; nth++) for (const auto & c : cfg) {
                    int64_t fused = 0;
                    const std::vector<float> got = run(c[0], c[1], c[2], nth, &fused, 1);
                    nc++;
                    if (fused < 0 || memcmp(got.data(), ref.data(), sizeof(float) * ref.size()) != 0) {
                        nbad++; printf("  DISPATCH MISMATCH %s %s: kernel %d pf %d fuse %d -t %d\n", sh.what, clamped ? "clamped" : "swiglu", c[0], c[1], c[2], nth);
                    }
                }
                printf("check 2d (op, %s, %s, ik's expert split on, -t 1..12): %d of %d runs bit-equal to the reference\n",
                       sh.what, clamped ? "clamped" : "swiglu", nc - nbad, nc);
                fails += nbad;
            }
        }
    }

#ifndef KB_NO_R8
    // ---- 3 (T100): the R8 kernels and the R8 path of mul_mat_id / the fused step, against the plain kernel ----
    {
        // 3a. kernel level: per row, R8 + Q8_0 against ggml_vec_dot_mxfp4_q8_0 (only the float summation order differs),
        //     R8 + Q8_2_X4 against a double-precision sum of the same quantized values (bf16 scales, int8)
        int n_bad = 0, n_cmp = 0;
        double worst_q = 0, worst_k = 0;
        for (int K : { 96, 128, 800, 768, 2560, 2592 }) {   // 3, 4, 25, 24, 80, 81 blocks (Q8_2_X4 tails of 3, 0, 1, 0, 0, 1)
            const int nb = K / 32, rows = 64;
            std::vector<float> w((size_t) rows * K), x(K);
            for (auto & v : w) v = nd(rng);
            for (auto & v : x) v = nd(rng);
            std::vector<uint8_t> wq((size_t) rows * nb * 17), wr(wq.size()), xq((size_t) nb * 34), xk(ggml_mxfp4_q8_2_x4_row_size(K));
            ggml_quantize_chunk(GGML_TYPE_MXFP4, w.data(), wq.data(), 0, rows, K, nullptr);
            ggml_mxfp4_to_r8(wq.data(), wr.data(), rows, K);
            quantize_row_q8_0(x.data(), xq.data(), K);
            ggml_quantize_row_q8_2_x4(x.data(), xk.data(), K);
            std::vector<float> ref(rows), gq(rows), gk(rows);
            for (int r = 0; r < rows; r++) ggml_vec_dot_mxfp4_q8_0(K, &ref[r], 0, wq.data() + (size_t) r * nb * 17, 0, xq.data(), 0, 1);
            ggml_gemv_mxfp4_r8_q8_0(K, gq.data(), wr.data(), xq.data(), rows);
            ggml_gemv_mxfp4_r8_q8_2x4(K, gk.data(), wr.data(), xk.data(), rows);
            // the Q8_2_X4 reference: dequantize its blocks (4 x 36 B groups: 4 bf16 d, 4 int16 s, 128 int8; tail blocks 36 B)
            std::vector<double> xd(K);
            for (int b = 0; b < nb; b++) {
                const uint8_t * base = xk.data() + (size_t) (b / 4) * 144;
                uint16_t dbits; const int8_t * q;
                if (b < 4 * (nb / 4)) { memcpy(&dbits, base + 2 * (b % 4), 2); q = (const int8_t *) (base + 16 + 32 * (b % 4)); }
                else { memcpy(&dbits, xk.data() + (size_t) b * 36, 2); q = (const int8_t *) (xk.data() + (size_t) b * 36 + 4); }
                uint32_t fb = (uint32_t) dbits << 16; float d; memcpy(&d, &fb, 4);
                for (int j = 0; j < 32; j++) xd[b * 32 + j] = (double) d * q[j];
            }
            static const int8_t kv[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
            for (int r = 0; r < rows; r++) {
                double acc = 0;
                for (int b = 0; b < nb; b++) {
                    const uint8_t * blk = wq.data() + ((size_t) r * nb + b) * 17;
                    const double sc = std::ldexp(1.0, (int) blk[0] - 128);
                    for (int j = 0; j < 16; j++) acc += sc * (kv[blk[1 + j] & 0xf] * xd[b * 32 + j] + kv[blk[1 + j] >> 4] * xd[b * 32 + j + 16]);
                }
                const double eq = std::fabs(gq[r] - ref[r]) / std::max(1e-6, (double) std::fabs(ref[r]));
                const double ek = std::fabs(gk[r] - acc) / std::max(1e-6, std::fabs(acc));
                worst_q = std::max(worst_q, eq); worst_k = std::max(worst_k, ek);
                n_cmp++;
                if (eq > 1e-4 || ek > 1e-4) { n_bad++; if (n_bad < 5) printf("  R8 KERNEL K=%d row %d: q8_0 rel %.3g, q8_2x4 rel %.3g\n", K, r, eq, ek); }
            }
        }
        printf("check 3a (R8 kernels, K of 3/4/25/24/80/81 blocks x 64 rows): %d of %d rows within 1e-4 (worst: Q8_0 vs plain %.3g, Q8_2_X4 vs its exact sum %.3g)\n",
               n_cmp - n_bad, n_cmp, worst_q, worst_k);
        fails += n_bad;
    }
    {
        // 3b. op level: one MoE layer (2560 x 768 experts like Ling, 16 of them) through ggml with R8 pools registered,
        //     act Q8_0 / Q8_2_X4, fused step off / on, -t 1..12 (odd thread counts make chunks that must be padded to
        //     8 rows), against the plain pools (all switches off, -t 8): max relative error per output row
        const int Kd = 2560, F = 768, E = 16;
        const size_t sz_gu = (size_t) F * (Kd / 32) * 17, sz_d = (size_t) Kd * (F / 32) * 17;
        std::vector<uint8_t *> plain(3), r8(3);
        const size_t szs[3] = { sz_gu, sz_gu, sz_d };
        for (int j = 0; j < 3; j++) { plain[j] = (uint8_t *) amalloc(E * szs[j]); r8[j] = (uint8_t *) amalloc(E * szs[j]); }
        {
            std::vector<float> w((size_t) F * Kd);
            for (int e = 0; e < E; e++) for (int j = 0; j < 3; j++) {
                for (auto & v : w) v = nd(rng) * 0.05f;
                const int rows = j < 2 ? F : Kd, kk = j < 2 ? Kd : F;
                ggml_quantize_chunk(GGML_TYPE_MXFP4, w.data(), plain[j] + e * szs[j], 0, rows, kk, nullptr);
                ggml_mxfp4_to_r8(plain[j] + e * szs[j], r8[j] + e * szs[j], rows, kk);
            }
        }
        std::vector<float> x(Kd); for (auto & v : x) v = nd(rng);
        std::vector<int32_t> ids(8); { std::vector<int> p(E); for (int i = 0; i < E; i++) p[i] = i; std::shuffle(p.begin(), p.end(), rng); for (int k = 0; k < 8; k++) ids[k] = p[k]; }
        auto run = [&](bool use_r8, int act, int fuse, int nth, int64_t * fused, int disp = 0) {
            ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead() + (1 << 16), nullptr, true };
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * Pg = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, Kd, F, E);
            ggml_tensor * Pu = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, Kd, F, E);
            ggml_tensor * Pd = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, F, Kd, E);
            std::vector<ggml_backend_buffer_t> bufs;
            uint8_t ** src = use_r8 ? r8.data() : plain.data();
            ggml_tensor * T[3] = { Pg, Pu, Pd };
            for (int j = 0; j < 3; j++) {
                ggml_backend_buffer_t b = ggml_backend_cpu_buffer_from_ptr(src[j], ggml_nbytes(T[j]));
                T[j]->buffer = b; T[j]->data = src[j]; ggml_backend_buffer_init_tensor(b, T[j]); bufs.push_back(b);
                if (use_r8 && !ggml_cpu_mxfp4_r8_register(src[j], ggml_nbytes(T[j]))) { fprintf(stderr, "R8 register failed\n"); exit(2); }
            }
            ggml_tensor * tx = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, Kd, 1, 1);
            ggml_tensor * ti = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, 1);
            uint8_t * small = (uint8_t *) amalloc(65536);
            ggml_backend_buffer_t sb = ggml_backend_cpu_buffer_from_ptr(small, 65536); bufs.push_back(sb);
            tx->buffer = sb; tx->data = small; ti->buffer = sb; ti->data = small + 32768;
            memcpy(tx->data, x.data(), sizeof(float) * Kd); memcpy(ti->data, ids.data(), sizeof(int32_t) * 8);
            ggml_tensor * act_t = ggml_swiglu_split(ctx, ggml_mul_mat_id(ctx, Pg, tx, ti), ggml_mul_mat_id(ctx, Pu, tx, ti));
            ggml_tensor * out = ggml_mul_mat_id(ctx, Pd, act_t, ti);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);
            ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
            if (!ggml_gallocr_alloc_graph(ga, gf)) { fprintf(stderr, "gallocr failed\n"); exit(1); }
            ggml_cpu_set_mxfp4_r8_act(act);
            ggml_cpu_set_moe_fuse(fuse);
            ggml_cpu_set_moe_dispatch(disp);
            const int64_t d0 = ggml_cpu_moe_dispatch_count();
            ggml_threadpool_params tpp = ggml_threadpool_params_default(nth);
            ggml_threadpool_t tp = ggml_threadpool_new(&tpp);
            ggml_backend_cpu_set_threadpool(be, tp);
            ggml_backend_cpu_set_n_threads(be, nth);
            const int64_t f0 = ggml_cpu_moe_fuse_count();
            ggml_backend_graph_compute(be, gf);
            *fused = ggml_cpu_moe_fuse_count() - f0;
            if (disp && ggml_cpu_moe_dispatch_count() == d0) { printf("  DISPATCH did not run\n"); *fused = -1; }
            std::vector<float> res((size_t) ggml_nelements(out));
            memcpy(res.data(), out->data, sizeof(float) * res.size());
            ggml_backend_cpu_set_threadpool(be, nullptr);
            ggml_threadpool_free(tp);
            ggml_cpu_set_mxfp4_r8_act(0); ggml_cpu_set_moe_fuse(0); ggml_cpu_set_moe_dispatch(0);
            if (use_r8) for (int j = 0; j < 3; j++) ggml_cpu_mxfp4_r8_unregister(src[j]);
            ggml_gallocr_free(ga);
            for (auto b : bufs) ggml_backend_buffer_free(b);
            free(small);
            ggml_free(ctx);
            return res;
        };
        int64_t fz = 0;
        const std::vector<float> ref = run(false, 0, 0, 8, &fz);
        // plain must not be touched by the registry: the plain run after R8 runs must give the same bits
        for (int act = 0; act < 2; act++) {
            double worst = 0; int n_cmp = 0, n_bad = 0;
            for (int nth = 1; nth <= 12; nth++) for (int fuse = 0; fuse < 2; fuse++) {
                int64_t fused = 0;
                const std::vector<float> got = run(true, act, fuse, nth, &fused);
                double w = 0;
                for (int k = 0; k < 8; k++) {
                    double dn = 0, rn = 0;
                    for (int i = 0; i < Kd; i++) { const double d = got[k * Kd + i] - ref[k * Kd + i]; dn += d * d; rn += (double) ref[k * Kd + i] * ref[k * Kd + i]; }
                    w = std::max(w, std::sqrt(dn / std::max(rn, 1e-30)));
                }
                worst = std::max(worst, w);
                n_cmp++;
                const double lim = act ? 3e-2 : 1e-5;   // Q8_0: summation order only; Q8_2_X4: bf16 activation scales
                const bool fuse_ok = (fuse != 0) == (fused > 0);
                if (w > lim || !fuse_ok) { n_bad++; printf("  R8 OP act %d fuse %d -t %d: max row rel %.3g, fused %lld\n", act, fuse, nth, w, (long long) fused); }
                // T100 1b: ik's expert split on the R8 path must give exactly the same bits as without it
                int64_t fd = 0;
                const std::vector<float> gd = run(true, act, fuse, nth, &fd, 1);
                if (fd < 0 || memcmp(gd.data(), got.data(), sizeof(float) * got.size()) != 0) { n_bad++; printf("  R8 DISPATCH act %d fuse %d -t %d: not bit-equal to the same run without it\n", act, fuse, nth); }
            }
            printf("check 3b (op, R8 pools, act %s, fuse off/on, -t 1..12; + ik's split bit-equal to without): %d of %d runs within %s of plain (worst max row rel %.3g)\n",
                   act ? "Q8_2_X4" : "Q8_0", n_cmp - n_bad, n_cmp, act ? "3e-2" : "1e-5", worst);
            fails += n_bad;
        }
        const std::vector<float> again = run(false, 0, 0, 8, &fz);
        const bool same = memcmp(again.data(), ref.data(), sizeof(float) * ref.size()) == 0;
        printf("check 3c (plain pools after the R8 runs, nothing registered): %s\n", same ? "bit-equal to before" : "DIFFERS");
        fails += !same;
        for (int j = 0; j < 3; j++) { free(plain[j]); free(r8[j]); }
    }
#else
    printf("check 3 (R8): skipped, built with -DKB_NO_R8\n");
#endif

    ggml_backend_free(be);
    printf("CHECK %s\n", fails ? "FAILED" : "PASSED");
    return fails ? 1 : 0;
}
#endif
#ifndef KB_IK
// =====================================================================================================================
// C18 (Flint): the K-quant / IQ expert kernels on Flint's real expert bytes (read from the GGUF), ggml's 1-row kernel
// against our 4-row ones. Flint's shapes (n_embd 2048, n_ff 512, top 6) and its own layer loop on the level-1 pieces
// above (SpinPool, ggml_chunks, amalloc); the MXFP4 arms are untouched.
//   --kq <flint.gguf> --kq-type q5_K --kq-layers 3,12,32 [--per-layer 128 --words 30 --reps 5 --threads 8
//        --kq-arms A,Fy0,Fy2,F0,F2 --kq-word-layers 47]
//   --check-kq   (bit identity of the 4-row kernels on quantized random data and on random bytes; no file needed)
// Arms: A = ggml's vec_dot, one row per call; Fy<p> / F<p> = ours, 4 rows per call, 256 / 512-bit registers, prefetch p:
// 0 none, 1 = 4 super-blocks ahead in the row, 2 / 3 = the same super-block of the next / the second next 4-row group; R = the floor: the same driver
// and the same bytes, only read (no maths; its outputs are not compared).
// Every arm's outputs of word 0 (every layer of it) are compared with A's byte for byte.
// =====================================================================================================================
#include "gguf.h"
#include <sys/types.h>

extern "C" {
void quantize_row_q8_K(const float * x, void * y, int64_t k);
void ggml_vec_dot_q5_K_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_q5_K_q8_K_x4(int n, float * s, const void * vx, size_t bx, const void * vy, int width, int pf);
void ggml_vec_dot_iq4_xs_q8_K(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_iq4_xs_q8_K_x4(int n, float * s, const void * vx, size_t bx, const void * vy, int width, int pf);
}

namespace kq {

typedef void (*dot1_t)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
typedef void (*dot4_t)(int, float *, const void *, size_t, const void *, int, int);

struct Kind { const char * name; ggml_type type; dot1_t dot1; dot4_t dot4; };
static const Kind KINDS[] = {
    { "q5_K",   GGML_TYPE_Q5_K,   ggml_vec_dot_q5_K_q8_K,   ggml_vec_dot_q5_K_q8_K_x4 },
    { "iq4_xs", GGML_TYPE_IQ4_XS, ggml_vec_dot_iq4_xs_q8_K, ggml_vec_dot_iq4_xs_q8_K_x4 },
    // C18 item 9d: any formats, one per projection (the first named layer's; every named layer must have the same) - arms A and R only,
    // each projection through ggml's own vec_dot and activation type (Flint's small formats sit next to Q5_K in the same layers)
    { "mixed",  GGML_TYPE_COUNT,  nullptr,                  nullptr },
};
static const Kind * find_kind(const std::string & s) {
    for (const Kind & k : KINDS) if (s == k.name) return &k;
    fprintf(stderr, "--kq-type %s: not one of", s.c_str());
    for (const Kind & k : KINDS) fprintf(stderr, " %s", k.name);
    fprintf(stderr, "\n");
    exit(1);
}

static const int EMBD = 2048, FF = 512, USED = 6;
static const int P_K[3]    = { EMBD, EMBD, FF };   // gate, up, down: inner dimension
static const int P_ROWS[3] = { FF,   FF,   EMBD }; // output rows

static const int QK_K_KB = 256;   // the K-quant / IQ super-block

static std::string format_pct(double v, double ref) {
    char b[32];
    snprintf(b, sizeof(b), " %+.1f", 100.0 * (v / ref - 1.0));
    return b;
}

// random block bytes get finite fp16 scales: the byte offsets of the ggml_half fields of one block of each type
static void kind_fix_scales(ggml_type t, uint8_t * blocks, int n_blocks, std::mt19937 & rng) {
    std::uniform_real_distribution<float> ud(-0.05f, 0.05f);
    const size_t bs = ggml_type_size(t);
    std::vector<size_t> halves;
    if (t == GGML_TYPE_Q5_K)   halves = { 0, 2 };   // d, dmin
    if (t == GGML_TYPE_IQ4_XS) halves = { 0 };      // d
    for (int b = 0; b < n_blocks; b++) {
        for (const size_t off : halves) {
            const ggml_fp16_t h = ggml_fp32_to_fp16(ud(rng));
            memcpy(blocks + (size_t) b * bs + off, &h, sizeof(h));
        }
    }
}

static size_t q8k_bytes(int K) { return ggml_row_size(GGML_TYPE_Q8_K, K); }

// one arm: pf / width for ours; A = width 0
struct Arm { std::string name; int width = 0; int pf = 0; };   // width -1 = R

static Arm parse_arm(const std::string & t) {
    Arm a; a.name = t;
    if (t == "A") return a;
    if (t == "R") { a.width = -1; return a; }
    size_t i = 0;
    if (t.rfind("Fy", 0) == 0) { a.width = 256; i = 2; } else if (t.rfind("F", 0) == 0) { a.width = 512; i = 1; }
    else { fprintf(stderr, "unknown arm %s\n", t.c_str()); exit(1); }
    const int m = atoi(t.c_str() + i);
    a.pf = m == 0 ? 0 : m == 1 ? 4 : m == 2 ? -1 : m == 3 ? -2 : 0;
    if (m < 0 || m > 3 || i >= t.size()) { fprintf(stderr, "unknown arm %s (prefetch 0, 1, 2 or 3)\n", t.c_str()); exit(1); }
    return a;
}

// rows [r0, r1) of one slice
static void rows(const Kind & k, const Arm & a, int K, const uint8_t * slice, size_t rb, int r0, int r1, const void * q, float * out,
                 dot1_t d1 = nullptr) {
    int r = r0;
    if (a.width < 0) {   // R: read the rows, 32 B at a time (+ 8 B steps for a row that is not a multiple of 32)
        for (; r < r1; r++) {
            const uint8_t * p = slice + (size_t) r * rb;
            __m256i acc = _mm256_setzero_si256();
            size_t o = 0;
            for (; o + 32 <= rb; o += 32) acc = _mm256_xor_si256(acc, _mm256_loadu_si256((const __m256i *) (p + o)));
            uint64_t t = 0;
            for (; o < rb; o += 8) { uint64_t v; memcpy(&v, p + o, 8); t ^= v; }
            out[r] = (float) ((uint32_t) _mm256_extract_epi32(acc, 0) ^ (uint32_t) t);
        }
        return;
    }
    if (a.width > 0) {
        for (; r + 3 < r1; r += 4) k.dot4(K, out + r, slice + (size_t) r * rb, rb, q, a.width, a.pf);
    }
    const dot1_t f = d1 ? d1 : k.dot1;
    for (; r < r1; r++) f(K, out + r, 0, slice + (size_t) r * rb, 0, q, 0, 1);
}

struct Pools {
    int n_exp = 0;
    ggml_type t[3] = { GGML_TYPE_COUNT, GGML_TYPE_COUNT, GGML_TYPE_COUNT };   // each projection's format
    size_t rb[3] = {0, 0, 0}, slice[3] = {0, 0, 0};
    uint8_t * p[3] = { nullptr, nullptr, nullptr };
};

static Pools load(const std::string & path, const Kind & kind, const std::vector<int> & layers, int per_layer) {
    ggml_context * meta = nullptr;
    gguf_init_params ip = { /*no_alloc =*/ true, /*ctx =*/ &meta };
    gguf_context * g = gguf_init_from_file(path.c_str(), ip);
    if (!g) { fprintf(stderr, "cannot read the GGUF header of %s\n", path.c_str()); exit(1); }
    static const char * names[3] = { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" };
    Pools P;
    P.n_exp = (int) layers.size() * per_layer;
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    for (int j = 0; j < 3; j++) {
        P.t[j] = kind.type;
        if (kind.type == GGML_TYPE_COUNT) {   // mixed: the first named layer's format of this projection
            const std::string name = "blk." + std::to_string(layers[0]) + "." + names[j] + ".weight";
            const ggml_tensor * t = gguf_find_tensor(g, name.c_str()) >= 0 ? ggml_get_tensor(meta, name.c_str()) : nullptr;
            if (!t) { fprintf(stderr, "%s: missing\n", name.c_str()); exit(1); }
            P.t[j] = t->type;
        }
        P.rb[j]    = ggml_row_size(P.t[j], P_K[j]);
        P.slice[j] = P.rb[j] * P_ROWS[j];
        P.p[j]     = (uint8_t *) amalloc((size_t) P.n_exp * P.slice[j]);
    }
    int e = 0;
    for (const int il : layers) {
        for (int j = 0; j < 3; j++) {
            const std::string name = "blk." + std::to_string(il) + "." + names[j] + ".weight";
            const int64_t tid = gguf_find_tensor(g, name.c_str());
            const ggml_tensor * t = tid >= 0 ? ggml_get_tensor(meta, name.c_str()) : nullptr;
            if (!t || t->type != P.t[j] || t->ne[0] != P_K[j] || t->ne[1] != P_ROWS[j] || t->ne[2] < per_layer) {
                fprintf(stderr, "%s: missing, or not %s [%d, %d, >= %d] (type %s)\n", name.c_str(), ggml_type_name(P.t[j]), P_K[j], P_ROWS[j],
                        per_layer, t ? ggml_type_name(t->type) : "-");
                exit(1);
            }
            const size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, tid);
            uint8_t * dst = P.p[j] + (size_t) e * P.slice[j];
            const size_t n = (size_t) per_layer * P.slice[j];   // experts 0 .. per_layer-1, back to back as in the file
            if (pread(fd, dst, n, (off_t) off) != (ssize_t) n) { fprintf(stderr, "short read of %s\n", name.c_str()); exit(1); }
            posix_fadvise(fd, (off_t) off, (off_t) n, POSIX_FADV_DONTNEED);   // no page cache of the model kept behind
        }
        e += per_layer;
    }
    close(fd);
    gguf_free(g);
    ggml_free(meta);
    return P;
}

struct Word { std::vector<int32_t> ids; std::vector<float> w; };

struct Run {
    const Kind & kind;
    const Arm  & arm;
    const Pools & P;
    SpinPool & pool;
    std::vector<uint8_t> qx, qact;
    std::vector<float> g, u, a, d, out;
    std::atomic<int64_t> ctr[USED];
    dot1_t d1[3];                // each projection's ggml vec_dot (its own format)
    ggml_from_float_t qf_x, qf_a; // the activations' quantizers: gate/up's and down's vec_dot_type
    size_t qb_x, qb_a;           // their row bytes (K = 2048 / 512)

    Run(const Kind & k, const Arm & a_, const Pools & p, SpinPool & sp) : kind(k), arm(a_), P(p), pool(sp) {
        for (int j = 0; j < 3; j++) d1[j] = ggml_get_type_traits_cpu(P.t[j])->vec_dot;
        const ggml_type vx = ggml_get_type_traits_cpu(P.t[0])->vec_dot_type, va = ggml_get_type_traits_cpu(P.t[2])->vec_dot_type;
        qf_x = ggml_get_type_traits_cpu(vx)->from_float; qf_a = ggml_get_type_traits_cpu(va)->from_float;
        qb_x = ggml_row_size(vx, EMBD); qb_a = ggml_row_size(va, FF);
        qx.resize(qb_x); qact.resize((size_t) USED * qb_a);
        g.resize(USED * FF); u.resize(USED * FF); a.resize(USED * FF); d.resize((size_t) USED * EMBD); out.resize(EMBD);
    }

    // one layer as the graph runs it: gate, up (a barrier after each), SwiGLU + quantize, down (stolen chunks), weighted sum
    void layer(const float * x, const int32_t * ids, const float * w) {
        const int nth = pool.n;
        pool.run([&](int ith) {
            if (ith == 0) qf_x(x, qx.data(), EMBD);
            pool.barrier();
            for (int j = 0; j < 2; j++) {
                float * o = j == 0 ? g.data() : u.data();
                const Chunks c = ggml_chunks(FF, nth, 4);
                for (int k = 0; k < USED; k++) {
                    for (int64_t ch = ith; ch < c.nchunk; ch += nth) {
                        const int64_t r0 = ch * c.dr, r1 = std::min<int64_t>(r0 + c.dr, FF);
                        if (r0 < r1) rows(kind, arm, EMBD, P.p[j] + (size_t) ids[k] * P.slice[j], P.rb[j], (int) r0, (int) r1, qx.data(), o + (size_t) k * FF, d1[j]);
                    }
                }
                pool.barrier();
            }
            if (ith == 0) {
                for (int i = 0; i < USED * FF; i++) a[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
                for (int k = 0; k < USED; k++) {
                    qf_a(a.data() + (size_t) k * FF, qact.data() + (size_t) k * qb_a, FF);
                    ctr[k].store(nth, std::memory_order_relaxed);
                }
            }
            pool.barrier();
            {
                const Chunks c = ggml_chunks(EMBD, nth, 4);
                for (int k = 0; k < USED; k++) {
                    int64_t ch = ith;
                    while (ch < c.nchunk) {
                        const int64_t r0 = ch * c.dr, r1 = std::min<int64_t>(r0 + c.dr, EMBD);
                        if (r0 < r1) rows(kind, arm, FF, P.p[2] + (size_t) ids[k] * P.slice[2], P.rb[2], (int) r0, (int) r1,
                                          qact.data() + (size_t) k * qb_a, d.data() + (size_t) k * EMBD, d1[2]);
                        if (nth >= c.nchunk) break;
                        ch = ctr[k].fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
            pool.barrier();
            if (ith == 0) {
                for (int i = 0; i < EMBD; i++) {
                    float s = 0;
                    for (int k = 0; k < USED; k++) s += w[k] * d[(size_t) k * EMBD + i];
                    out[i] = s;
                }
            }
        });
    }
};

static int bench(const std::string & path, const std::string & type, const std::vector<int> & layers, int per_layer, int words,
                 int reps, int word_layers, int seed, const std::vector<int> & threads, const std::string & arms_s) {
    const Kind & kind = *find_kind(type);
    const double t0 = now_s();
    const Pools P = load(path, kind, layers, per_layer);
    const size_t exp_bytes = P.slice[0] + P.slice[1] + P.slice[2];
    printf("kq: %s (%s / %s / %s), %d experts (%zu layers x %d) of %zu B = %.2f GB, read in %.1f s\n", kind.name, ggml_type_name(P.t[0]),
           ggml_type_name(P.t[1]), ggml_type_name(P.t[2]), P.n_exp, layers.size(), per_layer, exp_bytes, (double) P.n_exp * exp_bytes / 1e9, now_s() - t0);

    // the plan: words x word_layers x 6 distinct experts + weights, one fixed layer input
    std::mt19937 rng((uint32_t) seed);
    std::uniform_real_distribution<float> uw(0.05f, 1.0f);
    std::vector<Word> plan((size_t) words * word_layers);
    for (Word & wd : plan) {
        wd.ids.resize(USED); wd.w.resize(USED);
        float sum = 0;
        for (int k = 0; k < USED; k++) {
            int c;
            do { c = (int) (rng() % (uint32_t) P.n_exp); } while (std::find(wd.ids.begin(), wd.ids.begin() + k, c) != wd.ids.begin() + k);
            wd.ids[k] = c; wd.w[k] = uw(rng); sum += wd.w[k];
        }
        for (float & v : wd.w) v /= sum;
    }
    std::vector<float> x(EMBD);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (float & v : x) v = nd(rng);
    const double word_bytes = (double) word_layers * USED * exp_bytes;
    printf("kq: %d words x %d layers x %d experts = %.1f MB of expert bytes per word\n", words, word_layers, USED, word_bytes / 1e6);

    std::vector<Arm> arms;
    for (const std::string & t : split(arms_s, ',')) arms.push_back(parse_arm(t));
    if (arms.empty() || arms[0].name != "A") { fprintf(stderr, "--kq-arms must start with A (the reference)\n"); return 1; }
    for (const Arm & a : arms) {
        if (a.width > 0 && kind.dot4 == nullptr) { fprintf(stderr, "--kq-type %s: arms A and R only\n", kind.name); return 1; }
    }

    int rc = 0;
    for (const int nth : threads) {
        SpinPool sp(nth);
        // word 0, every layer, for the byte comparison
        std::vector<std::vector<float>> ref;
        for (const Arm & a : arms) {
            Run run(kind, a, P, sp);
            std::vector<float> o;
            for (int l = 0; l < word_layers; l++) {
                run.layer(x.data(), plan[l].ids.data(), plan[l].w.data());
                o.insert(o.end(), run.out.begin(), run.out.end());
            }
            if (ref.empty()) { ref.push_back(o); continue; }
            if (a.width < 0) { printf("kq: -t %d arm %-4s word 0: read-only, not compared\n", nth, a.name.c_str()); continue; }
            const bool same = memcmp(o.data(), ref[0].data(), o.size() * sizeof(float)) == 0;
            size_t n_diff = 0;
            for (size_t i = 0; i < o.size(); i++) n_diff += memcmp(&o[i], &ref[0][i], sizeof(float)) != 0;
            printf("kq: -t %d arm %-4s word 0 (%d layers x %d outputs): %s vs A (%zu of %zu differ)\n", nth, a.name.c_str(), word_layers, EMBD,
                   same ? "bit-equal" : "DIFFERS", n_diff, o.size());
            if (!same) rc = 3;
        }
        // timing: arms interleaved within each rep, the first arm rotating from rep to rep (no arm always first)
        std::vector<std::vector<double>> ms(arms.size());
        for (int rep = 0; rep < reps; rep++) {
            for (size_t j = 0; j < arms.size(); j++) {
                const size_t ia = (j + (size_t) rep) % arms.size();
                Run run(kind, arms[ia], P, sp);
                const double ts = now_s();
                for (int wi = 0; wi < words; wi++) {
                    for (int l = 0; l < word_layers; l++) {
                        const Word & wd = plan[(size_t) wi * word_layers + l];
                        run.layer(x.data(), wd.ids.data(), wd.w.data());
                    }
                }
                ms[ia].push_back((now_s() - ts) * 1e3 / words);
            }
        }
        auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        const double a_med = median(ms[0]);
        printf("SUMMARY kq %s -t %d (ms per word, median of %d reps; vs A per rep)\n", kind.name, nth, reps);
        for (size_t ia = 0; ia < arms.size(); ia++) {
            const double m = median(ms[ia]);
            std::string per;
            for (int rep = 0; rep < reps; rep++) per += format_pct(ms[ia][rep], ms[0][rep]);
            printf("  %-4s %8.3f ms  %6.1f GB/s  %+6.1f %%  [%s]\n", arms[ia].name.c_str(), m, word_bytes / (m * 1e-3) / 1e9,
                   100.0 * (m / a_med - 1.0), per.c_str());
        }
    }
    return rc;
}

static int check() {
    std::mt19937 rng(18);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int n_cmp = 0, n_bad = 0;
    for (const Kind & kind : KINDS) {
        if (kind.dot4 == nullptr) continue;
        for (const int nsb : { 1, 2, 3, 8, 9 }) {
            const int K = nsb * QK_K_KB, rows_n = 64;
            const size_t rb = ggml_row_size(kind.type, K);
            for (int mode = 0; mode < 2; mode++) {   // 0 = quantized random floats, 1 = random bytes with finite scales
                std::vector<uint8_t> wq((size_t) rows_n * rb);
                if (mode == 0) {
                    std::vector<float> wf((size_t) rows_n * K);
                    for (float & v : wf) v = nd(rng);
                    ggml_quantize_chunk(kind.type, wf.data(), wq.data(), 0, rows_n, K, nullptr);
                } else {
                    for (uint8_t & b : wq) b = (uint8_t) rng();
                    kind_fix_scales(kind.type, wq.data(), rows_n * nsb, rng);
                }
                std::vector<float> xf(K);
                for (float & v : xf) v = nd(rng) * (mode == 0 ? 1.0f : 40.0f);   // mode 1: large activations, q8 near +-127
                std::vector<uint8_t> xq(q8k_bytes(K));
                quantize_row_q8_K(xf.data(), xq.data(), K);
                std::vector<float> want(rows_n), got(rows_n);
                for (int r = 0; r < rows_n; r++) kind.dot1(K, &want[r], 0, wq.data() + (size_t) r * rb, 0, xq.data(), 0, 1);
                for (const int width : { 256, 512 }) {
                    for (const int pf : { 0, 4, -1 }) {
                        std::fill(got.begin(), got.end(), -1.0f);
                        for (int r = 0; r + 3 < rows_n; r += 4) kind.dot4(K, &got[r], wq.data() + (size_t) r * rb, rb, xq.data(), width, pf);
                        n_cmp++;
                        if (memcmp(got.data(), want.data(), sizeof(float) * rows_n) != 0) {
                            n_bad++;
                            int r = 0;
                            while (memcmp(&got[r], &want[r], sizeof(float)) == 0) r++;
                            printf("  %s K %d %s width %d pf %d: row %d got %.9g want %.9g\n", kind.name, K, mode ? "bytes" : "quantized", width, pf, r, got[r], want[r]);
                        }
                    }
                }
            }
        }
    }
    printf("check-kq (4 rows vs 1 row, %d types x K of 1/2/3/8/9 super-blocks x quantized / random bytes x width 256/512 x prefetch none/4/next): %d of %d bit-equal\n",
           (int) std::count_if(std::begin(KINDS), std::end(KINDS), [](const Kind & k) { return k.dot4 != nullptr; }), n_cmp - n_bad, n_cmp);
    return n_bad == 0 ? 0 : 3;
}

}   // namespace kq
#endif

int main(int argc, char ** argv) {
    Opts o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(1); } return argv[++i]; };
        if      (a == "--sidecar")   o.sidecar = next();
        else if (a == "--offsets")   { for (auto & s : split(next(), ',')) o.offs.push_back((size_t) atoll(s.c_str())); }
        else if (a == "--per-layer") o.per_layer = atoi(next().c_str());
        else if (a == "--words")     o.words = atoi(next().c_str());
        else if (a == "--reps")      o.reps = atoi(next().c_str());
        else if (a == "--seed")      o.seed = atoi(next().c_str());
        else if (a == "--threads")   { o.threads.clear(); for (auto & s : split(next(), ',')) o.threads.push_back(atoi(s.c_str())); }
        else if (a == "--arms")      o.arms = next();
        else if (a == "--out-dir")   o.out_dir = next();
        else if (a == "--compare")   o.compare = next();
        else if (a == "--profile")   o.profile = next();
#ifndef KB_IK
        else if (a == "--check")     return do_check();
        else if (a == "--check-kq")  return kq::check();
        else if (a == "--kq")        o.kq = next();
        else if (a == "--kq-type")   o.kq_type = next();
        else if (a == "--kq-arms")   o.kq_arms = next();
        else if (a == "--kq-word-layers") o.kq_word_layers = atoi(next().c_str());
        else if (a == "--kq-layers") { for (auto & s : split(next(), ',')) o.kq_layers.push_back(atoi(s.c_str())); }
#endif
        else { fprintf(stderr, "unknown argument %s\n", a.c_str()); return 1; }
    }
    if (!o.compare.empty()) return do_compare(o);
#ifndef KB_IK
    if (!o.kq.empty()) {
        if (o.kq_layers.empty()) { fprintf(stderr, "--kq needs --kq-layers\n"); return 1; }
        return kq::bench(o.kq, o.kq_type, o.kq_layers, o.per_layer, o.words, o.reps, o.kq_word_layers, o.seed, o.threads, o.kq_arms);
    }
#endif
    if (o.sidecar.empty() || o.offs.empty() || (o.arms.empty() && o.profile.empty())) { fprintf(stderr, "need --sidecar, --offsets, --arms (or --profile)\n"); return 1; }
    if (o.arms.empty()) o.arms = o.profile;   // T100: the layouts the profiled arms need

    const double t_load = now_s();
    Pools P = load_pools(o.sidecar.c_str(), o.offs, o.per_layer);
    printf("loaded %d experts (%zu layers x %d) = %.2f GB per layout, O_DIRECT, %.1f s\n", P.n_exp, o.offs.size(), o.per_layer,
           (double) P.n_exp * BLOCK / 1e9, now_s() - t_load);
    // one plan for timing (words) and word 0 for the rounding check; the same seed in both builds
    Plan plan = make_plan(o.words, P.n_exp, (uint32_t) o.seed);
    printf("plan: %d words x %d layers x %d experts, %.1f MB of expert bytes per word\n", o.words, N_LAYERS, N_USED, WORD_BYTES / 1e6);

    // ---- the arms ----
    std::vector<std::string> tags = split(o.arms, ',');
    std::vector<ArmL1> l1;    // by name
    uint8_t * extra[3] = { nullptr, nullptr, nullptr };   // a second layout (C: 8x8, E: kt, D: R8)

#ifndef KB_IK
    const bool need_c = o.arms.find('C') != std::string::npos;
    const bool need_e = o.arms.find('E') != std::string::npos;
    uint8_t * pool_c[3] = { nullptr, nullptr, nullptr };
    uint8_t * pool_e[3] = { nullptr, nullptr, nullptr };
    if (need_c) {
        // ggml's own repack (repack_mxfp4_to_mxfp4_8_bl) through a CPU_REPACK buffer, one slice at a time
        const double t0 = now_s();
        for (int j = 0; j < 3; j++) {
            pool_c[j] = (uint8_t *) amalloc((size_t) P.n_exp * SLICE);
            ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
            ggml_context * c = ggml_init(ip);
            ggml_backend_buffer_t rb = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_repack_buffer_type(), SLICE);
            ggml_tensor * s = ggml_new_tensor_2d(c, GGML_TYPE_MXFP4, PROJ_K[j], PROJ_ROWS[j]);
            s->buffer = rb;
            s->data   = ggml_backend_buffer_get_base(rb);
            ggml_backend_buffer_init_tensor(rb, s);
            if (!s->extra) { fprintf(stderr, "no repack traits for MXFP4 [%d, %d]\n", PROJ_K[j], PROJ_ROWS[j]); return 2; }
            for (int e = 0; e < P.n_exp; e++) {
                ggml_backend_tensor_set(s, P.p[j] + (size_t) e * SLICE, 0, SLICE);
                memcpy(pool_c[j] + (size_t) e * SLICE, s->data, SLICE);
            }
            ggml_backend_buffer_free(rb);
            ggml_free(c);
        }
        printf("C layout (mxfp4_8x8): %.2f ms per expert (3 slices)\n", (now_s() - t0) * 1e3 / P.n_exp);
    }
    bool need_r8 = false;
    for (auto & t : split(o.arms, ',')) need_r8 = need_r8 || is_r8_arm(t.substr(0, t.size() - 1));
    uint8_t * pool_r8[3] = { nullptr, nullptr, nullptr };
    if (need_r8) {
        // T100: ik's 8-row interleave, one slice at a time (rows 768 / 2560, multiples of 8)
        const double t0 = now_s();
        for (int j = 0; j < 3; j++) {
            pool_r8[j] = (uint8_t *) amalloc((size_t) P.n_exp * SLICE);
            for (int e = 0; e < P.n_exp; e++) {
                ggml_mxfp4_to_r8(P.p[j] + (size_t) e * SLICE, pool_r8[j] + (size_t) e * SLICE, PROJ_ROWS[j], PROJ_K[j]);
            }
        }
        printf("R8 layout (ggml_mxfp4_to_r8): %.3f ms per expert (3 slices)\n", (now_s() - t0) * 1e3 / P.n_exp);
    }
    if (need_e) {
        const double t0 = now_s();
        for (int j = 0; j < 3; j++) {
            pool_e[j] = (uint8_t *) amalloc((size_t) P.n_exp * SLICE);
            for (int e = 0; e < P.n_exp; e++) {
                kt::to_kt(P.p[j] + (size_t) e * SLICE, pool_e[j] + (size_t) e * SLICE, PROJ_ROWS[j], PROJ_K[j]);
            }
        }
        printf("E layout (values + E8M0 scales): %.3f ms per expert (3 slices)\n", (now_s() - t0) * 1e3 / P.n_exp);
    }
    (void) extra;

    const size_t q8_2560 = (size_t) N_EMBD / 32 * 34, q8_768 = (size_t) N_FF / 32 * 34;
    auto row_bytes = [](int K) { return (size_t) K / 32 * 17; };
    {   // A
        ArmL1 a; a.name = "A"; a.align = 1; a.qbytes[0] = a.qbytes[1] = q8_2560; a.qbytes[2] = q8_768;
        a.quant = [](const float * x, int K, void * q) { quantize_row_q8_0(x, q, K); };
        a.rows = [&P, row_bytes](int proj, int e, int r0, int r1, const void * q, float * out) {
            const size_t rb = row_bytes(PROJ_K[proj]);
            const uint8_t * base = P.p[proj] + (size_t) e * SLICE;
            for (int r = r0; r < r1; r++) ggml_vec_dot_mxfp4_q8_0(PROJ_K[proj], out + r, 0, base + (size_t) r * rb, 0, q, 0, 1);
        };
        l1.push_back(a);
    }
    for (int fv = 0; fv < 6; fv++) {   // F0, F1, F2 (zmm), Fy0, Fy1, Fy2 (ymm); the digit is the prefetch mode
        ArmL1 a; a.name = std::string(fv < 3 ? "F" : "Fy") + std::to_string(fv % 3); a.align = 1; a.qbytes[0] = a.qbytes[1] = q8_2560; a.qbytes[2] = q8_768;
        a.quant = [](const float * x, int K, void * q) { quantize_row_q8_0(x, q, K); };
        const int width = fv < 3 ? 512 : 256, pf = fv % 3 == 1 ? 38 : fv % 3 == 2 ? -1 : 0;
        a.rows = [&P, row_bytes, width, pf](int proj, int e, int r0, int r1, const void * q, float * out) {
            const size_t rb = row_bytes(PROJ_K[proj]);
            const uint8_t * base = P.p[proj] + (size_t) e * SLICE;
            int r = r0;
            for (; r + 3 < r1; r += 4) ggml_vec_dot_mxfp4_q8_0_x4(PROJ_K[proj], out + r, base + (size_t) r * rb, rb, q, width, pf);
            for (; r < r1; r++) ggml_vec_dot_mxfp4_q8_0(PROJ_K[proj], out + r, 0, base + (size_t) r * rb, 0, q, 0, 1);
        };
        l1.push_back(a);
    }
    {   // R: the fp32 reference (rounding check only)
        ArmL1 a; a.name = "R"; a.align = 1; a.qbytes[0] = a.qbytes[1] = (size_t) N_EMBD * 4; a.qbytes[2] = (size_t) N_FF * 4;
        a.quant = [](const float * x, int K, void * q) { memcpy(q, x, sizeof(float) * K); };
        a.rows = [&P, row_bytes](int proj, int e, int r0, int r1, const void * q, float * out) {
            static const int8_t kv[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
            const int K = PROJ_K[proj];
            const size_t rb = row_bytes(K);
            const float * xf = (const float *) q;
            for (int r = r0; r < r1; r++) {
                const uint8_t * row = P.p[proj] + (size_t) e * SLICE + (size_t) r * rb;
                double acc = 0;
                for (int b = 0; b < K / 32; b++) {
                    const uint8_t * blk = row + (size_t) b * 17;
                    const double sc = std::ldexp(1.0, (int) blk[0] - 128);   // ggml's E8M0_HALF: kvalues are doubled
                    for (int j = 0; j < 16; j++) {
                        acc += sc * kv[blk[1 + j] & 0xf] * xf[b * 32 + j];
                        acc += sc * kv[blk[1 + j] >> 4]  * xf[b * 32 + j + 16];
                    }
                }
                out[r] = (float) acc;
            }
        };
        l1.push_back(a);
    }
    {   // C
        ArmL1 a; a.name = "C"; a.align = 8; a.qbytes[0] = a.qbytes[1] = q8_2560; a.qbytes[2] = q8_768;
        a.quant = [](const float * x, int K, void * q) { quantize_row_q8_0(x, q, K); };
        a.rows = [&pool_c, row_bytes](int proj, int e, int r0, int r1, const void * q, float * out) {
            const size_t rb = row_bytes(PROJ_K[proj]);
            const uint8_t * base = pool_c[proj] + (size_t) e * SLICE;
            ggml_gemv_mxfp4_8x8_q8_0(PROJ_K[proj], out + r0, (size_t) PROJ_ROWS[proj], base + (size_t) r0 * rb, q, 1, r1 - r0);
        };
        l1.push_back(a);
    }
    for (int act : { 0, 1 }) {   // T100: Q (our Q8_0 activations), K (ik's Q8_2_X4)
        ArmL1 a; a.name = act ? "K" : "Q"; a.align = 8;
        a.qbytes[0] = a.qbytes[1] = act ? ggml_mxfp4_q8_2_x4_row_size(N_EMBD) : q8_2560;
        a.qbytes[2] = act ? ggml_mxfp4_q8_2_x4_row_size(N_FF) : q8_768;
        if (act) a.quant = [](const float * x, int K, void * q) { ggml_quantize_row_q8_2_x4(x, q, K); };
        else     a.quant = [](const float * x, int K, void * q) { quantize_row_q8_0(x, q, K); };
        a.rows = [&pool_r8, row_bytes, act](int proj, int e, int r0, int r1, const void * q, float * out) {
            const size_t rb = row_bytes(PROJ_K[proj]);
            const uint8_t * base = pool_r8[proj] + (size_t) e * SLICE + (size_t) r0 * rb;
            if (act) ggml_gemv_mxfp4_r8_q8_2x4(PROJ_K[proj], out + r0, base, q, r1 - r0);
            else     ggml_gemv_mxfp4_r8_q8_0(PROJ_K[proj], out + r0, base, q, r1 - r0);
        };
        l1.push_back(a);
    }
    {   // E
        ArmL1 a; a.name = "E"; a.align = 4; a.qbytes[0] = a.qbytes[1] = (size_t) N_EMBD * 2; a.qbytes[2] = (size_t) N_FF * 2;
        a.quant = kt::quant_bf16;
        a.rows = [&pool_e](int proj, int e, int r0, int r1, const void * q, float * out) {
            kt::gemv(PROJ_K[proj], r0, r1, pool_e[proj] + (size_t) e * SLICE, PROJ_ROWS[proj], (const uint16_t *) q, out);
        };
        l1.push_back(a);
    }
#else
    const bool need_d = o.arms.find('D') != std::string::npos;
    uint8_t * pool_d[3] = { nullptr, nullptr, nullptr };
    if (need_d) {
        const double t0 = now_s();
        for (int j = 0; j < 3; j++) {
            pool_d[j] = (uint8_t *) amalloc((size_t) P.n_exp * SLICE);
            memcpy(pool_d[j], P.p[j], (size_t) P.n_exp * SLICE);
            ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
            ggml_context * c = ggml_init(ip);
            ggml_tensor * t = ggml_new_tensor_3d(c, GGML_TYPE_MXFP4, PROJ_K[j], PROJ_ROWS[j], P.n_exp);
            t->data = pool_d[j];
            iqk_repack_tensor(t);
            if (t->type != GGML_TYPE_MXFP4_R8) { fprintf(stderr, "iqk_repack_tensor did not give MXFP4_R8 (type %d)\n", (int) t->type); return 2; }
            ggml_free(c);
        }
        printf("D layout (MXFP4_R8): %.2f ms per expert (3 slices)\n", (now_s() - t0) * 1e3 / P.n_exp);
    }
    const auto q82 = ggml_internal_get_type_traits(GGML_TYPE_Q8_2_X4);
    const size_t q82_2560 = ggml_row_size(GGML_TYPE_Q8_2_X4, N_EMBD), q82_768 = ggml_row_size(GGML_TYPE_Q8_2_X4, N_FF);
    auto row_bytes = [](int K) { return (size_t) K / 32 * 17; };
    for (int is_r8 : { 0, 1 }) {
        ArmL1 a; a.name = is_r8 ? "D" : "B"; a.align = is_r8 ? 16 : 1;   // iqk's R8 GEMV takes rows in multiples of 16 a.qbytes[0] = a.qbytes[1] = q82_2560; a.qbytes[2] = q82_768;
        auto ff = q82.from_float;
        a.quant = [ff](const float * x, int K, void * q) { ff(x, q, K); };
        uint8_t ** src = is_r8 ? pool_d : P.p;
        const int typeA = is_r8 ? GGML_TYPE_MXFP4_R8 : GGML_TYPE_MXFP4;
        a.rows = [src, typeA, row_bytes](int proj, int e, int r0, int r1, const void * q, float * out) {
            const size_t rb = row_bytes(PROJ_K[proj]);
            const uint8_t * base = src[proj] + (size_t) e * SLICE;
            if (!iqk_mul_mat(r1 - r0, 1, PROJ_K[proj], typeA, base + (size_t) r0 * rb, (long) rb, GGML_TYPE_Q8_2_X4, q,
                             (long) ggml_row_size(GGML_TYPE_Q8_2_X4, PROJ_K[proj]), out + r0, r1 - r0, 0, 1)) {
                fprintf(stderr, "iqk_mul_mat refused type %d\n", typeA);
                exit(2);
            }
        };
        l1.push_back(a);
    }
#endif

    auto find_l1 = [&](const std::string & n) -> const ArmL1 * {
        for (auto & a : l1) if (a.name == n) return &a;
        return nullptr;
    };

#ifndef KB_IK
    // F arms: the last digit is the prefetch mode - 0 none, 1 = 38 blocks (~646 B) ahead, 2 = the next 4-row group
    auto f_prefetch = [](const std::string & n) { const char m = n.back(); return m == '1' ? 38 : m == '2' ? -1 : 0; };
#endif
    // ---- one block: arm tag (name + level digit), threads -> ms per word; out = word-0 outputs (or null) ----
    ggml_backend_t backend = ggml_backend_cpu_init();
    // T100 --profile: when set, the level-2 block times each stage's prefix graph (every op up to that boundary, one
    // graph compute, so the ops keep their barriers) instead of the whole graph; stage cost = prefix(k) - prefix(k-1)
    std::vector<std::pair<std::string, double>> * prof = nullptr;
    auto run_block = [&](const std::string & tag, int nth, int words, std::vector<float> * out0) -> double {
        const std::string name = tag.substr(0, tag.size() - 1);
        const int level = tag.back() - '0';
        double t = 0;
        if (level == 1) {
            const ArmL1 * a = find_l1(name);
            if (!a) { fprintf(stderr, "no arm %s at level 1\n", name.c_str()); exit(1); }
#ifndef KB_IK
#endif
            SpinPool pool(nth);
            L1Run run(*a, pool);
            auto word = [&](int wi, std::vector<float> * res) {
                for (int l = 0; l < N_LAYERS; l++) {
                    const size_t k = ((size_t) wi * N_LAYERS + l) * N_USED;
                    run.layer(plan.x.data(), &plan.ids[k], &plan.w[k]);
                    if (res) std::copy(run.out.begin(), run.out.end(), res->begin() + (size_t) l * N_EMBD);
                }
            };
            word(words - 1, nullptr);   // warm-up (threads, scratch), not timed
            if (out0) { word(0, out0); return 0; }
            const double t0 = now_s();
            for (int wi = 0; wi < words; wi++) word(wi, nullptr);
            t = now_s() - t0;
        } else {
#ifndef KB_IK
            const bool r8 = is_r8_arm(name);   // T100: Q, K, QG, KG
            uint8_t * const * pools = name == "C" ? pool_c : r8 ? pool_r8 : P.p;
            const bool repack = name == "C";
            const bool is_f = name[0] == 'F';
            const bool fuse = name == "G" || name == "H" || name == "QG" || name == "KG" || is_dispatch_arm(name);
            const bool disp = is_dispatch_arm(name);   // T100 1b
            ggml_cpu_set_moe_dispatch(disp);
            const int64_t disp0 = ggml_cpu_moe_dispatch_count();
            if (name[0] == 'H') {
                ggml_cpu_set_mxfp4_kernel(1, -1);   // = Fy2 (kernel 1 = 256-bit registers)
            } else {
                ggml_cpu_set_mxfp4_kernel(!is_f ? 0 : name.rfind("Fy", 0) == 0 ? 1 : 2, is_f ? f_prefetch(name) : 0);
            }
            ggml_cpu_set_moe_fuse(fuse);
            if (r8) {
                ggml_cpu_set_mxfp4_r8_act(name[0] == 'K' ? 1 : 0);
                for (int j = 0; j < 3; j++) {
                    if (!ggml_cpu_mxfp4_r8_register(pool_r8[j], (size_t) P.n_exp * SLICE)) { fprintf(stderr, "R8 register failed\n"); exit(2); }
                }
            }
            const int64_t fused0 = ggml_cpu_moe_fuse_count();
            const ggml_type type = GGML_TYPE_MXFP4;
#else
            uint8_t * const * pools = name == "D" ? pool_d : P.p;
            const bool repack = false;
            const ggml_type type = name == "D" ? GGML_TYPE_MXFP4_R8 : GGML_TYPE_MXFP4;
#endif
#ifndef KB_IK
            // a persistent threadpool, as common attaches one to the engine's context (common.cpp, common_threadpools)
            ggml_threadpool_params tpp = ggml_threadpool_params_default(nth);
            ggml_threadpool_t tp = ggml_threadpool_new(&tpp);
            ggml_backend_cpu_set_threadpool(backend, tp);
#endif
            ggml_backend_cpu_set_n_threads(backend, nth);
            {
                L2Graph G(backend, pools, P.n_exp, type, repack);
                std::vector<float> lay(N_EMBD);
                auto word = [&](int wi, std::vector<float> * res) {
                    for (int l = 0; l < N_LAYERS; l++) {
                        const size_t k = ((size_t) wi * N_LAYERS + l) * N_USED;
                        G.layer(backend, plan.x.data(), &plan.ids[k], &plan.w[k], res ? res->data() + (size_t) l * N_EMBD : nullptr);
                    }
                };
                word(words - 1, nullptr);
                if (prof) {
                    // stages inside the fused group are skipped when the fused step is on (a prefix ending at gate or up
                    // would run them unfused)
#ifndef KB_IK
                    const bool skip_inner = fuse;
#else
                    const bool skip_inner = false;
#endif
                    ggml_init_params ipg = { 8 * ggml_graph_overhead(), nullptr, true };
                    ggml_context * gctx = ggml_init(ipg);
                    for (auto & st : G.stages) {
                        if (skip_inner && (st.first == "gate" || st.first == "up")) continue;
                        ggml_cgraph * gs = ggml_new_graph(gctx);
                        ggml_build_forward_expand(gs, st.second);
                        auto wordg = [&](int wi) {
                            for (int l = 0; l < N_LAYERS; l++) {
                                const size_t k = ((size_t) wi * N_LAYERS + l) * N_USED;
                                G.layer(backend, plan.x.data(), &plan.ids[k], &plan.w[k], nullptr, gs);
                            }
                        };
                        wordg(words - 1);
                        const double t0 = now_s();
                        for (int wi = 0; wi < words; wi++) wordg(wi);
                        prof->push_back({ st.first, (now_s() - t0) * 1e3 / words });
                    }
                    ggml_free(gctx);
                    t = 0;
                } else if (out0) {
                    word(0, out0);
                } else {
                    const double t0 = now_s();
                    for (int wi = 0; wi < words; wi++) word(wi, nullptr);
                    t = now_s() - t0;
                }
            }
#ifndef KB_IK
            ggml_backend_cpu_set_threadpool(backend, nullptr);
            ggml_threadpool_free(tp);
            ggml_cpu_set_mxfp4_kernel(0, 0);
            ggml_cpu_set_moe_fuse(0);
            ggml_cpu_set_moe_dispatch(0);
            {   // the dispatch must have run on every layer's down op (the fused step covers gate/up), and only then
                const int64_t nd = ggml_cpu_moe_dispatch_count() - disp0;
                const int64_t wd = disp ? (int64_t) (1 + (out0 ? 1 : words)) * N_LAYERS : 0;
                if (!prof && nd != wd) { fprintf(stderr, "arm %s: %lld one-token dispatches, expected %lld\n", name.c_str(), (long long) nd, (long long) wd); exit(3); }
            }
            if (r8) {
                for (int j = 0; j < 3; j++) ggml_cpu_mxfp4_r8_unregister(pool_r8[j]);
                ggml_cpu_set_mxfp4_r8_act(0);
            }
            // the setting must have reached the op: G/H/QG/KG fused every layer, the others none
            const int64_t fused = ggml_cpu_moe_fuse_count() - fused0;
            const int64_t want  = fuse ? (int64_t) (1 + (out0 ? 1 : words)) * N_LAYERS : 0;
            if (!prof && fused != want) {   // (profile mode computes prefix graphs: other counts)
                fprintf(stderr, "arm %s: %lld fused steps, expected %lld\n", name.c_str(), (long long) fused, (long long) want);
                exit(3);
            }
#endif
            if (out0) return 0;
        }
        return t * 1e3 / words;
    };

    if (!o.profile.empty()) {
        // T100: reps x arms, rotated; per arm and stage the median prefix time and the stage's own cost
        const auto ptags = split(o.profile, ',');
        const int nth = o.threads[0];
        std::vector<std::vector<std::vector<std::pair<std::string, double>>>> all(ptags.size());
        for (int rep = 0; rep < o.reps; rep++) {
            for (size_t ai = 0; ai < ptags.size(); ai++) {
                const size_t a = (ai + rep) % ptags.size();
                std::vector<std::pair<std::string, double>> v;
                prof = &v;
                run_block(ptags[a], nth, o.words, nullptr);
                prof = nullptr;
                all[a].push_back(v);
            }
        }
        printf("PROFILE (%d threads, %d words x %d reps; ms per word, median over reps; prefix = every op up to the stage, cost = prefix - previous prefix)\n",
               nth, o.words, o.reps);
        for (size_t a = 0; a < ptags.size(); a++) {
            const size_t ns = all[a][0].size();
            double prev = 0;
            for (size_t s = 0; s < ns; s++) {
                std::vector<double> x;
                for (auto & r : all[a]) x.push_back(r[s].second);
                std::sort(x.begin(), x.end());
                const double med = x[x.size() / 2];
                printf("%-5s %-12s prefix %8.3f  cost %8.3f   [prefix min %.3f max %.3f]\n", ptags[a].c_str(), all[a][0][s].first.c_str(),
                       med, med - prev, x.front(), x.back());
                prev = med;
            }
        }
        ggml_backend_free(backend);
        return 0;
    }

    // ---- 1. word-0 outputs of every arm (8 threads), for the rounding check ----
    for (auto & tag : tags) {
        std::vector<float> res((size_t) N_LAYERS * N_EMBD);
        run_block(tag, 8, o.words, &res);
        write_f32(o.out_dir + "/out_" + tag + ".f32", res);
    }
    printf("word-0 outputs written to %s/out_<arm>.f32\n", o.out_dir.c_str());

    // R is the rounding reference only: not timed
    tags.erase(std::remove_if(tags.begin(), tags.end(), [](const std::string & t) { return t[0] == 'R'; }), tags.end());

    // ---- 2. timing: reps x (threads x arms), arms rotated every rep ----
    printf("\n%-4s %-5s %3s %5s %10s %8s\n", "rep", "arm", "thr", "words", "ms/word", "GB/s");
    std::vector<std::vector<double>> res(tags.size() * o.threads.size());
    for (int rep = 0; rep < o.reps; rep++) {
        for (size_t ti = 0; ti < o.threads.size(); ti++) {
            for (size_t ai = 0; ai < tags.size(); ai++) {
                const size_t a = (ai + rep) % tags.size();
                const double ms = run_block(tags[a], o.threads[ti], o.words, nullptr);
                res[a * o.threads.size() + ti].push_back(ms);
                printf("%-4d %-5s %3d %5d %10.3f %8.2f\n", rep, tags[a].c_str(), o.threads[ti], o.words, ms, WORD_BYTES / (ms * 1e-3) / 1e9);
                fflush(stdout);
            }
        }
    }

    printf("\nSUMMARY (ms per word = %d layers x 8 experts, %.1f MB; median of %d reps, [min, max])\n", N_LAYERS, WORD_BYTES / 1e6, o.reps);
    printf("%-5s %3s %10s %20s %8s\n", "arm", "thr", "median", "[min, max]", "GB/s");
    for (size_t a = 0; a < tags.size(); a++) {
        for (size_t ti = 0; ti < o.threads.size(); ti++) {
            auto v = res[a * o.threads.size() + ti];
            std::sort(v.begin(), v.end());
            const double med = v[v.size() / 2];
            printf("%-5s %3d %10.3f   [%7.3f, %7.3f] %8.2f\n", tags[a].c_str(), o.threads[ti], med, v.front(), v.back(),
                   WORD_BYTES / (med * 1e-3) / 1e9);
        }
    }
    ggml_backend_free(backend);
    return 0;
}
