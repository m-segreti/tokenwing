// GGML_OP_MOE_GROUP_TOPK on CUDA.
//
// The op replaces the ~11-kernel argsort chain that llm_graph_context::build_moe_topk builds for grouped routing (the
// idea of a fused grouped top-k is from ik_llama.cpp's ggml_cuda_op_grouped_topk / cuda_bailingmoev2_experts,
// ggml/src/ggml-cuda/argsort.cu there, MIT, Copyright (c) 2024-2025 The ik_llama.cpp authors; these kernels are written
// for our op and keep no code of theirs).
//
// The result must be the chain's to the bit, equal scores included (MOE_STREAM_TEST_FUSED_CHECK found 3 exact ties per 4k-token prompt on
// Flash, and a "lower/higher index first" rule changed the text). So:
//  - the group scores are the sum of the k_group largest values (k_group <= 2 on the fast path: v1 + v2 is the chain's
//    sum_rows result whatever the order of equal values);
//  - the best groups and the top k come from plain comparisons when the values involved are all different (then every
//    correct descending sort, the chain's bitonic argsort included, gives the same indices in the same order), and from
//    the very network of ggml's bitonic argsort (k_argsort_f32_i32 in argsort.cu) when two of them are equal.
// The CPU reference (ggml-cpu/ops.cpp, ggml_compute_forward_moe_group_topk) always runs that network, so the two backends
// agree too (test-backend-ops, with and without ties).
//
// One warp per row. The fast kernel gives each lane E = ne00/32 adjacent experts in registers (a group = n_per/E lanes):
// group top-2 by a few shuffles, group ranks by counting, the top k + 1 by merging the lanes' sorted lists (k + 1 rounds
// of a 5-step shuffle arg-max); the exact routine below is its fallback and the generic kernel for other shapes.

#include "moe-group-topk.cuh"

#include <climits>

static __device__ __forceinline__ bool moe_gtk_better(const float v, const int e, const float bv, const int be) {
    return v > bv || (v == bv && e > be);
}

// all lanes end with the same (v, e): the order is total and e is unique, so every butterfly gives the same winner
static __device__ __forceinline__ void moe_gtk_warp_best(float & v, int & e) {
#pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, v, off, WARP_SIZE);
        const int   oe = __shfl_xor_sync(0xffffffff, e, off, WARP_SIZE);
        if (moe_gtk_better(ov, oe, v, e)) {
            v = ov;
            e = oe;
        }
    }
}

// one compare-exchange of ggml's bitonic argsort (k_argsort_f32_i32 in argsort.cu, descending order)
static __device__ __forceinline__ void moe_gtk_bitonic_step(int * idx, const float * x, const int ncols, const int col,
        const int j, const int k) {
    const int ixj = col ^ j;
    if (ixj > col) {
        if ((col & k) == 0) {
            if (idx[col] >= ncols || (idx[ixj] < ncols && x[idx[col]] < x[idx[ixj]])) {
                const int t = idx[col]; idx[col] = idx[ixj]; idx[ixj] = t;
            }
        } else {
            if (idx[ixj] >= ncols || (idx[col] < ncols && x[idx[col]] > x[idx[ixj]])) {
                const int t = idx[col]; idx[col] = idx[ixj]; idx[ixj] = t;
            }
        }
    }
}

// the bitonic argsort network over x[0, ncols) into idx[0, ncols_pad): by lane 0 alone (lanes == 1) or by the warp
static __device__ void moe_gtk_bitonic(int * idx, const float * x, const int ncols, const int ncols_pad, const int lane,
        const int lanes) {
    if (lane < lanes) {
        for (int c = lane; c < ncols_pad; c += lanes) {
            idx[c] = c;
        }
    }
    __syncwarp();
    for (int kk = 2; kk <= ncols_pad; kk *= 2) {
        for (int j = kk/2; j > 0; j /= 2) {
            // the pairs (c, c ^ j) of one step are disjoint, so the lanes do not race
            if (lane < lanes) {
                for (int c = lane; c < ncols_pad; c += lanes) {
                    moe_gtk_bitonic_step(idx, x, ncols, c, j, kk);
                }
            }
            __syncwarp();
        }
    }
}

// the whole selection of one row with the chain's networks: exact and slow (~20 us for 512 experts)
static __device__ void moe_gtk_row_exact(
        const float * __restrict__ ar, const float * __restrict__ bias, int * __restrict__ out, float * smem,
        const int lane, const int ne00, const int ne00_pad, const int n_groups, const int n_groups_pad,
        const int n_group_used, const int k_group, const int k) {
    // ne00 scores, n_groups group scores, ne00_pad + n_groups_pad sort indices
    float * s    = smem;
    float * g    = smem + ne00;
    int   * idx  = (int *) (g + n_groups);
    int   * gidx = idx + ne00_pad;

    for (int e = lane; e < ne00; e += WARP_SIZE) {
        s[e] = bias ? ar[e] + bias[e] : ar[e];
    }
    __syncwarp();

    const int n_per = ne00 / n_groups;

    // group scores: the sum of the k_group largest, largest first
    for (int grp = 0; grp < n_groups; ++grp) {
        float tv  = INFINITY;
        int   te  = INT_MAX;
        float sum = 0.0f;
        for (int r = 0; r < k_group; ++r) {
            float bv = -INFINITY;
            int   be = INT_MIN;
            for (int e = grp*n_per + lane; e < (grp + 1)*n_per; e += WARP_SIZE) {
                const float v = s[e];
                if (moe_gtk_better(tv, te, v, e) && moe_gtk_better(v, e, bv, be)) {
                    bv = v;
                    be = e;
                }
            }
            moe_gtk_warp_best(bv, be);
            sum = r == 0 ? bv : sum + bv;
            tv  = bv;
            te  = be;
        }
        if (lane == 0) {
            g[grp] = sum;
        }
    }
    __syncwarp();

    moe_gtk_bitonic(gidx, g, n_groups, n_groups_pad, lane, 1);
    uint64_t keep = 0;
    for (int r = 0; r < n_group_used; ++r) {
        keep |= 1ull << gidx[r];
    }

    // the chain's fill(-inf) + set_rows of the kept groups, then its argsort
    for (int e = lane; e < ne00; e += WARP_SIZE) {
        if (!((keep >> (e / n_per)) & 1)) {
            s[e] = -INFINITY;
        }
    }
    __syncwarp();
    moe_gtk_bitonic(idx, s, ne00, ne00_pad, lane, WARP_SIZE);
    for (int r = lane; r < k; r += WARP_SIZE) {
        out[r] = idx[r];
    }
}

static __global__ void k_moe_group_topk(
        const float * __restrict__ a, const float * __restrict__ bias, const int64_t bias_rs, int * __restrict__ dst,
        const int ne00, const int ne00_pad, const int n_groups, const int n_groups_pad, const int n_group_used,
        const int k_group, const int k) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    // bias_rs = 0 - one bias for every row; ne00 - a bias per row (the combined guess's N targets)
    moe_gtk_row_exact(a + (int64_t) row*ne00, bias ? bias + row*bias_rs : nullptr, dst + (int64_t) row*k, smem, threadIdx.x,
            ne00, ne00_pad, n_groups, n_groups_pad, n_group_used, k_group, k);
}

// fast path: E = ne00/32 experts per lane, n_per % E == 0 (a lane is inside one group), k_group <= 2, n_groups <= 32,
// k + 1 <= KMAX
template <int E, int KMAX>
// The early start (PDL, as cpy_small_rows): it lets the next kernel (the ids copy) launch at once and waits
// for the kernel before it (the sigmoid) before it reads; launched with the PDL attribute (ggml_cuda_kernel_launch), so it can itself start
// while the sigmoid ends. Plain pointers, no __restrict__: PDL and __restrict__ must not be mixed (common.cuh; GGML_CUDA_RESTRICT differs
// between nvcc's host and device passes, so a template kernel's parameters cannot carry it - as cpy_small_rows, plain)
static __global__ void k_moe_group_topk_fast(
        const float * a, const float * bias_all, const int64_t bias_rs, int * dst,
        const int ne00, const int ne00_pad, const int n_groups, const int n_groups_pad, const int n_group_used,
        const int k_group, const int k) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    extern __shared__ float smem[];

    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    const float * bias = bias_all ? bias_all + row*bias_rs : nullptr;   // as k_moe_group_topk

    const float * ar  = a   + (int64_t) row*ne00;
    int         * out = dst + (int64_t) row*k;

    const int n_per = ne00 / n_groups;
    const int L     = n_per / E;        // lanes per group, a power of 2
    const int e0    = lane*E;
    const int grp   = e0 / n_per;

    float v[E];
#pragma unroll
    for (int i = 0; i < E; ++i) {
        v[i] = bias ? ar[e0 + i] + bias[e0 + i] : ar[e0 + i];
    }

    // group scores: the 2 largest values of the lane, merged over the group's lanes
    float m1 = -INFINITY;
    float m2 = -INFINITY;
#pragma unroll
    for (int i = 0; i < E; ++i) {
        const float x = v[i];
        if (x > m1) {
            m2 = m1;
            m1 = x;
        } else if (x > m2) {
            m2 = x;
        }
    }
    for (int off = 1; off < L; off <<= 1) {
        const float o1 = __shfl_xor_sync(0xffffffff, m1, off, WARP_SIZE);
        const float o2 = __shfl_xor_sync(0xffffffff, m2, off, WARP_SIZE);
        const float n2 = m1 >= o1 ? fmaxf(m2, o1) : fmaxf(m1, o2);
        m1 = fmaxf(m1, o1);
        m2 = n2;
    }
    const float gsum = k_group == 1 ? m1 : m1 + m2;

    // the best groups: by rank when all group scores differ, else with the chain's network
    float * g = smem + ne00;
    if (lane % L == 0) {
        g[grp] = gsum;
    }
    __syncwarp();
    int  rank = 0;
    bool geq  = false;
    if (lane < n_groups) {
        const float gl = g[lane];
        for (int m = 0; m < n_groups; ++m) {
            rank += g[m] > gl;
            geq   = geq || (m != lane && g[m] == gl);
        }
    }
    uint32_t keep;
    if (__any_sync(0xffffffff, geq)) {
        int * gidx = (int *) (g + n_groups) + ne00_pad;
        moe_gtk_bitonic(gidx, g, n_groups, n_groups_pad, lane, 1);
        keep = 0;
        for (int r = 0; r < n_group_used; ++r) {
            keep |= 1u << gidx[r];
        }
    } else {
        keep = __ballot_sync(0xffffffff, lane < n_groups && rank < n_group_used);
    }
    const bool kept = (keep >> grp) & 1;

    // the lane's own top KMAX of its kept experts, sorted (value, then index: only a tie can make the second matter, and
    // a tie takes the exact path)
    float lv[KMAX];
    int   li[KMAX];
#pragma unroll
    for (int t = 0; t < KMAX; ++t) {
        lv[t] = -INFINITY;
        li[t] = INT_MIN;
    }
    if (kept) {
#pragma unroll
        for (int i = 0; i < E; ++i) {
            float x  = v[i];
            int   xi = e0 + i;
#pragma unroll
            for (int t = 0; t < KMAX; ++t) {
                if (x > lv[t]) {
                    const float tvv = lv[t]; lv[t] = x;  x  = tvv;
                    const int   tii = li[t]; li[t] = xi; xi = tii;
                }
            }
        }
    }

    // the top k + 1 over the warp: k + 1 rounds of a shuffle arg-max over the lanes' heads
    bool  tie  = false;
    int   pick = INT_MIN;   // lane r keeps the r-th index
    float tv   = INFINITY;
    for (int r = 0; r <= k; ++r) {
        float bv = lv[0];
        int   be = li[0];
        moe_gtk_warp_best(bv, be);
        tie = tie || (r > 0 && bv == tv);
        if (r < k && r == lane) {
            pick = be;
        }
        if (li[0] == be && be != INT_MIN) {
            // this lane's head was taken: pop it
#pragma unroll
            for (int t = 0; t < KMAX - 1; ++t) {
                lv[t] = lv[t + 1];
                li[t] = li[t + 1];
            }
            lv[KMAX - 1] = -INFINITY;
            li[KMAX - 1] = INT_MIN;
        }
        tv = bv;
    }

    if (!tie) {
        if (lane < k) {
            out[lane] = pick;
        }
        return;
    }
    __syncwarp();
    moe_gtk_row_exact(ar, bias, out, smem, lane, ne00, ne00_pad, n_groups, n_groups_pad, n_group_used, k_group, k);
}

static int moe_gtk_next_pow2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

void ggml_cuda_op_moe_group_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(dst));

    const int n_groups     = ggml_get_op_params_i32(dst, 0);
    const int n_group_used = ggml_get_op_params_i32(dst, 1);
    const int k_group      = ggml_get_op_params_i32(dst, 2);
    const int k            = (int) dst->ne[0];
    const int ne00         = (int) src0->ne[0];
    const int64_t nrows    = ggml_nrows(src0);

    if (nrows == 0) {
        return;
    }

    const int ne00_pad     = moe_gtk_next_pow2(ne00);
    const int n_groups_pad = moe_gtk_next_pow2(n_groups);
    const size_t smem = (size_t) (ne00 + n_groups) * sizeof(float) + (size_t) (ne00_pad + n_groups_pad) * sizeof(int);

    const float * a    = (const float *) src0->data;
    const float * bias = src1 ? (const float *) src1->data : nullptr;
    // A bias of one row is for every row, a bias with a row per row of src0 is per row (ggml_moe_group_topk)
    const int64_t bias_rs = src1 && ggml_nelements(src1) != ne00 ? ne00 : 0;
    int         * d    = (int *) dst->data;
    cudaStream_t  st   = ctx.stream();

    constexpr int KMAX = 17;   // the guesses take up to 16 (+1 to see a tie at the boundary)
    const int  E      = ne00 / WARP_SIZE;
    const int  n_per  = ne00 / n_groups;
    const int  L      = E > 0 && n_per % E == 0 ? n_per / E : 0;
    const bool fast   = ne00 % WARP_SIZE == 0 && L > 0 && (L & (L - 1)) == 0 && n_groups <= 32 && k_group <= 2 &&
                        k + 1 <= KMAX;

    const ggml_cuda_kernel_launch_params gtk_params = ggml_cuda_kernel_launch_params((dim3) (unsigned) nrows, WARP_SIZE, smem, st);
#define MOE_GTK_FAST(EE) \
    ggml_cuda_kernel_launch(k_moe_group_topk_fast<EE, KMAX>, gtk_params, \
        a, bias, bias_rs, d, ne00, ne00_pad, n_groups, n_groups_pad, n_group_used, k_group, k)

    if (fast && E == 1) {
        MOE_GTK_FAST(1);
    } else if (fast && E == 2) {
        MOE_GTK_FAST(2);
    } else if (fast && E == 4) {
        MOE_GTK_FAST(4);
    } else if (fast && E == 8) {
        MOE_GTK_FAST(8);
    } else if (fast && E == 16) {
        MOE_GTK_FAST(16);
    } else if (fast && E == 32) {
        MOE_GTK_FAST(32);
    } else {
        k_moe_group_topk<<<(unsigned) nrows, WARP_SIZE, smem, st>>>(
            a, bias, bias_rs, d, ne00, ne00_pad, n_groups, n_groups_pad, n_group_used, k_group, k);
    }
#undef MOE_GTK_FAST
    CUDA_CHECK(cudaGetLastError());
}
