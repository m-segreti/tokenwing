// What ik_llama.cpp's expert kernels need that our ggml does not have, kept inside this port (no new public ggml types):
// ik's interleaved block layouts (generated from its ggml-common.h, MIT, Copyright (C) Iwan Kawrakow) and a private number for each
// of its own types. These numbers are switch labels only; a ggml_tensor never has one of these types.
#pragma once
#include "ggml-common.h"
#include <assert.h>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#include "ggml-cpu-impl.h"

#define GGML_TYPE_Q2_K_R4 1000
#define GGML_TYPE_Q3_K_R4 1001
#define GGML_TYPE_Q4_K_R4 1002
#define GGML_TYPE_Q5_K_R4 1003
#define GGML_TYPE_Q6_K_R4 1004
#define GGML_TYPE_IQ2_S_R4 1005
#define GGML_TYPE_IQ2_XS_R4 1006
#define GGML_TYPE_IQ2_XXS_R4 1007
#define GGML_TYPE_IQ3_KS_R16 1008
#define GGML_TYPE_IQ3_S_R4 1009
#define GGML_TYPE_IQ3_XXS_R4 1010
#define GGML_TYPE_IQ4_KS_R16 1011
#define GGML_TYPE_IQ4_NL_R4 1012
#define GGML_TYPE_IQ4_XS_R8 1013
#define GGML_TYPE_IQ5_KS_R16 1014
#define GGML_TYPE_MXFP4_R8 1015
#define GGML_TYPE_Q2_0_R8 1016
#define GGML_TYPE_Q4_0_R8 1017
#define GGML_TYPE_Q5_0_R4 1018
#define GGML_TYPE_Q6_0 1019
#define GGML_TYPE_Q6_0_R4 1020
#define GGML_TYPE_Q8_0_R16 1021
#define GGML_TYPE_Q8_0_R8 1022
#define GGML_TYPE_Q8_0_X4 1023
#define GGML_TYPE_Q8_1_X4 1024
#define GGML_TYPE_Q8_2_X4 1025
#define GGML_TYPE_Q8_K32 1026
#define GGML_TYPE_Q8_K_R16 1027
#define GGML_TYPE_Q8_K_R8 1028
#define GGML_TYPE_Q8_KR8 1029
#define GGML_TYPE_Q8_KV 1030
#define GGML_TYPE_Q8_KV_R8 1031

#define GGML_Q8_K_R16_ALIGN 64
#define QK1_0_G128 128
#define QK1_0_G128_R8_ROWS 8
#define QK2_0_R8_ROWS 8
#define QK6_0 32
#define QK8_2 32
#define QK_IQ1BN 64
#define QK_IQ2BN 64
#define QK_PQ2_0 128
#define QK_PQ2_0_R8_ROWS 8
#define QK_PTQ1_0 128
#define QK_PTQ1_0_R8_ROWS 8
#define QK_TQ2_T 128

typedef struct {
    ggml_half d[QK2_0_R8_ROWS];
    uint8_t   qs[QK2_0_R8_ROWS*(QK2_0/4)];
} block_q2_0_r8;
static_assert(sizeof(block_q2_0_r8) == QK2_0_R8_ROWS*sizeof(block_q2_0), "wrong q2_0_r8 block size/padding");
typedef struct {
    uint8_t e[8]; // E8M0
    uint8_t qs[4*QK_MXFP4];
} block_mxfp4_r8;
static_assert(sizeof(block_mxfp4_r8) == 8*sizeof(block_mxfp4), "wrong mxfp4_r8 block size/padding");
typedef struct {
    ggml_half d[4];        // delta
    uint8_t qh[QK5_0/2];   // 5-th bit of quants
    uint8_t qs[QK5_0*2];   // nibbles / quants
} block_q5_0_r4;
static_assert(sizeof(block_q5_0_r4) == 4*sizeof(ggml_half) + QK5_0*2 + QK5_0/2, "wrong q5_0_r4 block size/padding");
typedef struct {
    ggml_half d;         // delta
    uint8_t qh[QK6_0/4]; // 5+6-th bit of quants
    uint8_t qs[QK6_0/2]; // nibbles / quants
} block_q6_0;
static_assert(sizeof(block_q6_0) == sizeof(ggml_half) + QK6_0/2 + QK6_0/4, "wrong q6_0 block size/padding");
typedef struct {
    ggml_half d[4];      // delta
    uint8_t qh[QK6_0];   // 5+6-th bit of quants
    uint8_t qs[QK6_0*2]; // nibbles / quants
} block_q6_0_r4;
static_assert(sizeof(block_q6_0_r4) == 4*sizeof(ggml_half) + QK6_0*2 + QK6_0, "wrong q6_0_r4 block size/padding");
typedef struct {
    ggml_half d[8];
    int8_t qs[4*QK8_1];
} block_q8_1_x4;
static_assert(sizeof(block_q8_1_x4) == 4*sizeof(block_q8_1), "wrong q8_1_x4 block size/padding");
typedef struct {
    ggml_half d[4];
    int8_t qs[4*QK8_0];
} block_q8_0_x4;
static_assert(sizeof(block_q8_0_x4) == 4*sizeof(block_q8_0), "wrong q8_0_x4 block size/padding");
typedef struct {
    ggml_half d[8];
    int8_t qs[8*QK8_0];
} block_q8_0_r8;
static_assert(sizeof(block_q8_0_r8) == 8*sizeof(block_q8_0), "wrong q8_0_r8 block size/padding");
typedef struct {
    float   d[16];
    uint8_t qs[16*QK8_0];
} block_q8_0_r16;
static_assert(sizeof(block_q8_0_r16) == 16*sizeof(float) + 16*QK8_0, "wrong q8_0_r16 block size/padding");
typedef struct {
    ggml_half d[4];        // deltas for 4 q4_0 blocks
    uint8_t qs[QK4_0 * 2]; // nibbles / quants for 4 q4_0 blocks
} block_q4_0x4;
static_assert(sizeof(block_q4_0x4) == 4 * sizeof(ggml_half) + QK4_0 * 2, "wrong q4_0x4 block size/padding");
typedef struct {
    ggml_half d[8];        // deltas for 8 q4_0 blocks
    uint8_t qs[QK4_0 * 4]; // nibbles / quants for 8 q4_0 blocks
} block_q4_0x8;
static_assert(sizeof(block_q4_0x8) == 8 * sizeof(ggml_half) + QK4_0 * 4, "wrong q4_0x8 block size/padding");
typedef struct {
    ggml_half d[4];        // deltas for 4 q8_0 blocks
    int8_t qs[QK8_0 * 4];  // quants for 4 q8_0 blocks
} block_q8_0x4;
static_assert(sizeof(block_q8_0x4) == 4 * sizeof(ggml_half) + QK8_0 * 4, "wrong q8_0x4 block size/padding");
typedef struct {
    ggml_half d[8];        // deltas for 8 q8_0 blocks
    int8_t qs[QK8_0 * 8];  // quants for 8 q8_0 blocks
} block_q8_0x8;
static_assert(sizeof(block_q8_0x8) == 8 * sizeof(ggml_half) + QK8_0 * 8, "wrong q8_0x8 block size/padding");
typedef struct {
    uint16_t d;
    uint16_t s;
    int8_t qs[QK8_2]; // quants
} block_q8_2;
static_assert(sizeof(block_q8_2) == sizeof(ggml_half) + sizeof(int16_t) + QK8_2, "wrong q8_2 block size/padding");
typedef struct {
    uint16_t d[8];
    int8_t qs[4*QK8_2];
} block_q8_2_x4;
static_assert(sizeof(block_q8_2_x4) == 4*sizeof(block_q8_2), "wrong q8_2_x4 block size/padding");
typedef struct {
    ggml_half d[8];
    uint8_t scales[QK_K/4]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K];       // quants
} block_q2_k_r4;
static_assert(sizeof(block_q2_k_r4) == 8*sizeof(ggml_half) + QK_K/4 + QK_K, "wrong q2_k_r4 block size/padding");
typedef struct {
    ggml_half d[4];             // super-block scales
    uint8_t scales_h[QK_K/16];  // scales quantized with 6 bits (high 2 bits)
    uint8_t scales_l[QK_K/8];   // scales quantized with 6 bits (low  4 bits)
    uint8_t qh[QK_K/2];         // quants - high bit
    uint8_t qs[QK_K];           // quants - low 2 bits
} block_q3_k_r4;
static_assert(sizeof(block_q3_k_r4) == 4*sizeof(ggml_half) + QK_K/16 + QK_K/8 + QK_K/2 + QK_K, "wrong q3_k_r4 block size/padding");
typedef struct {
    ggml_half d[8];
    uint8_t scales_h[QK_K/16];// scales and mins, quantized with 6 bits
    uint8_t scales_l[QK_K/8]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K*2];           // 4--bit quants
} block_q4_k_r4;
static_assert(sizeof(block_q4_k_r4) == 8*sizeof(ggml_half) + QK_K/16 + QK_K/8 + QK_K*2, "wrong q4_k_r4 block size/padding");
typedef struct {
    ggml_half d[8];
    uint8_t scales_h[QK_K/16];// scales and mins, quantized with 6 bits
    uint8_t scales_l[QK_K/8]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/2];           // quants, high bit
    uint8_t qs[QK_K*2];           // quants, low 4 bits
} block_q5_k_r4;
static_assert(sizeof(block_q5_k_r4) == 8*sizeof(ggml_half) + QK_K/16 + QK_K/8 + QK_K/2 + QK_K*2, "wrong q5_k_r4 block size/padding");
typedef struct {
    ggml_half d[4];          // super-block scale
    int8_t  scales[QK_K/4];  // scales, quantized with 8 bits
    uint8_t qh[QK_K];        // quants, upper 2 bits
    uint8_t ql[QK_K*2];      // quants, lower 4 bits
} block_q6_k_r4;
static_assert(sizeof(block_q6_k_r4) == 4*sizeof(ggml_half) + QK_K/4 + 3*QK_K, "wrong q6_k_r4 block size/padding");
typedef struct {
    float   d;            // delta
    int8_t  qs[64];       // quants
} block_q8_K64;
static_assert(sizeof(block_q8_K64) == sizeof(float) + 64, "wrong q8_K64 block size/padding");
typedef struct {
    float   d;              // delta
    int32_t s;              // bsums[0] + bsums[1] + bsums[2] + bsums[3]
    int16_t bsums[4];       // quant sums for blocks of 32
    int8_t  qs[128];        // quants
} block_q8_K128;
static_assert(sizeof(block_q8_K128) == sizeof(float) + sizeof(int32_t) + 4*sizeof(int16_t) + 128, "wrong q8_K128 block size/padding");
typedef struct {
    ggml_half d[8];         // delta
    int8_t    qs[8*QK_K];   // quants, stored as unsigned ints
} block_q8_k_r8;
static_assert(sizeof(block_q8_k_r8) == 8*sizeof(ggml_half) + 8*QK_K, "wrong q8_k_r8 block size/padding");
typedef struct {
    float     d[16];         // delta
    int8_t    qs[16*QK_K];   // quants, stored as unsigned ints
} block_q8_k_r16;
static_assert(16*sizeof(float) == GGML_Q8_K_R16_ALIGN, "q8_k_r16 header must be exactly one alignment unit");
typedef struct {
    ggml_half d[4];
    uint8_t   sas[QK_K/2];
    uint8_t   qs[QK_K/2];
} block_iq2_xxs_r4;
static_assert(sizeof(block_iq2_xxs_r4) == 4*sizeof(block_iq2_xxs), "wrong iq2_xxs_r4 block size/padding");
typedef struct {
    ggml_half d[4];
    uint16_t qs[QK_K/2];
    uint8_t  scales[QK_K/8];
} block_iq2_xs_r4;
static_assert(sizeof(block_iq2_xs_r4) == 4*sizeof(block_iq2_xs), "wrong iq2_xs_r4 block size/padding");
typedef struct {
    ggml_half d[4];
    uint8_t qs[QK_K/2];
    uint8_t qh[QK_K/8];
    uint8_t signs[QK_K/2];
    uint8_t scales[QK_K/8];
} block_iq2_s_r4;
static_assert(sizeof(block_iq2_s_r4) == 4*sizeof(block_iq2_s), "wrong iq2_s_r4 block size/padding");
typedef struct {
    ggml_half d[4];
    uint8_t   sas[QK_K/2];
    uint8_t   qs[QK_K];
} block_iq3_xxs_r4;
static_assert(sizeof(block_iq3_xxs_r4) == 4*sizeof(block_iq3_xxs), "wrong iq3_xxs_r4 block size/padding");
typedef struct {
    ggml_half d[4];
    uint8_t qs[QK_K];
    uint8_t qh[QK_K/8];
    uint8_t signs[QK_K/2];
    uint8_t scales[4*IQ3S_N_SCALE];
} block_iq3_s_r4;
static_assert(sizeof(block_iq3_s_r4) == 4*sizeof(block_iq3_s), "wrong iq3_s_r4 block size/padding");
typedef struct {
    uint8_t  qs[16];
    uint16_t qh[4];
} block_iq1_s_r4;
static_assert(sizeof(block_iq1_s_r4) == 24, "wrong iq1_s_r4 block size/padding");
typedef struct {
    uint8_t  qs[16];     // grid index, low 8 bits
    uint8_t  qh[ 8];     // grid index, high 3 bits + grid shift bits (for two groups of 8)
    uint8_t  scales[4];  // 4-bit block scales
} block_iq1_m_r4;
static_assert(sizeof(block_iq1_m_r4) == 28, "wrong iq1_m_r4 block size/padding");
typedef struct {
    ggml_half  d;
    uint8_t    qs[QK1_0_G128 / 8];
} block_q1_0_g128;
static_assert(sizeof(block_q1_0_g128) == sizeof(ggml_half) + QK1_0_G128 / 8, "wrong q1_0_g128 block size/padding");
typedef struct {
    ggml_half d[QK1_0_G128_R8_ROWS];
    uint8_t   qs[QK1_0_G128];
} block_q1_0_g128_r8;
static_assert(sizeof(block_q1_0_g128_r8) == QK1_0_G128_R8_ROWS*sizeof(ggml_half) + QK1_0_G128, "wrong q1_0_g128_r8 block size/padding");
typedef struct {
    ggml_half d;                  // delta (scale)
    uint8_t   qs[QK_PQ2_0 / 4];   // 2 bits per weight
} block_pq2_0;
static_assert(sizeof(block_pq2_0) == sizeof(ggml_half) + QK_PQ2_0 / 4, "wrong pq2_0 block size/padding");
typedef struct {
    uint8_t   qs[(QK_PTQ1_0 - 4*QK_PTQ1_0/64)/5]; // 5 base-3 trits per byte
    uint8_t   qh[QK_PTQ1_0/64];                    // high trits
    ggml_half d;                                   // delta (scale)
} block_ptq1_0;
static_assert(sizeof(block_ptq1_0) == 28, "wrong ptq1_0 block size/padding");
typedef struct {
    ggml_half d[QK_PQ2_0_R8_ROWS];
    uint8_t   qs[QK_PQ2_0_R8_ROWS*(QK_PQ2_0/4)];
} block_pq2_0_r8;
static_assert(sizeof(block_pq2_0_r8) == QK_PQ2_0_R8_ROWS*sizeof(block_pq2_0), "wrong pq2_0_r8 block size/padding");
typedef struct {
    uint8_t ql[12];
    uint8_t extra;
} block_iq1_bn;
static_assert(sizeof(block_iq1_bn) == 13, "wrong iq1_bn block size/padding");
typedef struct {
    ggml_half    d[QK_PTQ1_0_R8_ROWS];
    block_iq1_bn qs[QK_PTQ1_0_R8_ROWS*QK_PTQ1_0/QK_IQ1BN];
} block_ptq1_0_r8;
static_assert(sizeof(block_ptq1_0_r8) == QK_PTQ1_0_R8_ROWS*sizeof(block_ptq1_0), "wrong ptq1_0_r8 block size/padding");
typedef struct {
    ggml_half d;              // scale
    uint8_t   qs[4*6];        // 32 steps x 6 bits
} block_tqk6;
static_assert(sizeof(block_tqk6) == sizeof(ggml_half) + 4*6, "wrong tqk6 block size/padding");
typedef struct {
    ggml_half d;              // scale
    uint8_t   qs[4*7];        // 32 steps x 7 bits
} block_tqk7;
static_assert(sizeof(block_tqk7) == sizeof(ggml_half) + 4*7, "wrong tqk7 block size/padding");
typedef struct {
    ggml_half d;              // scale
    uint8_t   qs[QK_TQ2_T/4]; // one trellis byte per 4 weights
} block_tq2_t;
static_assert(sizeof(block_tq2_t) == sizeof(ggml_half) + QK_TQ2_T/4, "wrong tq2_t block size/padding");
typedef struct {
    ggml_half d;              // scale
    uint8_t   qs[8*5];        // 64 steps x 5 bits
} block_tqp5;
static_assert(sizeof(block_tqp5) == sizeof(ggml_half) + 8*5, "wrong tqp5 block size/padding");
typedef struct {
    ggml_half d;              // scale
    uint8_t   qs[8*6];        // 64 steps x 6 bits
} block_tqp6;
static_assert(sizeof(block_tqp6) == sizeof(ggml_half) + 8*6, "wrong tqp6 block size/padding");
typedef struct {
    ggml_half d;              // scale
    uint8_t   qs[8*7];        // 64 steps x 7 bits
} block_tqp7;
static_assert(sizeof(block_tqp7) == sizeof(ggml_half) + 8*7, "wrong tqp7 block size/padding");
typedef struct {
    uint8_t qs[QK_IQ2BN/4];
} block_iq2_bn;
static_assert(sizeof(block_iq2_bn) == QK_IQ2BN/4, "wrong iq2_bn block size/padding");
typedef struct {
    ggml_half d[4];
    uint8_t qs[2*QK4_NL];
} block_iq4_nl_r4;
static_assert(sizeof(block_iq4_nl_r4) == 4*sizeof(ggml_half) + 2*QK4_NL, "wrong iq4_nl_r4 block size/padding");
typedef struct {
    ggml_half d[8];
    uint8_t qs[4*QK4_NL];
} block_iq4_nl_r8;
static_assert(sizeof(block_iq4_nl_r8) == 8*sizeof(ggml_half) + 4*QK4_NL, "wrong iq4_nl_r8 block size/padding");
typedef struct {
    ggml_half d[8];
    uint8_t scales_h[QK_K/16];
    uint8_t scales_l[QK_K/ 8];
    uint8_t qs[QK_K*4];
} block_iq4_xs_r8;
static_assert(sizeof(block_iq4_xs_r8) == 8*sizeof(block_iq4_xs), "wrong iq4_xs_rs block size/padding");
typedef struct {
    uint8_t  scales[QK_K/32];
    uint8_t  qs[QK_K/2];
} block_iq4_ks;
static_assert(sizeof(block_iq4_ks) == QK_K/32 + QK_K/2, "wrong iq4_ks block size/padding");
typedef struct {
    uint8_t  scales[QK_K/8];
    uint8_t  qs[QK_K*2];
} block_iq4_ks_r4;
static_assert(sizeof(block_iq4_ks_r4) == 4*sizeof(block_iq4_ks), "wrong iq4_ks_r4 block size/padding");
typedef struct {
    uint8_t  scales[16];
    uint8_t  qs[8*QK8_0];
} block_iq4_ks_r16;
static_assert(sizeof(block_iq4_ks_r16) == 16 + 8*QK8_0, "wrong iq4_ks_r16 block size/padding");
typedef struct {
    uint32_t qs[QK_K/8];
} block_iq4_kss;
static_assert(sizeof(block_iq4_kss) == QK_K/8*sizeof(uint32_t), "wrong iq4_kss block size/padding");
typedef struct {
    ggml_half d;
    uint16_t extra;
    uint8_t  scales[QK_K/32];
    uint8_t  qs[QK_K/4];
} block_iq2_k;
static_assert(sizeof(block_iq2_k) == sizeof(ggml_half) + sizeof(uint16_t) + QK_K/32 + QK_K/4, "wrong iq2_k block size/padding");
typedef struct {
    uint16_t scales_h;
    uint8_t  scales_l[QK_K/64];
    uint8_t  qs[QK_K/4];
    uint8_t  qh[QK_K/16];
} block_iq2_kl;
static_assert(sizeof(block_iq2_kl) == sizeof(uint16_t) + QK_K/64 + QK_K/4 + QK_K/16, "wrong iq2_kl block size/padding");
typedef struct {
    ggml_half d[4];
    uint8_t  extra[8];
    uint8_t  scales[QK_K/8];
    uint8_t  qs[QK_K];
} block_iq2_k_r4;
static_assert(sizeof(block_iq2_k_r4) == 4*sizeof(block_iq2_k), "wrong iq2_k_r4 block size/padding");
typedef struct {
    uint16_t extra;
    uint8_t  scales[QK_K/64];
    uint8_t  qs[QK_K/4];
} block_iq2_ks;
static_assert(sizeof(block_iq2_ks) == sizeof(uint16_t) + QK_K/64 + QK_K/4, "wrong iq2_ks block size/padding");
typedef struct {
    uint8_t  sh[QK_K/32]; // 4-bit scales + 13th bits for groups of 8
    uint8_t  ql[QK_K/8];  //  low 8 bits for groups of 8
    uint8_t  qh[QK_K/16]; // high 4 bits for groups of 8
} block_iq1_kt;
static_assert(sizeof(block_iq1_kt) == QK_K/8 + QK_K/16 + QK_K/32, "wrong iq1_kt block size/padding");
typedef struct {
    uint8_t  scales[QK_K/64];
    uint8_t  ql[QK_K/4];
} block_iq2_kt;
static_assert(sizeof(block_iq2_kt) == QK_K/4 + QK_K/64, "wrong iq2_kt block size/padding");
typedef struct {
    uint8_t  scales[QK_K/64];
    uint8_t  ql[QK_K/4];
    uint8_t  qh[QK_K/8];
} block_iq3_kt;
static_assert(sizeof(block_iq3_kt) == QK_K/4 + QK_K/8 + QK_K/64, "wrong iq3_kt block size/padding");
typedef struct {
    uint32_t qs[QK_K/8];
} block_iq4_kt;
static_assert(sizeof(block_iq4_kt) == QK_K/2, "wrong iq4_kt block size/padding");
typedef struct {
    ggml_half d;
    uint16_t extra;
    uint16_t scales_h;
    uint8_t scales_l[QK_K/32];
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/8];
} block_iq3_k;
static_assert(sizeof(block_iq3_k) == sizeof(ggml_half) + 2*sizeof(uint16_t) + QK_K/32 + QK_K/4 + QK_K/8, "wrong iq3_k block size/padding");
typedef struct {
    uint16_t extra;
    uint8_t scales[QK_K/64];
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/8];
} block_iq3_ks;
static_assert(sizeof(block_iq3_ks) == sizeof(uint16_t) + QK_K/64 + QK_K/4 + QK_K/8, "wrong iq3_ks block size/padding");
typedef struct {
    uint32_t extra;      // bit r: scale bit 4 of row r; bit 16+r: codebook page of row r
    uint8_t  scales[8];  // low nibble: scale bits 0..3 of rows 0..7; high nibble: rows 8..15
    uint8_t  qs[128];    // 2 low bits of the 512 quants
    uint8_t  qh[64];     // high bit of the 512 quants
} block_iq3_ks_r16;
static_assert(sizeof(block_iq3_ks_r16) == 204, "wrong iq3_ks_r16 block size/padding");
typedef struct {
    ggml_half d[4];
    uint8_t extra[8];
    uint8_t scales_h[QK_K/32];
    uint8_t scales_l[QK_K/8];
    uint8_t qs[QK_K];
    uint8_t qh[QK_K/2];
} block_iq3_k_r4;
static_assert(sizeof(block_iq3_k_r4) == 4*sizeof(block_iq3_k), "wrong iq3_k_r4 block size/padding");
typedef struct {
    ggml_half d;
    uint16_t extra;
    uint8_t  scales_h[QK_K/64];
    uint8_t  scales_l[QK_K/32];
    uint8_t  qs[QK_K/2];
} block_iq4_k;
static_assert(sizeof(block_iq4_k) == sizeof(ggml_half) + sizeof(uint16_t) + QK_K/2 + 3*QK_K/64, "wrong iq4_k block size/padding");
typedef struct {
    ggml_half d[4];
    uint8_t  extra[8];
    uint8_t  scales_h[QK_K/16];
    uint8_t  scales_l[QK_K/8];
    uint8_t  qs[QK_K*2];
} block_iq4_k_r4;
static_assert(sizeof(block_iq4_k_r4) == 4*sizeof(block_iq4_k), "wrong iq4_k_r4 block size/padding");
typedef struct {
    ggml_half d;
    uint16_t extra;
    uint8_t  scales_h[QK_K/64];
    uint8_t  scales_l[QK_K/32];
    uint8_t  qs[QK_K/2];
    uint8_t  qh[QK_K/8];
} block_iq5_k;
static_assert(sizeof(block_iq5_k) == sizeof(ggml_half) + sizeof(uint16_t) + QK_K/2 + QK_K/8 + 3*QK_K/64, "wrong iq5_k block size/padding");
typedef struct {
    ggml_half d[4];
    uint8_t  extra[8];
    uint8_t  scales_h[QK_K/16];
    uint8_t  scales_l[QK_K/8 ];
    uint8_t  qs[QK_K*2];
    uint8_t  qh[QK_K/2];
} block_iq5_k_r4;
static_assert(sizeof(block_iq5_k_r4) == 4*sizeof(block_iq5_k), "wrong iq5_k_r4 block size/padding");
typedef struct {
    ggml_half d;
    uint16_t extra;
    int8_t   scales[QK_K/16];
    uint8_t  qs[QK_K/2];
    uint8_t  qh[QK_K/4];
} block_iq6_k;
static_assert(sizeof(block_iq6_k) == sizeof(ggml_half) + sizeof(uint16_t) + QK_K/2 + QK_K/4 + QK_K/16, "wrong iq6_k block size/padding");
typedef struct {
    uint8_t  scales[QK_K/32];
    uint8_t  qs[QK_K/2];
    uint8_t  qh[QK_K/8];
} block_iq5_ks;
static_assert(sizeof(block_iq5_ks) == QK_K/32 + QK_K/2 + QK_K/8, "wrong iq5_ks block size/padding");
typedef struct {
    uint8_t  scales[QK_K/8];
    uint8_t  qs[QK_K*2];
    uint8_t  qh[QK_K/2];
} block_iq5_ks_r4;
static_assert(sizeof(block_iq5_ks_r4) == 4*sizeof(block_iq5_ks), "wrong iq5_ks_r4 block size/padding");
typedef struct {
    uint8_t  scales[16];
    uint8_t  qs[8*QK8_0];
    uint8_t  qh[2*QK8_0];
} block_iq5_ks_r16;
static_assert(sizeof(block_iq5_ks_r16) == 16 + 10*QK8_0, "wrong iq5_ks_r16 block size/padding");

// ik's activation block: our block_q8_K plus the sum of all its quants (its kernels read it); see iqk_port.cpp, which fills these rows
typedef struct {
    float   d;              // delta
    float   sum;            // sum of quants in the entire block
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K_ik;
static_assert(sizeof(block_q8_K_ik) == 2*sizeof(float) + QK_K + QK_K/16*sizeof(int16_t), "wrong q8_K (ik) block size/padding");

static const int8_t iq3nl_values[16] = {-63, -40, -23, -10, 1, 13, 28,  47,
    -59, -36, -19,  -6, 5, 17, 32,  51,};
static const int8_t iq4k_values[32] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117};
static const int8_t iq5nl_values[64] = {-126, -114, -103, -92, -83, -74, -65, -57, -50, -43, -36, -30, -24, -18, -12, -6, -1, 5, 11, 17, 23, 29, 36, 43, 51, 59, 68, 77, 87, 97, 109, 121,
    -124, -112, -101, -90, -81, -72, -63, -55, -48, -41, -34, -28, -22, -16, -10, -4,  1, 7, 13, 19, 25, 31, 38, 45, 53, 61, 70, 79, 89, 99, 111, 123,};
