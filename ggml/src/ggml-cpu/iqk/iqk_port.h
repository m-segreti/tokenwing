// The entry to ik_llama.cpp's expert matmul kernels for our MUL_MAT_ID (see iqk_port.cpp). x86 with AVX2 or AVX-512 only: elsewhere
// iqk_port_available() is 0 and the engine keeps the plain path. The pools stay in the GGUF's own bytes: the weights are unpacked per thread, in
// the call, into a work buffer of that thread.
#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
int    iqk_port_available(void);
// can these weights ([ne00, ne01, n]) take the kernels: format + shape. q6_K is excluded (ik is slower at prompt sizes)
int    iqk_port_node_ok(int type, long ne00, long ne01);
// the activation layout ik's kernels want for these weights: 1 = ik's q8_K (our q8_K rows, one more field per block), 2 = q8_2_x4 (from the floats)
int    iqk_port_b_kind(int type);
size_t iqk_port_b_row_size(int type, long n);
void   iqk_port_q8_2_x4_row(const float * x, void * vy, long n);
size_t iqk_port_q8k_row_size(long n);
// blocks [first_block, end_block) of one activation row quantized by ik's quantizer: into ik's q8_K layout (ik_row, may be NULL) and ours (ours_row, may be NULL)
void   iqk_port_q8k_quant_range(const float * x, void * ours_row, void * ik_row, long first_block, long end_block);
// can the kernels take an expert of this format with Ny rows
int    iqk_port_can(int typeA, long Ny, long ne00);
// this thread's share (by weight rows) of one expert; 0 = the kernels could not take it (the caller uses the plain path)
int    iqk_port_mmid(long Nx, long Ny, long ne00, int ne11, int typeA, const void * A, long strideA,
                     const void * B, long strideB, float * C, long nb1, long nb2,
                     const void * row_mapping, int ith, int nth);
#ifdef __cplusplus
}
#endif
