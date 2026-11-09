#pragma once
#include "common.cuh"

// GGML_CUDA_FUSE_AMAX=1: build_amax_src1_f16's chain (#30861: ABS, POOL_1D max over the whole row, SCALE, LOG,
// SCALE, CEIL, CLAMP, SCALE, EXP, DIV) as one kernel - writes the EXP node (e, one value per row) and the DIV node (x / e)
bool ggml_cuda_fuse_amax_on();
void ggml_cuda_op_moe_amax_fused(ggml_backend_cuda_context & ctx, ggml_tensor * const * nodes);
