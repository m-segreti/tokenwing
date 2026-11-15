#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// GGML_CUDA_FUSE_NORM_QUANT=1: see norm.cu
bool ggml_cuda_fuse_norm_q8_on();
bool ggml_cuda_norm_q8_shape_ok(const ggml_tensor * norm, const ggml_tensor * mul_tensor);
void ggml_cuda_op_rms_norm_mul_q8_1(ggml_backend_cuda_context & ctx, ggml_tensor * norm, ggml_tensor * mul_tensor);
// a tagged grouped norm (several rows) whose output a quantized matvec reads as one row (mm_src1): it also writes that q8_1 copy
bool ggml_cuda_norm_rows_q8_shape_ok(const ggml_tensor * norm, const ggml_tensor * mul_tensor, const ggml_tensor * mm_src1);
void ggml_cuda_op_rms_norm_mul_rows_q8_1(ggml_backend_cuda_context & ctx, ggml_tensor * norm, ggml_tensor * mul_tensor,
                                         const ggml_tensor * mm_src1);
