#include "common.cuh"

// ggml_moe_group_topk - the group-limited expert selection (sigmoid scores + selection bias -> group
// scores -> the best groups -> top k) in one kernel
void ggml_cuda_op_moe_group_topk(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
