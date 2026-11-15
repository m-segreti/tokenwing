// The GPU half of the test quant plugin (ext-q4_0.c): plain, slow matmul kernels for its Q4_0-identical type. One thread
// per output value, the weights dequantized block by block and multiplied with the F32 input, so the results match
// ggml's Q4_0 kernels up to rounding (those quantize the input to 8 bits first).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

namespace {

struct block_q4_0 {
    __half  d;        // scale
    uint8_t qs[16];   // 32 values of 4 bits: value j in the low half of byte j, value j + 16 in its high half
};
static_assert(sizeof(block_q4_0) == 18, "a Q4_0 block is 18 bytes");

// one line on stderr the first time each function runs, so a test can see that the GPU path was taken
void say_once(bool & said, const char * what) {
    if (!said) {
        said = true;
        fprintf(stderr, "ext-q4_0: first %s call\n", what);
    }
}

__device__ float dot_row(const block_q4_0 * x, const float * y, int64_t n_blocks) {
    float sum = 0.0f;
    for (int64_t b = 0; b < n_blocks; ++b) {
        const float d = __half2float(x[b].d);
        const float * yb = y + b*32;
        float s = 0.0f;
        for (int j = 0; j < 16; ++j) {
            const uint8_t q = x[b].qs[j];
            s += (float) ((int) (q & 0x0F) - 8)*yb[j] + (float) ((int) (q >> 4) - 8)*yb[j + 16];
        }
        sum += d*s;
    }
    return sum;
}

// dst[i01, i11, i12, i13] = row i01 of weight matrix (i12, i13) (broadcast over the input's batch dims) . column (i11, i12, i13)
__global__ void k_mul_mat(const char * vx, const char * vy, char * dst,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        size_t nb01, size_t nb02, size_t nb03,
        int64_t ne11, int64_t ne12, int64_t ne13,
        size_t nb11, size_t nb12, size_t nb13,
        size_t nb1, size_t nb2, size_t nb3) {
    const int64_t i01 = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t i11 = blockIdx.y;
    const int64_t i12 = blockIdx.z % ne12;
    const int64_t i13 = blockIdx.z / ne12;
    if (i01 >= ne01 || i11 >= ne11 || i13 >= ne13) {
        return;
    }
    const int64_t i02 = i12/(ne12/ne02);
    const int64_t i03 = i13/(ne13/ne03);
    const block_q4_0 * x = (const block_q4_0 *) (vx + i01*nb01 + i02*nb02 + i03*nb03);
    const float      * y = (const float *) (vy + i11*nb11 + i12*nb12 + i13*nb13);
    *(float *) (dst + i01*sizeof(float) + i11*nb1 + i12*nb2 + i13*nb3) = dot_row(x, y, ne00/32);
}

// pair (j, t): expert ids[j, t]; an id at or past first_zero (an all-zero slice) gives 0 without reading it
__global__ void k_mul_mat_id(const char * vx, const char * vy, const char * ids, char * dst,
        int64_t ne00, int64_t ne01, int64_t ne02, size_t nb01, size_t nb02,
        int64_t ne11, size_t nb11, size_t nb12,
        size_t ids_nb0, size_t ids_nb1, size_t nb1, size_t nb2, int32_t first_zero) {
    const int64_t i01 = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const int64_t j   = blockIdx.y;
    const int64_t t   = blockIdx.z;
    if (i01 >= ne01) {
        return;
    }
    const int32_t e = *(const int32_t *) (ids + j*ids_nb0 + t*ids_nb1);
    float r = 0.0f;
    if (e >= 0 && e < ne02 && !(first_zero >= 0 && e >= first_zero)) {
        const block_q4_0 * x = (const block_q4_0 *) (vx + e*nb02 + i01*nb01);
        const float      * y = (const float *) (vy + (j % ne11)*nb11 + t*nb12);
        r = dot_row(x, y, ne00/32);
    }
    *(float *) (dst + i01*sizeof(float) + j*nb1 + t*nb2) = r;
}

} // namespace

extern "C" int32_t ext_q4_0_mul_mat_cuda(
        int32_t type, const void * vx, const float * vy, float * dst,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        size_t nb01, size_t nb02, size_t nb03,
        int64_t ne11, int64_t ne12, int64_t ne13,
        size_t nb11, size_t nb12, size_t nb13,
        size_t nb1, size_t nb2, size_t nb3,
        int32_t device, void * stream) {
    (void) type;
    (void) device;   // ggml has made the device current
    static bool said = false;
    say_once(said, "mul_mat_cuda");
    const dim3 block(128);
    const dim3 grid((unsigned) ((ne01 + 127)/128), (unsigned) ne11, (unsigned) (ne12*ne13));
    k_mul_mat<<<grid, block, 0, (cudaStream_t) stream>>>((const char *) vx, (const char *) vy, (char *) dst,
            ne00, ne01, ne02, ne03, nb01, nb02, nb03, ne11, ne12, ne13, nb11, nb12, nb13, nb1, nb2, nb3);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" int32_t ext_q4_0_mul_mat_id_cuda(
        int32_t type, const void * vx, const float * vy, const int32_t * ids, float * dst,
        int64_t ne00, int64_t ne01, int64_t ne02,
        size_t nb01, size_t nb02,
        int64_t ne11, int64_t ne12,
        size_t nb11, size_t nb12,
        int64_t n_expert_used,
        size_t ids_nb0, size_t ids_nb1,
        size_t nb1, size_t nb2,
        int32_t first_zero,
        int32_t device, void * stream) {
    (void) type;
    (void) device;
    static bool said = false;
    say_once(said, "mul_mat_id_cuda");
    const dim3 block(128);
    const dim3 grid((unsigned) ((ne01 + 127)/128), (unsigned) n_expert_used, (unsigned) ne12);
    k_mul_mat_id<<<grid, block, 0, (cudaStream_t) stream>>>((const char *) vx, (const char *) vy, (const char *) ids, (char *) dst,
            ne00, ne01, ne02, nb01, nb02, ne11, nb11, nb12, ids_nb0, ids_nb1, nb1, nb2, first_zero);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
