// How fast does the host-to-GPU link move an expert slice from pinned host memory to the GPU?
// Staging a layer's experts through VRAM is bounded by this number: a copy of one expert costs
// size / bandwidth, and the matmul has to read those same bytes.
//
// usage: pcie_h2d [MiB] [repeats] [expert_bytes]
//   MiB is the size of each copy (default 3), repeats the number of copies (default 200); with expert_bytes
//   (the size of one expert slice of your model) it also prints the time one expert takes at that rate.
// build: nvcc -O2 -o pcie_h2d pcie_h2d.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

int main(int argc, char ** argv) {
    const size_t miB    = argc > 1 ? (size_t) atoi(argv[1]) : 3;
    const int    repeats = argc > 2 ? atoi(argv[2]) : 200;
    const double expert_bytes = argc > 3 ? atof(argv[3]) : 0.0;
    const size_t bytes  = miB*1024*1024;

    void * h = NULL;
    if (cudaHostAlloc(&h, bytes, cudaHostAllocDefault) != cudaSuccess) {
        printf("pcie_h2d: cudaHostAlloc of %zu MiB failed\n", miB);
        return 1;
    }
    void * d = NULL;
    if (cudaMalloc(&d, bytes) != cudaSuccess) {
        printf("pcie_h2d: cudaMalloc of %zu MiB failed\n", miB);
        return 1;
    }
    for (size_t i = 0; i < bytes/sizeof(float); i++) {
        ((float *) h)[i] = 1.0f;
    }

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // warm up both directions
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaEventRecord(t0);
    for (int i = 0; i < repeats; i++) {
        cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, 0);
    }
    cudaEventRecord(t1);
    cudaDeviceSynchronize();

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    const double per_copy_us = 1000.0*ms/repeats;
    printf("pcie_h2d: %zu MiB x %d copies, %.3f ms total, %.3f ms each, %.2f GB/s\n",
           miB, repeats, ms, per_copy_us/1000.0, (double) bytes*(double) repeats/1e9/(ms/1e3));
    // bytes per microsecond is bytes/us, so the time for n bytes is n/(bytes/us) microseconds
    if (expert_bytes > 0.0) {
        printf("pcie_h2d: one expert (%.0f B) at that rate: %.3f ms\n",
               expert_bytes, expert_bytes/((double) bytes/per_copy_us)/1000.0);
    }

    cudaFree(d);
    cudaFreeHost(h);
    return 0;
}
