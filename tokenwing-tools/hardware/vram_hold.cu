// Make the GPU look like a smaller card. Takes free VRAM until only <MiB> is left free, prints
// what it holds, then waits until it is killed (SIGTERM / SIGINT); the memory goes back when it exits.
// Start it before the engine; the engine then sizes itself from the free VRAM it sees, as on a card of that size.
//
// usage: vram_hold <MiB left free, e.g. 6144> [device]
// build: nvcc -O2 -o vram_hold vram_hold.cu
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cuda_runtime.h>

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: vram_hold <MiB left free> [device]\n");
        return 2;
    }
    const size_t keep = (size_t) atoll(argv[1]) << 20;
    if (cudaSetDevice(argc > 2 ? atoi(argv[2]) : 0) != cudaSuccess) {
        fprintf(stderr, "vram_hold: no such CUDA device\n");
        return 1;
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);   // also creates the context, so its own memory is already counted
    if (free_b <= keep) {
        printf("vram_hold: only %zu MiB free, nothing to hold\n", free_b >> 20);
        return 0;
    }
    // one block first, then 64 MiB steps until free VRAM is at the target (the driver keeps some for itself)
    void * p = nullptr;
    size_t held = 0;
    const size_t step = 64u << 20;
    if (cudaMalloc(&p, free_b - keep - step) == cudaSuccess) {
        held += free_b - keep - step;
    }
    for (;;) {
        cudaMemGetInfo(&free_b, &total_b);
        if (free_b <= keep + step) {
            break;
        }
        if (cudaMalloc(&p, step) != cudaSuccess) {
            break;
        }
        held += step;
    }
    cudaMemGetInfo(&free_b, &total_b);
    printf("vram_hold: holding %zu MiB, %zu MiB of %zu MiB free\n", held >> 20, free_b >> 20, total_b >> 20);
    fflush(stdout);

    signal(SIGTERM, on_signal);
    signal(SIGINT, on_signal);
    while (!g_stop) {
        pause();
    }
    return 0;   // the driver frees everything when the process ends
}
