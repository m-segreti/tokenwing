// How fast can RAM be read the way an MoE expert streaming engine reads it?
//
// The engine reads one expert slice per expert, token and layer from a pool in RAM, with a fixed number
// of worker threads (the -t of the engine). Two patterns:
//   seq   each thread streams its own share of the buffer
//   chunk the threads read chunk_bytes chunks, each thread taking every nthreads-th chunk of the buffer
// which is what a pool read looks like: many slices per token, each in a different slot.
//
// usage: ram_bw <GiB> <threads> <seq|chunk> [chunk_bytes]
//   GiB is the buffer size (default 8), threads the thread count (default 8, at most 64), chunk_bytes the
//   size of one chunk in chunk mode (default 1 MiB; use the expert slice size of your model).
// build: cc -O2 -o ram_bw ram_bw.c -lpthread
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static unsigned char * buf = NULL;
static size_t nbytes = 0;
static size_t chunk = 1 << 20;
static int nthreads = 8;
static int mode_seq = 1;
static unsigned long long per_thread[64];

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9*ts.tv_nsec;
}

static void * worker(void * arg) {
    const int id = (int) (intptr_t) arg;
    unsigned long long sum = 0;
    if (mode_seq) {
        const size_t per = nbytes/nthreads;
        const unsigned char * p = buf + id*per;
        for (size_t i = 0; i < per; i += 64) {
            sum += p[i];
        }
    } else {
        // every thread takes every nthreads-th slice, so the threads sweep the buffer together the
        // way the engine's threads work through the expert slices of a layer
        for (size_t off = (size_t) id*chunk; off + chunk <= nbytes; off += (size_t) nthreads*chunk) {
            for (size_t i = 0; i < chunk; i += 64) {
                sum += buf[off + i];
            }
        }
    }
    per_thread[id] = sum;
    return NULL;
}

int main(int argc, char ** argv) {
    const size_t gib  = argc > 1 ? (size_t) atoi(argv[1]) : 8;
    const int    thr  = argc > 2 ? atoi(argv[2]) : 8;
    const char * mode = argc > 3 ? argv[3] : "seq";
    if (argc > 4) {
        chunk = (size_t) atoll(argv[4]);
    }
    nthreads = thr;
    mode_seq = strcmp(mode, "chunk") != 0;
    nbytes = gib<<30;

    if (posix_memalign((void **) &buf, 4096, nbytes) != 0) {
        fprintf(stderr, "ram_bw: cannot allocate %zu GiB\n", gib);
        return 1;
    }
    memset(buf, 1, nbytes);   // touch every page, so nothing is a missing mapping

    pthread_t th[64];
    const double t0 = now_s();
    for (int i = 0; i < nthreads; i++) {
        pthread_create(&th[i], NULL, worker, (void *) (intptr_t) i);
    }
    unsigned long long sum = 0;
    for (int i = 0; i < nthreads; i++) {
        pthread_join(th[i], NULL);
        sum += per_thread[i];
    }
    const double dt = now_s() - t0;

    printf("ram_bw: %s %zu GiB with %d threads, %.2f s, %.2f GB/s (%llu)\n",
            mode, gib, nthreads, dt, (double) nbytes/dt/1e9, sum);
    free(buf);
    return 0;
}