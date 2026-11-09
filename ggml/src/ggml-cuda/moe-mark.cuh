#pragma once

#include "common.cuh"

// The GPU producers of --moe-stream-needed-early 1.
// The streaming engine (src/, no CUDA) marks a node with a flag bit it chooses and the layer index in the node's last op param.
// After the node (or the fused group that contains it) is dispatched, the node loop enqueues for that layer:
//   1. one kernel: the node's data (and, for a "router" mark, the data of the node's src[1..]) into the layer's pinned host slot,
//      a system fence, then the call counter = the host's call number (read by the kernel when it runs, so a replayed CUDA graph
//      writes this call's number);
//   2. the layer's event (an external event-record node when capturing, a plain record when run directly).
// Two kinds of marks: "guess" (the 1-ahead guess, read by the guess helper) and "router" (the inputs of the
// layer's CPU split, read by the scheduler's input hook).
// Nothing is registered without the setting: the flag test is then one AND per dispatched node against a zero mask.
//
// NVIDIA only: the producer needs cudaEventRecordWithFlags / cudaEventRecordExternal, which the HIP and
// MUSA vendor headers do not map; those builds compile empty stubs and export no register function, so the engine keeps the host producer.

// Copy-ahead (one copy set): a "done" mark copies nothing - only the counter: the copier learns that the node (a wave's
// gate/up activation) has run, so the places whose gate/up parts it read are free.
enum ggml_cuda_moe_mark_kind {
    GGML_CUDA_MOE_MARK_GUESS  = 0,
    GGML_CUDA_MOE_MARK_ROUTER = 1,
    GGML_CUDA_MOE_MARK_DONE   = 2,
    GGML_CUDA_MOE_MARK_N      = 3,
};

// one kind's table, n_layers entries; an unused layer has slot == nullptr. slot / counter / expected are pinned host memory (the
// CUDA host buffer type); cap = the slot's bytes for the data (the counter is separate).
struct ggml_cuda_moe_mark_table {
    uint32_t                flag_bit;
    int32_t                 skip_layer;   // >= 0: test only (MOE_STREAM_TEST_SKIP_EVENT): never written nor recorded
    void * const          * slots;
    const int64_t         * caps;
    uint32_t * const      * counters;
    const uint32_t * const * expected;
};

// owner: the engine that registers (one per process - a second one is refused). device: the CUDA device the marked
// nodes run on (the events are created on it; a mark seen on another device aborts). Returns false with *why set.
bool ggml_cuda_moe_mark_register(const void * owner, int device, int32_t n_layers,
                                 const ggml_cuda_moe_mark_table * tables, int n_tables, const char ** why);
// waits for layer il's event of the given kind; false (with a printed reason) when there is none or on a CUDA error
bool ggml_cuda_moe_mark_wait(int32_t kind, int32_t il);
// waits until the registered device has run everything enqueued (after an aborted call); false on a CUDA error
bool ggml_cuda_moe_mark_sync(void);
// removes every entry and destroys the events - only when owner registered them
void ggml_cuda_moe_mark_unregister(const void * owner);

// the node loop: after nodes [i0, i1] were dispatched on stream (of device), for every marked node among them
void ggml_cuda_moe_mark_after(const ggml_cgraph * cgraph, int i0, int i1, cudaStream_t stream, bool capturing, int device);
// the bits of all registered kinds (the node loop's fast test; 0 when nothing is registered)
uint32_t ggml_cuda_moe_mark_bits(void);

// GGML_CUDA_MOE_MARK_IDS=1: true when this node's router mark copies the ids itself - the node is then not computed
bool ggml_cuda_moe_mark_takes_node(const ggml_tensor * node);

// TEST ONLY (MOE_STREAM_TEST_VF_PLACE_CHECK): one thread spins for us microseconds of GPU time on stream,
// so the work queued after it comes that much later. NVIDIA only (%globaltimer); false elsewhere.
bool ggml_cuda_moe_test_spin(cudaStream_t stream, int64_t us);
