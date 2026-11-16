// autotuner: measures the parts of this PC that decide TokenWing's CPU settings, without loading a model, and prints
// one JSON line. The streamed experts' CPU matmul reads quantized weights from RAM, so its speed is bound by the memory
// system: the tool runs a quantized matrix-vector product over 0.6 GB of weights (more than any CPU cache) at 1, 2, ...
// physical-core threads and reports the speed of each, the smallest thread count within 5 % of the best, and the
// recommended -t. It also names the fast cores of a CPU with fast and slow cores (Intel's P-cores).
//
// With --file it also reads the model file directly (no file cache) at 1, 2, 4, 8, 16, 32 reads in flight, each the size
// of one expert's projection in that model (the engine reads an expert as three such pieces), at random places, and
// recommends the engine's SSD reader count (--moe-stream-io-threads): the fewest within 5 % of the best.
// It also lists the graphics cards the engine's backends see (CUDA, ROCm, Vulkan ...) with their memory, so the caller
// can choose by card without a vendor's own library.
//
//   autotuner [--seconds S] [--file MODEL.gguf]

#include "common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

// The bytes of one expert's projection in a model file (its first expert tensor / its experts), rounded up to 4 KiB;
// 2 MiB when the file has no expert tensors or cannot be read
static size_t expert_piece(const std::string & path) {
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    size_t piece = 2u << 20;
    if (g && meta) {
        for (ggml_tensor * t = ggml_get_first_tensor(meta); t; t = ggml_get_next_tensor(meta, t)) {
            if (strstr(t->name, "_exps.weight") && t->ne[2] > 1) {
                piece = (ggml_nbytes(t)/(size_t) t->ne[2] + 4095)/4096*4096;
                break;
            }
        }
    }
    if (meta) { ggml_free(meta); }
    if (g) { gguf_free(g); }
    return piece;
}

// GB/s of direct reads of `path`: `qd` threads, each reading `blk`-byte pieces at random 4 KiB-aligned places, for
// `seconds`. -1 when the file cannot be read that way.
static double read_gbps(const std::string & path, int qd, double seconds, size_t blk) {
    std::vector<double> got(qd, 0.0);
    std::vector<std::thread> th;
    bool ok = true;
    for (int i = 0; i < qd; i++) {
        th.emplace_back([&, i] {
#ifdef _WIN32
            HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING, nullptr);
            if (h == INVALID_HANDLE_VALUE) { ok = false; return; }
            LARGE_INTEGER sz; GetFileSizeEx(h, &sz); const uint64_t size = (uint64_t) sz.QuadPart;
            void * b = VirtualAlloc(nullptr, blk, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
            const int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
            if (fd < 0) { ok = false; return; }
            const uint64_t size = (uint64_t) lseek(fd, 0, SEEK_END);
            void * b = nullptr;
            if (posix_memalign(&b, 4096, blk) != 0) { close(fd); ok = false; return; }
#endif
            std::mt19937_64 rng(1234 + i);
            const auto t0 = std::chrono::steady_clock::now();
            double n = 0;
            while (size > 2*blk && std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
                const uint64_t off = (rng() % ((size - blk)/4096))*4096;
#ifdef _WIN32
                OVERLAPPED ov{}; ov.Offset = (DWORD) off; ov.OffsetHigh = (DWORD) (off >> 32);
                DWORD r = 0;
                if (!ReadFile(h, b, (DWORD) blk, &r, &ov)) { ok = false; break; }
#else
                const ssize_t r = pread(fd, b, blk, (off_t) off);
                if (r <= 0) { ok = false; break; }
#endif
                n += (double) r;
            }
            got[i] = n/std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
#ifdef _WIN32
            VirtualFree(b, 0, MEM_RELEASE); CloseHandle(h);
#else
            free(b); close(fd);
#endif
        });
    }
    for (auto & t : th) {
        t.join();
    }
    double sum = 0;
    for (double g : got) {
        sum += g;
    }
    return ok ? sum/1e9 : -1.0;
}

// The fast cores of a CPU with two kinds of cores, as a --cpu-range value ("0-15"), or "" when all cores are alike.
// Intel only, on both systems: AMD's compact cores (Zen 5c next to Zen 5) also carry a lower efficiency class on Windows,
// but they run the memory-bound expert matmul nearly as fast - pinning Ling to the 4 Zen 5 cores of a Ryzen AI 9 HX 370 measured
// 25.8 tok/s against 35.1 on all cores. (The Linux branch reads Intel's cpu_core / cpu_atom files, so it never names AMD cores)
static std::string fast_cpus() {
#ifdef _WIN32
    int id[4] = {};
    __cpuid(id, 0);
    char vendor[13] = {};
    memcpy(vendor, &id[1], 4); memcpy(vendor + 4, &id[3], 4); memcpy(vendor + 8, &id[2], 4);
    if (strcmp(vendor, "GenuineIntel") != 0) {
        return "";
    }
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<char> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
        return "";
    }
    // a CPU's fast cores carry the highest efficiency class; with one class all cores are alike
    int lo = 255, hi = 0;
    for (DWORD off = 0; off < len;) {
        auto * p = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (buf.data() + off);
        lo = std::min<int>(lo, p->Processor.EfficiencyClass);
        hi = std::max<int>(hi, p->Processor.EfficiencyClass);
        off += p->Size;
    }
    if (lo == hi) {
        return "";
    }
    int first = -1, last = -1;
    for (DWORD off = 0; off < len;) {
        auto * p = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (buf.data() + off);
        if (p->Processor.EfficiencyClass == hi) {
            const KAFFINITY m = p->Processor.GroupMask[0].Mask;
            for (int b = 0; b < (int) (8*sizeof(KAFFINITY)); b++) {
                if (m & ((KAFFINITY) 1 << b)) {
                    first = first < 0 ? b : std::min(first, b);
                    last  = std::max(last, b);
                }
            }
        }
        off += p->Size;
    }
    // shortcut: assumes the fast cores are numbered first and in one run (as on Intel's hybrid CPUs); a split set is not named
    return first == 0 && last > 0 ? "0-" + std::to_string(last) : "";
#else
    std::ifstream f("/sys/devices/cpu_core/cpus");   // Intel hybrid CPUs on Linux: the P-cores' logical CPUs, e.g. "0-15"
    std::string s;
    std::getline(f, s);
    std::ifstream e("/sys/devices/cpu_atom/cpus");    // ... and only when E-cores exist too
    std::string t;
    std::getline(e, t);
    return !s.empty() && !t.empty() && s.find(',') == std::string::npos ? s : "";
#endif
}

int main(int argc, char ** argv) {
    double seconds = 0.15;   // timed work per thread count
    std::string file;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seconds") && i + 1 < argc) {
            seconds = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--file") && i + 1 < argc) {
            file = argv[++i];
        }
    }
    const int phys = std::max(1, (int) common_cpu_get_num_physical_cores());

    // four 4096 x 36864 Q4_K matrices (~85 MB each, 0.34 GB) read in turn by one graph, so no cache holds them
    const int64_t K = 4096, N = 36864, M = 4;
    ggml_init_params ip = { /*.mem_size =*/ 64*ggml_tensor_overhead() + ggml_graph_overhead(), /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, K);
    std::vector<ggml_tensor *> w;
    ggml_cgraph * gf = ggml_new_graph(ctx);
    for (int i = 0; i < M; i++) {
        w.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, K, N));
        ggml_build_forward_expand(gf, ggml_mul_mat(ctx, w.back(), x));
    }
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!be || !buf) {
        fprintf(stderr, "autotuner: cannot allocate the test matrices\n");
        return 1;
    }
    ggml_backend_buffer_clear(buf, 0);   // zero blocks: the same bytes are read as with real weights, and no NaN / denormal
    size_t bytes = 0;
    for (auto * t : w) {
        bytes += ggml_nbytes(t);
    }

    std::vector<double> gbps(phys + 1, 0.0);
    for (int t = 1; t <= phys; t++) {
        ggml_backend_cpu_set_n_threads(be, t);
        ggml_backend_graph_compute(be, gf);   // warm-up (thread pool, pages)
        std::vector<double> runs;
        const auto t0 = std::chrono::steady_clock::now();
        do {
            const auto a = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(be, gf);
            runs.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count());
        } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds || runs.size() < 3);
        std::nth_element(runs.begin(), runs.begin() + runs.size()/2, runs.end());
        gbps[t] = bytes/runs[runs.size()/2]/1e9;   // the median run: a system hiccup does not move it
    }
    const double best = *std::max_element(gbps.begin(), gbps.end());
    int sat = phys;
    for (int t = 1; t <= phys; t++) {
        if (gbps[t] >= 0.95*best) {
            sat = t;
            break;
        }
    }
    // the engine's own threads (8 SSD readers, the guess helper) need cores too: at most all physical cores but two
    // (measured: 6 cores, -t 4 = 39 tok/s against -t 6 = 25)
    const int threads = std::max(1, std::min(sat, phys - 2));

    // the cards the engine's backends see: discrete GPUs only (an integrated GPU shares the system's RAM)
    ggml_backend_load_all();
    std::string gpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(d, &free, &total);
        gpus += (gpus.empty() ? "" : ", ") + std::string("{\"name\": \"") + ggml_backend_dev_name(d) + "\", \"mib\": " + std::to_string(total >> 20) + "}";
    }

    printf("{\"gpus\": [%s], \"physical_cores\": %d, \"threads\": %d, \"saturation\": %d, \"fast_cpus\": \"%s\", \"gbps\": [", gpus.c_str(), phys, threads, sat, fast_cpus().c_str());
    for (int t = 1; t <= phys; t++) {
        printf("%s%.1f", t > 1 ? ", " : "", gbps[t]);
    }
    printf("]");
    if (!file.empty()) {
        const int qds[] = { 1, 2, 4, 8, 16, 32 };
        const size_t piece = expert_piece(file);
        double rd[6], rbest = 0;
        for (int j = 0; j < 6; j++) {
            rd[j] = read_gbps(file, qds[j], 0.4, piece);
            rbest = std::max(rbest, rd[j]);
        }
        int io = 8;   // the engine's default, kept when the file cannot be read directly
        for (int j = 0; j < 6 && rbest > 0; j++) {
            // at least 4: needed reads and guesses share them; at most 16: the engine was never measured with more readers
            if (rd[j] >= 0.95*rbest) { io = std::min(16, std::max(4, qds[j])); break; }
        }
        printf(", \"io_threads\": %d, \"piece_kib\": %zu, \"read_gbps\": [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]", io, piece >> 10, rd[0], rd[1], rd[2], rd[3], rd[4], rd[5]);
    }
    printf("}\n");

    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    return 0;
}
