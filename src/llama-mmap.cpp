#include "llama-mmap.h"

#include "llama-impl.h"

#include "ggml.h"

#include <cstring>
#include <climits>
#include <cstdlib>
#include <stdexcept>
#include <cerrno>
#include <algorithm>
#include <mutex>

#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #include <fcntl.h>
        #include <sys/stat.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/mman.h>
        #endif
        #if defined(_POSIX_MEMLOCK_RANGE)
            #include <sys/resource.h>
        #endif
    #endif
#endif

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #include <winioctl.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <fcntl.h>
    #include <io.h>
    #include <malloc.h>
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#ifdef _WIN32
#    define llama_mmap_ftell _ftelli64
#    define llama_mmap_fseek _fseeki64
#else
#    define llama_mmap_ftell ftello
#    define llama_mmap_fseek fseeko
#endif

// TODO: consider moving to llama-impl.h if needed in more places
#if defined(_WIN32)
static std::string llama_format_win_err(DWORD err) {
    LPSTR buf;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, NULL);
    if (!size) {
        return "FormatMessageA failed";
    }
    std::string ret(buf, size);
    LocalFree(buf);
    return ret;
}
#endif

// llama_file

struct llama_file::impl {
#if defined(_WIN32)
    HANDLE fp_win32 = INVALID_HANDLE_VALUE;
    int fd = -1;
    bool direct_io = false;
    mutable size_t pos = 0;

    std::string GetErrorMessageWin32(DWORD error_code) const {
        std::string ret;
        LPSTR lpMsgBuf = NULL;
        DWORD bufLen = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                    NULL, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&lpMsgBuf, 0, NULL);
        if (!bufLen) {
            ret = format("Win32 error code: %lx", error_code);
        } else {
            ret = lpMsgBuf;
            LocalFree(lpMsgBuf);
        }
        return ret;
    }

    bool init_direct_io(const char * fname) {
        const int wlen = MultiByteToWideChar(CP_UTF8, 0, fname, -1, NULL, 0);
        if (wlen <= 0) {
            return false;
        }
        std::wstring wname((size_t) wlen, L'\0');
        MultiByteToWideChar(CP_UTF8, 0, fname, -1, &wname[0], wlen);
        HANDLE h = CreateFileW(wname.c_str(), GENERIC_READ,
                               FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               NULL, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, NULL);
        if (h == INVALID_HANDLE_VALUE) {
            return false;
        }
        LARGE_INTEGER fsz{};
        if (!GetFileSizeEx(h, &fsz)) {
            CloseHandle(h);
            return false;
        }
        // unbuffered reads must be on the volume's logical sector size; the storage property query does not answer on a file
        // handle, the file's storage information does
        size_t align = 4096;
        FILE_STORAGE_INFO si{};
        if (GetFileInformationByHandleEx(h, FileStorageInfo, &si, sizeof(si)) &&
                si.LogicalBytesPerSector >= 512 && (si.LogicalBytesPerSector & (si.LogicalBytesPerSector - 1)) == 0) {
            align = (size_t) si.LogicalBytesPerSector;
        }
        const int cfd = _open_osfhandle((intptr_t) h, _O_RDONLY | _O_BINARY);
        if (cfd == -1) {
            CloseHandle(h);
            return false;
        }
        fp_win32  = h;
        fd        = cfd;
        size      = (size_t) fsz.QuadPart;
        alignment = align;
        direct_io = true;
        pos       = 0;
        return true;
    }

    impl(const char * fname, const char * mode, const bool use_direct_io = false) {
        if (use_direct_io && std::strcmp(mode, "rb") == 0 && init_direct_io(fname)) {
            return;
        }
        fp = ggml_fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        fp_win32 = (HANDLE) _get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    impl(FILE * file) : owns_fp(false) {
        fp = file;
        fp_win32 = (HANDLE) _get_osfhandle(_fileno(fp));
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        if (direct_io) {
            return pos;
        }
        LARGE_INTEGER li;
        li.QuadPart = 0;
        BOOL ret = SetFilePointerEx(fp_win32, li, &li, FILE_CURRENT);
        if (!ret) {
            throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }
        return li.QuadPart;
    }

    void seek(size_t offset, int whence) const {
        static_assert(SEEK_SET == FILE_BEGIN, "SEEK_SET != FILE_BEGIN");
        static_assert(SEEK_CUR == FILE_CURRENT, "SEEK_CUR != FILE_CURRENT");
        static_assert(SEEK_END == FILE_END, "SEEK_END != FILE_END");
        if (direct_io) {
            pos = whence == SEEK_SET ? offset : (whence == SEEK_CUR ? pos + offset : size + offset);
            return;
        }
        LARGE_INTEGER li;
        li.QuadPart = offset;
        BOOL ret = SetFilePointerEx(fp_win32, li, NULL, whence);
        if (!ret) {
            throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }
    }

    size_t read_overlapped_aligned(void * dst, size_t len, size_t offset) const {
        struct tls_ev {
            HANDLE h = CreateEventW(NULL, TRUE, FALSE, NULL);
            ~tls_ev() { if (h) { CloseHandle(h); } }
        };
        static thread_local tls_ev ev;
        size_t total = 0;
        while (total < len) {
            const DWORD chunk = (DWORD) std::min<size_t>(len - total, 64*1024*1024);
            const uint64_t cur = (uint64_t) (offset + total);
            OVERLAPPED ov{};
            ov.Offset     = (DWORD) (cur & 0xffffffffu);
            ov.OffsetHigh = (DWORD) (cur >> 32);
            ov.hEvent     = ev.h;
            if (ev.h) { ResetEvent(ev.h); }
            DWORD got = 0;
            BOOL ok = ReadFile(fp_win32, (char *) dst + total, chunk, &got, &ov);
            if (!ok) {
                DWORD err = GetLastError();
                if (err == ERROR_IO_PENDING) {
                    if (!GetOverlappedResult(fp_win32, &ov, &got, TRUE)) {
                        err = GetLastError();
                        if (err == ERROR_HANDLE_EOF) { break; }
                        throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(err).c_str()));
                    }
                } else if (err == ERROR_HANDLE_EOF) {
                    break;
                } else {
                    throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(err).c_str()));
                }
            }
            if (got == 0) { break; }
            total += got;
            if (got < chunk) { break; }
        }
        return total;
    }

    void read_raw(void * ptr, size_t len) {
        if (len == 0) {
            return;
        }
        if (direct_io) {
            if (pos + len > size) {
                throw std::runtime_error("unexpectedly reached end of file");
            }
            if (pos % alignment == 0 && len % alignment == 0 && ((uintptr_t) ptr) % alignment == 0) {
                if (read_overlapped_aligned(ptr, len, pos) != len) {
                    throw std::runtime_error("unexpectedly reached end of file");
                }
                pos += len;
                return;
            }
            size_t done = 0;
            while (done < len) {
                const size_t cur = pos + done;
                const size_t step = std::min<size_t>(len - done, 32*1024*1024);
                const size_t a0 = cur & ~(alignment - 1);
                const size_t head = cur - a0;
                const size_t want = (head + step + alignment - 1) & ~(alignment - 1);
                struct aligned_deleter { void operator()(void * p) const { _aligned_free(p); } };
                std::unique_ptr<void, aligned_deleter> bounce(_aligned_malloc(want, alignment));
                if (!bounce) {
                    throw std::runtime_error("failed to allocate aligned direct-io buffer");
                }
                const size_t got = read_overlapped_aligned(bounce.get(), want, a0);
                if (got < head + step) {
                    throw std::runtime_error("unexpectedly reached end of file");
                }
                std::memcpy((char *) ptr + done, (const char *) bounce.get() + head, step);
                done += step;
            }
            pos += len;
            return;
        }
        size_t bytes_read = 0;
        while (bytes_read < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_read, 64*1024*1024);
            DWORD chunk_read = 0;
            BOOL result = ReadFile(fp_win32, reinterpret_cast<char*>(ptr) + bytes_read, (DWORD) chunk_size, &chunk_read, NULL);
            if (!result) {
                throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_read < chunk_size || chunk_read == 0) {
                throw std::runtime_error("unexpectedly reached end of file");
            }

            bytes_read += chunk_read;
        }
    }

    // With direct reads the loader reads the aligned range around a tensor, so for the file's last tensor it asks for up to one
    // alignment unit past the end of the file. Those padding bytes read as zeros, as in read_raw_unsafe on other systems;
    // anything further past the end is still an error.
    void read_raw_unsafe(void * ptr, size_t len) {
        if (direct_io && pos + len > size && pos <= size && pos + len - size < alignment) {
            const size_t tail = pos + len - size;
            const size_t p0   = pos;
            read_raw(ptr, len - tail);
            std::memset((char *) ptr + (len - tail), 0, tail);
            pos = p0 + len;
            return;
        }
        read_raw(ptr, len);
    }

    uint32_t read_u32() {
        uint32_t val;
        read_raw(&val, sizeof(val));
        return val;
    }

    void write_raw(const void * ptr, size_t len) const {
        size_t bytes_written = 0;
        while (bytes_written < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_written, 64*1024*1024);
            DWORD chunk_written = 0;
            BOOL result = WriteFile(fp_win32, reinterpret_cast<char const*>(ptr) + bytes_written, (DWORD) chunk_size, &chunk_written, NULL);
            if (!result) {
                throw std::runtime_error(format("write error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_written < chunk_size || chunk_written == 0) {
                throw std::runtime_error("unexpectedly failed to write bytes");
            }

            bytes_written += chunk_written;
        }
    }

    void write_u32(uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    bool has_direct_io() const {
        return direct_io;
    }

    ~impl() {
        if (fd != -1) {
            _close(fd);
        } else if (fp && owns_fp) {
            std::fclose(fp);
        }
    }
#else
    impl(const char * fname, const char * mode, [[maybe_unused]] const bool use_direct_io = false) : fname(fname) {
#ifdef __linux__
        // Try unbuffered I/O for read only
        if (use_direct_io && std::strcmp(mode, "rb") == 0) {
            if (init_fd()) {
                return;
            }
            LLAMA_LOG_WARN("Failed to open file '%s' with error: %s. Falling back to buffered I/O",
                           fname, strerror(errno));
        }
#endif
        init_fp(mode);
    }

#ifdef __linux__
    bool init_fd() {
        fd = open(fname.c_str(), O_RDONLY | O_DIRECT);

        if (fd != -1) {
            struct stat file_stats{};
            fstat(fd, &file_stats);

            size = file_stats.st_size;
            alignment = file_stats.st_blksize;

            off_t ret = lseek(fd, 0, SEEK_SET);
            if (ret == -1) {
                throw std::runtime_error(format("seek error: %s", strerror(errno)));
            }
            return true;
        }
        return false;
    }
#endif

    void init_fp(const char * mode) {
        fp = ggml_fopen(fname.c_str(), mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname.c_str(), strerror(errno)));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    impl(FILE * file) : fname("(file*)"), owns_fp(false) {
        fp = file;
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        if (fd == -1) {
            off_t ret = llama_mmap_ftell(fp);
            if (ret == -1) {
                throw std::runtime_error(format("ftell error: %s", strerror(errno)));
            }

            return (size_t) ret;
        }

        off_t pos = lseek(fd, 0, SEEK_CUR);
        if (pos == -1) {
            throw std::runtime_error(format("lseek error: %s", strerror(errno)));
        }
        return (size_t) pos;
    }

    void seek(size_t offset, int whence) const {
        off_t ret = 0;
        if (fd == -1) {
            ret = llama_mmap_fseek(fp, offset, whence);
        } else {
            ret = lseek(fd, offset, whence);
        }
        if (ret == -1) {
            throw std::runtime_error(format("seek error: %s", strerror(errno)));
        }
    }

    void read_raw_unsafe(void * ptr, size_t len) {
        if (len == 0) {
            return;
        }
        errno = 0;
        if (fd == -1) {
            const size_t curr_off = tell();
            const size_t to_read = std::min(len, size - curr_off);

            std::size_t ret = std::fread(ptr, to_read, 1, fp);
            if (ferror(fp)) {
                throw std::runtime_error(format("read error: %s", strerror(errno)));
            }
            if (to_read > 0 && ret != 1) {
                throw std::runtime_error("unexpectedly reached end of file");
            }
        } else {
            size_t bytes_read = 0;
            while (bytes_read < len) {
                const size_t to_read = len - bytes_read;
                ssize_t ret = ::read(fd, reinterpret_cast<char *>(ptr) + bytes_read, to_read);

                if (ret == -1) {
                    if (errno == EINTR) {
                        continue;  // Interrupted by signal, retry
                    }
                    // Fallback to std::fread in case the DMA controller cannot access the buffer
                    if (errno == EFAULT || errno == EINVAL) {
                        LLAMA_LOG_WARN("%s: Falling back to buffered IO due to %s\n", __func__, strerror(errno));
                        auto curr_off = tell();
                        close(fd);
                        fd = -1;
                        alignment = 1;
                        init_fp("rb");
                        seek(curr_off, SEEK_SET);
                        read_raw_unsafe(ptr, len);
                        return;
                    }
                    throw std::runtime_error(format("read error: %s", strerror(errno)));
                }
                if (ret == 0) {
                    // EOF: allow if this read was only pulling alignment padding past file end
                    off_t pos = lseek(fd, 0, SEEK_CUR);
                    if (pos != -1 && (size_t) pos == size) {
                        std::memset(reinterpret_cast<char *>(ptr) + bytes_read, 0, len - bytes_read);
                        return;
                    }
                    throw std::runtime_error("unexpectedly reached end of file");
                }

                bytes_read += (size_t) ret;
            }
        }
    }

    void read_aligned_chunk(void * dest, size_t size) {
        size_t offset = tell();
        off_t aligned_offset = offset & ~(alignment - 1);
        off_t offset_from_alignment = offset - aligned_offset;
        size_t bytes_to_read = (offset_from_alignment + size + alignment - 1) & ~(alignment - 1);

        // stage through a bounded buffer, so that a large tensor is not held in memory twice while it loads
        const size_t buffer_size = std::min<size_t>(bytes_to_read, LLAMA_DIRECT_IO_BUFFER_SIZE);

        void * raw_buffer = nullptr;
        int ret = posix_memalign(&raw_buffer, alignment, buffer_size);
        if (ret != 0) {
            throw std::runtime_error(format("posix_memalign failed with error %d", ret));
        }

        struct aligned_buffer_deleter {
            void operator()(void * p) const { free(p); }
        };
        std::unique_ptr<void, aligned_buffer_deleter> buffer(raw_buffer);

        seek(aligned_offset, SEEK_SET);

        size_t skip   = offset_from_alignment;
        size_t copied = 0;
        for (size_t done = 0; done < bytes_to_read; ) {
            const size_t n = std::min(buffer_size, bytes_to_read - done);
            read_raw_unsafe(buffer.get(), n);

            const size_t count = std::min(n - skip, size - copied);
            memcpy(reinterpret_cast<char *>(dest) + copied, reinterpret_cast<char *>(buffer.get()) + skip, count);

            copied += count;
            skip    = 0;
            done   += n;
        }
    }

    void read_raw(void * ptr, size_t len) {
        if (has_direct_io()) {
            read_aligned_chunk(ptr, len);
        } else {
            read_raw_unsafe(ptr, len);
        }
    }

    uint32_t read_u32() {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    void write_raw(const void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, len, 1, fp);
        if (ret != 1) {
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    bool has_direct_io() const {
        return fd != -1 && alignment > 1;
    }

    ~impl() {
        if (fd != -1) {
            close(fd);
        } else if (owns_fp) {
            std::fclose(fp);
        }
    }
    int fd = -1;
    std::string fname;
#endif

    size_t read_alignment() const {
        return alignment;
    }

    size_t alignment = 1;

    FILE * fp{};
    size_t size{};
    bool owns_fp = true;
};

llama_file::llama_file(const char * fname, const char * mode, const bool use_direct_io) :
    pimpl(std::make_unique<impl>(fname, mode, use_direct_io)) {}

llama_file::llama_file(FILE * file) : pimpl(std::make_unique<impl>(file)) {}

llama_file::~llama_file() = default;

size_t llama_file::tell() const { return pimpl->tell(); }
size_t llama_file::size() const { return pimpl->size; }

size_t llama_file::read_alignment() const { return pimpl->read_alignment(); }
bool llama_file::has_direct_io() const { return pimpl->has_direct_io(); }

int llama_file::file_id() const {
#ifdef _WIN32
    if (pimpl->fd != -1) {
        return pimpl->fd;
    }
    return _fileno(pimpl->fp);
#else
    if (pimpl->fd != -1) {
        return pimpl->fd;
    }
#if defined(fileno)
    return fileno(pimpl->fp);
#else
    return ::fileno(pimpl->fp);
#endif
#endif
}

void llama_file::seek(size_t offset, int whence) const { pimpl->seek(offset, whence); }
void llama_file::read_raw(void * ptr, size_t len) { pimpl->read_raw(ptr, len); }
void llama_file::read_raw_unsafe(void * ptr, size_t len) { pimpl->read_raw_unsafe(ptr, len); }

uint32_t llama_file::read_u32() { return pimpl->read_u32(); }

void llama_file::write_raw(const void * ptr, size_t len) const { pimpl->write_raw(ptr, len); }
void llama_file::write_u32(uint32_t val) const { pimpl->write_u32(val); }

// llama_mmap

#if defined(_POSIX_MAPPED_FILES) || defined(_WIN32)
// merge `ranges` and return their complement within [0, limit)
static llama_mmap::ranges ranges_complement(llama_mmap::ranges ranges, size_t limit) {
    llama_mmap::ranges res;
    std::sort(ranges.begin(), ranges.end());

    size_t pos = 0;
    for (const auto & range : ranges) {
        const size_t beg = std::min(range.first,  limit);
        const size_t end = std::min(range.second, limit);
        if (beg > pos) {
            res.emplace_back(pos, beg);
        }
        pos = std::max(pos, end);
    }
    if (pos < limit) {
        res.emplace_back(pos, limit);
    }

    return res;
}
#endif

#if defined(_WIN32)
// A reserved address range standing for a whole file: its pages are committed and filled by unbuffered reads ahead of use
// (llama_prefetch) or on first touch (a vectored exception handler). It serves the lazy tables like a file mapping, but the
// file has no mapped view, so NTFS keeps sending the stream's unbuffered reads of the same file to the drive in parallel.
struct llama_win_read_region {
    static constexpr size_t PAGE   = 4096;
    // filled bytes kept (1 GiB); the next prefetch over it drops them all first. LLAMA_WIN_LAZY_BUDGET_MIB=N (test only) sets it
    static size_t budget() {
        static const size_t b = getenv("LLAMA_WIN_LAZY_BUDGET_MIB") ? (size_t) atoll(getenv("LLAMA_WIN_LAZY_BUDGET_MIB")) << 20 : 1ull << 30;
        return b;
    }

    char *  base = nullptr;
    size_t  size = 0;
    HANDLE  h    = INVALID_HANDLE_VALUE;   // a duplicate of the loader's unbuffered, overlapped handle: it outlives the loader
    std::mutex mu;                         // guards everything below
    std::vector<uint8_t> filled;           // per page
    std::vector<std::pair<size_t, size_t>> runs_filled;   // [first page, n pages)
    size_t  filled_bytes = 0;
    size_t  n_reads = 0, read_bytes = 0;   // for the line at exit
    HANDLE  evs[64] = {};

    llama_win_read_region(HANDLE hf, size_t n);
    ~llama_win_read_region();

    // fill the pages of these [first, last) page runs that are not filled yet; mu held
    void fill_locked(const std::vector<std::pair<size_t, size_t>> & runs) {
        struct io { OVERLAPPED ov; size_t p0, np; bool pending; };
        std::vector<io> q;
        for (const auto & [a, b] : runs) {
            for (size_t p = a; p < b;) {
                if (filled[p]) { p++; continue; }
                size_t e = p;
                while (e < b && !filled[e] && e - p < 256) { filled[e++] = 1; }   // at most 1 MiB per read
                q.push_back({ {}, p, e - p, false });
                p = e;
            }
        }
        for (size_t i = 0; i < q.size(); i += 64) {
            const size_t n = std::min<size_t>(64, q.size() - i);
            for (size_t k = 0; k < n; k++) {
                io & x = q[i + k];
                char * dst = base + x.p0*PAGE;
                if (!VirtualAlloc(dst, x.np*PAGE, MEM_COMMIT, PAGE_READWRITE)) {
                    GGML_ABORT("lazy table: commit failed: %s", llama_format_win_err(GetLastError()).c_str());
                }
                const uint64_t off = (uint64_t) x.p0*PAGE;
                x.ov.Offset     = (DWORD) off;
                x.ov.OffsetHigh = (DWORD) (off >> 32);
                x.ov.hEvent     = evs[k];
                ResetEvent(evs[k]);
                DWORD got = 0;
                if (ReadFile(h, dst, (DWORD) (x.np*PAGE), &got, &x.ov)) {
                    x.pending = true;   // done already; the result call below returns at once
                } else if (GetLastError() == ERROR_IO_PENDING) {
                    x.pending = true;
                } else if (GetLastError() != ERROR_HANDLE_EOF) {
                    GGML_ABORT("lazy table: read failed: %s", llama_format_win_err(GetLastError()).c_str());
                }
            }
            for (size_t k = 0; k < n; k++) {
                io & x = q[i + k];
                DWORD got = 0;
                if (x.pending && !GetOverlappedResult(h, &x.ov, &got, TRUE) && GetLastError() != ERROR_HANDLE_EOF) {
                    GGML_ABORT("lazy table: read failed: %s", llama_format_win_err(GetLastError()).c_str());
                }
                runs_filled.emplace_back(x.p0, x.np);
                filled_bytes += x.np*PAGE;
                n_reads++;
                read_bytes += x.np*PAGE;
            }
        }
    }

    void check_locked(const std::vector<std::pair<size_t, size_t>> & runs);

    void drop_all_locked() {
        for (const auto & [p0, np] : runs_filled) {
            VirtualFree(base + p0*PAGE, np*PAGE, MEM_DECOMMIT);
            std::fill(filled.begin() + p0, filled.begin() + p0 + np, 0);
        }
        runs_filled.clear();
        filled_bytes = 0;
    }
};

static std::mutex                           g_win_regions_mu;   // taken before a region's own mutex
static std::vector<llama_win_read_region *> g_win_regions;

static LONG CALLBACK llama_win_read_region_veh(EXCEPTION_POINTERS * e) {
    const EXCEPTION_RECORD * er = e->ExceptionRecord;
    if (er->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || er->NumberParameters < 2) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const char * a = (const char *) er->ExceptionInformation[1];
    std::lock_guard<std::mutex> lk(g_win_regions_mu);
    for (auto * r : g_win_regions) {
        if (a >= r->base && a < r->base + r->size) {
            std::lock_guard<std::mutex> lr(r->mu);
            const size_t p = (size_t) (a - r->base) / llama_win_read_region::PAGE;
            r->fill_locked({ { p, p + 1 } });   // a page another thread filled meanwhile is skipped
            r->check_locked({ { p, p + 1 } });
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

llama_win_read_region::llama_win_read_region(HANDLE hf, size_t n) : size(n) {
    if (!DuplicateHandle(GetCurrentProcess(), hf, GetCurrentProcess(), &h, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
        throw std::runtime_error(format("lazy table: DuplicateHandle failed: %s", llama_format_win_err(GetLastError()).c_str()));
    }
    base = (char *) VirtualAlloc(nullptr, n, MEM_RESERVE, PAGE_READWRITE);
    if (!base) {
        CloseHandle(h);
        throw std::runtime_error(format("lazy table: reserving %zu bytes failed: %s", n, llama_format_win_err(GetLastError()).c_str()));
    }
    filled.assign((n + PAGE - 1) / PAGE, 0);
    for (auto & ev : evs) { ev = CreateEventW(nullptr, TRUE, FALSE, nullptr); }
    static const PVOID veh = AddVectoredExceptionHandler(1, llama_win_read_region_veh);
    GGML_ASSERT(veh != nullptr);
    std::lock_guard<std::mutex> lk(g_win_regions_mu);
    g_win_regions.push_back(this);
    LLAMA_LOG_INFO("%s: lazy tables of this file (%.2f GiB) are filled by unbuffered reads, not mapped (LLAMA_WIN_LAZY_MAP=1 maps)\n",
            __func__, n/1073741824.0);
}

llama_win_read_region::~llama_win_read_region() {
    {
        std::lock_guard<std::mutex> lk(g_win_regions_mu);
        g_win_regions.erase(std::find(g_win_regions.begin(), g_win_regions.end(), this));
    }
    LLAMA_LOG_INFO("%s: lazy tables: %zu reads, %.1f MiB read\n", __func__, n_reads, read_bytes/1048576.0);
    VirtualFree(base, 0, MEM_RELEASE);
    for (auto & ev : evs) { if (ev) { CloseHandle(ev); } }
    CloseHandle(h);
}

// fill the ranges that lie in a read region (parallel reads) and zero their size so the caller skips them
static void llama_win_read_region_prefetch(llama_memory_ranges & mr) {
    using R = llama_win_read_region;
    std::lock_guard<std::mutex> lk(g_win_regions_mu);
    for (auto * r : g_win_regions) {
        std::vector<std::pair<size_t, size_t>> runs;
        for (auto & m : mr) {
            const char * a = (const char *) m.addr;
            if (m.size && a >= r->base && a < r->base + r->size) {
                const size_t p0 = (size_t) (a - r->base) / R::PAGE;
                const size_t p1 = std::min(r->filled.size(), ((size_t) (a - r->base) + m.size + R::PAGE - 1) / R::PAGE);
                runs.emplace_back(p0, p1);
                m.size = 0;
            }
        }
        if (runs.empty()) {
            continue;
        }
        std::sort(runs.begin(), runs.end());
        std::lock_guard<std::mutex> lr(r->mu);
        if (r->filled_bytes > R::budget()) {
            r->drop_all_locked();   // between graph runs: nothing reads the table now
        }
        r->fill_locked(runs);
        r->check_locked(runs);
    }
}

// LLAMA_WIN_LAZY_CHECK=1 (test): every prefetched page run is read again into a buffer of its own and compared with the region
void llama_win_read_region::check_locked(const std::vector<std::pair<size_t, size_t>> & runs) {
    static const bool on = getenv("LLAMA_WIN_LAZY_CHECK") && atoi(getenv("LLAMA_WIN_LAZY_CHECK")) != 0;
    if (!on) {
        return;
    }
    static size_t n_ok = 0;
    for (const auto & [a, b] : runs) {
        const size_t len = (b - a)*PAGE;
        char * tmp = (char *) VirtualAlloc(nullptr, len, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        GGML_ASSERT(tmp);
        OVERLAPPED ov{};
        const uint64_t off = (uint64_t) a*PAGE;
        ov.Offset = (DWORD) off; ov.OffsetHigh = (DWORD) (off >> 32); ov.hEvent = evs[0]; ResetEvent(evs[0]);
        DWORD got = 0;
        if (!ReadFile(h, tmp, (DWORD) len, &got, &ov) && GetLastError() == ERROR_IO_PENDING) {
            GetOverlappedResult(h, &ov, &got, TRUE);
        }
        if (got == 0 || memcmp(tmp, base + a*PAGE, got) != 0) {
            GGML_ABORT("lazy table check: pages %zu..%zu differ from the file", a, b);
        }
        VirtualFree(tmp, 0, MEM_RELEASE);
        if (++n_ok == 1 || n_ok % 100000 == 0) {
            LLAMA_LOG_INFO("lazy table check: %zu page runs match the file\n", n_ok);
        }
    }
}
#endif

struct llama_mmap::impl {
#ifdef _POSIX_MAPPED_FILES
    std::vector<std::pair<size_t, size_t>> mapped_fragments;

    impl(struct llama_file * file, size_t prefetch, bool numa, const llama_mmap::ranges & lazy_ranges) {
        size = file->size();
        int fd = file->file_id();
        int flags = MAP_SHARED;
        if (numa) { prefetch = 0; }
#ifdef __linux__
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
            LLAMA_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n",
                    strerror(errno));
        }
        // MAP_POPULATE would fault in the lazy ranges too
        if (prefetch && lazy_ranges.empty()) { flags |= MAP_POPULATE; }
#endif
        addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }

        // page-aligned madvise over [beg, end), clamped to the file
        auto advise = [&](size_t beg, size_t end, int advice, const char * name) {
            const size_t page_size = sysconf(_SC_PAGESIZE);
            beg = beg & ~(page_size - 1);
            end = std::min((end + page_size - 1) & ~(page_size - 1), file->size());
            if (beg >= end) {
                return;
            }
            if (posix_madvise((char *) addr + beg, end - beg, advice)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., %s) failed: %s\n", name, strerror(errno));
            }
        };

        if (prefetch > 0) {
            for (const auto & range : ranges_complement(lazy_ranges, std::min(file->size(), prefetch))) {
                advise(range.first, range.second, POSIX_MADV_WILLNEED, "POSIX_MADV_WILLNEED");
            }
        }
        for (const auto & range : lazy_ranges) {
            advise(range.first, range.second, POSIX_MADV_RANDOM, "POSIX_MADV_RANDOM");
        }
        if (numa) {
            if (posix_madvise(addr, file->size(), POSIX_MADV_RANDOM)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n",
                        strerror(errno));
            }
        }

        mapped_fragments.emplace_back(0, file->size());
    }

    static void align_range(size_t * first, size_t * last, size_t page_size) {
        size_t offset_in_page = *first & (page_size - 1);
        size_t offset_to_page = offset_in_page == 0 ? 0 : page_size - offset_in_page;
        *first += offset_to_page;

        *last = *last & ~(page_size - 1);

        if (*last <= *first) {
            *last = *first;
        }
    }

    void unmap_fragment(size_t first, size_t last) {
        int page_size = sysconf(_SC_PAGESIZE);
        align_range(&first, &last, page_size);
        size_t len = last - first;

        if (len == 0) {
            return;
        }

        GGML_ASSERT(first % page_size == 0);
        GGML_ASSERT(last % page_size == 0);
        GGML_ASSERT(last > first);

        void * next_page_start = (uint8_t *) addr + first;

        if (munmap(next_page_start, len)) {
            LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
        }

        std::vector<std::pair<size_t, size_t>> new_mapped_fragments;
        for (const auto & frag : mapped_fragments) {
            if (frag.first < first && frag.second > last) {
                new_mapped_fragments.emplace_back(frag.first, first);
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first < first && frag.second > first) {
                new_mapped_fragments.emplace_back(frag.first, first);
            } else if (frag.first < last && frag.second > last) {
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first >= first && frag.second <= last) {
            } else {
                new_mapped_fragments.push_back(frag);
            }
        }
        mapped_fragments = std::move(new_mapped_fragments);
    }

    ~impl() {
        for (const auto & frag : mapped_fragments) {
            if (munmap((char *) addr + frag.first, frag.second - frag.first)) {
                LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
            }
        }
    }
#elif defined(_WIN32)
    HANDLE hMapping = nullptr;
    std::unique_ptr<llama_win_read_region> region;

    impl(struct llama_file * file, size_t prefetch, bool numa, const llama_mmap::ranges & lazy_ranges) {
        GGML_UNUSED(numa);

        size = file->size();

        HANDLE hFile = (HANDLE) _get_osfhandle(file->file_id());

        // Direct-I/O load, mapped only for lazy tables: any mapped view makes NTFS send the stream's unbuffered expert reads to
        // the drive one at a time, so the table's pages are filled with our own unbuffered reads instead (LLAMA_WIN_LAZY_MAP=1:
        // the old file mapping)
        static const bool keep_map = getenv("LLAMA_WIN_LAZY_MAP") && atoi(getenv("LLAMA_WIN_LAZY_MAP")) != 0;
        if (file->has_direct_io() && prefetch == 0 && !keep_map) {
            region = std::make_unique<llama_win_read_region>(hFile, size);
            addr   = region->base;
            return;
        }

        hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);

        if (hMapping == NULL) {
            DWORD error = GetLastError();
            throw std::runtime_error(format("CreateFileMappingA failed: %s", llama_format_win_err(error).c_str()));
        }

        addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        DWORD error = GetLastError();

        if (addr == NULL) {
            CloseHandle(hMapping);
            throw std::runtime_error(format("MapViewOfFile failed: %s", llama_format_win_err(error).c_str()));
        }

        if (prefetch > 0) {
#if _WIN32_WINNT >= 0x602
            BOOL (WINAPI *pPrefetchVirtualMemory) (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
            HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

            pPrefetchVirtualMemory = (decltype(pPrefetchVirtualMemory))(void *) GetProcAddress(hKernel32, "PrefetchVirtualMemory");

            if (pPrefetchVirtualMemory) {
                std::vector<WIN32_MEMORY_RANGE_ENTRY> entries;
                for (const auto & range : ranges_complement(lazy_ranges, std::min(size, prefetch))) {
                    WIN32_MEMORY_RANGE_ENTRY entry;
                    entry.VirtualAddress = (char *) addr + range.first;
                    entry.NumberOfBytes  = (SIZE_T) (range.second - range.first);
                    entries.push_back(entry);
                }
                if (!entries.empty() &&
                        !pPrefetchVirtualMemory(GetCurrentProcess(), (ULONG_PTR) entries.size(), entries.data(), 0)) {
                    LLAMA_LOG_WARN("warning: PrefetchVirtualMemory failed: %s\n",
                            llama_format_win_err(GetLastError()).c_str());
                }
            }
#else
            LLAMA_LOG_DEBUG("skipping PrefetchVirtualMemory because _WIN32_WINNT < 0x602\n");
#endif
        }
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);
    }

    ~impl() {
        if (region) {
            addr = nullptr;   // the region frees its own reservation
        }
        if (hMapping) {
            if (addr) {
                if (!UnmapViewOfFile(addr)) {
                    LLAMA_LOG_WARN("warning: UnmapViewOfFile failed: %s\n",
                            llama_format_win_err(GetLastError()).c_str());
                }
            }
            if (!CloseHandle(hMapping)) {
                LLAMA_LOG_WARN("warning: CloseHandle failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
            }
        }
    }
#else
    impl(struct llama_file * file, size_t prefetch, bool numa, const llama_mmap::ranges & lazy_ranges) {
        GGML_UNUSED(file);
        GGML_UNUSED(prefetch);
        GGML_UNUSED(numa);
        GGML_UNUSED(lazy_ranges);

        throw std::runtime_error("mmap not supported");
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);

        throw std::runtime_error("mmap not supported");
    }
#endif

    void * addr;
    size_t size;
};

llama_mmap::llama_mmap(struct llama_file * file, size_t prefetch, bool numa,
        const ranges & lazy_ranges) : pimpl(std::make_unique<impl>(file, prefetch, numa, lazy_ranges)) {}
llama_mmap::~llama_mmap() = default;

size_t llama_mmap::size() const { return pimpl->size; }
void * llama_mmap::addr() const { return pimpl->addr; }

void llama_mmap::unmap_fragment(size_t first, size_t last) { pimpl->unmap_fragment(first, last); }

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool llama_mmap::SUPPORTED  = true;
#else
const bool llama_mmap::SUPPORTED  = false;
#endif

// llama_mlock

struct llama_mlock::impl {
#ifdef _POSIX_MEMLOCK_RANGE
    static size_t lock_granularity() {
        return (size_t) sysconf(_SC_PAGESIZE);
    }

    bool raw_lock(const void * addr, size_t size) const {
        if (!mlock(addr, size)) {
            return true;
        }

#ifdef __APPLE__
#define MLOCK_SUGGESTION \
        "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
        "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MEMLOCK (ulimit -l).\n"
#else
#define MLOCK_SUGGESTION \
        "Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).\n"
#endif

        char* errmsg = std::strerror(errno);
        bool suggest = (errno == ENOMEM);
#if defined(TARGET_OS_VISION) || defined(TARGET_OS_TV) || defined(_AIX) || defined(__HAIKU__)
        // visionOS/tvOS/Haiku don't support RLIMIT_MEMLOCK
        // Skip resource limit checks on these platforms
        suggest = false;
#else
        struct rlimit lock_limit;
        if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
            suggest = false;
        }
        if (suggest && ((uint64_t)lock_limit.rlim_max > (uint64_t)lock_limit.rlim_cur + size)) {
            suggest = false;
        }
#endif

        LLAMA_LOG_WARN("warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s",
                size, this->size, errmsg, suggest ? MLOCK_SUGGESTION : "");
        return false;
    }

    static void raw_unlock(void * addr, size_t size) {
        if (munlock(addr, size)) {
            LLAMA_LOG_WARN("warning: failed to munlock buffer: %s\n", std::strerror(errno));
        }
    }
#elif defined(_WIN32)
    static size_t lock_granularity() {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (size_t) si.dwPageSize;
    }

    bool raw_lock(void * ptr, size_t len) const {
        for (int tries = 1; ; tries++) {
            if (VirtualLock(ptr, len)) {
                return true;
            }
            if (tries == 2) {
                LLAMA_LOG_WARN("warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s\n",
                    len, size, llama_format_win_err(GetLastError()).c_str());
                return false;
            }

            SIZE_T min_ws_size, max_ws_size;
            if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size, &max_ws_size)) {
                LLAMA_LOG_WARN("warning: GetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
            size_t increment = len + 1048576;
            min_ws_size += increment;
            max_ws_size += increment;
            if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size, max_ws_size)) {
                LLAMA_LOG_WARN("warning: SetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
        }
    }

    static void raw_unlock(void * ptr, size_t len) {
        if (!VirtualUnlock(ptr, len)) {
            LLAMA_LOG_WARN("warning: failed to VirtualUnlock buffer: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    static size_t lock_granularity() {
        return (size_t) 65536;
    }

    bool raw_lock(const void * addr, size_t len) const {
        LLAMA_LOG_WARN("warning: mlock not supported on this system\n");
        return false;
    }

    static void raw_unlock(const void * addr, size_t len) {}
#endif

    impl() : addr(NULL), size(0), failed_already(false) {}

    void init(void * ptr) {
        GGML_ASSERT(addr == NULL && size == 0);
        addr = ptr;
    }

    void grow_to(size_t target_size) {
        GGML_ASSERT(addr);
        if (failed_already) {
            return;
        }
        size_t granularity = lock_granularity();
        target_size = (target_size + granularity - 1) & ~(granularity - 1);
        if (target_size > size) {
            if (raw_lock((uint8_t *) addr + size, target_size - size)) {
                size = target_size;
            } else {
                failed_already = true;
            }
        }
    }

    void * addr;
    size_t size;

    bool failed_already;
};

llama_mlock::llama_mlock() : pimpl(std::make_unique<impl>()) {}
llama_mlock::~llama_mlock() = default;

void llama_mlock::init(void * ptr) { pimpl->init(ptr); }
void llama_mlock::grow_to(size_t target_size) { pimpl->grow_to(target_size); }

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool llama_mlock::SUPPORTED = true;
#else
const bool llama_mlock::SUPPORTED = false;
#endif

void llama_prefetch(llama_memory_ranges mr) {
#if defined(__linux__) || defined(__APPLE__) || (defined(_WIN32) && _WIN32_WINNT >= 0x602)
    if (mr.empty()) {
        return;
    }

#if defined(_WIN32)
    llama_win_read_region_prefetch(mr);

    using prefetch_virtual_memory_t = BOOL (WINAPI *)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static const auto pPrefetchVirtualMemory = (prefetch_virtual_memory_t) (void *) GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory");
    if (!pPrefetchVirtualMemory) {
        return;
    }

    static const long page_size = [] {
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        return (long) info.dwPageSize;
    }();
#else
    static const long page_size = sysconf(_SC_PAGESIZE);
#endif
    if (page_size <= 0) {
        return;
    }

    const size_t page = (size_t) page_size;
    std::sort(mr.begin(), mr.end(), [](const llama_memory_range & a, const llama_memory_range & b) {
        return (uintptr_t) a.addr < (uintptr_t) b.addr;
    });

    uintptr_t begin = 0, end = 0;
    size_t n_ranges = 0;
    for (const auto & range : mr) {
        if (!range.addr || range.size == 0) {
            continue;
        }
        const uintptr_t pointer = (uintptr_t) range.addr;
        const uintptr_t first = pointer / page * page;
        const uintptr_t last = (pointer + range.size + page - 1) / page * page;
        if (end && first > end) {
            mr[n_ranges++] = { (const void *) begin, end - begin };
            end = 0;
        }
        if (!end) {
            begin = first;
        }
        end = std::max(end, last);
    }
    if (end) {
        mr[n_ranges++] = { (const void *) begin, end - begin };
    }
    mr.resize(n_ranges);
#if defined(_WIN32)
    // prefetch the merged ranges in one call, so the reads can be issued concurrently
    std::vector<WIN32_MEMORY_RANGE_ENTRY> entries;
    entries.reserve(mr.size());
    for (const auto & range : mr) {
        entries.push_back({ const_cast<PVOID>(range.addr), (SIZE_T) range.size });
    }
    if (!entries.empty() && !pPrefetchVirtualMemory(GetCurrentProcess(), (ULONG_PTR) entries.size(), entries.data(), 0)) {
        LLAMA_LOG_WARN("llama_prefetch: PrefetchVirtualMemory failed: %s\n",
                llama_format_win_err(GetLastError()).c_str());
    }
#else
    for (const auto & range : mr) {
        // TODO: in a follow-up, use posix_madvise(.., POSIX_MADV_WILLNEED) to match the rest of this file
        if (madvise(const_cast<void *>(range.addr), range.size, MADV_WILLNEED) != 0) {
            LLAMA_LOG_WARN("llama_prefetch: madvise(MADV_WILLNEED) failed: %s\n", strerror(errno));
            return;
        }
    }
#endif
#else
    GGML_UNUSED(mr);
#endif
}

size_t llama_path_max() {
    return PATH_MAX;
}
