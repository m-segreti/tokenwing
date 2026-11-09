"""Read benchmarks for an expert sidecar: how fast and how evenly the SSD serves expert-sized O_DIRECT reads.

usage: readbench.py <sidecar> [mode] [mode arguments]
  <sidecar> is the experts file written by expert_repack.py. Its <sidecar>.json gives the size of one expert block,
  the read size most modes use. The reads are O_DIRECT, so the file must be on a filesystem that supports it.
  (no mode)                       bandwidth over 1, 8 and 32 threads: expert-block reads and 4 KiB random reads
  latency [samples]               one expert block read at queue depth 1, whole and cut into parallel chunks
  runs <trace dir> [widths...]    how many reads a layer's wanted set costs when neighbouring expert ids are merged
  coalesce [samples]              one read of n neighbouring experts vs n reads of one
  qd [samples]                    one demand read under a background of d independent readers
  mergeload [seconds per arm]     the same speculative bytes as one merged read or eight, under a background load
  batch [repeats] [n reads]       scattered reads in random order vs sorted by file offset
  sortload [seconds per arm] [background threads]   the demand read's latency when the speculative order is sorted
  buffers [samples]               read into a recycled buffer vs a freshly mapped one
  scatter [samples] [n]           the memcpy a merged read needs to fill n pool slots
  abort [samples]                 latency of the pieces a read can be cut into
  release [reads per thread] [threads]   cost of giving a RAM slot back and taking it again, or moving its pages
"""
import os, mmap, random, threading, time, sys, json
if len(sys.argv) < 2:
    sys.exit(__doc__)
path = sys.argv[1]
size = os.path.getsize(path)
with open(path + ".json") as _f:
    _meta = json.load(_f)
# One expert block of the sidecar. A sidecar whose layers have different block sizes has no single one: take the largest.
BLOCK = _meta["block_bytes"] or max(_meta["block_bytes_per_layer"].values())
def bench(chunk, threads, secs=6, direct=True):
    chunk = (chunk // 4096) * 4096
    total = [0]; lock = threading.Lock(); stop = time.time() + secs
    def worker(seed):
        rnd = random.Random(seed)
        fd = os.open(path, os.O_RDONLY | (os.O_DIRECT if direct else 0))
        buf = mmap.mmap(-1, chunk); n = 0
        while time.time() < stop:
            off = rnd.randrange(0, (size - chunk) // 4096) * 4096
            n += os.preadv(fd, [buf], off)
        os.close(fd)
        with lock: total[0] += n
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]
    return total[0] / (time.time() - t0) / 1e9
def sweep():
    for nbytes, label in [(BLOCK, f"expert block ({BLOCK/1e6:.1f}MB)"), (4096, "4KB random")]:
        for th in (1, 8, 32):
            gbps = bench(nbytes, th)
            print(f"{label:22} threads={th:2}  {gbps:6.2f} GB/s" + (f"  ({gbps*1e9/4096/1e3:.0f}k IOPS)" if nbytes == 4096 else ""), flush=True)

# The latency of one expert read (the sidecar's block size, O_DIRECT, QD1) against the same bytes read as
# N parallel page-aligned chunks. That is the premise of the chunked demand read: decode has about one
# miss per layer, so a demand read is a single preadv on a single I/O thread while the other I/O threads idle.
# The workers are started once and released with a barrier per sample: starting a thread per sample
# would add ~0.2 ms and hide the effect. A 4 KiB control read gives the barrier's own cost.
# usage: readbench.py <sidecar> latency [samples]
def latency(samples=200, block=BLOCK, page=4096):
    import statistics
    n_pages = block // page
    print(f"latency: {samples} samples, a {block} B expert block ({n_pages} pages) at random offsets, O_DIRECT")
    for n_chunks in (1, 3, 6, 12, 24):
        per = n_pages // n_chunks          # page-aligned chunks, the last one takes the remainder
        iov = []
        for i in range(n_chunks):
            first = i*per
            last = n_pages if i == n_chunks-1 else (i+1)*per
            iov.append((first*page, (last-first)*page))
        if n_chunks == 1:
            iov = [(0, block)]
        bufs = [mmap.mmap(-1, ln if n_chunks > 1 else block) for _, ln in iov]
        fds = [os.open(path, os.O_RDONLY | os.O_DIRECT) for _ in range(n_chunks)]
        base_holder = [0]
        go = threading.Barrier(n_chunks + 1)
        done = threading.Barrier(n_chunks + 1)
        def worker(i):
            while True:
                go.wait()
                off, ln = iov[i]
                try:
                    os.preadv(fds[i], [bufs[i]], base_holder[0] + off)
                except OSError:
                    pass
                done.wait()
        ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n_chunks)]
        [t.start() for t in ths]
        rnd = random.Random(27)
        lat = []
        for s_i in range(samples):
            base_holder[0] = rnd.randrange(0, (size - block) // 4096) * 4096
            t0 = time.perf_counter()
            go.wait()
            done.wait()
            lat.append((time.perf_counter() - t0)*1e3)
        for t in ths: pass
        for fd in fds: os.close(fd)
        for b in bufs: b.close()
        lat.sort()
        p50, p90 = lat[len(lat)//2], lat[int(0.9*len(lat))]
        print(f"  {n_chunks:2d} chunk(s) of {iov[0][1]//1024:5d} KiB: p50 {p50:6.3f} ms, p90 {p90:6.3f} ms, "
              f"{block/(p50/1e3)/1e9:5.2f} GB/s at p50", flush=True)
    # the barrier and the syscall overhead alone
    bufs = [mmap.mmap(-1, 4096)]
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    lat = []
    for _ in range(samples):
        t0 = time.perf_counter()
        os.preadv(fd, bufs, 0)
        lat.append((time.perf_counter() - t0)*1e3)
    lat.sort()
    print(f"  control, one 4 KiB read on the main thread: p50 {lat[len(lat)//2]:6.3f} ms")


# ================================================================================================
# Read shape and queue depth, measured against the sidecar without a model. Every expert of a layer is
# at base_offset[layer] + e*block_bytes and block_bytes is a whole number of 4096 B pages, so a *run*
# of neighbouring experts is one aligned read of n*block_bytes - that is what "coalesce" exploits, and
# what it costs on the CPU (a scatter into n pool slots) is measured here too.
PAGESZ = 4096
assert BLOCK % PAGESZ == 0

def _open_odirect():
    return os.open(path, os.O_RDONLY | os.O_DIRECT)

def _rand_base(rnd, nbytes):
    return rnd.randrange(0, (size - nbytes) // PAGESZ) * PAGESZ

# ---- how often are the experts we want to read neighbours in the sidecar? ----------
# usage: readbench.py <sidecar> runs <trace dir> [top widths...]
# <trace dir> holds llama-moe-trace recordings: <prefix>.bin (int32 [n_tokens][n_moe_layers][k]) and <prefix>.json.
def runs(trace_dir, widths=(1, 2, 4, 8)):
    import glob, json as _json
    import numpy as np
    print(f"merge runs: how many reads a layer's wanted set costs, from the routing traces in {trace_dir}")
    print(f"{'M':>3} {'set':>7} {'runs':>6} {'merge x':>8} {'P(adj pair)':>11} {'gap p50':>8} "
          f"{'runs p90':>9}   (set = wanted ids, runs = maximal runs of consecutive ids)")
    tot = {}
    for b in sorted(glob.glob(os.path.join(trace_dir, "*.bin"))):
        if b.endswith(".routing.bin") or b.endswith(".hin.bin"):
            continue
        j = b[:-4] + ".json"
        if not os.path.exists(j):
            continue
        m = _json.load(open(j))
        n, k = m["n_tokens"], m["k"]
        a = np.fromfile(b, dtype=np.int32)
        if a.size % (n * k) != 0:
            continue
        L = a.size // n // k
        a = a.reshape(n, L, k)
        for M in widths:
            s = np.sort(a[:, :, :M], axis=2)                 # [n, L, M] sorted ids
            diff = np.diff(s, axis=2)
            adj = (diff == 1)
            n_runs = 1 + (diff != 1).sum(axis=2)              # maximal runs of consecutive ids
            pairs = adj.sum()
            gaps = diff[diff > 1]
            tot.setdefault(M, []).append((n_runs.ravel(), pairs, gaps, M, adj.sum(axis=2).size))
    for M, parts in sorted(tot.items()):
        nr = np.concatenate([p[0] for p in parts])
        pairs = sum(p[1] for p in parts); tot_pairs = sum(p[2].size for p in parts)
        gaps = np.concatenate([p[2] for p in parts]) if any(len(p[2]) for p in parts) else np.array([0])
        print(f"{M:3d} {M:7d} {nr.mean():6.3f} {M/np.clip(nr.mean(),1,None):8.2f} "
              f"{pairs/max(tot_pairs,1):11.4f} {np.median(gaps):8.0f} {np.percentile(nr,90):9.1f}")
    print("  a set of M ids costs one read per run; 'runs p90' is the worst 10 % of (token, layer) sets")

# ---- one read of n neighbouring experts vs n reads of one ----------------------------------------
# usage: readbench.py <sidecar> coalesce [samples]
def coalesce(samples=150, ns=(1, 2, 4, 8, 16)):
    print(f"coalesce: one O_DIRECT read of n x {BLOCK} B (a run of neighbouring experts of one layer), "
          f"QD1, {samples} samples")
    print(f"{'n':>3} {'MiB':>7} {'p50 ms':>8} {'p90 ms':>8} {'p99 ms':>8} {'GB/s p50':>9} {'ms/expert':>10}")
    for n in ns:
        nbytes = n * BLOCK
        fd = _open_odirect()
        buf = mmap.mmap(-1, nbytes)
        rnd = random.Random(101)
        base_holder = [0]
        go = threading.Barrier(2); done = threading.Barrier(2)
        def worker():
            while True:
                go.wait()
                try:
                    os.preadv(fd, [buf], base_holder[0])
                except OSError:
                    pass
                done.wait()
        th = threading.Thread(target=worker, daemon=True); th.start()
        lat = []
        for _ in range(samples):
            base_holder[0] = _rand_base(rnd, nbytes)
            t0 = time.perf_counter(); go.wait(); done.wait()
            lat.append((time.perf_counter() - t0) * 1e3)
        os.close(fd); buf.close()
        lat.sort()
        p50, p90, p99 = lat[len(lat)//2], lat[int(0.9*len(lat))], lat[int(0.99*len(lat))]
        print(f"{n:3d} {nbytes/2**20:7.1f} {p50:8.3f} {p90:8.3f} {p99:8.3f} "
              f"{nbytes/(p50/1e3)/1e9:9.2f} {p50/n:10.3f}", flush=True)

# ---- where is the knee? one demand read under a background of depth d ---------------------------
# usage: readbench.py <sidecar> qd [samples]
def qd(samples=150, secs=2.0, depths=(0, 1, 2, 4, 8, 12, 16)):
    print(f"queue depth: one {BLOCK} B O_DIRECT demand read under a background of d independent readers")
    print(f"{'d':>3} {'p50 ms':>8} {'p90 ms':>8} {'p99 ms':>8} {'bg GB/s':>8} {'bg/s':>7}")
    for d in depths:
        stop = [False]; lk = threading.Lock(); live = []
        def bg(i):
            f = _open_odirect(); b = mmap.mmap(-1, BLOCK); r = random.Random(1000+i); loc = [0]
            live.append(loc)
            while not stop[0]:
                try:
                    loc[0] += os.preadv(f, [b], _rand_base(r, BLOCK))
                except OSError:
                    pass
            os.close(f); b.close()
        ths = [threading.Thread(target=bg, args=(i,), daemon=True) for i in range(d)]
        t_start = time.time()
        [t.start() for t in ths]
        time.sleep(min(1.0, secs))                     # let the background reach steady state
        fd = _open_odirect(); buf = mmap.mmap(-1, BLOCK); rnd = random.Random(27)
        base_holder = [0]
        go = threading.Barrier(2); done = threading.Barrier(2)
        def worker():
            while True:
                go.wait()
                try:
                    os.preadv(fd, [buf], base_holder[0])
                except OSError:
                    pass
                done.wait()
        th = threading.Thread(target=worker, daemon=True); th.start()
        bytes_at_start = sum(c[0] for c in live)          # the warm-up's bytes are not in the window
        lat = []; t0all = time.perf_counter()
        for _ in range(samples):
            base_holder[0] = _rand_base(rnd, BLOCK)
            t0 = time.perf_counter(); go.wait(); done.wait()
            lat.append((time.perf_counter() - t0) * 1e3)
        elapsed = time.perf_counter() - t0all
        stop[0] = True
        [t.join(timeout=5) for t in ths]
        moved = sum(c[0] for c in live) - bytes_at_start
        os.close(fd); buf.close()
        lat.sort()
        print(f"{d:3d} {lat[len(lat)//2]:8.3f} {lat[int(0.9*len(lat))]:8.3f} {lat[int(0.99*len(lat))]:8.3f} "
              f"{moved/1e9/max(elapsed,1e-9):8.2f} {samples/max(elapsed,1e-9):7.1f}", flush=True)
        time.sleep(0.3)

# ---- coalescing against a real queue: same speculative bytes, one merged read or eight -----------
# usage: readbench.py <sidecar> mergeload [seconds per arm]
def mergeload(secs=3.0):
    print(f"merge load: a background of 8 readers moves the same bytes as either 8 x {BLOCK} B reads "
          f"or 1 x {8*BLOCK} B; the probe is one {BLOCK} B demand read")
    print(f"{'arm':>10} {'probe p50':>10} {'p90':>8} {'p99':>8} {'bg GB/s':>8} {'bg reads/s':>11}")
    for arm, nbytes, nreads in (("8 separate", BLOCK, 8), ("1 merged", 8*BLOCK, 1)):
        stop = [False]; got = [0]; cnt = [0]; lk = threading.Lock()
        def bg(i):
            f = _open_odirect(); b = mmap.mmap(-1, nbytes); r = random.Random(2000+i); loc = [0]; lc = [0]
            while not stop[0]:
                for _ in range(nreads):
                    if stop[0]:
                        break
                    try:
                        loc[0] += os.preadv(f, [b], _rand_base(r, nbytes))
                    except OSError:
                        pass
                    lc[0] += 1
            with lk: got[0] += loc[0]; cnt[0] += lc[0]
            os.close(f); b.close()
        ths = [threading.Thread(target=bg, args=(i,), daemon=True) for i in range(8)]
        t_start = time.time(); [t.start() for t in ths]
        time.sleep(0.8)
        fd = _open_odirect(); buf = mmap.mmap(-1, BLOCK); rnd = random.Random(27)
        base_holder = [0]
        go = threading.Barrier(2); done = threading.Barrier(2)
        def worker():
            while True:
                go.wait()
                try:
                    os.preadv(fd, [buf], base_holder[0])
                except OSError:
                    pass
                done.wait()
        th = threading.Thread(target=worker, daemon=True); th.start()
        lat = []; n = 0; t0all = time.perf_counter()
        while time.perf_counter() - t0all < secs:
            base_holder[0] = _rand_base(rnd, BLOCK)
            t0 = time.perf_counter(); go.wait(); done.wait()
            lat.append((time.perf_counter() - t0) * 1e3); n += 1
            time.sleep(0.002)
        elapsed = time.perf_counter() - t0all
        stop[0] = True; [t.join(timeout=5) for t in ths]
        os.close(fd); buf.close(); lat.sort()
        print(f"{arm:>10} {lat[len(lat)//2]:10.3f} {lat[int(0.9*len(lat))]:8.3f} "
              f"{lat[min(len(lat)-1,int(0.99*len(lat)))]:8.3f} {got[0]/1e9/elapsed:8.2f} "
              f"{cnt[0]/elapsed:11.1f}", flush=True)
        time.sleep(0.3)

# ---- does the order of a deadline class matter? 40 scattered reads, random vs by offset ---------
# usage: readbench.py <sidecar> batch [repeats] [n reads]
def batch(repeats=15, nreads=40):
    print(f"batch: {nreads} scattered {BLOCK/2**20:.1f} MiB reads ({nreads*BLOCK/2**30:.2f} GiB), "
          f"{repeats} repeats, submitted one at a time at QD1 - random order vs sorted by file offset")
    print(f"{'arm':>16} {'total s':>9} {'per read ms':>12} {'GB/s':>7}")
    for arm in ("random", "by offset"):
        tot = []
        for r in range(repeats):
            rnd = random.Random(4000 + r)
            offs = [_rand_base(rnd, BLOCK) for _ in range(nreads)]
            if arm == "by offset":
                offs.sort()
            fd = _open_odirect(); buf = mmap.mmap(-1, BLOCK)
            # one warm read so the first sample is not the open's own cost
            try: os.preadv(fd, [buf], offs[0])
            except OSError: pass
            t0 = time.perf_counter()
            for o in offs:
                try: os.preadv(fd, [buf], o)
                except OSError: pass
            tot.append(time.perf_counter() - t0)
            os.close(fd); buf.close()
        tot.sort(); med = tot[len(tot)//2]
        print(f"{arm:>16} {med:9.3f} {med/nreads*1e3:12.3f} {nreads*BLOCK/med/1e9:7.2f}", flush=True)

# ---- the number that matters: does the speculative order change the *demand* read's latency? ----
# usage: readbench.py <sidecar> sortload [seconds per arm] [background threads]
def sortload(secs=3.0, nbg=8):
    print(f"sort load: a background of {nbg} readers, the same speculative bytes submitted in random "
          f"order or sorted by file offset; the probe is one {BLOCK} B demand read")
    print(f"{'arm':>10} {'probe p50':>10} {'p90':>8} {'p99':>8} {'bg GB/s':>8} {'bg reads/s':>11}")
    for arm in ("random", "by offset"):
        stop = [False]; live = []; lk = threading.Lock()
        def bg(i):
            f = _open_odirect(); b = mmap.mmap(-1, BLOCK); r = random.Random(3000+i)
            loc = [0]; pending = []
            while not stop[0]:
                pending = [_rand_base(r, BLOCK) for _ in range(8)]
                if arm == "by offset":
                    pending.sort()
                for o in pending:
                    if stop[0]:
                        break
                    try:
                        loc[0] += os.preadv(f, [b], o)
                    except OSError:
                        pass
            with lk: live.append(loc)
            os.close(f); b.close()
        ths = [threading.Thread(target=bg, args=(i,), daemon=True) for i in range(nbg)]
        [t.start() for t in ths]
        time.sleep(1.0)                     # reach steady state and register the byte counters
        fd = _open_odirect(); buf = mmap.mmap(-1, BLOCK); rnd = random.Random(27)
        base_holder = [0]
        go = threading.Barrier(2); done = threading.Barrier(2)
        def worker():
            while True:
                go.wait()
                try:
                    os.preadv(fd, [buf], base_holder[0])
                except OSError:
                    pass
                done.wait()
        th = threading.Thread(target=worker, daemon=True); th.start()
        lat = []; t0all = time.perf_counter()
        while time.perf_counter() - t0all < secs:
            base_holder[0] = _rand_base(rnd, BLOCK)
            t0 = time.perf_counter(); go.wait(); done.wait()
            lat.append((time.perf_counter() - t0) * 1e3)
            time.sleep(0.002)
        elapsed = time.perf_counter() - t0all
        stop[0] = True; [t.join(timeout=5) for t in ths]
        moved = sum(c[0] for c in live)
        os.close(fd); buf.close(); lat.sort()
        n_reads = moved // BLOCK
        print(f"{arm:>10} {lat[len(lat)//2]:10.3f} {lat[int(0.9*len(lat))]:8.3f} "
              f"{lat[min(len(lat)-1,int(0.99*len(lat)))]:8.3f} {moved/1e9/elapsed:8.2f} "
              f"{n_reads/elapsed:11.1f}", flush=True)
        time.sleep(0.3)

# ---- what does a fresh (unfaulted) destination cost, and does the engine pay it? ----------------
# usage: readbench.py <sidecar> buffers [samples]
def buffers(samples=120):
    print(f"buffers: the same {BLOCK} B O_DIRECT read into a *recycled* buffer (what the engine does: "
          f"it reads straight into the pool slot) vs a freshly mmap'd one that is faulted in by the read")
    print(f"{'arm':>22} {'p50 ms':>8} {'p90 ms':>8} {'p99 ms':>8} {'GB/s':>7}")
    for arm in ("recycled", "fresh mmap", "fresh + memset"):
        fd = _open_odirect(); rnd = random.Random(27)
        buf = mmap.mmap(-1, BLOCK) if arm == "recycled" else None
        try: os.preadv(fd, [buf or mmap.mmap(-1, BLOCK)], 0)     # warm the recycled one
        except OSError: pass
        lat = []
        for _ in range(samples):
            b = buf
            if arm != "recycled":
                b = mmap.mmap(-1, BLOCK)
                if arm == "fresh + memset":
                    b.write(b"\0" * BLOCK)
            t0 = time.perf_counter()
            try: os.preadv(fd, [b], _rand_base(rnd, BLOCK))
            except OSError: pass
            lat.append((time.perf_counter() - t0) * 1e3)
            if b is not buf:
                b.close()
        os.close(fd)
        if buf is not None: buf.close()
        lat.sort()
        print(f"{arm:>22} {lat[len(lat)//2]:8.3f} {lat[int(0.9*len(lat))]:8.3f} "
              f"{lat[min(len(lat)-1,int(0.99*len(lat)))]:8.3f} {BLOCK/(lat[len(lat)//2]/1e3)/1e9:7.2f}",
              flush=True)

# ---- the other side of coalescing: the scatter copy. A merged read lands in scratch, the pool wants n slots ----
# usage: readbench.py <sidecar> scatter [samples] [n]
def scatter(samples=40, n=8):
    # bytearray, not mmap: CPython's mmap.write() refuses a second full-length write of an
    # anonymous mapping (ValueError "data out of range"), and a bytearray slice assignment is the
    # same memcpy. Two footprints: "hot" is 2 x n blocks (cache-resident if they fit, the optimistic
    # case) and "big" strides the destinations 8 blocks apart so both ends come from DRAM, which is
    # what the engine would do (a layer's pool is far larger than the cache).
    print(f"scatter: the copy a merged read would need - {n} experts of one layer into {n} pool "
          f"slots, one {BLOCK:,} B memcpy per expert")
    for arm, stride in (("hot (L3)", BLOCK), ("big (DRAM)", 8 * BLOCK)):
        src = bytearray(n * stride)
        src[:] = b"\x5a" * (n * stride)
        dsts = [bytearray(stride) for _ in range(n)]
        lat = []
        for _ in range(samples):
            t0 = time.perf_counter()
            for i2, d in enumerate(dsts):
                d[:BLOCK] = src[i2*stride:i2*stride + BLOCK]
            lat.append((time.perf_counter() - t0) * 1e3)
        lat.sort(); p50 = lat[len(lat)//2]
        print(f"  {arm:>11}: {n} experts in p50 {p50:6.3f} ms = {p50/n:6.3f} ms/expert, "
              f"{n*BLOCK/(p50/1e3)/1e9:5.2f} GB/s")

# ---- the granularity at which a read can be abandoned --------------------------------------------
# usage: readbench.py <sidecar> abort [samples]
def abort(samples=150):
    print(f"abort: a {BLOCK} B read cut into page-aligned pieces - the piece is what a cancel can "
          f"stop at, so the piece's own latency is the price of giving up early")
    print(f"{'piece KiB':>10} {'n':>4} {'p50 ms':>8} {'p90 ms':>8} {'GB/s':>7}")
    for kib in (64, 256, 512, 1024, BLOCK // 1024):
        nbytes = kib * 1024
        n = BLOCK // nbytes
        fd = _open_odirect(); buf = mmap.mmap(-1, nbytes); rnd = random.Random(27)
        lat = []
        for _ in range(samples):
            base = _rand_base(rnd, nbytes)
            t0 = time.perf_counter()
            try: os.preadv(fd, [buf], base)
            except OSError: pass
            lat.append((time.perf_counter() - t0) * 1e3)
        os.close(fd); buf.close(); lat.sort()
        print(f"{kib:10d} {n:4d} {lat[len(lat)//2]:8.3f} {lat[int(0.9*len(lat))]:8.3f} "
              f"{nbytes/(lat[len(lat)//2]/1e3)/1e9:7.2f}", flush=True)

# ---- what does moving a RAM slot between layers cost? --------------------------------------------
# usage: readbench.py <sidecar> release [reads per thread] [threads]
# Each thread owns two slot-sized regions (BLOCK bytes, page-aligned anonymous memory) and reads random expert-sized blocks with
# O_DIRECT into them, like the engine's I/O threads. Arms:
#   resident - the slot's pages stay (a slot is reused in its own layer);
#   released - madvise(MADV_DONTNEED) before each read (a slot given back and taken again: the read faults in fresh pages the
#              kernel clears);
#   moved    - the pages of the slot read last are moved onto the slot read next with mremap(MREMAP_FIXED) and the old range is
#              mapped again empty (the Linux page move: no clearing).
# Prints per arm: GB/s over all threads, the read's p50 / p90 ms, and the process CPU seconds (user + system) per GB read.
def release(nreads=150, nthreads=8):
    import ctypes, resource
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mremap.restype = ctypes.c_void_p
    libc.mremap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    PROT_RW, MAP_PRIV_ANON, MAP_FIXED, MADV_DONTNEED, MREMAP_MAYMOVE, MREMAP_FIXED = 3, 0x22, 0x10, 4, 1, 2
    def amap(addr=None):
        p = libc.mmap(addr, BLOCK, PROT_RW, MAP_PRIV_ANON | (MAP_FIXED if addr else 0), -1, 0)
        assert p not in (None, ctypes.c_void_p(-1).value), "mmap failed"
        return p
    def buf(addr):
        return (ctypes.c_char * BLOCK).from_address(addr)
    print(f"release: {nthreads} threads x {nreads} O_DIRECT reads of {BLOCK} B per arm into slot-sized anonymous memory "
          f"(MADV_DONTNEED = giving a slot back; mremap = the Linux page move)")
    print(f"{'arm':>10} {'GB/s':>7} {'p50 ms':>8} {'p90 ms':>8} {'CPU s/GB':>9} {'sys s/GB':>9}")
    for arm in (os.environ.get("RB_ARMS") or "resident released moved resident").split():
        lat = []; lock = threading.Lock()
        def worker(i):
            fd = _open_odirect(); rnd = random.Random(1000 + i)
            slot = [amap(), amap()]
            for a in slot:
                ctypes.memset(a, 1, BLOCK)          # resident pages to start with
            my = []
            for n in range(nreads):
                dst = slot[n % 2]
                t0 = time.perf_counter()
                if arm == "released":
                    libc.madvise(dst, BLOCK, MADV_DONTNEED)
                elif arm == "moved":
                    src = slot[(n + 1) % 2]         # the slot read last: its pages move onto dst
                    r = libc.mremap(src, BLOCK, BLOCK, MREMAP_MAYMOVE | MREMAP_FIXED, dst)
                    assert r == dst, "mremap failed: errno %d" % ctypes.get_errno()
                    amap(src)                       # the old range mapped again, empty (no RAM); next round dst is read
                                                    # and its pages move onto this range, so every read lands in moved pages
                os.preadv(fd, [buf(dst)], _rand_base(rnd, BLOCK))
                my.append((time.perf_counter() - t0) * 1e3)
            os.close(fd)
            with lock: lat.extend(my)
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        ts = [threading.Thread(target=worker, args=(i,)) for i in range(nthreads)]
        t0 = time.time(); [t.start() for t in ts]; [t.join() for t in ts]; dt = time.time() - t0
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        gb = nthreads * nreads * BLOCK / 1e9
        cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime); sy = r1.ru_stime - r0.ru_stime
        lat.sort()
        print(f"{arm:>10} {gb/dt:7.2f} {lat[len(lat)//2]:8.3f} {lat[int(0.9*len(lat))]:8.3f} {cpu/gb:9.3f} {sy/gb:9.3f}", flush=True)

if len(sys.argv) > 3 and sys.argv[2] == "runs":
    runs(sys.argv[3], tuple(int(x) for x in sys.argv[4:]) or (1, 2, 4, 8))
elif len(sys.argv) > 2 and sys.argv[2] == "coalesce":
    coalesce(int(sys.argv[3]) if len(sys.argv) > 3 else 150)
elif len(sys.argv) > 2 and sys.argv[2] == "qd":
    qd(int(sys.argv[3]) if len(sys.argv) > 3 else 150)
elif len(sys.argv) > 2 and sys.argv[2] == "mergeload":
    mergeload(float(sys.argv[3]) if len(sys.argv) > 3 else 3.0)
elif len(sys.argv) > 2 and sys.argv[2] == "batch":
    batch(int(sys.argv[3]) if len(sys.argv) > 3 else 15, int(sys.argv[4]) if len(sys.argv) > 4 else 40)
elif len(sys.argv) > 2 and sys.argv[2] == "buffers":
    buffers(int(sys.argv[3]) if len(sys.argv) > 3 else 120)
elif len(sys.argv) > 2 and sys.argv[2] == "sortload":
    sortload(float(sys.argv[3]) if len(sys.argv) > 3 else 3.0,
             int(sys.argv[4]) if len(sys.argv) > 4 else 8)
elif len(sys.argv) > 2 and sys.argv[2] == "scatter":
    scatter(int(sys.argv[3]) if len(sys.argv) > 3 else 40, int(sys.argv[4]) if len(sys.argv) > 4 else 8)
elif len(sys.argv) > 2 and sys.argv[2] == "abort":
    abort(int(sys.argv[3]) if len(sys.argv) > 3 else 150)
elif len(sys.argv) > 2 and sys.argv[2] == "release":
    release(int(sys.argv[3]) if len(sys.argv) > 3 else 150, int(sys.argv[4]) if len(sys.argv) > 4 else 8)
elif len(sys.argv) > 2 and sys.argv[2] == "latency":
    latency(int(sys.argv[3]) if len(sys.argv) > 3 else 200)
else:
    sweep()
