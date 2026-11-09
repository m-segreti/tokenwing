"""Read routed-expert slices out of a GGUF file with O_DIRECT.

Uses the JSON produced by expert_index.py. Only the GGUF header is needed at run
time, so one index works for a fully downloaded model and for a partial one as
long as the requested bytes are already on disk.

A library module, not run directly: ExpertIndex.load("index.json") gives the
offsets, open_direct() and read_expert_slice() read the bytes.
"""

import json
import mmap
import os
import sys

GGUF_PY = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
if GGUF_PY not in sys.path:
    sys.path.insert(0, GGUF_PY)

PROJS = ("gate", "up", "down")
DEFAULT_ALIGN = 4096


class ExpertIndex:
    """Random access to expert slices of one GGUF, by (layer, expert, projection)."""

    def __init__(self, data):
        self.data = data
        self.model = data["model"]
        self.layers = data["layers"]
        self.align = data.get("io_align", DEFAULT_ALIGN)
        self.n_expert = self.model["n_expert"]
        self.n_expert_used = self.model["n_expert_used"]
        self.moe_layers = [int(il) for il in self.layers]
        self.path = self.model["path"]
        # a split GGUF keeps its tensor data in several files; every projection names the one it
        # is in and its offset is inside that file
        self.shards = self.model.get("shards") or [{"path": self.path, "size": self.model["file_size"]}]

    @classmethod
    def load(cls, json_path):
        with open(json_path) as f:
            return cls(json.load(f))

    @classmethod
    def from_model(cls, model_path, align=DEFAULT_ALIGN):
        import expert_index

        return cls(expert_index.build_index(model_path, align))

    def shard(self, layer, proj):
        """The index of the file this projection's bytes are in (0 unless the GGUF is split)."""
        return int(self.tensor(layer, proj).get("shard", 0))

    def shard_path(self, layer, proj):
        return self.shards[self.shard(layer, proj)]["path"]

    def open_shards(self, flags=os.O_RDONLY):
        """One fd per file of the model, for a reader that walks the layers."""
        return [open_direct(s["path"], flags) for s in self.shards]

    def layer(self, layer):
        try:
            return self.layers[str(layer)]["projections"]
        except KeyError:
            raise KeyError(
                f"layer {layer} has no routed experts (index has {self.moe_layers[:4]}...)") from None

    def tensor(self, layer, proj):
        return self.layer(layer)[proj]

    def slice_size(self, layer, proj):
        return self.tensor(layer, proj)["slice_size"]

    def slice_offset(self, layer, expert, proj):
        d = self.tensor(layer, proj)
        if not 0 <= expert < self.n_expert:
            raise IndexError(f"expert {expert} out of range (n_expert={self.n_expert})")
        return d["offset"] + expert * d["slice_size"]

    def slice_offsets(self, layer, expert):
        return {p: self.slice_offset(layer, expert, p) for p in PROJS}

    def bytes_per_expert(self, layer):
        return sum(self.slice_size(layer, p) for p in PROJS)

    def aligned_range(self, layer, expert, proj, align=None):
        """The 4096-aligned superset of one slice: the only range O_DIRECT can read."""
        align = align or self.align
        off = self.slice_offset(layer, expert, proj)
        size = self.slice_size(layer, proj)
        return off - off % align, -(-(off + size) // align) * align

    def read_plan(self, layer, expert, align=None):
        """The (file_offset, nbytes) ranges needed to read one whole expert."""
        return [self.aligned_range(layer, expert, p, align) for p in PROJS]


def open_direct(path, flags=os.O_RDONLY):
    """Open a GGUF for O_DIRECT reads (offset, length and buffer must be aligned)."""
    return os.open(path, flags | os.O_DIRECT)


def _pread_aligned(fd, start, end):
    """pread [start, end) into a page-aligned anonymous mapping; returns the bytes."""
    length = end - start
    buf = mmap.mmap(-1, length)
    try:
        view = memoryview(buf)
        try:
            got = 0
            while got < length:
                n = os.preadv(fd, [view[got:]], start + got)
                if n == 0:
                    raise EOFError(f"short read at {start + got}, wanted {length} bytes")
                got += n
        finally:
            view.release()
        return bytes(buf)
    finally:
        buf.close()


def read_expert_slice(fd, index, layer, expert, proj):
    """Return exactly slice_size(layer, proj) bytes of one expert projection."""
    off = index.slice_offset(layer, expert, proj)
    size = index.slice_size(layer, proj)
    start, end = index.aligned_range(layer, expert, proj)
    blob = _pread_aligned(fd, start, end)
    return blob[off - start:off - start + size]


def read_expert(fd, index, layer, expert):
    """Return {'gate': bytes, 'up': bytes, 'down': bytes} for one expert."""
    return {p: read_expert_slice(fd, index, layer, expert, p) for p in PROJS}


def read_expert_buffered(path, index, layer, expert, proj):
    """The same bytes as read_expert_slice, but through the normal page cache."""
    off = index.slice_offset(layer, expert, proj)
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(index.slice_size(layer, proj))


def load_sidecar(json_path):
    """Read the <sidecar>.json written by expert_repack.py."""
    with open(json_path) as f:
        return json.load(f)


def sidecar_block_range(meta, layer, expert):
    """The (file_offset, nbytes) of one expert block inside the sidecar."""
    try:
        blk = meta["blocks"][str(layer)]
    except KeyError:
        raise KeyError(f"sidecar has no layer {layer} (has {list(meta['blocks'])[:4]}...)") from None
    n_expert = meta["n_expert"]
    if not 0 <= expert < n_expert:
        raise IndexError(f"expert {expert} out of range (n_expert={n_expert})")
    return blk["base_offset"] + expert * blk["block_bytes"], blk["block_bytes"]


def read_expert_block(fd, meta, layer, expert):
    """Return (gate, up, down) for one expert with a single O_DIRECT read of its block."""
    off, block = sidecar_block_range(meta, layer, expert)
    if off % DEFAULT_ALIGN or block % DEFAULT_ALIGN:
        raise ValueError(f"sidecar block at {off} of {block} B is not {DEFAULT_ALIGN}-aligned")
    blob = _pread_aligned(fd, off, off + block)
    projs = meta["blocks"][str(layer)]["projections"]
    return tuple(blob[projs[p]["offset"]:projs[p]["offset"] + projs[p]["size"]] for p in PROJS)
