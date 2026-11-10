#!/usr/bin/env python3
"""Tests for make_model_file.py (no pytest needed, no model needed: the test writes a small GGUF of its own).

Usage: python3 test_make_model_file.py
  Writes into the system temp directory and removes everything at the end.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent.parent / "gguf-py"))

import gguf  # noqa: E402

SCRIPT = TOOLS / "make_model_file.py"
WORK = Path(tempfile.mkdtemp(prefix="make_model_file_test_"))
N_EXPERT = 5


def write_gguf(path, slice_floats):
    """A model of 2 dense-ish tensors and 2 MoE layers; one expert slice is slice_floats * 4 bytes."""
    w = gguf.GGUFWriter(str(path), "bailingmoe3")
    w.add_string("general.name", "tiny test")
    w.add_array("tokenizer.ggml.tokens", ["a", "b", "c"])
    w.add_uint32("bailingmoe3.block_count", 3)
    w.add_custom_alignment(32)
    rng = np.random.default_rng(1)
    w.add_tensor("token_embd.weight", rng.standard_normal((3, 7)).astype(np.float32))
    for il in (1, 2):
        for p in ("gate", "up", "down"):
            w.add_tensor(f"blk.{il}.ffn_{p}_exps.weight", rng.standard_normal((N_EXPERT, 2, slice_floats // 2)).astype(np.float32))
        w.add_tensor(f"blk.{il}.attn_norm.weight", rng.standard_normal((9,)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def main():
    failed = 0
    try:
        src = WORK / "m.gguf"
        out = WORK / "m.tw.gguf"
        write_gguf(src, 2048)   # 8192 bytes per expert slice
        seed = WORK / "seed.json"
        seed.write_text(json.dumps({"format": "tokenwing usage v1", "n_expert": N_EXPERT, "layers": {}}))

        cases = []

        def case(fn):
            cases.append(fn)
            return fn

        @case
        def converts_and_checks():
            r = run(src, "-o", out, "--usage-seed", seed, "--jobs", 2)
            assert r.returncode == 0, r.stderr + r.stdout
            assert "all " in r.stdout and "tensor bytes identical" in r.stdout, r.stdout
            assert not Path(str(out) + ".tmp").exists()
            rd = gguf.GGUFReader(str(out))
            assert rd.alignment == 4096 and rd.data_offset % 4096 == 0
            assert rd.fields["tokenwing.format"].contents() == "tokenwing model v1"
            assert len(rd.fields["tokenwing.sums.values"].contents()) == 2 * 9
            off = int(rd.fields["tokenwing.blob.usage_seed.offset"].contents())
            size = int(rd.fields["tokenwing.blob.usage_seed.size"].contents())
            assert off % 4096 == 0 and out.read_bytes()[off:off + size] == seed.read_bytes()
            assert out.stat().st_size == off + size

        @case
        def refuses_a_split_name():
            split = WORK / "x-00001-of-00002.gguf"
            shutil.copy(src, split)
            r = run(split, "-o", WORK / "x.out.gguf")
            assert r.returncode == 1 and "split" in r.stderr, r.stderr
            assert not (WORK / "x.out.gguf").exists()

        @case
        def refuses_unaligned_slices():
            odd = WORK / "odd.gguf"
            write_gguf(odd, 30)   # 120 bytes per slice
            r = run(odd, "-o", WORK / "odd.out.gguf")
            assert r.returncode == 1 and "not a multiple of 4096" in r.stderr and "blk.1.ffn_gate_exps.weight" in r.stderr, r.stderr
            assert not (WORK / "odd.out.gguf").exists() and not (WORK / "odd.out.gguf.tmp").exists()

        @case
        def refuses_a_wrong_blob():
            bad = WORK / "bad.json"
            bad.write_text(json.dumps({"format": "something else"}))
            r = run(src, "-o", WORK / "b.out.gguf", "--usage-seed", bad)
            assert r.returncode == 1 and "tokenwing usage v1" in r.stderr, r.stderr

        for fn in cases:
            try:
                fn()
                print(f"PASS {fn.__name__}")
            except Exception:
                failed += 1
                print(f"FAIL {fn.__name__}")
                traceback.print_exc()
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
    print(f"{failed} failures")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
