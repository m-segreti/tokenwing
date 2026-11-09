#!/usr/bin/env python3
"""Compare two builds, or two sets of settings, on your own machine: is the new one faster, and does it write the same text?

    python3 compare.py --old BUILD_A --new BUILD_B --model M.gguf --experts M.experts.json [--prompt FILE] [--words N] [--pairs N] [--ub N]
                       [--old-args "..."] [--new-args "..."] [-- more arguments for both]

To compare two settings, give the same build twice and put the settings in --old-args and --new-args. Runs alternate old / new so a machine
that warms up or slows down affects both alike, and every run gets the same command otherwise. Keep the machine quiet and plugged in while it runs.
Differences of a few percent between runs are normal; trust a small gain only when most pairs agree. Different text is not an error by itself:
rounding differs between execution paths. Text that stops making sense is.
"""
import argparse
import glob
import os
import shlex
import statistics
import sys

from runlib import PROMPTS, bin_dir, generate, numbers


def on_battery():
    online = glob.glob("/sys/class/power_supply/A*/online")
    return bool(online) and all(open(p).read().strip() == "0" for p in online)


def main():
    ap = argparse.ArgumentParser(description="Compare two builds or settings on this machine.")
    ap.add_argument("--old", required=True, help="build directory of the old side")
    ap.add_argument("--new", required=True, help="build directory of the new side")
    ap.add_argument("--model", required=True)
    ap.add_argument("--experts", required=True, help="the model's experts file (the .json)")
    ap.add_argument("--prompt", default=os.path.join(PROMPTS, "05_math_word_problem.txt"))
    ap.add_argument("--words", type=int, default=256, help="words per run")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--ub", type=int, default=2048)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 8))
    ap.add_argument("--old-args", default="", help="arguments only for the old side")
    ap.add_argument("--new-args", default="", help="arguments only for the new side")
    ap.add_argument("--work", default="tokenwing-compare", help="directory for logs")
    ap.add_argument("more", nargs="*", help="arguments for both sides, after --")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    if on_battery():
        print("Warning: this machine is on battery; its speed numbers will not be reliable.")
    common = ["--moe-stream", a.experts, "-ngl", "99"] + a.more
    sides = {"old": (bin_dir(a.old), common + shlex.split(a.old_args)), "new": (bin_dir(a.new), common + shlex.split(a.new_args))}
    res = {"old": [], "new": []}; texts = {"old": [], "new": []}
    print("pair   side     ms/word   SSD wait ms/word   words")
    for p in range(a.pairs):
        for side in (("old", "new") if p % 2 == 0 else ("new", "old")):
            bindir, args = sides[side]
            rc, text, log = generate(bindir, a.model, a.prompt, a.words, a.ub, args, a.work, "%s_%d" % (side, p), ctx=a.ctx, batch=min(a.ctx, 2048), threads=a.threads)
            if rc:
                sys.exit("the %s run of pair %d failed (rc %d): see %s/%s_%d.log" % (side, p, rc, a.work, side, p))
            n = numbers(log); res[side].append(n); texts[side].append(text)
            print("%-6d %-6s %9.2f   %16s   %5d" % (p, side, n["ms_word"], "%.2f" % n["ssd_wait"] if n["ssd_wait"] is not None else "-", n["words"]), flush=True)
    o, w = [r["ms_word"] for r in res["old"]], [r["ms_word"] for r in res["new"]]
    faster = sum(y < x for x, y in zip(o, w))
    print("\nold %.2f ms/word, new %.2f ms/word: %+.2f ms/word (%+.1f %%), the new side was faster in %d of %d pairs (spread between old runs: %.2f)" % (
        statistics.mean(o), statistics.mean(w), statistics.mean(w) - statistics.mean(o),
        100 * (statistics.mean(w) / statistics.mean(o) - 1), faster, a.pairs, statistics.pstdev(o)))
    print("same text as the old side: %s" % ("yes" if all(x == y for x, y in zip(texts["old"], texts["new"])) else "no"))


if __name__ == "__main__":
    main()
