"""Measure texconv's fixed per-invocation cost, and how far batching and concurrency amortise it.

`clarity probe` already shows the problem: a 64x64 encode takes ~0.3 s on hardware that does the
actual work in microseconds. That 0.3 s is process creation plus D3D11 device init plus compiling
the BC7 compute shader, and `texio._texconv` pays it once per texture. Over a 375k-row run that is
about 31 hours before a single useful FLOP.

texconv accepts many input files per invocation, so the fix is to stop invoking it per texture.
This script measures what that is worth *on this machine* rather than guessing:

  * files per invocation: 1, 2, 4, 8, 16, 32, 64
  * concurrent invocations of one file each: 1, 2, 4, 8
  * the write/read I/O separately from the subprocess, so the value of moving CLARITY_SCRATCH
    off the archive drive is visible on its own

Run from the repo root:

    uv run python scripts\\bench_texconv.py
    uv run python scripts\\bench_texconv.py --size 2048        # a 2x pass over a 1024 source
    uv run python scripts\\bench_texconv.py --scratch C:\\temp  # compare a local disk

Writes nothing outside the scratch directory and reads no game files.
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from clarity import paths, texio

FLAGS = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


def make_source(h: int, w: int, seed: int = 0) -> bytes:
    """One uncompressed RGBA DDS, the shape texio hands texconv."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.dstack([(xx * 3) & 255, (yy * 5) & 255, ((xx + yy) * 2) & 255])
    rgb = np.clip(rgb + rng.integers(-8, 9, rgb.shape), 0, 255)
    rgba = np.dstack([rgb, np.full((h, w), 255)]).astype(np.uint8)
    return texio._dds_rgba(rgba)


def run_texconv(paths_in: list[str], out_dir: str, fmt: str = "BC7_UNORM") -> float:
    """One invocation over `paths_in`. Returns wall seconds for the subprocess alone."""
    cmd = [texio.TEXCONV, "-nologo", "-y", "-f", fmt, "-m", "0", "-o", out_dir]
    if fmt.startswith("BC7"):
        cmd += ["-bc", "x"]
    cmd += ["-gpu", texio.TEXCONV_GPU, *paths_in]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, creationflags=FLAGS, check=False)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        tail = " ".join(((r.stdout or "") + (r.stderr or "")).split())[-300:]
        raise RuntimeError(f"texconv exit {r.returncode}: {tail}")
    return dt


def bench_batch(dds: bytes, counts: list[int], repeats: int, scratch: str) -> list[tuple]:
    rows = []
    for n in counts:
        spawn, io_in, io_out = [], [], []
        for _ in range(repeats):
            with tempfile.TemporaryDirectory(dir=scratch) as td:
                src_dir = os.path.join(td, "in")
                out_dir = os.path.join(td, "out")
                os.makedirs(src_dir)
                os.makedirs(out_dir)
                t0 = time.perf_counter()
                names = []
                for i in range(n):
                    p = os.path.join(src_dir, f"t{i:04d}.dds")
                    with open(p, "wb") as f:
                        f.write(dds)
                    names.append(p)
                io_in.append(time.perf_counter() - t0)
                spawn.append(run_texconv(names, out_dir))
                t0 = time.perf_counter()
                total = 0
                for i in range(n):
                    with open(os.path.join(out_dir, f"t{i:04d}.dds"), "rb") as f:
                        total += len(f.read())
                io_out.append(time.perf_counter() - t0)
        rows.append(
            (
                n,
                statistics.median(spawn),
                statistics.median(spawn) / n,
                statistics.median(io_in) / n,
                statistics.median(io_out) / n,
            )
        )
    return rows


def bench_concurrency(dds: bytes, levels: list[int], repeats: int, scratch: str) -> list[tuple]:
    rows = []
    for k in levels:
        walls = []
        for _ in range(repeats):
            with tempfile.TemporaryDirectory(dir=scratch) as td:
                jobs = []
                for i in range(k):
                    d = os.path.join(td, f"w{i}")
                    os.makedirs(os.path.join(d, "out"))
                    p = os.path.join(d, "in.dds")
                    with open(p, "wb") as f:
                        f.write(dds)
                    jobs.append(([p], os.path.join(d, "out")))
                t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=k) as pool:
                    list(pool.map(lambda j: run_texconv(*j), jobs))
                walls.append(time.perf_counter() - t0)
        rows.append((k, statistics.median(walls), statistics.median(walls) / k))
    return rows


def bench_combined(
    dds: bytes, n: int, levels: list[int], repeats: int, scratch: str
) -> list[tuple]:
    """k concurrent invocations of n files each: the cell `--encode-workers` is chosen from."""
    rows = []
    for k in levels:
        walls = []
        for _ in range(repeats):
            with tempfile.TemporaryDirectory(dir=scratch) as td:
                jobs = []
                for i in range(k):
                    d = os.path.join(td, f"w{i}")
                    os.makedirs(os.path.join(d, "out"))
                    names = []
                    for j in range(n):
                        p = os.path.join(d, f"in_{j:03d}.dds")
                        with open(p, "wb") as f:
                            f.write(dds)
                        names.append(p)
                    jobs.append((names, os.path.join(d, "out")))
                t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=k) as pool:
                    list(pool.map(lambda j: run_texconv(*j), jobs))
                walls.append(time.perf_counter() - t0)
        rows.append((k, statistics.median(walls), statistics.median(walls) / (k * n)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size", type=int, default=512, help="square source edge (default 512)")
    ap.add_argument("--height", type=int, default=None, help="override height (default = --size)")
    ap.add_argument("--repeats", type=int, default=3, help="median of N runs (default 3)")
    ap.add_argument("--scratch", default=None, help="scratch dir (default CLARITY_SCRATCH)")
    ap.add_argument("--max-batch", type=int, default=64)
    a = ap.parse_args()

    if not texio.use_texconv():
        print("texconv is not available here; this benchmark needs it.")
        return 2

    scratch = a.scratch or paths.SCRATCH
    os.makedirs(scratch, exist_ok=True)
    w, h = a.size, a.height or a.size
    dds = make_source(h, w)

    free = shutil.disk_usage(scratch).free / 1e9
    print(f"texconv : {texio.TEXCONV}")
    print(f"scratch : {scratch}  ({free:.0f} GB free)")
    print(f"source  : {w}x{h} RGBA -> BC7, {len(dds) / 1e6:.1f} MB per input file")
    print(f"repeats : {a.repeats} (median reported)\n")

    counts = [n for n in (1, 2, 4, 8, 16, 32, 64) if n <= a.max_batch]
    print("FILES PER INVOCATION")
    print(
        f"  {'n':>4}  {'call (s)':>10}  {'per file':>10}  {'write':>8}  {'read':>8}  {'speedup':>8}"
    )
    rows = bench_batch(dds, counts, a.repeats, scratch)
    base = rows[0][2]
    for n, call, per, wi, ro in rows:
        print(f"  {n:>4}  {call:>10.3f}  {per:>10.4f}  {wi:>8.4f}  {ro:>8.4f}  {base / per:>7.1f}x")

    print("\nCONCURRENT INVOCATIONS (one file each)")
    print(f"  {'k':>4}  {'wall (s)':>10}  {'per file':>10}  {'speedup':>8}")
    crows = bench_concurrency(dds, [1, 2, 4, 8], a.repeats, scratch)
    cbase = crows[0][2]
    for k, wall, per in crows:
        print(f"  {k:>4}  {wall:>10.3f}  {per:>10.4f}  {cbase / per:>7.1f}x")

    best_n, _c, best_per, _wi, _ro = min(rows, key=lambda r: r[2])
    # Both at once: batched calls, several in flight. Measured on an OTHERWISE IDLE card, which
    # a run never has: during `clarity run` the upscaling models own most of the VRAM, and every
    # extra worker is another texconv process creating its own D3D11 device on what is left. The
    # table is still worth reading -- it says how far texconv is from saturating the GPU alone --
    # but the suggestion below stays at one worker. This script once printed the best k here as
    # the flag to use; a run with --encode-workers 4 was part of what put a 12 GB card 20 GB over.
    n_c = min(32, a.max_batch)
    print(f"\nCONCURRENT BATCHED INVOCATIONS ({n_c} files each)")
    print(f"  {'k':>4}  {'wall (s)':>10}  {'per file':>10}  {'vs k=1':>8}")
    brows = bench_combined(dds, n_c, [1, 2, 4], a.repeats, scratch)
    bbase = brows[0][2]
    for k, wall, per in brows:
        print(f"  {k:>4}  {wall:>10.3f}  {per:>10.4f}  {bbase / per:>7.2f}x")
    best_k, _w, best_kper = min(brows, key=lambda r: r[2])
    queued = 375814
    print(
        f"\nBest batched cost: {best_per:.4f} s/file at n={best_n}"
        f"  (from {base:.4f} s at n=1)"
        f"\nOver {queued:,} queued rows: {queued * base / 3600:.1f} h -> "
        f"{queued * best_per / 3600:.1f} h"
        f"\nWith {best_k} worker(s) of {n_c}: {best_kper:.4f} s/file -> "
        f"{queued * best_kper / 3600:.1f} h  (idle-card figure; see the note in the source)"
        f"\n\nSuggested: clarity run --encode-batch {n_c}"
        f"\n  (--encode-workers stays at 1: the models hold the VRAM this table was measured"
        f" without; the encode overlaps the model pass anyway)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
