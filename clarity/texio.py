"""Read vanilla `.tex` → RGBA uint8; write `.tex` back as BC7 (with mips), BC3, or B8G8R8A8.

Encoders, in order of preference:
* `texconv.exe` (DirectXTex, in ffxiv_7_0_toolbox/scripts) — GPU BC7, seconds per 4K texture.
  Set CLARITY_TEXCONV to its path (the default looks next to this repo).
* `bc7enc` from the KB tools — pure numpy, modes 6+5, ~20 s per 2048² on CPU. Used when
  texconv is not found (the sandbox) and for tests.
"""

import os
import pathlib
import struct
import subprocess
import tempfile

import numpy as np

from . import ffxiv as kb
from . import paths

_HERE = os.path.dirname(os.path.abspath(__file__))
# Search order: CLARITY_TEXCONV, then a release texconv dropped into <project>/tools/, then the
# toolbox copy. The toolbox copy is a *Debug* build (imports ucrtbased.dll): it asks D3D11 for the
# debug layer, does not get it, and silently falls back to an unoptimised CPU codec — 54 s for one
# 512×1024 (measured 2026-09-05). A release build from microsoft/DirectXTex encodes on the GPU.
_CANDIDATES = [
    os.environ.get("CLARITY_TEXCONV"),
    paths.TEXCONV,  # vendor-tools\texconv\texconv.exe
    os.path.normpath(os.path.join(_HERE, "..", "tools", "texconv.exe")),
    os.path.normpath(
        os.path.join(_HERE, "..", "..", "ffxiv_7_0_toolbox", "scripts", "texconv.exe")
    ),
]
TEXCONV: str = next(
    (c for c in _CANDIDATES if c and c != "none" and os.path.isfile(c)), _CANDIDATES[-1] or ""
)
TEXCONV_GPU = os.environ.get("CLARITY_TEXCONV_GPU", "0")  # DXGI adapter index; 0 = the RTX here
BC7, BC3, BC1, BGRA8 = 0x6432, 0x3431, 0x3420, 0x1450
_probe = {}


def use_texconv():
    return (
        os.name == "nt"
        and os.environ.get("CLARITY_TEXCONV", "") != "none"
        and os.path.isfile(TEXCONV)
    )


def is_debug_build(path=None):
    """A Debug texconv imports the debug CRT; it cannot create a D3D device without the SDK layers."""
    try:
        exe = path or TEXCONV
        if not exe:
            return False
        with open(exe, "rb") as f:
            return b"ucrtbased.dll" in f.read()
    except OSError:
        return False


def probe_texconv():
    """Encode a 64×64 block once and read texconv's own device line. -> dict(ok, gpu, seconds, text)."""
    if _probe:
        return _probe
    if not use_texconv():
        _probe.update(
            ok=False,
            gpu=False,
            seconds=0.0,
            text="texconv not available; numpy encoder",
        )
        return _probe
    import time

    rgba = (np.indices((64, 64)).sum(0) % 2 * 255).astype(np.uint8)
    rgba = np.dstack([rgba, rgba, rgba, np.full_like(rgba, 255)])
    t0 = time.time()
    try:
        _, text = _texconv(rgba, "BC7_UNORM", 1, want_text=True)
        ok = True
    except Exception as e:
        text, ok = str(e), False
    gpu = "Using DirectCompute" in text
    _probe.update(
        ok=ok,
        gpu=gpu,
        seconds=time.time() - t0,
        text=text.strip(),
        debug=is_debug_build(),
    )
    return _probe


def read(path_or_bytes):
    """-> (TexHeader, rgba uint8 (h, w, 4)) of mip 0."""
    raw = kb.game().read(path_or_bytes) if isinstance(path_or_bytes, str) else path_or_bytes
    hdr = kb.texfile.TexHeader(raw)
    return hdr, kb.texdecode.decode(raw, 0)


def read_raw(path):
    return kb.game().read(path)


def full_mips(w, h):
    return int(np.floor(np.log2(max(w, h)))) + 1


# ---------------------------------------------------------------- DDS wrapping (texconv path)
_DDS_MAGIC = b"DDS "


def _dds_rgba(rgba):
    """Uncompressed 32-bit RGBA DDS (what texconv reads as input)."""
    h, w = rgba.shape[:2]
    hdr = bytearray(124)
    struct.pack_into("<7I", hdr, 0, 124, 0x1 | 0x2 | 0x4 | 0x1000 | 0x8, h, w, w * 4, 0, 1)
    # pixel format: size 32, flags RGB|ALPHAPIXELS, bitcount 32, masks R G B A
    struct.pack_into(
        "<IIIIIIII",
        hdr,
        72,
        32,
        0x41,
        0,
        32,
        0x000000FF,
        0x0000FF00,
        0x00FF0000,
        0xFF000000,
    )
    struct.pack_into("<I", hdr, 104, 0x1000)  # caps: TEXTURE
    return _DDS_MAGIC + bytes(hdr) + np.ascontiguousarray(rgba, np.uint8).tobytes()


def _dds_split(dds, block_bytes, w, h, mips):
    """Block payload of each mip out of a DX10 DDS produced by texconv."""
    assert dds[:4] == _DDS_MAGIC
    off = 4 + 124
    fourcc = dds[84:88]
    if fourcc == b"DX10":
        off += 20
    out, cw, ch = [], w, h
    for _ in range(mips):
        n = ((cw + 3) // 4) * ((ch + 3) // 4) * block_bytes
        out.append(dds[off : off + n])
        off += n
        cw, ch = max(1, cw // 2), max(1, ch // 2)
    return out


TEXCONV_TRIES = 2


def _texconv_cmd(fmt_name, mips, out_dir):
    """The texconv command line up to (not including) the input files."""
    cmd = [
        TEXCONV,
        "-nologo",
        "-y",
        "-f",
        fmt_name,
        "-m",
        str(mips),
        "-o",
        out_dir,
        "-gpu",
        TEXCONV_GPU,
    ]
    if fmt_name == "BC7_UNORM":
        cmd += ["-bc", "x"]  # 3-subset modes too: free on the GPU codec
    return cmd


def _texconv(rgba, fmt_name, mips, want_text=False):
    """Encode through texconv, out of reach of the operator's Ctrl+C, reporting the END of a failure.

    CTRL+C WAS KILLING THE ENCODER, AND THAT IS WHAT THE THREE "FAILURES" WERE. On Windows a Ctrl+C
    is delivered to the whole process GROUP, and a child started by subprocess.run inherits that
    group -- so texconv.exe, which is a subprocess and is the slowest single step, was being killed
    whenever the run was stopped by hand. It returned non-zero, the caller recorded the row as
    failed, and the count went from "0 failed" on the line before the interrupt to "1 failed" on the
    line after it. Exactly one per run, one per Ctrl+C, three runs, three failures. CREATE_NEW_
    PROCESS_GROUP takes the child out of that group, so an in-flight encode now finishes -- which is
    what cmd_run's interrupt handler already promised to do and could not.

    (My first reading of this was a transient in the GPU codec, on the grounds that 6,076 textures of
    the identical shape succeeded. The shape was a red herring: big textures spend longest in the
    encoder, so a big one is simply the likeliest to be in flight when the key is pressed. The log
    said "0 failed" immediately before each interrupt and "1 failed" immediately after, which is not
    a pattern a random fault makes.)

    The retry that came with that wrong diagnosis is kept, but honestly labelled: no transient has
    ever actually been observed here. One retry is cheap insurance on a job measured in hundreds of
    hours, not something the evidence demanded.

    THE MESSAGE WAS ALSO BEING THROWN AWAY. It was built as stdout + stderr and truncated to 200
    characters by the caller, and texconv prints its device banner FIRST, so all three rows recorded
    the banner and the "reading ... as" line while the line naming the reason never fit. There was
    no reason to find, as it turns out -- the process was killed mid-sentence -- but a failure that
    cannot say why is a bad thing to keep. The end of the stream is what is kept now.
    """
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    last = ""
    # dir=paths.SCRATCH, not the tempfile default.
    #
    # The default is %TEMP%, which on Windows is on C:. The file written below is an UNCOMPRESSED
    # RGBA DDS of the image *after* upscaling -- 8192x8192x4 = 256 MiB for a 4x pass over a
    # 2048-square source -- and texconv's output lands in the same directory. So the encode step
    # pushed ~340 MiB through the system drive per texture while every configured path pointed at P:,
    # and the only disk check in the tool measures --out, which is somewhere else entirely. The
    # warning arrived during upscaling and never during packing for exactly that reason.
    #
    # It is cleaned up per texture by the `with`, so this is peak, not cumulative -- unless the
    # process is killed between the write and the exit, which leaves the directory behind. Putting
    # them under one known root at least makes the leftovers findable.
    os.makedirs(paths.SCRATCH, exist_ok=True)
    for _attempt in range(TEXCONV_TRIES):
        with tempfile.TemporaryDirectory(dir=paths.SCRATCH) as td:
            src = os.path.join(td, "in.dds")
            pathlib.Path(src).write_bytes(_dds_rgba(rgba))
            cmd = [*_texconv_cmd(fmt_name, mips, td), src]
            r = subprocess.run(
                cmd, capture_output=True, text=True, creationflags=flags, check=False
            )
            if r.returncode == 0:
                out = pathlib.Path(td, "in.dds").read_bytes()
                return (out, r.stdout + r.stderr) if want_text else out
            text = " ".join((r.stdout or "").split()) + " | " + " ".join((r.stderr or "").split())
            last = "rc=%d ...%s" % (r.returncode, text[-300:])
    raise RuntimeError("texconv failed after %d tries: %s" % (TEXCONV_TRIES, last))


def _texconv_many(rgbas, fmt_name, mips):
    """Several images through ONE texconv invocation.

    -> list of DDS bytes, or an exception in the slot of any image that could not be encoded
    (the others are unaffected).

    Almost all of a texconv call is fixed cost: process creation, D3D11 device creation and the
    BC7 compute-shader compile. Measured on the RTX 4070 SUPER with scripts/bench_texconv.py, a
    512x512 BC7 costs 0.32 s alone and 0.06 s each in a call of 64, and the run's queue is mostly
    textures far smaller than that. So the run hands the encoder a batch, and this is the batch.

    texconv converts the files it is given one after another; a file it cannot convert is
    reported and skipped, and the exit code is non-zero at the END if any was. So a non-zero
    exit does not mean the batch failed: inputs go in `in/`, outputs come out in `out/`, and an
    output that is not there is the one that failed. Those are retried one at a time through
    `_texconv`, which is where the retry lives and which raises a message that can name a file.
    """
    if not rgbas:
        return []
    if len(rgbas) == 1:
        try:
            return [_texconv(rgbas[0], fmt_name, mips)]
        except Exception as e:
            return [e]
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    os.makedirs(paths.SCRATCH, exist_ok=True)
    out: list = [None] * len(rgbas)
    with tempfile.TemporaryDirectory(dir=paths.SCRATCH) as td:
        src_dir, out_dir = os.path.join(td, "in"), os.path.join(td, "out")
        os.makedirs(src_dir)
        os.makedirs(out_dir)
        names = ["%04d.dds" % i for i in range(len(rgbas))]
        for name, rgba in zip(names, rgbas, strict=True):
            pathlib.Path(src_dir, name).write_bytes(_dds_rgba(rgba))
        cmd = _texconv_cmd(fmt_name, mips, out_dir) + [os.path.join(src_dir, n) for n in names]
        subprocess.run(cmd, capture_output=True, text=True, creationflags=flags, check=False)
        for i, name in enumerate(names):
            produced = pathlib.Path(out_dir, name)
            if produced.is_file():
                out[i] = produced.read_bytes()
    for i, rgba in enumerate(rgbas):
        if out[i] is None:
            try:
                out[i] = _texconv(rgba, fmt_name, mips)
            except Exception as e:
                out[i] = e
    return out


# The formats texconv is asked for, by .tex format code. This is also the set of block
# formats this module can encode at all: a format outside it must be refused the same way
# on every platform, which is why the lookup happens before the texconv branch rather than
# inside it (on Windows an unknown format used to reach the dict and raise KeyError, while
# on Linux it fell through to the ValueError below -- so the error depended on the machine).
TEXCONV_FORMAT = {BC7: "BC7_UNORM", BC3: "BC3_UNORM", BC1: "BC1_UNORM"}


def encode(rgba, fmt=BC7, mips=None, attributes=None):
    """RGBA uint8 (h, w, 4) -> .tex bytes in `fmt` with a full mip chain (or `mips` levels)."""
    a = np.ascontiguousarray(rgba, np.uint8)
    h, w = a.shape[:2]
    if mips is None:
        mips = full_mips(w, h)
    attr = kb.texwrite.ATTR_2D if attributes is None else attributes
    if fmt == BGRA8:
        return kb.texwrite.write(a, attributes=attr, mips=mips)
    if fmt not in TEXCONV_FORMAT:
        raise ValueError(f"no encoder for format {fmt:#x}")
    if use_texconv():
        name = TEXCONV_FORMAT[fmt]
        dds = _texconv(a, name, mips)
        blocks = _dds_split(dds, kb.texwrite.BLOCK_BYTES[fmt], w, h, mips)
        return kb.texwrite.write_blocks(fmt, blocks, w, h, attr)
    if fmt == BC7:
        return kb.texwrite.write_bc7(a, mips=mips, attributes=attr)
    if fmt == BC3:
        from PIL import Image

        levels = kb.texwrite.mip_chain(a, mips)
        blocks = []
        for L in levels:
            im = Image.fromarray(L, "RGBA")
            # Pillow's DDS plugin cannot emit raw blocks; go through its DXT5 encoder
            import io

            buf = io.BytesIO()
            im.save(buf, "DDS", pixel_format="DXT5")
            d = buf.getvalue()
            blocks.append(_dds_split(d, 16, L.shape[1], L.shape[0], 1)[0])
        return kb.texwrite.write_blocks(BC3, blocks, w, h, attr)
    # Reachable only for a format in TEXCONV_FORMAT with no pure-Python encoder -- BC1 --
    # on a machine where texconv is unavailable. Say that, rather than "no encoder", which
    # reads as "this format is not supported" and sends the reader to the wrong place.
    raise ValueError(
        f"format {fmt:#x} can only be encoded by texconv, which is not available here "
        "(Windows only; see `clarity probe`)"
    )


def float_chain(rgba_u8, n):
    """Box-filtered mip chain computed in float from the top level and rounded once per level —

    what texconv does internally, and what the old per-tier numpy path did — so the same pixels
    come out whichever encoder runs. -> list of uint8 RGBA levels, largest first.
    """
    cur = np.asarray(rgba_u8, np.float32) / 255.0  # same arithmetic as roles' old box_down path
    levels = [np.asarray(rgba_u8, np.uint8)]
    for _ in range(1, n):
        h, w = cur.shape[:2]
        if h <= 1 and w <= 1:
            break
        hh, ww = max(1, h // 2), max(1, w // 2)
        if h >= 2 and w >= 2:
            cur = cur[: hh * 2, : ww * 2].reshape(hh, 2, ww, 2, 4).mean((1, 3))
        elif h >= 2:
            cur = cur[: hh * 2].reshape(hh, 2, w, 4).mean(1)
        else:
            cur = cur[:, : ww * 2].reshape(h, ww, 2, 4).mean(2)
        levels.append(np.clip(np.rint(cur * 255.0), 0, 255).astype(np.uint8))
    return levels


def _bgra_levels(levels_rgba):
    return [np.ascontiguousarray(L[:, :, [2, 1, 0, 3]], np.uint8).tobytes() for L in levels_rgba]


def _bgra_tex(levels_bytes, w, h, attributes):
    """B8G8R8A8 .tex from per-mip BGRA byte blocks, largest first (mirrors texwrite.write)."""
    d = bytearray(kb.texwrite.HDR)
    struct.pack_into("<IIHHHBB", d, 0, attributes, BGRA8, w, h, 1, len(levels_bytes), 1)
    struct.pack_into("<3I", d, 16, kb.texwrite.HDR, kb.texwrite.HDR, kb.texwrite.HDR)
    offs, at = [], kb.texwrite.HDR
    for L in levels_bytes:
        offs.append(at)
        at += len(L)
    offs += [0] * (13 - len(offs))
    struct.pack_into("<13I", d, 28, *offs[:13])
    return bytes(d) + b"".join(levels_bytes)


def encode_tiers(rgba_top, fmt=BC7, attributes=None, offsets=(0,), keep_mips=True):
    """Encode the top-tier image ONCE with its full mip chain, then cut one .tex per requested tier:

    tier at offset k = mip levels k.. of that chain (dims >> k). Mip k of a box-filtered chain is
    exactly the 2×2 average the numpy path computed, so the tiers are the same pixels as before,
    encoded once instead of once per tier. `keep_mips=False` writes a single level per tier (for
    sources that ship without a chain, e.g. UI). -> {offset: tex bytes}
    """
    a = np.ascontiguousarray(rgba_top, np.uint8)
    h, w = a.shape[:2]
    n = full_mips(w, h)
    attr = kb.texwrite.ATTR_2D if attributes is None else attributes
    if fmt != BGRA8 and fmt not in TEXCONV_FORMAT:
        raise ValueError(f"no chain encoder for format {fmt:#x}")
    if fmt == BGRA8:
        levels = _bgra_levels(float_chain(a, n))
    elif use_texconv():
        name = TEXCONV_FORMAT[fmt]
        levels = _dds_split(
            _texconv(a, name, 0), kb.texwrite.BLOCK_BYTES[fmt], w, h, n
        )  # -m 0: the full chain
    elif fmt == BC7:
        levels = [kb.bc7enc.encode(L) for L in float_chain(a, n)]
    elif fmt == BC3:
        import io

        from PIL import Image

        levels = []
        for L in float_chain(a, n):
            buf = io.BytesIO()
            Image.fromarray(L, "RGBA").save(buf, "DDS", pixel_format="DXT5")
            levels.append(_dds_split(buf.getvalue(), 16, L.shape[1], L.shape[0], 1)[0])
    else:
        raise ValueError(f"no chain encoder for format {fmt:#x}")
    return _cut_tiers(levels, fmt, w, h, attr, offsets, keep_mips)


def _cut_tiers(levels, fmt, w, h, attr, offsets, keep_mips):
    """One .tex per requested tier out of an encoded mip chain (see encode_tiers). -> {offset: bytes}"""
    out = {}
    for k in offsets:
        if k >= len(levels):
            continue
        cw, ch = max(1, w >> k), max(1, h >> k)
        lv = levels[k:] if keep_mips else levels[k : k + 1]
        out[k] = (
            _bgra_tex(lv, cw, ch, attr)
            if fmt == BGRA8
            else kb.texwrite.write_blocks(fmt, lv, cw, ch, attr)
        )
    return out


def encode_tiers_many(items):
    """`encode_tiers` for a batch, one result per item and one item's failure never the batch's.

    items are (rgba_top, fmt, attributes, offsets, keep_mips); the result has one entry per item,
    in order -- the {offset: bytes} dict, or the exception that item raised.

    On the texconv path every item of one block format goes through a single texconv invocation
    (see `_texconv_many`); the pixels are exactly those `encode_tiers` would produce for the item
    alone, since texconv converts each file independently. Everything else is `encode_tiers`
    per item.
    """
    results: list = [None] * len(items)
    by_fmt: dict = {}
    for i, (rgba, fmt, attributes, offsets, keep_mips) in enumerate(items):
        if use_texconv() and fmt in TEXCONV_FORMAT:
            by_fmt.setdefault(fmt, []).append(i)
            continue
        try:
            results[i] = encode_tiers(rgba, fmt, attributes, offsets=offsets, keep_mips=keep_mips)
        except Exception as e:
            results[i] = e
    for fmt, idx in by_fmt.items():
        arrays = [np.ascontiguousarray(items[i][0], np.uint8) for i in idx]
        encoded = _texconv_many(arrays, TEXCONV_FORMAT[fmt], 0)  # -m 0: the full chain
        for i, a, dds in zip(idx, arrays, encoded, strict=True):
            if isinstance(dds, Exception):
                results[i] = dds
                continue
            _rgba, _fmt, attributes, offsets, keep_mips = items[i]
            h, w = a.shape[:2]
            n = full_mips(w, h)
            attr = kb.texwrite.ATTR_2D if attributes is None else attributes
            try:
                levels = _dds_split(dds, kb.texwrite.BLOCK_BYTES[fmt], w, h, n)
                results[i] = _cut_tiers(levels, fmt, w, h, attr, offsets, keep_mips)
            except Exception as e:
                results[i] = e
    return results


def out_format(src_fmt_name, role):
    """Output format policy: block formats become BC7 (BC3 stays BC3 for UI), uncompressed stays."""
    if src_fmt_name in ("B8G8R8A8", "B8G8R8X8", "B5G5R5A1", "B4G4R4A4", "L8", "A8"):
        return BGRA8
    if src_fmt_name in ("BC2", "BC3") and role in ("icon", "ui"):
        return BC3
    return BC7
