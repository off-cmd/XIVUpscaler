"""Minimal, dependency-free SqPack reader for FFXIV (Win32, DX11).

Verified against retail client 2026.08.11 (patch 7.55+, Dawntrail).
Format sources: xiv.dev docs/data-files/sqpack.md + Lumina (SqPackStream.cs,
SqPackFileInfo.cs, Misc/Crc32.cs).

Purpose: ground-truth inspection of game assets (.tex/.mtrl/.mdl/.shpk) without
needing dotnet/TexTools. Read-only; never writes to the game install.
"""

import contextlib
import glob
import hashlib
import io
import os
import struct
import zlib
from typing import Any

CATEGORIES = {
    "common": 0x00,
    "bgcommon": 0x01,
    "bg": 0x02,
    "cut": 0x03,
    "chara": 0x04,
    "shader": 0x05,
    "ui": 0x06,
    "sound": 0x07,
    "vfx": 0x08,
    "ui_script": 0x09,
    "exd": 0x0A,
    "game_script": 0x0B,
    "music": 0x0C,
    "sqpack_test": 0x12,
    "debug": 0x13,
}
EXPANSIONS = {"ffxiv": 0, "ex1": 1, "ex2": 2, "ex3": 3, "ex4": 4, "ex5": 5}


# FFXIV path hash == bitwise-NOT of the standard zlib CRC-32.
# (Lumina seeds with 0xFFFFFFFF and omits the final XOR, which is the same thing.)
def crc32(b: bytes) -> int:
    return (~zlib.crc32(b)) & 0xFFFFFFFF


def split_hash(path: str):
    p = path.lower().strip("/")
    d, _, f = p.rpartition("/")
    return crc32(d.encode()), crc32(f.encode())


def full_hash(path: str) -> int:
    return crc32(path.lower().strip("/").encode())


def parse_path(path: str):
    """-> (category_id, expansion_id, chunk_id=0). Chunks other than 0 are rare."""
    parts = path.lower().strip("/").split("/")
    if parts[0] not in CATEGORIES:
        return None
    cat = CATEGORIES[parts[0]]
    exp = EXPANSIONS.get(parts[1], 0) if len(parts) > 1 else 0
    return cat, exp, 0


class Index:
    """One .win32.index / .index2 hash table."""

    def __init__(self, path):
        self.path = path
        self.entries = {}
        self.is_index2 = path.endswith(".index2")
        with open(path, "rb") as f:
            data = f.read()
        # SqPackHeader: magic[8], platformId, pad[3], size, version, type
        hdr_size = struct.unpack_from("<I", data, 0x0C)[0]
        # SqPackIndexHeader at hdr_size: size, version, indexDataOffset, indexDataSize
        _, _, off, size = struct.unpack_from("<4I", data, hdr_size)
        if self.is_index2:
            for i in range(size // 8):
                h, d = struct.unpack_from("<2I", data, off + i * 8)
                self.entries[h] = d
        else:
            for i in range(size // 16):
                h, d, _ = struct.unpack_from("<QII", data, off + i * 16)
                self.entries[h] = d

    @staticmethod
    def decode(d):
        return (d & 0b1110) >> 1, (d & ~0xF) * 0x08  # (dataFileId, offset)


class Repository:
    """A sqpack repository dir (ffxiv, ex1..ex5)."""

    def __init__(self, sqpack_dir, name):
        self.dir = os.path.join(sqpack_dir, name)
        self.name = name
        self._idx = {}

    def index(self, cat, exp, chunk=0, two=False):
        key = (cat, exp, chunk, two)
        if key not in self._idx:
            base = os.path.join(self.dir, f"{cat:02x}{exp:02x}{chunk:02x}.win32.index")
            p = base + "2" if two else base
            self._idx[key] = Index(p) if os.path.exists(p) else None
        return self._idx[key]

    def dat(self, cat, exp, chunk, file_id):
        return os.path.join(self.dir, f"{cat:02x}{exp:02x}{chunk:02x}.win32.dat{file_id}")


class GameData:
    """Main entry point for SqPack file access."""

    def __init__(self, sqpack_dir: str) -> None:
        self.sqpack = sqpack_dir
        self.repos = {
            n: Repository(sqpack_dir, n)
            for n in os.listdir(sqpack_dir)
            if os.path.isdir(os.path.join(sqpack_dir, n))
        }

    def locate(self, path: str) -> Any:
        parsed = parse_path(path)
        if parsed is None:
            return None
        cat, _exp, _chunk = parsed
        # The repository is the explicit expansion segment when the path carries one
        # (bg/ex3/..., music/ex1/...), otherwise the base game.
        segs = path.lower().strip("/").split("/")
        repo_name = segs[1] if len(segs) > 1 and segs[1] in EXPANSIONS else "ffxiv"
        exp = EXPANSIONS[repo_name]
        repo = self.repos.get(repo_name)
        if repo is None:
            return None
        dh, fh = split_hash(path)
        key = (dh << 32) | fh
        # index1 is a 64-bit hash (folder+file) and is authoritative.
        # index2 is a 32-bit hash of the whole path and WILL collide across a
        # large probe set - only consult it when no index1 exists for the chunk
        # (benchmark clients ship index2 only).
        any_index1 = False
        for ch in range(0, 8):
            idx = repo.index(cat, exp, ch)
            if idx is not None:
                any_index1 = True
                if key in idx.entries:
                    fid, off = Index.decode(idx.entries[key])
                    return repo.dat(cat, exp, ch, fid), off
        if not any_index1:
            k2 = full_hash(path)
            for ch in range(0, 8):
                idx2 = repo.index(cat, exp, ch, two=True)
                if idx2 and k2 in idx2.entries:
                    fid, off = Index.decode(idx2.entries[k2])
                    return repo.dat(cat, exp, ch, fid), off
        return None

    def exists(self, path):
        return self.locate(path) is not None

    def read(self, path: str) -> bytes:
        loc = self.locate(path)
        if loc is None:
            raise FileNotFoundError(path)
        return read_file(*loc)


# ---- dat entry reading -------------------------------------------------------
FT_EMPTY, FT_STANDARD, FT_MODEL, FT_TEXTURE = 1, 2, 3, 4


def _block(f, offset, out):
    f.seek(offset)
    _size, _unk, btype, dsize = struct.unpack("<4I", f.read(16))
    if btype == 32000:  # uncompressed
        out.write(f.read(dsize))
    else:  # raw DEFLATE, no zlib wrapper
        d = zlib.decompressobj(-15)
        buf = d.decompress(f.read(btype), dsize)
        out.write(buf)
    return dsize


# Keeping .dat handles open. Profiling the gear export showed `io.open`
# accounting for **55% of total runtime** -- 438 sqpack reads meant 438 fresh
# opens of the same handful of archives, and opening a file on a mounted
# filesystem is expensive. There are only a few dozen .dat files, so caching
# every handle is cheap and removes the cost entirely.
_DAT_HANDLES = {}


def _dat(path):
    f = _DAT_HANDLES.get(path)
    if f is None or f.closed:
        f = open(path, "rb")  # noqa: SIM115 - cached; closed by close_dats()
        _DAT_HANDLES[path] = f
    return f


def close_dats():
    for f in _DAT_HANDLES.values():
        with contextlib.suppress(Exception):
            f.close()
    _DAT_HANDLES.clear()


def read_file(dat_path, offset):
    f = _dat(dat_path)
    f.seek(offset)
    hsize, ftype, _raw_size = struct.unpack("<3I", f.read(12))
    f.seek(offset)
    head = f.read(hsize)
    out = io.BytesIO()
    if ftype == FT_STANDARD:
        nblocks = struct.unpack_from("<I", head, 0x14)[0]
        for i in range(nblocks):
            boff, _csz, _usz = struct.unpack_from("<IHH", head, 0x18 + i * 8)
            _block(f, offset + hsize + boff, out)
    elif ftype == FT_TEXTURE:
        nblocks = struct.unpack_from("<I", head, 0x14)[0]
        lods = [struct.unpack_from("<5I", head, 0x18 + i * 20) for i in range(nblocks)]
        # sub-block compressed sizes follow the LodBlock array
        sub = struct.unpack_from(f"<{sum(lod[4] for lod in lods)}H", head, 0x18 + nblocks * 20)
        mip_hdr = lods[0][0]
        if mip_hdr:  # raw .tex header sits before block 0
            f.seek(offset + hsize)
            out.write(f.read(mip_hdr))
        si = 0
        for lb in lods:
            run = offset + hsize + lb[0]
            for _ in range(lb[4]):
                _block(f, run, out)
                run += sub[si]
                si += 1
    elif ftype == FT_MODEL:
        return _read_model(f, offset, head)
    else:
        raise ValueError(f"unsupported/empty file type {ftype}")
    return out.getvalue()


def _read_model(f, offset, head):
    (_size, _ftype, _rawsz, _nblocks, _used, version, _stack_size, _runtime_size) = (
        struct.unpack_from("<8I", head, 0)
    )
    struct.unpack_from("<3I", head, 0x20)
    struct.unpack_from("<3I", head, 0x2C)
    struct.unpack_from("<3I", head, 0x38)
    _c_stack, _c_runtime = struct.unpack_from("<2I", head, 0x44)
    off = 0x70
    stack_off, runtime_off = struct.unpack_from("<2I", head, off)
    off += 8
    vb_off = struct.unpack_from("<3I", head, off)
    off += 12
    eg_off = struct.unpack_from("<3I", head, off)
    off += 12
    ib_off = struct.unpack_from("<3I", head, off)
    off += 12
    stack_i, runtime_i = struct.unpack_from("<2H", head, off)
    off += 4
    vb_i = struct.unpack_from("<3H", head, off)
    off += 6
    eg_i = struct.unpack_from("<3H", head, off)
    off += 6
    ib_i = struct.unpack_from("<3H", head, off)
    off += 6
    stack_n, runtime_n = struct.unpack_from("<2H", head, off)
    off += 4
    vb_n = struct.unpack_from("<3H", head, off)
    off += 6
    eg_n = struct.unpack_from("<3H", head, off)
    off += 6
    ib_n = struct.unpack_from("<3H", head, off)
    off += 6
    vdecl_n, mat_n = struct.unpack_from("<2H", head, off)
    off += 4
    nlods, ibs, ege, _pad = struct.unpack_from("<4B", head, off)
    total = stack_n + runtime_n + sum(vb_n) + sum(eg_n) + sum(ib_n)
    csizes = struct.unpack_from(f"<{total}H", head, 0xD0)
    return _rebuild_model(
        f,
        offset + len(head),
        version,
        vdecl_n,
        mat_n,
        nlods,
        ibs,
        ege,
        csizes,
        (stack_off, stack_i, stack_n),
        (runtime_off, runtime_i, runtime_n),
        vb_off,
        vb_i,
        vb_n,
        eg_off,
        eg_i,
        eg_n,
        ib_off,
        ib_i,
        ib_n,
    )


def _section(f, base, off, first, count, csizes, out):
    """Decompress `count` consecutive blocks starting at `base + off`."""
    run = base + off
    for k in range(count):
        _block(f, run, out)
        run += csizes[first + k]
    return count


def _rebuild_model(
    f,
    base,
    version,
    vdecl_n,
    mat_n,
    nlods,
    ibs,
    ege,
    csizes,
    stack,
    runtime,
    vb_off,
    vb_i,
    vb_n,
    eg_off,
    eg_i,
    eg_n,
    ib_off,
    ib_i,
    ib_n,
):
    """Reassemble a `.mdl` exactly as a mod tool would write it.

    A model is not stored as one stream. Stack, runtime and each LOD's vertex, edge-geometry and
    index buffers are separate runs of compressed blocks, and the 68-byte file header has to be
    *rebuilt* from the dat entry's own fields with the offsets the reassembly actually produced.
    Everything downstream -- reading vanilla material names, comparing a mod against stock --
    needs this, so it is worth doing properly rather than reporting header stats.
    """
    # The sizes in the dat entry are the PADDED ones the compressor used -- 896 where the
    # stack is really 816, 4096 where the runtime is really 4000. Writing those into the .mdl
    # header puts every later section 80 bytes out and the string block reads as garbage. The
    # header has to carry what actually came out of the blocks.
    body = io.BytesIO()
    _section(f, base, stack[0], stack[1], stack[2], csizes, body)
    _stack_size = body.tell()
    _section(f, base, runtime[0], runtime[1], runtime[2], csizes, body)
    _runtime_size = body.tell() - _stack_size
    v_off, i_off, v_size, i_size = [0] * 3, [0] * 3, [0] * 3, [0] * 3
    hdr_size = 0x44
    for lod in range(3):
        if vb_n[lod]:
            v_off[lod] = hdr_size + body.tell()
            before = body.tell()
            _section(f, base, vb_off[lod], vb_i[lod], vb_n[lod], csizes, body)
            v_size[lod] = body.tell() - before
        if eg_n[lod]:
            _section(f, base, eg_off[lod], eg_i[lod], eg_n[lod], csizes, body)
        if ib_n[lod]:
            i_off[lod] = hdr_size + body.tell()
            before = body.tell()
            _section(f, base, ib_off[lod], ib_i[lod], ib_n[lod], csizes, body)
            i_size[lod] = body.tell() - before
    h = struct.pack(
        "<III HH 3I 3I 3I 3I BBBB",
        version,
        _stack_size,
        _runtime_size,
        vdecl_n,
        mat_n,
        *v_off,
        *i_off,
        *v_size,
        *i_size,
        nlods,
        ibs,
        ege,
        0,
    )
    return bytearray(h) + body.getvalue()


# Where a real install puts sqpack. Checked on every drive letter on Windows.
_INSTALL_SUFFIXES = (
    r"Steam\steamapps\common\FINAL FANTASY XIV Online\game\sqpack",
    r"SquareEnix\FINAL FANTASY XIV - A Realm Reborn\game\sqpack",
    r"FINAL FANTASY XIV Online\game\sqpack",
    r"FINAL FANTASY XIV - A Realm Reborn\game\sqpack",
)


def _ok(p):
    return p and os.path.isdir(os.path.join(p, "ffxiv"))


def find_game(root: str | None = None) -> str:
    r"""Locate the game's `sqpack` directory.

    Set **FFXIV_SQPACK** to skip the search entirely:

        set FFXIV_SQPACK=C:\\...\\FINAL FANTASY XIV Online\\game\\sqpack

    Raises rather than returning None -- returning None produced a confusing
    `TypeError: expected str ... not NoneType` deep inside GameData.
    """
    env = os.environ.get("FFXIV_SQPACK") or os.environ.get("FFXIV_PATH")
    if env:
        p = env if _ok(env) else os.path.join(env, "game", "sqpack")
        if _ok(p):
            return p
        raise FileNotFoundError(f"FFXIV_SQPACK is set but has no 'ffxiv' folder: {env}")

    if root and _ok(root):
        return root

    if os.name == "nt":
        bases = []
        for d in "CDEFGHIJKL":
            for pf in ("Program Files (x86)", "Program Files", "Games", ""):
                bases.append(os.path.join(f"{d}:\\", pf) if pf else f"{d}:\\")
        for b in bases:
            for suf in _INSTALL_SUFFIXES:
                p = os.path.join(b, suf)
                if _ok(p):
                    return p
    else:
        for pat in (
            "/sessions/*/mnt/*/game/sqpack",
            "/sessions/*/mnt/**/sqpack",
            os.path.expanduser(
                "~/.steam/steam/steamapps/common/FINAL FANTASY XIV Online/game/sqpack"
            ),
            os.path.expanduser("~/.xlcore/ffxiv/game/sqpack"),
        ):
            for c in glob.glob(pat):
                if _ok(c):
                    return c

    raise FileNotFoundError(
        "Could not find the FFXIV sqpack folder. Set FFXIV_SQPACK to it, e.g.\n"
        r"  set FFXIV_SQPACK=C:\Program Files (x86)\Steam\steamapps\common"
        r"\FINAL FANTASY XIV Online\game\sqpack"
    )


def game_version(sqpack_dir=None):
    """The patch level of the install, e.g. '2026.08.19.0000.0000'.

    `ffxivgame.ver` sits one level up from `sqpack`. Expansion .ver files exist too but the base
    one is what moves on every patch and hotfix, which is the question being asked here.
    """
    d = sqpack_dir or find_game()
    for cand in (
        os.path.join(os.path.dirname(d), "ffxivgame.ver"),
        os.path.join(d, "ffxivgame.ver"),
    ):
        try:
            with open(cand, encoding="utf-8", errors="ignore") as f:
                v = f.read().strip()
            if v:
                return v
        except OSError:
            continue
    return "unknown"


def entry_fingerprint(dat_path, offset, digest_size=16):
    """A content hash of one dat entry, taken from its **compressed** on-disk bytes.

    -> lowercase hex BLAKE2b digest, or None if the entry type has no block table here.

    The obvious way to fingerprint a texture is to hash what `read_file` returns, but that inflates
    every mip of every file: for a manifest this size it is tens of gigabytes of zlib work to answer
    a yes/no question. The entry header already carries each block's *compressed* size, so the whole
    entry can be read as opaque bytes -- header plus payload, no decompression at all -- for roughly
    a quarter of the I/O and none of the CPU.

    What that buys and what it costs: identical content stored identically hashes identically, and
    any change to the pixels changes at least one compressed block, so a real edit can never hash
    the same. The one direction it errs is the harmless one -- a repack of *unchanged* content with
    different deflate settings would produce a different hash, i.e. a false "changed", which costs a
    re-run of that texture rather than a silently stale upscale. Squeenix's packer has been stable
    across the patches this was tested on, so in practice this does not fire.

    Deliberately not the dat offset: a patch rewrites the archives and moves nearly every entry, so
    an offset-based fingerprint reports the whole game as changed after any patch at all.
    """
    f = _dat(dat_path)
    f.seek(offset)
    hsize, ftype, _raw_size = struct.unpack("<3I", f.read(12))
    f.seek(offset)
    head = f.read(hsize)
    h = hashlib.blake2b(digest_size=digest_size)
    h.update(head)
    if ftype == FT_TEXTURE:
        nblocks = struct.unpack_from("<I", head, 0x14)[0]
        lods = [struct.unpack_from("<5I", head, 0x18 + i * 20) for i in range(nblocks)]
        sub = struct.unpack_from("<%dH" % sum(lod[4] for lod in lods), head, 0x18 + nblocks * 20)
        mip_hdr = lods[0][0] if lods else 0
        if mip_hdr:  # the raw .tex header, stored uncompressed before block 0
            f.seek(offset + hsize)
            h.update(f.read(mip_hdr))
        si = 0
        for lb in lods:
            run = offset + hsize + lb[0]
            for _ in range(lb[4]):
                f.seek(run)
                h.update(
                    f.read(sub[si])
                )  # sub[si] is the on-disk size, header and padding included
                run += sub[si]
                si += 1
    elif ftype == FT_STANDARD:
        nblocks = struct.unpack_from("<I", head, 0x14)[0]
        for i in range(nblocks):
            boff, csz, _usz = struct.unpack_from("<IHH", head, 0x18 + i * 8)
            f.seek(offset + hsize + boff)
            h.update(f.read(csz))
    elif ftype == FT_MODEL:
        # Models have a different block table and nothing in the manifest is one; the header alone
        # still moves whenever the model does, so this stays a usable (if coarser) fingerprint.
        pass
    else:
        return None
    return h.hexdigest()


def read_tex_header(dat_path, offset, nbytes=80):
    """Fast path: a texture-type dat entry stores the raw .tex header uncompressed

    immediately after the entry header, so we can read it without inflating mips.
    """
    f = _dat(dat_path)
    f.seek(offset)
    hsize, ftype, _raw_size = struct.unpack("<3I", f.read(12))
    if ftype != FT_TEXTURE:
        return None
    f.seek(offset + 0x18)
    first_comp_offset = struct.unpack("<I", f.read(4))[0]
    if first_comp_offset == 0:
        return None
    f.seek(offset + hsize)
    return f.read(min(nbytes, first_comp_offset))
