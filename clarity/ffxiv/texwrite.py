"""Write an FFXIV `.tex`.

Only uncompressed **B8G8R8A8** (format 0x1450), single mip, 2D. That is deliberate: it is the
one format that needs no block encoder, the game reads it perfectly well, and for the index and
mask maps -- which must not be lossy -- it is the *correct* choice regardless. A 4K BGRA map is
64 MB, so keep these at 1K/2K unless there is a reason.

Header is 80 bytes:

     0  u32  attributes          copy from a real texture rather than guessing the bits
     4  u32  format              0x1450
     8  u16  width
    10  u16  height
    12  u16  depth               1
    14  u8   mip flag            mip count in the low 7 bits
    15  u8   array size          1
    16  u32[3]   lod offsets
    28  u32[13]  surface offsets

BGRA byte order, not RGBA -- swapping them silently gives a blue garment.
"""

import struct

import numpy as np

HDR = 80
B8G8R8A8 = 0x1450
BC7 = 0x6432
BC5 = 0x6230
BC3 = 0x3431
ATTR_2D = 0x00800000  # the bit texfile.TexHeader reports as "2D"
BLOCK_BYTES = {BC7: 16, BC5: 16, BC3: 16, 0x3420: 8, 0x6120: 8}


def write(rgba, attributes=ATTR_2D, mips=1):
    """rgba: (h, w, 4) uint8 in R,G,B,A order -> .tex bytes.

    `mips` writes a box-filtered chain. Vanilla textures always carry one; a single-mip file
    works, but declaring the chain removes any question of what the loader does when it asks
    for a level that is not there.
    """
    a = np.asarray(rgba)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.dstack([a, a, a, np.full(a.shape, 255, np.uint8)])
    if a.shape[2] == 3:
        a = np.dstack([a, np.full(a.shape[:2], 255, np.uint8)])
    h, w = a.shape[:2]
    levels = mip_chain(a, mips)
    body = b"".join(L[:, :, [2, 1, 0, 3]].tobytes() for L in levels)
    d = bytearray(HDR)
    struct.pack_into("<IIHHHBB", d, 0, attributes, B8G8R8A8, w, h, 1, len(levels), 1)
    struct.pack_into("<3I", d, 16, HDR, HDR, HDR)
    offs, at = [], HDR
    for L in levels:
        offs.append(at)
        at += L.shape[0] * L.shape[1] * 4
    offs += [0] * (13 - len(offs))
    struct.pack_into("<13I", d, 28, *offs[:13])
    return bytes(d) + body


def write_like(rgba, reference_tex, mips=1):
    """Same, but inherit the attribute bits from an existing texture of the same role."""
    attr = struct.unpack_from("<I", reference_tex, 0)[0]
    return write(rgba, attributes=attr, mips=mips)


def mip_chain(rgba, mips):
    """Box-filtered RGBA levels, largest first, down to 1x1 or ``mips`` levels.

    Each edge halves and clamps at 1 on its own, so a 16x4 source has 16x4, 8x2, 4x1, 2x1,
    1x1 -- the same rule as ``TexHeader.mip_dimensions`` and ``texio.float_chain``. The
    earlier loop stopped as soon as *either* edge reached 1, which gave non-square textures
    a shorter chain here than ``encode_tiers`` wrote for the same image.
    """
    a = np.asarray(rgba, np.uint8)
    levels, cur = [], a
    for _ in range(max(1, mips)):
        levels.append(cur)
        if cur.shape[0] <= 1 and cur.shape[1] <= 1:
            break
        cur = _box_down(cur)
    return levels


def _box_down(cur):
    """One mip step: a 2x2 box average, degrading to 2x1 or 1x2 once an edge has reached 1."""
    h, w = cur.shape[:2]
    hh, ww = max(1, h // 2), max(1, w // 2)
    f = cur[: 2 * hh if h > 1 else 1, : 2 * ww if w > 1 else 1].astype(np.float32)
    if h > 1 and w > 1:
        m = f.reshape(hh, 2, ww, 2, 4).mean((1, 3))
    elif h > 1:
        m = f.reshape(hh, 2, 1, 4).mean(1)
    else:
        m = f.reshape(1, ww, 2, 4).mean(2)
    return m.round().astype(np.uint8)


def write_blocks(fmt, levels_blocks, w, h, attributes=ATTR_2D):
    """A block-compressed .tex: `levels_blocks` are the encoded bytes of each mip, largest

    first, each ceil(w/4)*ceil(h/4)*block_bytes long for its own dimensions.
    """
    d = bytearray(HDR)
    struct.pack_into("<IIHHHBB", d, 0, attributes, fmt, w, h, 1, len(levels_blocks), 1)
    struct.pack_into("<3I", d, 16, HDR, HDR, HDR)
    offs, at = [], HDR
    cw, ch = w, h
    for i, blk in enumerate(levels_blocks):
        need = ((cw + 3) // 4) * ((ch + 3) // 4) * BLOCK_BYTES[fmt]
        assert len(blk) == need, "mip %d: %d bytes, expected %d for %dx%d" % (
            i,
            len(blk),
            need,
            cw,
            ch,
        )
        offs.append(at)
        at += len(blk)
        cw, ch = max(1, cw // 2), max(1, ch // 2)
    offs += [0] * (13 - len(offs))
    struct.pack_into("<13I", d, 28, *offs[:13])
    return bytes(d) + b"".join(levels_blocks)


def write_bc7(rgba, mips=None, attributes=ATTR_2D, modes=(6, 5)):
    """RGBA (h, w, 4) -> BC7 .tex with a full mip chain (tools/bc7enc). `modes=(6,)` is about

    twice as fast and only loses a little on blocks whose alpha varies sharply.
    """
    from . import bc7enc

    a = np.asarray(rgba, np.uint8)
    h, w = a.shape[:2]
    if mips is None:
        mips = int(np.floor(np.log2(max(w, h)))) + 1
    levels = mip_chain(a, mips)
    return write_blocks(BC7, [bc7enc.encode(L, modes=modes) for L in levels], w, h, attributes)
