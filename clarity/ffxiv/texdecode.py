"""Decode FFXIV .tex mip 0 to an RGBA numpy array.

Uses `texture2ddecoder` when present (handles BC7/BC6H); falls back to pure-python
decoders for BC1/BC3/BC4/BC5, which cover ~90% of gear textures.
    pip install texture2ddecoder --break-system-packages
"""

import numpy as np

from . import tex as texfile

try:
    import texture2ddecoder as _t2d
except Exception:
    _t2d = None


def _bc1_blocks(data, w, h, four_colour=False):
    """Decode 8-byte BC1 colour blocks.

    `four_colour` forces the 4-colour palette regardless of the c0 > c1 test. BC2 and BC3 embed a
    BC1 colour block that is ALWAYS in 4-colour mode -- they carry alpha separately, so the encoder
    never needs the 3-colour-plus-transparent variant and decoders must not infer it. Consulting
    c0 > c1 there silently picks the wrong interpolants for any block that happens to have c0 <= c1.
    """
    bw, bh = (w + 3) // 4, (h + 3) // 4
    blocks = np.frombuffer(data[: bw * bh * 8], dtype="<u2").reshape(bh, bw, 4)
    c0, c1 = blocks[..., 0].astype(np.uint32), blocks[..., 1].astype(np.uint32)
    bits = (blocks[..., 3].astype(np.uint32) << 16) | blocks[..., 2].astype(np.uint32)

    def unpack565(c):
        r = ((c >> 11) & 0x1F) * 255 // 31
        g = ((c >> 5) & 0x3F) * 255 // 63
        b = (c & 0x1F) * 255 // 31
        return np.stack([r, g, b], -1).astype(np.float32)

    p0, p1 = unpack565(c0), unpack565(c1)
    wide = np.ones_like(c0, dtype=bool) if four_colour else (c0 > c1)
    pal = np.zeros((*p0.shape[:2], 4, 3), np.float32)
    pal[..., 0, :], pal[..., 1, :] = p0, p1
    pal[..., 2, :] = np.where(wide[..., None], (2 * p0 + p1) / 3, (p0 + p1) / 2)
    pal[..., 3, :] = np.where(wide[..., None], (p0 + 2 * p1) / 3, 0)
    a = np.ones((*p0.shape[:2], 4), np.float32) * 255
    a[..., 3] = np.where(wide, 255, 0)
    out = np.zeros((bh * 4, bw * 4, 4), np.uint8)
    for py in range(4):
        for px in range(4):
            sel = (bits >> (2 * (py * 4 + px))) & 3
            idx = np.take_along_axis(pal, sel[..., None, None].repeat(3, -1), axis=2)[:, :, 0, :]
            av = np.take_along_axis(a, sel[..., None], axis=2)[:, :, 0]
            out[py::4, px::4, :3] = idx.astype(np.uint8)
            out[py::4, px::4, 3] = av.astype(np.uint8)
    return out[:h, :w]


def _bc_alpha_plane(data, w, h, stride, off):
    """BC4-style 8-byte alpha block decoder -> (h,w) uint8."""
    bw, bh = (w + 3) // 4, (h + 3) // 4
    raw = np.frombuffer(data, dtype=np.uint8).reshape(bh, bw, stride)[:, :, off : off + 8]
    a0 = raw[..., 0].astype(np.float32)
    a1 = raw[..., 1].astype(np.float32)
    bits = np.zeros(raw.shape[:2], np.uint64)
    for i in range(6):
        bits |= raw[..., 2 + i].astype(np.uint64) << np.uint64(8 * i)
    pal = np.zeros((*raw.shape[:2], 8), np.float32)
    pal[..., 0], pal[..., 1] = a0, a1
    gt = a0 > a1
    for i in range(1, 7):
        pal[..., i + 1] = np.where(
            gt,
            ((7 - i) * a0 + i * a1) / 7,
            ((5 - i) * a0 + i * a1) / 5 if i <= 5 else 0,
        )
    pal[..., 6] = np.where(gt, pal[..., 6], 0)
    pal[..., 7] = np.where(gt, pal[..., 7], 255)
    out = np.zeros((bh * 4, bw * 4), np.uint8)
    for py in range(4):
        for px in range(4):
            sel = ((bits >> np.uint64(3 * (py * 4 + px))) & np.uint64(7)).astype(np.int64)
            out[py::4, px::4] = np.take_along_axis(pal, sel[..., None], axis=2)[:, :, 0].astype(
                np.uint8
            )
    return out[:h, :w]


def decode(tex_bytes: bytes, mip: int = 0) -> np.ndarray:
    """Decode one mip of a 2.0-era `.tex` file -> (H, W, 4) uint8 RGBA."""
    h = texfile.TexHeader(tex_bytes)
    if not 0 <= mip < h.mip_count or h.surface_offsets[mip] == 0:
        # Falling back to mip 0 here returned the wrong surface silently; a caller that
        # asks for a level the file does not have should hear about it.
        raise ValueError(f"mip {mip} not present (file has {h.mip_count})")
    off = h.surface_offsets[mip]
    w, ht, _d = h.mip_dimensions(mip)
    size = h.surface_size(mip)
    return decode_raw(h.format_name, tex_bytes[off : off + size], w, ht)


def decode_raw(f: str, data: bytes, w: int, ht: int) -> np.ndarray:
    """Decode a bare surface of known format -> (H, W, 4) uint8 RGBA.

    Split out from `decode` so the 1.0 `GTEX` reader can share the same block
    decoders -- 1.0's DXT1/DXT3/DXT5 are byte-identical to BC1/BC2/BC3.
    """
    if f in ("BC7", "BC6H"):
        # There is no pure-python fallback for these. Without the decoder the old code fell through
        # every remaining branch and returned whatever the last one produced -- garbage of the right
        # shape, which is the failure mode that took a checkerboard three weeks to notice. Fail here
        # instead, naming the fix.
        if _t2d is None:
            raise RuntimeError(
                f"{f} needs texture2ddecoder, which is not installed. Add `--with texture2ddecoder` "
                "to the uv run, and pin --python 3.12: it ships no wheels for 3.13+."
            )
        fn = {"BC7": _t2d.decode_bc7, "BC6H": _t2d.decode_bc6}[f]
        bgra = np.frombuffer(fn(data, w, ht), np.uint8).reshape(ht, w, 4)
        return bgra[..., [2, 1, 0, 3]].copy()
    if f == "BC2":
        # 16-byte block: 8 bytes of raw 4-bit alpha, then a BC1 colour block
        # that is always in 4-colour mode (c0 > c1 is not consulted).
        if _t2d is not None and hasattr(_t2d, "decode_bc2"):
            bgra = np.frombuffer(_t2d.decode_bc2(data, w, ht), np.uint8).reshape(ht, w, 4)
            return bgra[..., [2, 1, 0, 3]].copy()
        bw, bh = (w + 3) // 4, (ht + 3) // 4
        blocks = np.frombuffer(data[: bw * bh * 16], np.uint8).reshape(bh, bw, 16)
        rgb = _bc1_blocks(blocks[:, :, 8:].tobytes(), w, ht, four_colour=True)
        a = np.zeros((bh * 4, bw * 4), np.uint8)
        for py in range(4):
            for px in range(4):
                i = py * 4 + px
                byte = blocks[..., i // 2]
                nib = (byte & 0x0F) if i % 2 == 0 else (byte >> 4)
                a[py::4, px::4] = (nib.astype(np.uint16) * 17).astype(np.uint8)
        rgb[..., 3] = a[:ht, :w]
        return rgb
    if f == "BC1":
        if _t2d is not None:
            bgra = np.frombuffer(_t2d.decode_bc1(data, w, ht), np.uint8).reshape(ht, w, 4)
            return bgra[..., [2, 1, 0, 3]].copy()
        return _bc1_blocks(data, w, ht)
    if f == "BC3":
        if _t2d is not None:
            bgra = np.frombuffer(_t2d.decode_bc3(data, w, ht), np.uint8).reshape(ht, w, 4)
            return bgra[..., [2, 1, 0, 3]].copy()
        # A BC3 block is SIXTEEN bytes: 8 of BC4-style alpha, then 8 of BC1 colour. The colour half
        # therefore has to be extracted before it is handed to a decoder that walks 8-byte blocks
        # back to back. Passing `data` straight through -- under a comment correctly noting that the
        # colour half sits at +8 -- makes every second "colour block" actually be alpha data, and
        # doubles the apparent block grid. It decodes to a fine checkerboard of the right SHAPE,
        # which is why it survived: at thumbnail size it reads as texture rather than as corruption.
        bw, bh = (w + 3) // 4, (ht + 3) // 4
        blocks = np.frombuffer(data[: bw * bh * 16], np.uint8).reshape(bh, bw, 16)
        rgb = _bc1_blocks(blocks[:, :, 8:].tobytes(), w, ht, four_colour=True)
        rgb[..., 3] = _bc_alpha_plane(data, w, ht, 16, 0)
        return rgb
    if f == "BC5":
        if _t2d is not None:
            bgra = np.frombuffer(_t2d.decode_bc5(data, w, ht), np.uint8).reshape(ht, w, 4)
            return bgra[..., [2, 1, 0, 3]].copy()
        r = _bc_alpha_plane(data, w, ht, 16, 0)
        g = _bc_alpha_plane(data, w, ht, 16, 8)
        out = np.zeros((ht, w, 4), np.uint8)
        out[..., 0] = r
        out[..., 1] = g
        out[..., 3] = 255
        return out
    if f == "BC4":
        r = _bc_alpha_plane(data, w, ht, 8, 0)
        out = np.zeros((ht, w, 4), np.uint8)
        out[..., 0] = out[..., 1] = out[..., 2] = r
        out[..., 3] = 255
        return out
    if f in ("B8G8R8A8", "B8G8R8X8"):
        a = np.frombuffer(data, np.uint8).reshape(ht, w, 4)
        out = a[..., [2, 1, 0, 3]].copy()
        if f == "B8G8R8X8":
            out[..., 3] = 255  # X is not alpha; the sampler reads 1.0
        return out
    if f == "B4G4R4A4":
        v = np.frombuffer(data, "<u2").reshape(ht, w).astype(np.uint32)
        out = np.zeros((ht, w, 4), np.uint8)
        out[..., 2] = ((v >> 0) & 0xF) * 17
        out[..., 1] = ((v >> 4) & 0xF) * 17
        out[..., 0] = ((v >> 8) & 0xF) * 17
        out[..., 3] = ((v >> 12) & 0xF) * 17
        return out
    if f == "B5G5R5A1":
        # 16 bits little-endian: B in 0..4, G in 5..9, R in 10..14, one bit of alpha at 15.
        # 5 -> 8 bits by bit replication ((v << 3) | (v >> 2)), which maps 31 to 255 exactly;
        # a plain <<3 would cap white at 248 and tint every icon.
        v = np.frombuffer(data, "<u2").reshape(ht, w).astype(np.uint32)
        out = np.empty((ht, w, 4), np.uint8)
        for ch, sh in ((2, 0), (1, 5), (0, 10)):
            c5 = (v >> sh) & 0x1F
            out[..., ch] = ((c5 << 3) | (c5 >> 2)).astype(np.uint8)
        out[..., 3] = np.where((v >> 15) & 1, 255, 0).astype(np.uint8)
        return out
    if f == "L8":
        lum = np.frombuffer(data, np.uint8).reshape(ht, w)
        out = np.empty((ht, w, 4), np.uint8)
        out[..., 0] = out[..., 1] = out[..., 2] = lum
        out[..., 3] = 255
        return out
    if f == "A8":
        # Alpha only. D3D samples RGB as 0 from an A8 surface, so anything converting this to a
        # colour format has to write zeros there or the shader reading it sees white where it
        # used to see black.
        a = np.frombuffer(data, np.uint8).reshape(ht, w)
        out = np.zeros((ht, w, 4), np.uint8)
        out[..., 3] = a
        return out
    raise NotImplementedError(f"no decoder for {f}")
