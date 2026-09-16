"""Rewrite a `.mdl`'s string table, with names of any length.

A model's runtime section opens with `u16 count, u16 pad, u32 size` and then a blob of
null-terminated names: attributes, bones, material paths, shape names. Everything that refers to
one stores a **byte offset into that blob**, and those offset tables sit *after* the blob, so
changing any name by even one byte moves them and every section behind them.

Until now the answer to "point this mesh at a different material" was to pick a replacement name
that happened to be the same length. That works right up until it does not -- `_ckbra` swaps for
`_top_a` at 25 bytes each, `_ckundies` has no 28-byte equivalent worth having -- and choosing
asset names to suit a byte count is how a mod ends up with paths nobody can read. This does it
properly: rebuild the blob, re-point every offset, and shift the file header's own vertex and
index offsets by the difference.

Offsets that reference the blob, all u32 unless noted:

    attribute name offsets   attribute_count, immediately after the mesh records
    material name offsets    material_count,  after the submesh and terrain-shadow records
    bone name offsets        bone_count,      after the material offsets
    shape name offset        first field of each 16-byte Shape record
"""

import struct

from . import mdlpatch

HDR = 0x44


def _tables(m):
    """Absolute file offsets of every table that stores a string offset."""
    o = m.mesh_off + 36 * len(m.meshes)
    attr = o
    o += 4 * m.counts["attribute"]
    o += 20 * m.counts["tsm"]
    o += 16 * m.counts["submesh"]
    o += 10 * m.counts["tss"]
    mat = o
    o += 4 * m.counts["material"]
    bone = o
    o += 4 * m.counts["bone"]
    bta = struct.unpack_from("<f 9H BB H BB ff HH BBBB H H I 4B", m.d, m.radius_off)[23]
    o += 4 * m.counts["bone_table"] + 2 * bta
    shape = o
    return attr, mat, bone, shape


def read(raw):
    """-> (names in blob order, {kind: [name, ...]})."""
    m = mdlpatch.Mdl(raw)
    _c, _p, size = struct.unpack_from("<HHI", m.d, m.rt0)
    blob = bytes(m.d[m.rt0 + 8 : m.rt0 + 8 + size])
    at, mt, bn, sh = _tables(m)

    def names(off, n, stride=4):
        out = []
        for i in range(n):
            p = struct.unpack_from("<I", m.d, off + i * stride)[0]
            out.append(blob[p : blob.index(b"\0", p)].decode())
        return out

    return blob, {
        "attribute": names(at, m.counts["attribute"]),
        "material": names(mt, m.counts["material"]),
        "bone": names(bn, m.counts["bone"]),
        "shape": names(sh, m.counts["shape"], 16),
    }


def rewrite(raw, mapping):
    """Return a new model with every name in `mapping` replaced. Lengths may differ."""
    m = mdlpatch.Mdl(raw)
    _c, _p, size = struct.unpack_from("<HHI", m.d, m.rt0)
    blob_at = m.rt0 + 8
    post = blob_at + size
    blob = bytes(m.d[blob_at:post])
    at, mt, bn, sh = _tables(m)

    # Rebuild the blob in its original order, so unrelated tooling that walks it sequentially
    # still sees what it expects.
    parts, new_off, cur = [], {}, 0
    p = 0
    for _ in range(_c):  # exactly string_count names; what follows is alignment padding
        e = blob.index(b"\0", p)
        old = blob[p:e].decode()
        new = mapping.get(old, old).encode()
        new_off[p] = cur
        parts.append(new + b"\0")
        cur += len(new) + 1
        p = e + 1
    while cur % 4:  # keep the u32 tables behind it aligned
        parts.append(b"\0")
        cur += 1
    new_blob = b"".join(parts)
    delta = len(new_blob) - size

    out = bytearray(m.d[:blob_at]) + bytearray(new_blob) + bytearray(m.d[post:])
    struct.pack_into("<HHI", out, m.rt0, _c, _p, len(new_blob))

    def repoint(off, n, stride=4):
        for i in range(n):
            a = off + delta + i * stride
            old = struct.unpack_from("<I", out, a)[0]
            struct.pack_into("<I", out, a, new_off[old])

    repoint(at, m.counts["attribute"])
    repoint(mt, m.counts["material"])
    repoint(bn, m.counts["bone"])
    repoint(sh, m.counts["shape"], 16)

    # The runtime section changed length, so the file header's own offsets move with it.
    runtime = struct.unpack_from("<I", out, 8)[0] + delta
    struct.pack_into("<I", out, 8, runtime)
    for k in range(3):
        for base in (16, 28):  # vertex_offsets, index_offsets
            v = struct.unpack_from("<I", out, base + 4 * k)[0]
            if v:
                struct.pack_into("<I", out, base + 4 * k, v + delta)
    # The 60-byte LOD record ends with 4 u32: vertex_buffer_size, index_buffer_size,
    # vertex_data_offset, index_data_offset -- the last two at 52 and 56, not 0x38 and 0x3C.
    for k in range(3):
        rec = m.lod_off_0 + delta + 60 * k
        for fld in (52, 56):
            v = struct.unpack_from("<I", out, rec + fld)[0]
            if v:
                struct.pack_into("<I", out, rec + fld, v + delta)
    return bytes(out)
