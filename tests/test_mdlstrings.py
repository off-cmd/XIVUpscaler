"""``clarity.ffxiv.mdlstrings`` and the ``mdlpatch.Mdl`` section walk it depends on.

A minimal but *structurally valid* v6 ``.mdl`` is built in memory: one vertex declaration, one
mesh of two vertices, and one attribute, material, bone and shape name in the string blob. Every
count the section walk consumes is set, and ``Mdl.__init__``'s own assertion (header + stack +
runtime == LOD0 vertex_data_offset) is what proves the synthetic layout is self-consistent.

File layout (little-endian)::

     0  header (68)     version, stack_size, runtime_size, vdc, material_count,
                        vertex_offsets[3], index_offsets[3], vertex_buffer_size[3],
                        index_buffer_size[3], lod_count, 3 x u8
    68  stack           vdc x 136-byte vertex declarations (8-byte elements, 0xFF-terminated)
        runtime         u16 string_count, u16 pad, u32 blob_size, blob of "name\\0"s,
                        56-byte model header (counts), 3 x 60-byte LOD records,
                        36-byte mesh records, then the offset tables into the blob
                        (attribute, material, bone; shape records carry theirs inline),
                        submesh bone map, a pad byte + padding, bounding boxes
        vertex data     LOD0 at vertex_data_offset; index data at index_data_offset
"""

import struct

import numpy as np
import pytest

from clarity.ffxiv import mdlpatch, mdlstrings

VERSION = 0x01000006
ATTRIBUTE = "atr_lod"
MATERIAL = "/mt_c0101e0001_top_a.mtrl"
BONE = "j_kao"
SHAPE = "shp_a"
NAMES = [ATTRIBUTE, MATERIAL, BONE, SHAPE]
POSITIONS = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 8.0]])
INDICES = (0, 1, 0)


def _blob(names: list[str]) -> tuple[bytes, list[int]]:
    """Null-terminated names, padded to 4 bytes; returns (blob, offset of each name)."""
    offsets, blob = [], b""
    for n in names:
        offsets.append(len(blob))
        blob += n.encode() + b"\0"
    while len(blob) % 4:
        blob += b"\0"
    return blob, offsets


def build_mdl(names: list[str] = NAMES) -> bytes:
    blob, name_off = _blob(names)
    attr_off, mat_off, bone_off, shape_off = name_off

    # -- stack: one declaration, one element: stream 0, offset 0, type 2 (3 x f32), usage 0
    # (position), usage_index 0; then the 0xFF end marker for the remaining slots.
    stack = struct.pack("<5B3x", 0, 0, 2, 0, 0) + b"\xff" * (mdlpatch.DECL_SLOT - 8)

    # -- runtime section, assembled so its length is known before the header is written
    rt = bytearray()
    rt += struct.pack("<HHI", len(names), 0, len(blob)) + blob
    radius_off = len(rt)
    # 56-byte model header: radius, then the counts the section walk consumes.
    rt += struct.pack("<f", 1.0)
    # mesh attribute submesh material bone bone_table shape shape_mesh shape_value
    rt += struct.pack("<9H", 1, 1, 1, 1, 1, 0, 1, 0, 0)
    rt += struct.pack("<BB", 1, 0)  # lod_count, flags1
    rt += struct.pack("<H", 0)  # element_id_count
    rt += struct.pack("<BB", 0, 0)  # tsm_count, flags2 (bit 4 would add extra LOD records)
    rt += struct.pack("<ff", 0.0, 0.0)  # model / shadow clip-out distances
    rt += struct.pack("<HH", 0, 0)  # culling type, tss_count
    rt += struct.pack("<BBBB", 0, 0, 0, 0)  # flags3, bg change/crest material, neck_morph_count
    rt += struct.pack("<HH", 0, 0)  # bone_table_array_count_total, unknown
    rt += struct.pack("<I", 0)  # face_data_count
    rt += bytes(4)
    assert len(rt) - radius_off == 56
    lod_off = len(rt)
    rt += bytes(60 * 3)  # LOD records: filled once vertex_data_offset is known
    mesh_off = len(rt)
    # 36-byte mesh record
    rt += struct.pack("<HH", len(POSITIONS), 0)  # vertex_count, pad
    rt += struct.pack("<I", len(INDICES))  # index_count
    rt += struct.pack("<4H", 0, 0, 1, 0)  # material, submesh_index, submesh_count, bone_table
    rt += struct.pack("<I", 0)  # start_index
    rt += struct.pack("<3I", 0, 0, 0)  # vertex buffer offset per stream
    rt += struct.pack("<4B", 12, 0, 0, 0)  # vertex stride per stream, pad
    assert len(rt) - mesh_off == 36
    attr_table = len(rt)
    rt += struct.pack("<I", attr_off)
    rt += bytes(16)  # one submesh record (index_offset, index_count, attribute_mask, bones)
    mat_table = len(rt)
    rt += struct.pack("<I", mat_off)
    bone_table = len(rt)
    rt += struct.pack("<I", bone_off)
    shape_table = len(rt)
    rt += struct.pack("<I", shape_off) + bytes(12)  # shape record: name offset + 3 u32
    rt += struct.pack("<I", 4) + bytes(4)  # submesh bone map: u32 size, then that many bytes
    rt += bytes([3, 0, 0, 0])  # pad byte = 3, then 3 bytes of padding
    bbox_off = len(rt)
    rt += bytes(32 * 5)  # 4 model boxes + 1 bone box, 8 x f32 each
    runtime_size = len(rt)

    vertex_data_offset = mdlpatch.HDR_SIZE + len(stack) + runtime_size
    vertex_bytes = b"".join(struct.pack("<3f", *p) for p in POSITIONS)
    index_data_offset = vertex_data_offset + len(vertex_bytes)
    index_bytes = struct.pack("<3H", *INDICES)
    # LOD 0 (60 bytes): mesh_index, mesh_count, then 44 bytes of ranges/water/shadow/edge/neck
    # fields that stay zero, then vertex_buffer_size, index_buffer_size, vertex_data_offset,
    # index_data_offset at +44..+60.
    struct.pack_into("<2H", rt, lod_off, 0, 1)
    struct.pack_into(
        "<4I",
        rt,
        lod_off + 44,
        len(vertex_bytes),
        len(index_bytes),
        vertex_data_offset,
        index_data_offset,
    )

    hdr = struct.pack("<III", VERSION, len(stack), runtime_size)
    hdr += struct.pack("<HH", 1, 1)  # vertex declaration count, material count
    hdr += struct.pack("<3I", vertex_data_offset, 0, 0)  # vertex_offsets per LOD
    hdr += struct.pack("<3I", index_data_offset, 0, 0)  # index_offsets per LOD
    hdr += struct.pack("<3I", len(vertex_bytes), 0, 0)  # vertex_buffer_size per LOD
    hdr += struct.pack("<3I", len(index_bytes), 0, 0)  # index_buffer_size per LOD
    hdr += struct.pack("<BBBB", 1, 0, 0, 0)  # lod_count, flags, pad
    assert len(hdr) == mdlpatch.HDR_SIZE
    out = hdr + stack + bytes(rt) + vertex_bytes + index_bytes
    # Sanity of the builder itself, against the module's absolute-offset bookkeeping.
    m = mdlpatch.Mdl(out)
    rt_base = mdlpatch.HDR_SIZE + len(stack)
    assert m.rt0 == rt_base
    assert m.radius_off == rt_base + radius_off
    assert m.lod_off_0 == rt_base + lod_off
    assert m.mesh_off == rt_base + mesh_off
    assert m.bbox_off == rt_base + bbox_off
    assert mdlstrings._tables(m) == tuple(
        rt_base + t for t in (attr_table, mat_table, bone_table, shape_table)
    )
    return out


# ---------------------------------------------------------------- mdlpatch


def test_mdl_parses_the_synthetic_model() -> None:
    m = mdlpatch.Mdl(build_mdl())
    assert m.version == VERSION
    assert m.counts["mesh"] == 1
    assert m.counts["material"] == 1
    assert m.counts["bone"] == 1
    assert m.counts["shape"] == 1
    assert m.decls == [[{"stream": 0, "offset": 0, "type": 2, "usage": 0, "usage_index": 0}]]
    assert m.meshes[0]["vertex_count"] == 2
    assert m.meshes[0]["stride"] == [12, 0, 0]
    assert m.lod_of(0) == 0
    np.testing.assert_array_equal(m.positions(0), POSITIONS)


def test_patch_positions_rewrites_vertices_and_bounds_only() -> None:
    raw = build_mdl()
    out, n = mdlpatch.patch_positions(raw, lambda _mi, p: p * 2.0)
    assert n == 2
    assert len(out) == len(raw)
    m = mdlpatch.Mdl(out)
    np.testing.assert_array_equal(m.positions(0), POSITIONS * 2.0)
    # bb[1] is the tight box, bb[0] extends it to contain the origin
    lo, hi = POSITIONS.min(0) * 2, POSITIONS.max(0) * 2
    bb1 = struct.unpack_from("<8f", out, m.bbox_off + mdlpatch.BBOX)
    assert bb1 == (*lo, 1.0, *hi, 1.0)
    bb0 = struct.unpack_from("<8f", out, m.bbox_off)
    assert bb0 == (*np.minimum(lo, 0), 1.0, *np.maximum(hi, 0), 1.0)
    assert struct.unpack_from("<f", out, m.radius_off)[0] == pytest.approx(np.linalg.norm(hi))
    # Everything outside the positions, boxes and radius is untouched.
    v0 = m.lods[0]["vertex_data_offset"]
    assert out[: m.radius_off] == raw[: m.radius_off]
    assert out[m.radius_off + 4 : m.bbox_off] == raw[m.radius_off + 4 : m.bbox_off]
    assert out[m.bbox_off + 2 * mdlpatch.BBOX : v0] == raw[m.bbox_off + 2 * mdlpatch.BBOX : v0]
    assert out[v0 + 24 :] == raw[v0 + 24 :]


# ---------------------------------------------------------------- mdlstrings.read


def test_read_returns_the_blob_and_names_by_kind() -> None:
    blob, names = mdlstrings.read(build_mdl())
    assert blob == _blob(NAMES)[0]
    assert names == {
        "attribute": [ATTRIBUTE],
        "material": [MATERIAL],
        "bone": [BONE],
        "shape": [SHAPE],
    }


# ---------------------------------------------------------------- mdlstrings.rewrite


def _check_rewritten(raw: bytes, mapping: dict[str, str]) -> bytes:
    out = mdlstrings.rewrite(raw, mapping)
    old, new = mdlpatch.Mdl(raw), mdlpatch.Mdl(out)
    old_size = struct.unpack_from("<I", raw, old.rt0 + 4)[0]
    new_size = struct.unpack_from("<I", out, new.rt0 + 4)[0]
    delta = new_size - old_size
    # The module keeps the old blob's alignment NULs as empty entries and pads again, so the
    # new blob is at least the minimal padded size and never more than one extra word.
    minimal = len(_blob([mapping.get(n, n) for n in NAMES])[0])
    assert minimal <= new_size <= minimal + 4
    assert new_size % 4 == 0
    assert len(out) == len(raw) + delta

    # The new file parses, with the same section walk shifted by delta.
    assert new.runtime_size == old.runtime_size + delta
    assert new.vertex_offsets == [old.vertex_offsets[0] + delta, 0, 0]
    assert new.index_offsets == [old.index_offsets[0] + delta, 0, 0]
    assert new.lods[0]["vertex_data_offset"] == old.lods[0]["vertex_data_offset"] + delta
    assert new.lods[0]["index_data_offset"] == old.lods[0]["index_data_offset"] + delta
    assert new.lods[1:] == old.lods[1:]  # unused LODs hold zero offsets and stay zero

    # Names re-resolve through the re-pointed tables, in blob order.
    blob, names = mdlstrings.read(out)
    assert blob.rstrip(b"\0") == b"\0".join(mapping.get(n, n).encode() for n in NAMES)
    assert names["attribute"] == [mapping.get(ATTRIBUTE, ATTRIBUTE)]
    assert names["material"] == [mapping.get(MATERIAL, MATERIAL)]
    assert names["bone"] == [mapping.get(BONE, BONE)]
    assert names["shape"] == [mapping.get(SHAPE, SHAPE)]

    # Geometry moved as a block and is still readable at its new offsets.
    np.testing.assert_array_equal(new.positions(0), POSITIONS)
    ido = new.lods[0]["index_data_offset"]
    assert struct.unpack_from("<3H", out, ido) == INDICES
    # Only runtime_size (header +8) and the offsets change ahead of the blob; the vertex
    # declarations and the blob's count word are untouched.
    assert out[:8] == raw[:8]
    assert out[mdlpatch.HDR_SIZE : old.rt0 + 4] == raw[mdlpatch.HDR_SIZE : old.rt0 + 4]
    return out


def test_rewrite_with_a_longer_material_path() -> None:
    _check_rewritten(build_mdl(), {MATERIAL: "/mt_c0101e0001_top_ckundies_longer.mtrl"})


def test_rewrite_with_a_shorter_name() -> None:
    _check_rewritten(build_mdl(), {MATERIAL: "/a.mtrl", ATTRIBUTE: "x"})


def test_rewrite_with_several_names_at_once() -> None:
    _check_rewritten(
        build_mdl(),
        {ATTRIBUTE: "atr_hidden_by_default", BONE: "j_kubi", SHAPE: "shp_bbb"},
    )


def test_rewrite_with_identity_mapping_is_byte_identical() -> None:
    raw = build_mdl()
    assert mdlstrings.rewrite(raw, {}) == raw
    assert mdlstrings.rewrite(raw, {"not-in-the-blob": "whatever"}) == raw


def test_rewrite_rejects_garbage() -> None:
    with pytest.raises((AssertionError, struct.error)):
        mdlstrings.rewrite(b"not a model", {})
    with pytest.raises((AssertionError, struct.error)):
        mdlstrings.rewrite(bytes(300), {})
