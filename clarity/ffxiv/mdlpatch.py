"""Rewrite the vertex positions of an FFXIV `.mdl` in place, leaving everything else byte-identical.

    import mdlpatch
    out = mdlpatch.patch_positions(open(p, "rb").read(), lambda mesh, pos: pos * 1.1)

Why in place rather than a writer
---------------------------------
Writing a `.mdl` from scratch means reproducing eleven interleaved sections, two redundant copies
of every buffer offset, a version fork in the bone tables, and a padding block whose size is not
derivable. Physis has that writer and it has three known defects — tangents written as zeros
(`mod.rs:1355`), a `Byte4` blend-weight arm marked `TODO: WRONG!` (line 1269), and no
`UnsignedShort4` write path at all, which is what every modern v6 character model uses for blend
weights and indices. Porting it would import those bugs.

For a **positions-only change with unchanged topology** none of that is necessary. Position is
always the `usage == 0` element of stream 0 at element offset 0, sharing that stream only with
blend weights and blend indices — never with normals, UVs, tangents or colours. Vertex counts,
strides, buffer sizes and every offset are functions of topology, so they do not move. The only
things that legitimately change are the position bytes and the bounding boxes.

Layout verified against `redstrate/Physis`, `file-formats/imhex/mdl.hexpat`, Lumina's
`MdlStructs.cs` and Xande's `MdlFileWriter.cs`, and asserted at runtime.
"""

import struct

import numpy as np

HDR = "<III HH 3I 3I 3I 3I BBBB"
HDR_SIZE = 68
DECL_SLOT = 17 * 8  # 136 bytes per mesh, sentinel-terminated
BBOX = 32

# VertexType -> (struct format, component count, bytes)
VT = {
    0: ("<f", 1, 4),
    1: ("<2f", 2, 8),
    2: ("<3f", 3, 12),
    3: ("<4f", 4, 16),
    5: (None, 4, 4),
    6: (None, 2, 4),
    7: (None, 4, 8),
    8: (None, 4, 4),
    9: (None, 2, 4),
    10: (None, 4, 8),
    13: ("<2e", 2, 4),
    14: ("<4e", 4, 8),
    16: (None, 2, 4),
    17: (None, 4, 8),
}


class Mdl:
    def __init__(self, data):
        self.d = bytearray(data)
        v = struct.unpack_from(HDR, self.d, 0)
        self.version = v[0]
        self.stack_size = v[1]
        self.runtime_size = v[2]
        self.vdc = v[3]
        self.material_count_h = v[4]
        self.vertex_offsets = list(v[5:8])
        self.index_offsets = list(v[8:11])
        self.vertex_buffer_size = list(v[11:14])
        self.index_buffer_size = list(v[14:17])
        self.lod_count = v[17]
        # Vanilla dat entries pad the stack section up to a 128-byte boundary, so the header
        # size is >= the declarations it contains. Mod tools write it exact. Accept both.
        assert self.vdc * DECL_SLOT <= self.stack_size < self.vdc * DECL_SLOT + 128, (
            "stack_size %d != vdc %d * 136 — version detection is wrong"
            % (self.stack_size, self.vdc)
        )
        self._decls()
        self._runtime()
        # The one assertion that validates the entire section walk at once.
        assert (
            HDR_SIZE + self.stack_size + self.runtime_size == self.lods[0]["vertex_data_offset"]
        ), "section walk mismatch: 0x44+%d+%d != %d" % (
            self.stack_size,
            self.runtime_size,
            self.lods[0]["vertex_data_offset"],
        )

    def _decls(self):
        self.decls = []
        for i in range(self.vdc):
            o = HDR_SIZE + i * DECL_SLOT
            els = []
            for k in range(17):
                st, off, ty, us, ui = struct.unpack_from("<5B", self.d, o + k * 8)
                if st == 0xFF:
                    break
                els.append(
                    {
                        "stream": st,
                        "offset": off,
                        "type": ty,
                        "usage": us,
                        "usage_index": ui,
                    }
                )
            self.decls.append(els)

    def _runtime(self):
        o = HDR_SIZE + self.stack_size
        self.rt0 = o
        _scount, _pad, ssize = struct.unpack_from("<HHI", self.d, o)
        o += 8 + ssize
        mh = struct.unpack_from("<f 9H BB H BB ff HH BBBB H H I 4B", self.d, o)
        self.radius_off = o
        (
            self.radius,
            mesh_count,
            attribute_count,
            submesh_count,
            material_count,
            bone_count,
            bone_table_count,
            shape_count,
            shape_mesh_count,
            shape_value_count,
            _lod_count,
            _flags1,
            element_id_count,
            tsm_count,
            flags2,
            _mclip,
            _sclip,
            _cull,
            tss_count,
            _flags3,
            _bgm,
            _bgc,
            neck_morph_count,
            bta_total,
            _unk8,
            face_data_count,
        ) = mh[:26]
        self.counts = dict(
            mesh=mesh_count,
            attribute=attribute_count,
            submesh=submesh_count,
            material=material_count,
            bone=bone_count,
            bone_table=bone_table_count,
            shape=shape_count,
            shape_mesh=shape_mesh_count,
            shape_value=shape_value_count,
            element_id=element_id_count,
            tsm=tsm_count,
            tss=tss_count,
            neck_morph=neck_morph_count,
            face_data=face_data_count,
        )
        o += 56
        o += 32 * element_id_count
        # 60 bytes, and the two f32 LOD ranges sit at +4 in the middle of the u16 run — flatten
        # them into the u16s and every field after them is off by one.
        #   2H mesh_index/count | 2f ranges | 8H water/shadow/tshadow/vfog pairs
        #   3I edge size/offset/polygon | BBH neck morph | 4I vbs/ibs/vdo/ido
        # That is four pairs, eight u16 — writing 10H makes the record 64 bytes against a real
        # 60, so LOD0's tail reads two fields early and LOD1/2 are pure garbage.
        self.lods = []
        self.lod_off_0 = o  # mdlwrite needs to rewrite these in place
        for _i in range(3):
            f = struct.unpack_from("<2H2f8H3IBBH4I", self.d, o)
            assert struct.calcsize("<2H2f8H3IBBH4I") == 60
            self.lods.append(
                dict(
                    mesh_index=f[0],
                    mesh_count=f[1],
                    water_mesh_index=f[4],
                    water_mesh_count=f[5],
                    vertex_buffer_size=f[18],
                    index_buffer_size=f[19],
                    vertex_data_offset=f[20],
                    index_data_offset=f[21],
                )
            )
            o += 60
        if flags2 & 0x10:
            o += 40 * 3
        self.meshes = []
        self.mesh_off = o
        for _i in range(mesh_count):
            f = struct.unpack_from("<HH I 4H I 3I 4B", self.d, o)
            self.meshes.append(
                dict(
                    vertex_count=f[0],
                    index_count=f[2],
                    material_index=f[3],
                    submesh_index=f[4],
                    submesh_count=f[5],
                    bone_table_index=f[6],
                    start_index=f[7],
                    vbo=list(f[8:11]),
                    stride=list(f[11:14]),
                )
            )
            o += 36
        o += 4 * attribute_count
        o += 20 * tsm_count
        o += 16 * submesh_count
        o += 10 * tss_count
        o += 4 * material_count
        o += 4 * bone_count
        if self.version >= 0x01000006:
            o += 4 * bone_table_count
            o += 2 * bta_total
        else:
            o += 132 * bone_table_count
        o += 16 * shape_count
        o += 12 * shape_mesh_count
        o += 4 * shape_value_count
        sbm = struct.unpack_from("<I", self.d, o)[0]
        o += 4 + sbm
        if self.version >= 0x01000006:
            o += 32 * neck_morph_count
            o += 16 * face_data_count
        pad = self.d[o]
        o += 1 + pad
        self.bbox_off = o  # 4 model boxes, then bone_count bone boxes

    # ---- mesh -> LOD, so the right vertex_data_offset is used
    def lod_of(self, mi):
        for i, lod in enumerate(self.lods):
            if lod["mesh_index"] <= mi < lod["mesh_index"] + lod["mesh_count"]:
                return i
            if (
                lod["water_mesh_count"]
                and lod["water_mesh_index"]
                <= mi
                < lod["water_mesh_index"] + lod["water_mesh_count"]
            ):
                return i
        return 0

    def _pos_element(self, mi):
        for e in self.decls[mi]:
            if e["usage"] == 0:
                return e
        return None

    def positions(self, mi):
        e = self._pos_element(mi)
        if e is None:
            return None
        m = self.meshes[mi]
        base = (
            self.lods[self.lod_of(mi)]["vertex_data_offset"] + m["vbo"][e["stream"]] + e["offset"]
        )
        stride = m["stride"][e["stream"]]
        fmt, _n, _sz = VT[e["type"]]
        if fmt is None:
            raise NotImplementedError("position vertex type %d" % e["type"])
        out = np.empty((m["vertex_count"], 3), np.float64)
        for k in range(m["vertex_count"]):
            out[k] = struct.unpack_from(fmt, self.d, base + stride * k)[:3]
        return out

    def set_positions(self, mi, pos):
        e = self._pos_element(mi)
        m = self.meshes[mi]
        base = (
            self.lods[self.lod_of(mi)]["vertex_data_offset"] + m["vbo"][e["stream"]] + e["offset"]
        )
        stride = m["stride"][e["stream"]]
        fmt, n, _sz = VT[e["type"]]
        for k in range(m["vertex_count"]):
            v = [float(c) for c in pos[k][:3]]
            if n == 4:
                v.append(1.0)  # w is always 1.0 for Half4 / Single4 positions
            struct.pack_into(fmt, self.d, base + stride * k, *v[:n])  # ty: ignore[invalid-argument-type]

    def update_bounds(self):
        """bb[1] is the tight AABB; bb[0] is that AABB extended to contain the origin.

        Derived by measurement, not assumption: on three shipped models bb[1] matched the
        geometry exactly while bb[0] had its min-y pinned to 0.0. Physis and Xande both collapse
        the two into one and still produce loadable files, so this affects culling and LOD
        selection rather than correctness — but a refit moves geometry enough to be worth it.
        """
        allp = [self.positions(i) for i in range(len(self.meshes))]
        allp = [p for p in allp if p is not None and len(p)]
        if not allp:
            return
        P = np.vstack(allp)
        lo, hi = P.min(0), P.max(0)
        bb1 = (lo, hi)
        bb0 = (np.minimum(lo, 0.0), np.maximum(hi, 0.0))
        for i, (a, b) in enumerate((bb0, bb1)):
            struct.pack_into(
                "<8f",
                self.d,
                self.bbox_off + i * BBOX,
                a[0],
                a[1],
                a[2],
                1.0,
                b[0],
                b[1],
                b[2],
                1.0,
            )
        struct.pack_into("<f", self.d, self.radius_off, float(np.linalg.norm(hi)))

    def bytes(self):
        return bytes(self.d)


def patch_positions(data, fn, update_bounds=True):
    """fn(mesh_index, positions Nx3 float64) -> Nx3. Everything else stays byte-identical."""
    m = Mdl(data)
    n = 0
    for mi in range(len(m.meshes)):
        p = m.positions(mi)
        if p is None or not len(p):
            continue
        q = np.asarray(fn(mi, p), float)
        assert q.shape == p.shape, "position count changed — this patcher cannot retopologise"
        m.set_positions(mi, q)
        n += len(q)
    if update_bounds:
        m.update_bounds()
    return m.bytes(), n
