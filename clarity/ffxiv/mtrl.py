"""Read **and write** an FFXIV `.mtrl`, and migrate a legacy material to the Dawntrail shader.

Layout (v `0x01030000`), confirmed against Lumina's `MtrlStructs.cs` and round-tripped
byte-identically over the 2,700 materials in the Atelier Anno collection:

    0x00 u32  version
    0x04 u16  fileSize          0x06 u16 dataSetSize
    0x08 u16  stringTableSize   0x0A u16 shaderPackageNameOffset
    0x0C u8   textureCount, uvSetCount, colorSetCount, additionalDataSize
    0x10      texture[]  (u16 nameOffset, u16 flags)
              uvSet[]    (u16 nameOffset, u8 index, u8 unk)
              colorSet[] (u16 nameOffset, u8 index, u8 unk)
              string table (NUL-terminated, padded)
              additional data
              data set      = colour table (+ dye table).  2048 = 32 rows x 64 B,
                              2176 = the same plus a 128 B dye table, 544 = the legacy 16-row shape
              u16 shaderValueListSize, shaderKeyCount, constantCount, samplerCount, unk1, unk2
              shaderKey[] (u32 category, u32 value)
              constant[]  (u32 id, u16 valueOffset, u16 valueSize)
              sampler[]   (u32 id, u32 flags, u8 textureIndex, u8[3] pad)
              float values[shaderValueListSize / 4]

The migration is the mechanical half of legacy -> Dawntrail, derived by diffing **44 materials that
Anno shipped in both forms** (see KB 40-community/31): swap the shader package, drop the shader key
and constants that exist only in `characterlegacy.shpk`, and leave everything else — textures,
samplers, colour table — untouched, so the result renders as it did before.

The *artistic* half is deliberately not automatic. The same diff showed Anno writes roughness/sheen
into only 29 % of rows, mixed within a single material, with **no correlation to any legacy field**
(|r| <= 0.07) — the values are hand-picked, clustering on 0.5 roughness and 0.2 sheen. `pbr_preset()`
applies those modal values to rows selected by a `Scalar3 >= 1.5` rule that recovers 94 % of the rows
Anno chose (F1 0.70 against 0.45 for upgrading everything). It is a starting point to tune, not a
reproduction.
"""

import pathlib
import struct
import sys

LEGACY_SHPK = "characterlegacy.shpk"
DT_SHPK = "character.shpk"

# Present in characterlegacy.shpk only — verified absent from character.shpk's key/param tables.
LEGACY_KEY = 0xC8BD1DEF
LEGACY_CONSTANTS = {0x36080AD0, 0xB500BB24, 0x575ABFB2}
# `g_SamplerDiffuse`, and the key Anno drops alongside it when moving colour to the colour table.
SAMPLER_DIFFUSE = 0x115306BE
DIFFUSE_KEY = 0xB616DC5A

# Colour-table column indices (KB 10-ffxiv-assets/06-colour-tables.md)
DIFFUSE, SHININESS, SPECULAR, SPECMASK = 0, 3, 4, 7
FLAG11, SHEEN_RATE, SHEEN_TINT, SHEEN_APERTURE = 11, 12, 13, 14
ROUGHNESS, METALNESS, ANISOTROPY = 16, 18, 19


class Mtrl:
    def __init__(self, data):
        self.version = struct.unpack_from("<I", data, 0)[0]
        _fs, self.dataset_size = struct.unpack_from("<HH", data, 4)
        sts, shoff = struct.unpack_from("<HH", data, 8)
        tc, uvc, csc, self.add_size = struct.unpack_from("<4B", data, 0x0C)
        p = 0x10
        tex = [struct.unpack_from("<HH", data, p + 4 * i) for i in range(tc)]
        p += 4 * tc
        uv = [struct.unpack_from("<HBB", data, p + 4 * i) for i in range(uvc)]
        p += 4 * uvc
        cs = [struct.unpack_from("<HBB", data, p + 4 * i) for i in range(csc)]
        p += 4 * csc
        strtab = data[p : p + sts]
        p += sts
        self.additional = data[p : p + self.add_size]
        p += self.add_size
        self.dataset = bytearray(data[p : p + self.dataset_size])
        p += self.dataset_size

        def string_at(o):
            e = strtab.find(b"\0", o)
            return strtab[o : e if e >= 0 else None].decode("ascii", "replace")

        self.textures = [(string_at(o), f) for o, f in tex]
        self.uvsets = [(string_at(o), i, u) for o, i, u in uv]
        self.colorsets = [(string_at(o), i, u) for o, i, u in cs]
        self.shpk = string_at(shoff)

        svl, kc, cc, sc, self.unk1, self.unk2 = struct.unpack_from("<6H", data, p)
        p += 12
        self.keys = [struct.unpack_from("<II", data, p + 8 * i) for i in range(kc)]
        p += 8 * kc
        cons = [struct.unpack_from("<IHH", data, p + 8 * i) for i in range(cc)]
        p += 8 * cc
        self.samplers = [struct.unpack_from("<IIB", data, p + 12 * i) for i in range(sc)]
        p += 12 * sc
        vals = list(struct.unpack_from("<%df" % (svl // 4), data, p))
        # keep each constant with its own slice of the value array, so dropping one is safe
        self.constants = [(cid, vals[off // 4 : (off + size) // 4]) for cid, off, size in cons]

    # ---------------------------------------------------------------- colour table

    @property
    def rows(self):
        """The colour table as a 32x32 list of floats, or None for the legacy 16-row shape."""
        if self.dataset_size < 2048:
            return None
        import numpy as np

        return np.frombuffer(bytes(self.dataset[:2048]), dtype="<f2").reshape(32, 32).astype("f4")

    def set_rows(self, arr):
        import numpy as np

        self.dataset[:2048] = np.asarray(arr, dtype="<f2").tobytes()

    # ---------------------------------------------------------------- write

    def build(self):
        strings, offsets = bytearray(), {}

        def put(s):
            if s not in offsets:
                offsets[s] = len(strings)
                strings.extend(s.encode("ascii") + b"\0")
            return offsets[s]

        tex = [(put(n), f) for n, f in self.textures]
        uv = [(put(n), i, u) for n, i, u in self.uvsets]
        cs = [(put(n), i, u) for n, i, u in self.colorsets]
        shoff = put(self.shpk)
        while len(strings) % 4:
            strings.append(0)

        vals, cons = [], []
        for cid, v in self.constants:
            cons.append((cid, len(vals) * 4, len(v) * 4))
            vals.extend(v)

        body = bytearray()
        for o, f in tex:
            body += struct.pack("<HH", o, f)
        for o, i, u in uv:
            body += struct.pack("<HBB", o, i, u)
        for o, i, u in cs:
            body += struct.pack("<HBB", o, i, u)
        body += strings + self.additional + self.dataset
        body += struct.pack(
            "<6H",
            len(vals) * 4,
            len(self.keys),
            len(cons),
            len(self.samplers),
            self.unk1,
            self.unk2,
        )
        for a, b in self.keys:
            body += struct.pack("<II", a, b)
        for cid, off, size in cons:
            body += struct.pack("<IHH", cid, off, size)
        for sid, fl, ti in self.samplers:
            body += struct.pack("<IIB3x", sid, fl, ti)
        for v in vals:
            body += struct.pack("<f", v)

        head = struct.pack(
            "<IHHHHBBBB",
            self.version,
            0x10 + len(body),
            len(self.dataset),
            len(strings),
            shoff,
            len(tex),
            len(uv),
            len(cs),
            len(self.additional),
        )
        return bytes(head + body)

    # ---------------------------------------------------------------- migration

    def is_legacy(self):
        return self.shpk.lower() == LEGACY_SHPK

    def migrate(self, shpk=DT_SHPK):
        """The provably safe half: shader package, and the key/constants that only legacy has.

        Returns a list of what changed.
        """
        log = []
        if self.shpk.lower() == shpk.lower():
            return log
        log.append(f"shpk {self.shpk} -> {shpk}")
        self.shpk = shpk
        n = len(self.keys)
        self.keys = [k for k in self.keys if (k[0] & 0xFFFFFFFF) != LEGACY_KEY]
        if len(self.keys) != n:
            log.append("dropped legacy key C8BD1DEF")
        n = len(self.constants)
        self.constants = [c for c in self.constants if (c[0] & 0xFFFFFFFF) not in LEGACY_CONSTANTS]
        if len(self.constants) != n:
            log.append("dropped %d legacy constant(s)" % (n - len(self.constants)))
        return log

    def drop_diffuse(self, style="remove"):
        """Anno's own choice on every conversion measured: unbind `g_SamplerDiffuse`, drop the

        `B616DC5A` key, and remove the `_d` texture entry entirely, letting colour come from the
        colour table instead of a diffuse map. Sampler texture indices above the removed slot are
        renumbered.

        Not the default here, because keeping the diffuse preserves appearance exactly and
        `character.shpk` declares `g_SamplerDiffuse` either way. Returns True if it changed
        anything.
        """
        hit = [s for s in self.samplers if (s[0] & 0xFFFFFFFF) == SAMPLER_DIFFUSE]
        if not hit:
            return False
        slot = hit[0][2]
        if style != "dummy":
            self.samplers = [
                (sid, fl, ti - 1 if ti > slot else ti)
                for sid, fl, ti in self.samplers
                if (sid & 0xFFFFFFFF) != SAMPLER_DIFFUSE
            ]
        if 0 <= slot < len(self.textures):
            flags = self.textures[slot][1]
            del self.textures[slot]
        else:
            flags = 0
        if style == "dummy":
            # Anno's other form, seen on Anis: keep the sampler and the key, but drop the `_d`
            # entry and append `dummy.tex` at the end, pointing the diffuse sampler there.
            # Both forms appear in their own releases; neither samples real albedo any more.
            self.textures.append(("dummy.tex", flags))
            last = len(self.textures) - 1
            self.samplers = [
                (
                    sid,
                    fl,
                    last
                    if (sid & 0xFFFFFFFF) == SAMPLER_DIFFUSE
                    else (ti - 1 if ti > slot else ti),
                )
                for sid, fl, ti in self.samplers
            ]
            return True
        self.keys = [k for k in self.keys if (k[0] & 0xFFFFFFFF) != DIFFUSE_KEY]
        return True

    def pbr_preset(self, threshold=1.5, roughness=0.5, sheen=(0.2, 0.2, 3.0), metalness=0.0):
        """The artistic half, as Anno's modal values on rows the classifier selects.

        Only touches rows that are non-empty and currently have no roughness or sheen.
        """
        arr = self.rows
        if arr is None:
            return 0
        n = 0
        for r in range(32):
            row = arr[r]
            if not row.any():
                continue
            if row[ROUGHNESS] > 0 or row[SHEEN_RATE] > 0:
                continue
            if row[SHININESS] < threshold:
                continue
            row[FLAG11] = 1.0
            row[ROUGHNESS] = roughness
            row[SHEEN_RATE], row[SHEEN_TINT], row[SHEEN_APERTURE] = sheen
            if metalness:
                row[METALNESS] = metalness
            n += 1
        if n:
            self.set_rows(arr)
        return n


def read(path):
    with open(path, "rb") as f:
        return Mtrl(f.read())


if __name__ == "__main__":
    for p in sys.argv[1:]:
        m = read(p)
        print(  # noqa: T201 - self-test entry point
            "%-52s %s  ds=%d tex=%d keys=%d cons=%d samp=%d  roundtrip=%s"
            % (
                p[-52:],
                m.shpk,
                m.dataset_size,
                len(m.textures),
                len(m.keys),
                len(m.constants),
                len(m.samplers),
                "OK" if m.build() == pathlib.Path(p).read_bytes() else "DIFFERS",
            )
        )
