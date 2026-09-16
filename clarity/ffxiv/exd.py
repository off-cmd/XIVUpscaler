"""Minimal FFXIV Excel (EXH/EXD) reader. Big-endian. Spec: Lumina Data/Structs/Excel."""

import struct


class Exh:
    def __init__(self, data):
        if data[:4] != b"EXHF":
            raise ValueError("not an EXHF file")
        (
            self.version,
            self.data_offset,
            self.column_count,
            self.page_count,
            self.language_count,
        ) = struct.unpack_from(">5H", data, 4)
        (self.row_count,) = struct.unpack_from(">I", data, 20)
        o = 32
        self.columns = []
        for _ in range(self.column_count):
            t, off = struct.unpack_from(">2H", data, o)
            o += 4
            self.columns.append((t, off))
        self.pages = []
        for _ in range(self.page_count):
            s, n = struct.unpack_from(">2I", data, o)
            o += 8
            self.pages.append((s, n))
        self.languages = [data[o + i] for i in range(self.language_count)]


class Exd:
    def __init__(self, data, exh):
        if data[:4] != b"EXDF":
            raise ValueError("not an EXDF file")
        self.data = data
        self.exh = exh
        (index_size,) = struct.unpack_from(">I", data, 8)
        self.offsets = {}
        for i in range(index_size // 8):
            rid, off = struct.unpack_from(">2I", data, 32 + i * 8)
            self.offsets[rid] = off

    def row(self, rid):
        off = self.offsets.get(rid)
        if off is None:
            return None
        _size, _count = struct.unpack_from(">IH", self.data, off)
        base = off + 6
        strings = base + self.exh.data_offset
        out = []
        for t, coff in self.exh.columns:
            p = base + coff
            if t == 0:  # string
                (so,) = struct.unpack_from(">I", self.data, p)
                s = strings + so
                e = self.data.index(b"\0", s)
                out.append(self.data[s:e].decode("utf8", "replace"))
            elif t == 1:
                out.append(bool(self.data[p]))
            elif t == 2:
                out.append(struct.unpack_from(">b", self.data, p)[0])
            elif t == 3:
                out.append(self.data[p])
            elif t == 4:
                out.append(struct.unpack_from(">h", self.data, p)[0])
            elif t == 5:
                out.append(struct.unpack_from(">H", self.data, p)[0])
            elif t in (6, 8):
                out.append(struct.unpack_from(">i", self.data, p)[0])
            elif t == 7:
                out.append(struct.unpack_from(">I", self.data, p)[0])
            elif t == 9:
                out.append(struct.unpack_from(">f", self.data, p)[0])
            elif t == 10:
                out.append(struct.unpack_from(">q", self.data, p)[0])
            elif t == 11:
                out.append(struct.unpack_from(">Q", self.data, p)[0])
            elif t >= 25:
                out.append(bool(self.data[p] & (1 << (t - 25))))
            else:
                out.append(None)
        return out


def load_sheet(gd, name, language="en"):
    exh = Exh(gd.read(f"exd/{name}.exh"))
    pages = []
    for start, _n in exh.pages:
        for suffix in [f"_{language}"] if exh.language_count > 1 else [""]:
            p = f"exd/{name}_{start}{suffix}.exd"
            if gd.exists(p):
                pages.append(Exd(gd.read(p), exh))
                break
    return exh, pages
