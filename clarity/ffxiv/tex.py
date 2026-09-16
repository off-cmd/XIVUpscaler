"""FFXIV .tex header parser. Spec: file-formats/imhex/tex.hexpat (verified 7.55+)."""

import struct

FORMATS = {
    0x1130: "L8",
    0x1131: "A8",
    0x1440: "B4G4R4A4",
    0x1441: "B5G5R5A1",
    0x1450: "B8G8R8A8",
    0x1451: "B8G8R8X8",
    0x2150: "R32F",
    0x2250: "R16G16F",
    0x2260: "R32G32F",
    0x2460: "R16G16B16A16F",
    0x2470: "R32G32B32A32F",
    0x3420: "BC1",
    0x3430: "BC2",
    0x3431: "BC3",
    0x6120: "BC4",
    0x6230: "BC5",
    0x6330: "BC6H",
    0x6432: "BC7",
    0x4140: "D16",
    0x4250: "D24S8",
    0x5100: "Null",
    0x5140: "Shadow16",
    0x5150: "Shadow24",
}
ATTR = [
    "discardPerFrame",
    "discardPerMap",
    "managed",
    "userManaged",
    "cpuRead",
    "locationMain",
    "noGpuRead",
    "alignedSize",
    "edgeCulling",
    "locationOnion",
    "readWrite",
    "immutable",
]


class TexHeader:
    __slots__ = (
        "array_size",
        "attributes",
        "bpp",
        "depth",
        "format",
        "format_name",
        "height",
        "kind",
        "lod_offsets",
        "mip_count",
        "surface_offsets",
        "width",
    )

    def __init__(self, data: bytes):
        (attr, fmt, w, h, d, mipflag, arr) = struct.unpack_from("<IIHHHBB", data, 0)
        self.attributes = attr
        self.format = fmt
        self.format_name = FORMATS.get(fmt, f"0x{fmt:04x}")
        self.width, self.height, self.depth = w, h, d
        self.mip_count = mipflag & 0x7F
        self.array_size = arr
        self.lod_offsets = struct.unpack_from("<3I", data, 16)
        self.surface_offsets = struct.unpack_from("<13I", data, 28)
        self.kind = (fmt & 0xF000) >> 12  # 1 int, 2 float, 3/6 block-compressed
        self.bpp = 1 << ((fmt & 0xF0) >> 4)  # bits per pixel (or per block-texel)

    @property
    def is_block_compressed(self):
        return self.kind in (3, 6)

    @property
    def texture_type(self):
        t = self.attributes
        if t & (1 << 28):
            return "2DArray"
        if t & (1 << 25):
            return "Cube"
        if t & (1 << 24):
            return "3D"
        if t & (1 << 23):
            return "2D"
        if t & (1 << 22):
            return "1D"
        return f"?(0x{t:08x})"

    def mip_dimensions(self, mip=0):
        """(width, height, depth) of one mip level.

        Direct3D and Lumina both halve with floor and clamp at 1, so a 12-wide texture is
        6, 3, 1, 1 down the chain -- never 2 at the third level. This is the single
        definition; `surface_size` and the decoder both use it, so the byte count and the
        pixel count of a surface can no longer disagree for non-power-of-two sizes.
        """
        return (
            max(1, self.width >> mip),
            max(1, self.height >> mip),
            max(1, self.depth >> mip),
        )

    def surface_size(self, mip=0):
        w, h, d = self.mip_dimensions(mip)
        if self.is_block_compressed:
            w = (w + 3) & ~3
            h = (h + 3) & ~3
        return ((w * h * self.bpp + 7) >> 3) * d

    def total_size(self):
        return sum(self.surface_size(m) for m in range(self.mip_count))

    def __repr__(self):
        return (
            f"{self.width}x{self.height} {self.format_name} "
            f"mips={self.mip_count} type={self.texture_type} "
            f"vram={self.total_size() / 1024:.1f}KiB"
        )
