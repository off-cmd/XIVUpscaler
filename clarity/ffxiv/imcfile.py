"""FFXIV .imc (Image Change) parser. Spec: file-formats/010editor/imc.bt

Maps an item VARIANT to the material folder / decal / vfx actually used.
This is the link between Item.ModelMain's variant number and which
material/v#### directory to load - they are NOT the same number.
"""

import struct

SLOT_PART = {
    "met": 0,
    "top": 1,
    "glv": 2,
    "dwn": 3,
    "sho": 4,
    "ear": 0,
    "nek": 1,
    "wrs": 2,
    "rir": 3,
    "ril": 4,
}


class ImcEntry:
    __slots__ = (
        "attribute_and_sound",
        "decal_id",
        "material_animation_id",
        "material_id",
        "vfx_id",
    )

    def __init__(self, m, d, a, v, ma):
        self.material_id = m
        self.decal_id = d
        self.attribute_and_sound = a
        self.vfx_id = v
        self.material_animation_id = ma

    @property
    def attribute_mask(self):
        return self.attribute_and_sound & 0x03FF

    @property
    def sound_id(self):
        return self.attribute_and_sound >> 10

    def __repr__(self):
        return (
            f"Imc(mat={self.material_id} decal={self.decal_id} "
            f"vfx={self.vfx_id} attr=0x{self.attribute_mask:03x})"
        )


class ImcFile:
    def __init__(self, data: bytes):
        count, part_mask = struct.unpack_from("<2H", data, 0)
        self.part_mask = part_mask
        self.count = count
        # Only the parts whose bit is set in part_mask are stored, in bit order. Equipment
        # ships 0x1F (all five), but the mask is the format's word on the matter and a
        # reader that ignores it reads the wrong slot from any file that does not.
        self.present = [i for i in range(16) if part_mask >> i & 1]
        self.part_count = len(self.present)
        self.variants = []  # [variant][stored index] -> ImcEntry
        o = 4
        for _ in range(count + 1):  # entry 0 is the default
            parts = []
            for _ in range(self.part_count):
                m, d, a, v, ma = struct.unpack_from("<2BH2B", data, o)
                o += 6
                parts.append(ImcEntry(m, d, a, v, ma))
            self.variants.append(parts)

    def entry(self, variant, slot):
        """The entry for ``variant`` (1-based, as stored in Item.ModelMain) and equipment ``slot``.

        Returns ``None`` when the variant is out of range or the slot's part is not present in
        this file. An unknown slot name is a programming error and raises ``KeyError``.
        """
        if slot not in SLOT_PART:
            raise KeyError(f"unknown equipment slot {slot!r}; one of {sorted(SLOT_PART)}")
        part = 0 if self.part_count == 1 else SLOT_PART[slot]
        if not 0 <= variant < len(self.variants) or part not in self.present:
            return None
        return self.variants[variant][self.present.index(part)]


def load(gd, eid, kind="equipment"):
    p = f"chara/{kind}/e{eid:04d}/e{eid:04d}.imc"
    if not gd.exists(p):
        return None
    return ImcFile(gd.read(p))
