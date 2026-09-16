"""``clarity.ffxiv.exd``: the big-endian Excel header (EXH) and data page (EXD) reader.

One sheet is built in memory, following the layout the module parses:

EXH::

     0  "EXHF"
     4  u16 version, u16 data_offset (size of a row's fixed part), u16 column_count,
        u16 page_count, u16 language_count
    20  u32 row_count
    32  column_count x (u16 type, u16 offset)   offset is into the row's fixed part
        page_count x (u32 start_row, u32 row_count)
        language_count x u8

EXD::

     0  "EXDF"
     8  u32 index_size (bytes; 8 per entry)
    32  index: (u32 row_id, u32 absolute file offset of the row)
        rows: u32 data_size, u16 subrow_count, then `data_offset` fixed bytes, then the
        null-terminated strings the string columns point into (offsets relative to that blob)
"""

import struct

import pytest

from clarity.ffxiv import exd

# Column types, in the order they are declared and returned. Type 28 is "bit 3 of the byte at
# the column offset" (packed bools are type 25 + bit).
COLUMNS = [
    (0, 0),  # string: u32 offset into the string blob
    (1, 4),  # bool
    (2, 5),  # int8
    (3, 6),  # uint8
    (4, 8),  # int16
    (5, 10),  # uint16
    (6, 12),  # int32
    (9, 16),  # float32
    (28, 20),  # packed bool, bit 3
    (5, 22),  # uint16, declared out of byte order to show columns are read by offset
]
FIXED = 24  # data_offset: bytes of fixed data per row; strings follow
PAGES = [(0, 2)]
LANGUAGES = [0, 1]  # none, ja

ROWS = {
    0: ("hello", True, -7, 200, -1234, 60000, -100000, 1.5, True, 4242),
    7: ("", False, 127, 0, 32767, 1, 2147483647, -0.25, False, 0),
}


def _exh() -> bytes:
    hdr = bytearray(32)
    hdr[:4] = b"EXHF"
    struct.pack_into(">5H", hdr, 4, 3, FIXED, len(COLUMNS), len(PAGES), len(LANGUAGES))
    struct.pack_into(">I", hdr, 20, len(ROWS))
    body = b"".join(struct.pack(">2H", t, off) for t, off in COLUMNS)
    body += b"".join(struct.pack(">2I", s, n) for s, n in PAGES)
    body += bytes(LANGUAGES)
    return bytes(hdr) + body


def _row(values: tuple) -> bytes:
    s, b, i8, u8, i16, u16, i32, f, packed, u16b = values
    fixed = bytearray(FIXED)
    struct.pack_into(">I", fixed, 0, 0)  # string 0 lives at blob offset 0
    fixed[4] = int(b)
    struct.pack_into(">b", fixed, 5, i8)
    fixed[6] = u8
    struct.pack_into(">h", fixed, 8, i16)
    struct.pack_into(">H", fixed, 10, u16)
    struct.pack_into(">i", fixed, 12, i32)
    struct.pack_into(">f", fixed, 16, f)
    fixed[20] = (0b1000 if packed else 0) | 0b0111  # other bits set: only bit 3 may be read
    struct.pack_into(">H", fixed, 22, u16b)
    blob = s.encode() + b"\0"
    return struct.pack(">IH", FIXED + len(blob), 1) + bytes(fixed) + blob


def _exd() -> bytes:
    hdr = bytearray(32)
    hdr[:4] = b"EXDF"
    index_size = 8 * len(ROWS)
    struct.pack_into(">I", hdr, 8, index_size)
    rows_at = 32 + index_size
    index, rows = b"", b""
    for rid, values in ROWS.items():
        index += struct.pack(">2I", rid, rows_at + len(rows))
        rows += _row(values)
    struct.pack_into(">I", hdr, 12, len(rows))
    return bytes(hdr) + index + rows


def test_exh_exposes_counts_columns_pages_and_languages() -> None:
    exh = exd.Exh(_exh())
    assert exh.version == 3
    assert exh.data_offset == FIXED
    assert exh.column_count == len(COLUMNS)
    assert exh.columns == COLUMNS
    assert exh.row_count == len(ROWS)
    assert exh.page_count == 1
    assert exh.pages == PAGES
    assert exh.language_count == 2
    assert exh.languages == LANGUAGES


def test_exd_indexes_rows_by_id() -> None:
    page = exd.Exd(_exd(), exd.Exh(_exh()))
    assert sorted(page.offsets) == [0, 7]
    assert page.row(3) is None


@pytest.mark.parametrize("rid", sorted(ROWS))
def test_row_decodes_every_column_type_in_column_order(rid: int) -> None:
    page = exd.Exd(_exd(), exd.Exh(_exh()))
    got = page.row(rid)
    assert got is not None
    expected = list(ROWS[rid])
    assert got == expected
    # bools come back as bool, not int, so `is` works downstream
    assert got[1] is expected[1]
    assert got[8] is expected[8]


def test_wrong_magic_raises() -> None:
    with pytest.raises(ValueError, match="not an EXHF"):
        exd.Exh(b"EXDF" + _exh()[4:])
    with pytest.raises(ValueError, match="not an EXDF"):
        exd.Exd(b"EXHF" + _exd()[4:], exd.Exh(_exh()))


def test_unknown_column_type_yields_none() -> None:
    exh = exd.Exh(_exh())
    exh.columns = [(12, 0)]  # type 12 has no reader
    assert exd.Exd(_exd(), exh).row(0) == [None]


class _FakeGameData:
    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.reads: list[str] = []

    def exists(self, path: str) -> bool:
        return path in self.files

    def read(self, path: str) -> bytes:
        self.reads.append(path)
        return self.files[path]


def test_load_sheet_picks_the_language_suffixed_page() -> None:
    gd = _FakeGameData({"exd/Item.exh": _exh(), "exd/Item_0_en.exd": _exd()})
    exh, pages = exd.load_sheet(gd, "Item", "en")
    assert exh.column_count == len(COLUMNS)
    assert len(pages) == 1
    assert pages[0].row(7)[0] == ""
    assert gd.reads == ["exd/Item.exh", "exd/Item_0_en.exd"]
