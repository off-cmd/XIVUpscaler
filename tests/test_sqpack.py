"""sqpack path hashing, path parsing, and the fast texture-header read.

Nothing here needs a game install: the hash functions are pure, and the header read is
exercised against a synthetic dat entry written to a temporary file.
"""

import struct
import zlib

import pytest

from clarity.ffxiv import sqpack


def test_crc32_is_the_bitwise_not_of_zlib() -> None:
    for b in (b"", b"a", b"chara/equipment/e0001/texture/v01_c0101e0001_top_n.tex"):
        assert sqpack.crc32(b) == (~zlib.crc32(b)) & 0xFFFFFFFF


def test_split_hash_lowercases_and_splits_at_the_last_slash() -> None:
    d, f = sqpack.split_hash("/Chara/Equipment/e0001/Texture/v01_top_n.tex/")
    assert d == sqpack.crc32(b"chara/equipment/e0001/texture")
    assert f == sqpack.crc32(b"v01_top_n.tex")


def test_full_hash_is_over_the_whole_lowercased_path() -> None:
    assert sqpack.full_hash("UI/Icon/000000/000001.tex") == sqpack.crc32(
        b"ui/icon/000000/000001.tex"
    )


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("common/graphics/texture/-nowloading.tex", (0x00, 0, 0)),
        ("chara/equipment/e0001/texture/x.tex", (0x04, 0, 0)),
        ("bg/ex3/01_nvt_n/twn/n4t1/texture/x.tex", (0x02, 3, 0)),
        ("ui/icon/000000/000001.tex", (0x06, 0, 0)),
        ("not-a-category/anything", None),
    ],
)
def test_parse_path(path, expected) -> None:
    assert sqpack.parse_path(path) == expected


def _texture_entry(tex_header: bytes) -> bytes:
    """A minimal FT_TEXTURE dat entry: entry header, one LOD block record whose first
    field says where the compressed data starts, then the uncompressed .tex header."""
    hsize = 0x18 + 20  # header fields + one <5I LodBlock
    head = bytearray(hsize)
    struct.pack_into("<3I", head, 0, hsize, sqpack.FT_TEXTURE, 0)
    struct.pack_into("<I", head, 0x14, 1)  # nblocks
    struct.pack_into("<5I", head, 0x18, len(tex_header), 0, 0, 0, 0)
    return bytes(head) + tex_header + b"\x00" * 64


def test_read_tex_header_returns_the_stored_header(tmp_path) -> None:
    hdr = bytes(range(80))
    dat = tmp_path / "040000.win32.dat0"
    dat.write_bytes(b"\xff" * 100 + _texture_entry(hdr))
    try:
        assert sqpack.read_tex_header(str(dat), 100) == hdr
        assert sqpack.read_tex_header(str(dat), 100, nbytes=16) == hdr[:16]
    finally:
        sqpack.close_dats()


def test_read_tex_header_refuses_non_texture_entries(tmp_path) -> None:
    dat = tmp_path / "040000.win32.dat0"
    head = bytearray(0x18)
    struct.pack_into("<3I", head, 0, 0x18, sqpack.FT_STANDARD, 0)
    dat.write_bytes(bytes(head))
    try:
        assert sqpack.read_tex_header(str(dat), 0) is None
    finally:
        sqpack.close_dats()


def test_read_tex_header_uses_the_shared_handle_cache(tmp_path) -> None:
    dat = tmp_path / "040000.win32.dat0"
    dat.write_bytes(_texture_entry(bytes(80)))
    try:
        sqpack.read_tex_header(str(dat), 0)
        assert str(dat) in sqpack._DAT_HANDLES
    finally:
        sqpack.close_dats()
    assert not sqpack._DAT_HANDLES
