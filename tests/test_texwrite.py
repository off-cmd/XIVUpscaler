"""``clarity.ffxiv.texwrite``: the 80-byte ``.tex`` header, the BGRA mip chain behind it, and the
block-compressed writer's per-level size check.

Every offset asserted here is read back through ``tex.TexHeader`` rather than hard-coded, so the
writer and the reader are held to the same layout.
"""

import struct

import numpy as np
import pytest

from clarity.ffxiv import tex, texwrite


def _red(h: int, w: int) -> np.ndarray:
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 0] = 255  # R
    img[..., 3] = 255  # A
    return img


def test_header_fields_for_a_single_mip() -> None:
    data = texwrite.write(_red(6, 10))
    assert len(data) == texwrite.HDR + 6 * 10 * 4
    hdr = tex.TexHeader(data)
    assert hdr.attributes == texwrite.ATTR_2D
    assert hdr.texture_type == "2D"
    assert hdr.format == texwrite.B8G8R8A8
    assert hdr.format_name == "B8G8R8A8"
    assert (hdr.width, hdr.height, hdr.depth) == (10, 6, 1)
    assert hdr.mip_count == 1
    assert hdr.array_size == 1
    assert hdr.lod_offsets == (texwrite.HDR,) * 3
    assert hdr.surface_offsets[0] == texwrite.HDR
    assert not any(hdr.surface_offsets[1:])  # unused slots are zero


def test_pixels_are_stored_bgra() -> None:
    data = texwrite.write(_red(2, 2))
    first = data[texwrite.HDR : texwrite.HDR + 4]
    assert first[0] == 0  # B
    assert first[1] == 0  # G
    assert first[2] == 255  # R
    assert first[3] == 255  # A


def test_mip_chain_offsets_follow_the_header_and_cumulative_sizes() -> None:
    rng = np.random.default_rng(0)
    src = rng.integers(0, 256, (12, 20, 4), dtype=np.uint8)
    data = texwrite.write(src, mips=8)  # 20x12 -> 10x6 -> 5x3 -> 2x1 -> 1x1
    hdr = tex.TexHeader(data)
    assert hdr.mip_count == 5
    expected = [(20, 12), (10, 6), (5, 3), (2, 1), (1, 1)]
    at = texwrite.HDR
    for k, dims in enumerate(expected):
        assert hdr.mip_dimensions(k)[:2] == dims
        assert hdr.surface_offsets[k] == at
        assert hdr.surface_size(k) == dims[0] * dims[1] * 4
        at += hdr.surface_size(k)
    assert len(data) == at
    assert hdr.total_size() == at - texwrite.HDR
    assert list(hdr.surface_offsets[:5]) == sorted(hdr.surface_offsets[:5])
    assert not any(hdr.surface_offsets[5:])
    # Mip 0 is the source, byte for byte (in BGRA).
    np.testing.assert_array_equal(
        np.frombuffer(data[texwrite.HDR : texwrite.HDR + 20 * 12 * 4], np.uint8).reshape(12, 20, 4),
        src[..., [2, 1, 0, 3]],
    )


def test_mip_levels_are_box_filtered() -> None:
    src = np.zeros((2, 2, 4), np.uint8)
    src[0, 0] = [100, 0, 0, 255]
    src[0, 1] = [0, 100, 0, 255]
    src[1, 0] = [0, 0, 100, 255]
    src[1, 1] = [100, 100, 100, 255]
    data = texwrite.write(src, mips=2)
    hdr = tex.TexHeader(data)
    off = hdr.surface_offsets[1]
    assert data[off : off + 4] == bytes([50, 50, 50, 255])  # BGRA of the 2x2 mean


def test_write_accepts_grey_rgb_and_float_input() -> None:
    grey = np.full((2, 2), 30, np.uint8)
    hdr_grey = tex.TexHeader(texwrite.write(grey))
    assert (hdr_grey.width, hdr_grey.height) == (2, 2)
    data = texwrite.write(grey)
    assert data[texwrite.HDR : texwrite.HDR + 4] == bytes([30, 30, 30, 255])

    rgb = np.zeros((1, 1, 3), np.uint8)
    rgb[0, 0] = [1, 2, 3]
    assert texwrite.write(rgb)[texwrite.HDR :] == bytes([3, 2, 1, 255])

    flt = np.array([[[300.0, -5.0, 12.4, 255.0]]])  # clipped, then truncated to uint8
    assert texwrite.write(flt)[texwrite.HDR :] == bytes([12, 0, 255, 255])


def test_mip_chain_halves_and_clamps_at_one() -> None:
    src = np.zeros((8, 2, 4), np.uint8)
    levels = texwrite.mip_chain(src, mips=10)
    # Each edge clamps at 1 on its own; the chain runs until both are 1 (D3D, Lumina,
    # TexHeader.mip_dimensions). It used to stop at (4, 1).
    assert [lvl.shape[:2] for lvl in levels] == [(8, 2), (4, 1), (2, 1), (1, 1)]
    square = texwrite.mip_chain(np.zeros((4, 4, 4), np.uint8), mips=10)
    assert [lvl.shape[:2] for lvl in square] == [(4, 4), (2, 2), (1, 1)]
    assert len(texwrite.mip_chain(np.zeros((4, 4, 4), np.uint8), mips=0)) == 1  # at least mip 0
    assert len(texwrite.mip_chain(np.zeros((4, 4, 4), np.uint8), mips=2)) == 2


def test_write_like_copies_the_attribute_word() -> None:
    reference = bytearray(texwrite.HDR)
    struct.pack_into("<I", reference, 0, 0x80C00000)
    data = texwrite.write_like(_red(2, 2), bytes(reference), mips=2)
    hdr = tex.TexHeader(data)
    assert hdr.attributes == 0x80C00000
    assert hdr.mip_count == 2
    assert hdr.format_name == "B8G8R8A8"


def test_write_blocks_lays_out_each_mip_at_its_own_block_count() -> None:
    # 10x6 BC7: mip 0 is 3x2 blocks, mip 1 (5x3) is 2x1, mip 2 (2x1) is 1x1.
    sizes = [3 * 2 * 16, 2 * 1 * 16, 1 * 1 * 16]
    levels = [bytes([k + 1]) * n for k, n in enumerate(sizes)]
    data = texwrite.write_blocks(texwrite.BC7, levels, 10, 6)
    hdr = tex.TexHeader(data)
    assert hdr.format_name == "BC7"
    assert hdr.is_block_compressed
    assert (hdr.width, hdr.height, hdr.mip_count) == (10, 6, 3)
    at = texwrite.HDR
    for k, n in enumerate(sizes):
        assert hdr.surface_offsets[k] == at
        assert hdr.surface_size(k) == n
        assert data[at : at + n] == levels[k]
        at += n
    assert len(data) == at
    assert not any(hdr.surface_offsets[3:])


def test_write_blocks_bc5_and_bc1_block_sizes() -> None:
    data = texwrite.write_blocks(texwrite.BC5, [bytes(16 * 4)], 8, 8)
    assert tex.TexHeader(data).surface_size(0) == 64
    data = texwrite.write_blocks(0x3420, [bytes(8 * 4)], 8, 8)  # BC1: 8 bytes per block
    assert tex.TexHeader(data).format_name == "BC1"
    assert tex.TexHeader(data).surface_size(0) == 32


def test_write_blocks_rejects_a_wrong_sized_level() -> None:
    # 8x8 -> 2x2 blocks (64 bytes); mip 1 is 4x4 -> one 16-byte block, not two.
    with pytest.raises(AssertionError, match="mip 1: 32 bytes, expected 16 for 4x4"):
        texwrite.write_blocks(texwrite.BC7, [bytes(4 * 16), bytes(32)], 8, 8)
