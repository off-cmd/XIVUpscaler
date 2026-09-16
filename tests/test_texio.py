"""``texio``: the output-format policy, the float mip chain, the DDS wrapper, and the
encoders on the numpy path (this box has no texconv, so ``use_texconv()`` is False).

Pins: ``out_format`` keeps uncompressed sources uncompressed, keeps BC2/BC3 as BC3 only for
the UI roles, and sends everything else to BC7; ``float_chain`` is a box filter computed in
float from the top level, rounded once per level, clamped at 1x1; ``_dds_rgba`` writes the
header texconv expects; ``encode`` / ``encode_tiers`` produce ``.tex`` files the package's
own decoder reads back (exactly for B8G8R8A8, within a small error for BC7 and BC3); and the
tier cut halves the dimensions once per offset.
"""

import os
import struct

import numpy as np
import pytest

from clarity import texio
from clarity.ffxiv import texdecode, texwrite


def _ramp(h: int = 16, w: int = 16, alpha: bool = False) -> np.ndarray:
    """A grey ramp: every block lies on one line in RGBA space, which BC7 mode 6 encodes well."""
    yy, xx = np.mgrid[0:h, 0:w]
    ramp = ((xx * 8 + yy * 4) & 255).astype(np.uint8)
    a = (255 - ramp // 2).astype(np.uint8) if alpha else np.full_like(ramp, 255)
    return np.dstack([ramp, ramp, ramp, a])


def _noise(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 4), dtype=np.uint8)


def _err(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return float(d.mean()), int(d.max())


# ---------------------------------------------------------------------------- environment
@pytest.mark.skipif(os.name == "nt", reason="texconv is a Windows-only encoder")
def test_use_texconv_is_false_off_windows() -> None:
    assert texio.use_texconv() is False


@pytest.mark.skipif(os.name == "nt", reason="texconv is a Windows-only encoder")
def test_probe_reports_the_numpy_encoder_off_windows() -> None:
    p = texio.probe_texconv()
    assert p["ok"] is False and p["gpu"] is False
    assert "numpy" in p["text"]
    assert texio.probe_texconv() is p  # cached


def test_is_debug_build(tmp_path) -> None:
    assert texio.is_debug_build(str(tmp_path / "missing.exe")) is False
    release = tmp_path / "release.exe"
    release.write_bytes(b"MZ" + b"\0" * 64 + b"ucrtbase.dll" + b"\0" * 16)
    assert texio.is_debug_build(str(release)) is False
    debug = tmp_path / "debug.exe"
    debug.write_bytes(b"MZ" + b"\0" * 64 + b"ucrtbased.dll" + b"\0" * 16)
    assert texio.is_debug_build(str(debug)) is True


def test_format_codes_match_texwrite() -> None:
    assert texio.BC7 == texwrite.BC7 == 0x6432
    assert texio.BC3 == texwrite.BC3 == 0x3431
    assert texio.BGRA8 == texwrite.B8G8R8A8 == 0x1450
    assert texio.BC1 == 0x3420 and texio.BC1 in texwrite.BLOCK_BYTES


@pytest.mark.parametrize(
    ("w", "h", "n"),
    [(1, 1, 1), (2, 2, 2), (8, 6, 4), (16, 16, 5), (1000, 1000, 10), (1024, 512, 11), (1, 64, 7)],
)
def test_full_mips(w, h, n) -> None:
    assert texio.full_mips(w, h) == n


# ------------------------------------------------------------------------------ out_format
UNCOMPRESSED = ["B8G8R8A8", "B8G8R8X8", "B5G5R5A1", "B4G4R4A4", "L8", "A8"]
ROLES = ["normal", "mask", "color", "icon", "ui"]


@pytest.mark.parametrize("fmt", UNCOMPRESSED)
@pytest.mark.parametrize("role", ROLES)
def test_out_format_uncompressed_stays_uncompressed(fmt, role) -> None:
    assert texio.out_format(fmt, role) == texio.BGRA8


@pytest.mark.parametrize("fmt", ["BC2", "BC3"])
def test_out_format_bc3_is_kept_only_for_the_ui_roles(fmt) -> None:
    assert texio.out_format(fmt, "icon") == texio.BC3
    assert texio.out_format(fmt, "ui") == texio.BC3
    for role in ("normal", "mask", "color"):
        assert texio.out_format(fmt, role) == texio.BC7, role


@pytest.mark.parametrize("fmt", ["BC1", "BC4", "BC5", "BC6H", "BC7", "R16G16F", "", "0xbeef"])
@pytest.mark.parametrize("role", ROLES)
def test_out_format_everything_else_is_bc7(fmt, role) -> None:
    assert texio.out_format(fmt, role) == texio.BC7


# ----------------------------------------------------------------------------- float_chain
def test_float_chain_is_a_box_filter_from_the_top_level() -> None:
    src = _noise(8, 8, seed=1)
    levels = texio.float_chain(src, 4)
    assert [lv.shape for lv in levels] == [(8, 8, 4), (4, 4, 4), (2, 2, 4), (1, 1, 4)]
    assert levels[0] is not None and np.array_equal(levels[0], src)
    # Level 1 is the 2x2 mean of the top level, rounded once.
    f = src.astype(np.float32) / 255.0
    l1 = f.reshape(4, 2, 4, 2, 4).mean((1, 3))
    np.testing.assert_array_equal(levels[1], np.clip(np.rint(l1 * 255.0), 0, 255).astype(np.uint8))
    # Level 2 is computed from the *float* level 1, not the rounded one.
    l2 = l1.reshape(2, 2, 2, 2, 4).mean((1, 3))
    np.testing.assert_array_equal(levels[2], np.clip(np.rint(l2 * 255.0), 0, 255).astype(np.uint8))
    # Level 3 is the mean of the float level 2 (the same arithmetic, one more step).
    l3 = l2.reshape(1, 2, 1, 2, 4).mean((1, 3))
    np.testing.assert_array_equal(levels[3], np.clip(np.rint(l3 * 255.0), 0, 255).astype(np.uint8))


def test_float_chain_constant_blocks_are_exact() -> None:
    src = np.zeros((4, 4, 4), np.uint8)
    src[:2, :2] = (10, 20, 30, 40)
    src[:2, 2:] = (50, 60, 70, 80)
    src[2:, :2] = (90, 100, 110, 120)
    src[2:, 2:] = (130, 140, 150, 160)
    levels = texio.float_chain(src, 3)
    np.testing.assert_array_equal(
        levels[1].reshape(4, 4),
        [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120], [130, 140, 150, 160]],
    )
    np.testing.assert_array_equal(levels[2].reshape(4), [70, 80, 90, 100])


def test_float_chain_clamps_at_one_pixel() -> None:
    levels = texio.float_chain(_noise(8, 2), 20)
    assert [lv.shape[:2] for lv in levels] == [(8, 2), (4, 1), (2, 1), (1, 1)]
    assert len(texio.float_chain(_noise(1, 1), 5)) == 1
    assert len(texio.float_chain(_noise(8, 8), 1)) == 1
    assert len(texio.float_chain(_noise(8, 8), 0)) == 1


def test_float_chain_one_dimensional_edges() -> None:
    row = np.zeros((1, 4, 4), np.uint8)
    row[0, :, 0] = [0, 100, 200, 50]
    levels = texio.float_chain(row, 3)
    assert [lv.shape[:2] for lv in levels] == [(1, 4), (1, 2), (1, 1)]
    assert list(levels[1][0, :, 0]) == [50, 125]
    assert levels[2][0, 0, 0] in (87, 88)  # 87.5 rounds to even
    col = np.zeros((4, 1, 4), np.uint8)
    col[:, 0, 1] = [0, 100, 200, 50]
    levels = texio.float_chain(col, 3)
    assert [lv.shape[:2] for lv in levels] == [(4, 1), (2, 1), (1, 1)]
    assert list(levels[1][:, 0, 1]) == [50, 125]


def test_float_chain_returns_uint8_and_a_copy_of_nothing_but_the_top() -> None:
    src = _noise(4, 4)
    levels = texio.float_chain(src, 3)
    assert all(lv.dtype == np.uint8 for lv in levels)
    assert np.shares_memory(levels[0], src)


# ------------------------------------------------------------------------------- _dds_rgba
def test_dds_rgba_header_fields() -> None:
    rgba = _noise(6, 10, seed=3)
    dds = texio._dds_rgba(rgba)
    assert dds[:4] == b"DDS "
    assert len(dds) == 4 + 124 + 6 * 10 * 4
    size, flags, h, w, pitch, depth, mips = struct.unpack_from("<7I", dds, 4)
    assert (size, h, w, pitch, depth, mips) == (124, 6, 10, 40, 0, 1)
    assert flags == 0x1 | 0x2 | 0x4 | 0x1000 | 0x8  # CAPS|HEIGHT|WIDTH|PIXELFORMAT|PITCH
    pf_size, pf_flags, fourcc, bits, rm, gm, bm, am = struct.unpack_from("<8I", dds, 4 + 72)
    assert (pf_size, pf_flags, fourcc, bits) == (32, 0x41, 0, 32)
    assert (rm, gm, bm, am) == (0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000)
    assert dds[84:88] != b"DX10"  # what _dds_split keys on
    assert struct.unpack_from("<I", dds, 4 + 104)[0] == 0x1000  # caps: TEXTURE
    # The payload is the RGBA bytes in memory order, untouched.
    assert dds[128:] == rgba.tobytes()


def test_dds_rgba_accepts_a_non_contiguous_view() -> None:
    rgba = _noise(8, 8)[::2, ::2]
    dds = texio._dds_rgba(rgba)
    assert dds[128:] == np.ascontiguousarray(rgba).tobytes()


def test_dds_split_walks_a_plain_dds_mip_chain() -> None:
    # A hand-built 8-byte-block DDS with three mips of a 8x4 image: 2, 1, 1 blocks.
    hdr = b"DDS " + b"\0" * 124
    payload = b"".join(bytes([i]) * 8 for i in (1, 2, 3, 4))
    mips = texio._dds_split(hdr + payload, 8, 8, 4, 3)
    assert mips == [b"\1" * 8 + b"\2" * 8, b"\3" * 8, b"\4" * 8]
    # A DX10 header adds 20 bytes before the payload.
    dx10 = b"DDS " + b"\0" * 80 + b"DX10" + b"\0" * 40 + b"\0" * 20
    assert texio._dds_split(dx10 + payload, 8, 8, 4, 1) == [b"\1" * 8 + b"\2" * 8]


# ---------------------------------------------------------------------------------- encode
def test_encode_bgra8_round_trips_exactly() -> None:
    src = _noise(6, 10, seed=4)
    data = texio.encode(src, texio.BGRA8)
    hdr, back = texio.read(data)
    assert (hdr.width, hdr.height, hdr.format_name, hdr.texture_type) == (10, 6, "B8G8R8A8", "2D")
    # One mip rule everywhere: 10x6, 5x3, 2x1, 1x1 -- the same chain encode_tiers writes.
    # (texwrite.write used to stop as soon as either edge reached 1 and said 3 here.)
    assert texio.full_mips(10, 6) == 4
    assert hdr.mip_count == 4
    np.testing.assert_array_equal(back, src)
    # The lower mips are box-filtered and readable too.
    assert texdecode.decode(data, 1).shape == (3, 5, 4)
    assert texdecode.decode(data, 2).shape == (1, 2, 4)
    assert texdecode.decode(data, 3).shape == (1, 1, 4)
    with pytest.raises(ValueError):
        texdecode.decode(data, 4)


def test_encode_bgra8_with_explicit_mip_count() -> None:
    data = texio.encode(_noise(8, 8), texio.BGRA8, mips=1)
    hdr, _ = texio.read(data)
    assert hdr.mip_count == 1
    assert len(data) == texwrite.HDR + 8 * 8 * 4


def test_encode_bgra8_keeps_the_attribute_bits() -> None:
    attr = texwrite.ATTR_2D | 0x1
    hdr, _ = texio.read(texio.encode(_noise(4, 4), texio.BGRA8, attributes=attr))
    assert hdr.attributes == attr and hdr.texture_type == "2D"


def test_encode_bgra8_matches_texwrite() -> None:
    src = _noise(8, 8, seed=5)
    assert texio.encode(src, texio.BGRA8, mips=2) == texwrite.write(src, mips=2)


@pytest.mark.skipif(texio.use_texconv(), reason="pins the numpy bc7enc path")
def test_encode_bc7_round_trips_within_tolerance() -> None:
    src = _ramp(16, 16, alpha=True)
    data = texio.encode(src, texio.BC7)
    hdr, back = texio.read(data)
    assert (hdr.width, hdr.height, hdr.format_name, hdr.mip_count) == (16, 16, "BC7", 5)
    assert hdr.is_block_compressed
    assert len(data) == texwrite.HDR + hdr.total_size()
    mean, peak = _err(back, src)
    assert mean < 1.0 and peak <= 2, (mean, peak)
    # Every mip is a valid, correctly sized surface.
    for mip, side in enumerate((16, 8, 4, 2, 1)):
        assert texdecode.decode(data, mip).shape == (side, side, 4)
    mean, peak = _err(texdecode.decode(data, 1), texio.float_chain(src, 2)[1])
    assert mean < 1.5 and peak <= 4, (mean, peak)


@pytest.mark.skipif(texio.use_texconv(), reason="pins the Pillow DXT5 path")
def test_encode_bc3_round_trips_within_tolerance() -> None:
    src = _ramp(h=8, w=16, alpha=True)
    data = texio.encode(src, texio.BC3)
    hdr, back = texio.read(data)
    # 16x8, 8x4, 4x2, 2x1, 1x1: the chain runs to 1x1 like every other writer here.
    assert (hdr.width, hdr.height, hdr.format_name, hdr.mip_count) == (16, 8, "BC3", 5)
    mean, peak = _err(back, src)
    assert mean < 4.0 and peak <= 12, (mean, peak)
    assert texdecode.decode(data, 3).shape == (1, 2, 4)


def test_encode_bc7_is_deterministic() -> None:
    src = _noise(8, 8, seed=6)
    assert texio.encode(src, texio.BC7, mips=1) == texio.encode(src, texio.BC7, mips=1)


def test_encode_non_multiple_of_four_pads_the_last_block() -> None:
    src = _ramp(6, 10)
    data = texio.encode(src, texio.BC7, mips=1)
    hdr, back = texio.read(data)
    assert (hdr.width, hdr.height) == (10, 6)
    assert back.shape == (6, 10, 4)
    assert len(data) == texwrite.HDR + 3 * 2 * 16  # ceil(10/4) x ceil(6/4) blocks


def test_encode_rejects_an_unknown_format() -> None:
    with pytest.raises(ValueError, match="no encoder"):
        texio.encode(_noise(4, 4), 0x9999)


def test_encode_accepts_any_uint8_convertible_input() -> None:
    src = _noise(4, 4).astype(np.int64)
    np.testing.assert_array_equal(texio.read(texio.encode(src, texio.BGRA8))[1], src)


# ---------------------------------------------------------------------------- encode_tiers
@pytest.mark.parametrize("fmt", [texio.BGRA8, texio.BC7])
def test_encode_tiers_halves_once_per_offset(fmt) -> None:
    src = _ramp(8, 16)
    out = texio.encode_tiers(src, fmt, offsets=(0, 1, 2))
    assert sorted(out) == [0, 1, 2]
    chain = texio.float_chain(src, texio.full_mips(16, 8))
    for k, (w, h, mips) in {0: (16, 8, 5), 1: (8, 4, 4), 2: (4, 2, 3)}.items():
        hdr, top = texio.read(out[k])
        assert (hdr.width, hdr.height, hdr.mip_count) == (w, h, mips), k
        if fmt == texio.BGRA8:
            np.testing.assert_array_equal(top, chain[k])
        else:
            # What encode_tiers claims is that tier k's top mip is float_chain[k] -- the same
            # pixels as before, encoded once instead of once per tier. The test of that claim
            # is that the tier is no worse than encoding that mip on its own, NOT a fixed
            # tolerance: the fixed one was calibrated against the numpy encoder and failed on
            # Windows, where texconv does the work and loses more on a 4x2 surface. This form
            # still catches a real disagreement between texconv's mip chain and float_chain,
            # because that would push the tier's error above the direct encode's.
            mean, peak = _err(top, chain[k])
            _hdr_direct, direct = texio.read(texio.encode(chain[k], fmt, mips=1))
            ref_mean, ref_peak = _err(direct, chain[k])
            assert mean <= ref_mean + 0.5, (k, mean, ref_mean)
            assert peak <= ref_peak + 2, (k, peak, ref_peak)
        # The tier's last mip is the 1x1 tail of the same chain.
        assert texdecode.decode(out[k], mips - 1).shape == (1, 1, 4)


def test_encode_tiers_without_mips_writes_one_level_per_tier() -> None:
    out = texio.encode_tiers(_ramp(8, 8), texio.BGRA8, offsets=(0, 1, 3), keep_mips=False)
    for k, side in {0: 8, 1: 4, 3: 1}.items():
        hdr, _ = texio.read(out[k])
        assert (hdr.width, hdr.height, hdr.mip_count) == (side, side, 1), k


def test_encode_tiers_skips_offsets_past_the_chain() -> None:
    out = texio.encode_tiers(_ramp(4, 4), texio.BGRA8, offsets=(0, 2, 3, 10))
    assert sorted(out) == [0, 2]  # a 4x4 has three levels: 0, 1, 2


def test_encode_tiers_default_is_the_top_only() -> None:
    out = texio.encode_tiers(_ramp(4, 4), texio.BGRA8)
    assert list(out) == [0]


def test_encode_tiers_offset_zero_matches_encode_to_within_rounding() -> None:
    src = _noise(8, 8, seed=7)
    via_tiers = texio.encode_tiers(src, texio.BGRA8)[0]
    via_encode = texio.encode(src, texio.BGRA8)
    # Same header and same top level. The lower mips are NOT byte-identical: encode() rounds
    # each level from uint8 (texwrite.write) while encode_tiers() rounds a float32 chain
    # (float_chain), and the two disagree at exact .5 ties.
    assert via_tiers[: texwrite.HDR] == via_encode[: texwrite.HDR]
    np.testing.assert_array_equal(texio.read(via_tiers)[1], texio.read(via_encode)[1])
    for mip in (1, 2, 3):
        mean, peak = _err(texdecode.decode(via_tiers, mip), texdecode.decode(via_encode, mip))
        assert peak <= 1, (mip, mean, peak)


def test_unsupported_format_is_refused_the_same_way_on_every_platform() -> None:
    """Regression: the format lookup sat *inside* the texconv branch, so an unknown format
    raised KeyError on Windows and ValueError off it. CI runs on Linux and never saw it.
    """
    assert 0x9999 not in texio.TEXCONV_FORMAT
    for call in (texio.encode, texio.encode_tiers):
        with pytest.raises(ValueError, match="format 0x9999"):
            call(_noise(4, 4), 0x9999)


def test_encode_tiers_rejects_an_unknown_format() -> None:
    with pytest.raises(ValueError, match="no chain encoder"):
        texio.encode_tiers(_noise(4, 4), 0x9999)


def test_encode_tiers_keeps_the_attribute_bits() -> None:
    attr = texwrite.ATTR_2D | 0x2
    out = texio.encode_tiers(_ramp(8, 8), texio.BC7, attributes=attr, offsets=(0, 1))
    assert all(texio.read(t)[0].attributes == attr for t in out.values())


# ------------------------------------------------------------------------------------ read
def test_read_round_trips_write() -> None:
    src = _noise(12, 5, seed=8)
    hdr, back = texio.read(texwrite.write(src, mips=3))
    assert (hdr.width, hdr.height, hdr.mip_count) == (5, 12, 3)
    np.testing.assert_array_equal(back, src)


def test_read_takes_bytes_not_a_path_when_given_bytes() -> None:
    # A bytes argument is decoded in place; only a str is looked up in the game archive.
    data = texwrite.write(_noise(2, 2))
    assert isinstance(texio.read(data)[0].width, int)
    assert isinstance(texio.read(bytes(bytearray(data)))[1], np.ndarray)
