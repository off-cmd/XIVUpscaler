"""Block and uncompressed decoders in ``clarity.ffxiv.texdecode``.

The block decoders are checked *differentially*: random block bytes are decoded once by the
compiled reference (``texture2ddecoder``, BGRA out) and once by the module's pure-numpy fallback,
and the two must agree within one count -- the reference rounds its interpolants, the numpy path
truncates a float32.

Two documented places where the reference is not the spec, so the tests steer around them:

* ``decode_bc1`` ignores BC1's 1-bit alpha: the "transparent black" index in 3-colour mode comes
  back with alpha 255. The numpy path follows the spec, so alpha is checked against the block
  bits, not against the reference.
* ``decode_bc3`` consults ``c0 > c1`` on the embedded colour block, but BC2/BC3 colour blocks are
  *always* 4-colour. The differential test forces ``c0 > c1`` so both agree, and a separate test
  pins the 4-colour behaviour on a ``c0 <= c1`` block.

``texture2ddecoder`` ships no ``decode_bc2``, so BC2 is checked against ``decode_bc1`` on the
colour half plus an independently computed 4-bit alpha plane.
"""

import numpy as np
import pytest
import texture2ddecoder as t2d

from clarity.ffxiv import bc7enc, texdecode

W = H = 16  # 4x4 blocks; small enough to be fast, big enough to hit every index value


def _bgra_to_rgba(buf: bytes, w: int, h: int) -> np.ndarray:
    """texture2ddecoder writes BGRA; the module swizzles to RGBA the same way."""
    return np.frombuffer(buf, np.uint8).reshape(h, w, 4)[..., [2, 1, 0, 3]].copy()


def _random_blocks(seed: int, block_bytes: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, ((H // 4) * (W // 4), block_bytes), dtype=np.uint8)


def _force_four_colour(colour_blocks: np.ndarray) -> np.ndarray:
    """Make ``c0 > c1`` in every 8-byte BC1 block (c0 and c1 are the first two u16, LE)."""
    ends = colour_blocks[:, :4].copy().view("<u2")  # (B, 2)
    c0, c1 = ends[:, 0].copy(), ends[:, 1].copy()
    lo, hi = np.minimum(c0, c1), np.maximum(c0, c1)
    hi = np.where(hi == lo, np.where(hi == 0xFFFF, hi, hi + 1), hi)  # equal -> still c0 > c1
    lo = np.where(hi == lo, lo - 1, lo)
    ends[:, 0], ends[:, 1] = hi, lo
    out = colour_blocks.copy()
    out[:, :4] = ends.view(np.uint8)
    return out


def _assert_within_one(a: np.ndarray, b: np.ndarray) -> None:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    assert diff.max() <= 1, f"max difference {diff.max()} at {np.argwhere(diff > 1)[:4]}"


def _numpy_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the compiled decoder so ``decode_raw`` takes the pure-numpy branch."""
    monkeypatch.setattr(texdecode, "_t2d", None)


# ---------------------------------------------------------------- BC1


def test_bc1_four_colour_blocks_match_reference() -> None:
    blocks = _force_four_colour(_random_blocks(1, 8))
    data = blocks.tobytes()
    ref = _bgra_to_rgba(t2d.decode_bc1(data, W, H), W, H)
    _assert_within_one(texdecode._bc1_blocks(data, W, H), ref)


def test_bc1_three_colour_blocks_match_reference_rgb_and_spec_alpha() -> None:
    blocks = _random_blocks(2, 8)  # unforced: roughly half the blocks are c0 <= c1
    data = blocks.tobytes()
    ref = _bgra_to_rgba(t2d.decode_bc1(data, W, H), W, H)
    out = texdecode._bc1_blocks(data, W, H)
    _assert_within_one(out[..., :3], ref[..., :3])

    # Expected alpha from the block bits: 0 only for index 3 of a c0 <= c1 block.
    u16 = blocks.view("<u2")  # (B, 4): c0, c1, idx lo, idx hi
    three_colour = u16[:, 0] <= u16[:, 1]
    assert three_colour.any(), "seed 2 must produce at least one 3-colour block"
    idx = (u16[:, 3].astype(np.uint32) << 16) | u16[:, 2].astype(np.uint32)
    expected = np.full((H, W), 255, np.uint8)
    for b in range(blocks.shape[0]):
        by, bx = divmod(b, W // 4)
        for t in range(16):
            if three_colour[b] and (idx[b] >> (2 * t)) & 3 == 3:
                expected[by * 4 + t // 4, bx * 4 + t % 4] = 0
    np.testing.assert_array_equal(out[..., 3], expected)
    assert (expected == 0).any()
    # And a transparent texel is transparent *black*, not the interpolant.
    assert not out[expected == 0][:, :3].any()


def test_bc1_decode_raw_prefers_the_compiled_decoder_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _force_four_colour(_random_blocks(3, 8)).tobytes()
    with_t2d = texdecode.decode_raw("BC1", data, W, H)
    _numpy_only(monkeypatch)
    without = texdecode.decode_raw("BC1", data, W, H)
    _assert_within_one(with_t2d, without)


# ---------------------------------------------------------------- BC2


def test_bc2_numpy_matches_reference_colour_and_nibble_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocks = _random_blocks(4, 16)  # 8 bytes of 4-bit alpha, then an 8-byte colour block
    blocks[:, 8:] = _force_four_colour(blocks[:, 8:])
    _numpy_only(monkeypatch)
    out = texdecode.decode_raw("BC2", blocks.tobytes(), W, H)

    ref_rgb = _bgra_to_rgba(t2d.decode_bc1(blocks[:, 8:].tobytes(), W, H), W, H)[..., :3]
    _assert_within_one(out[..., :3], ref_rgb)

    # Alpha: texel t is nibble t of the first 8 bytes, low nibble first, scaled 0..15 -> 0..255.
    nibbles = np.stack([blocks[:, :8] & 0x0F, blocks[:, :8] >> 4], -1).reshape(-1, 16)
    expected = (nibbles * 17).astype(np.uint8).reshape(H // 4, W // 4, 4, 4)
    expected = expected.transpose(0, 2, 1, 3).reshape(H, W)
    np.testing.assert_array_equal(out[..., 3], expected)


# ---------------------------------------------------------------- BC3


def test_bc3_numpy_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = _random_blocks(5, 16)  # 8 bytes of BC4-style alpha, then an 8-byte colour block
    blocks[:, 8:] = _force_four_colour(blocks[:, 8:])
    data = blocks.tobytes()
    ref = _bgra_to_rgba(t2d.decode_bc3(data, W, H), W, H)
    _numpy_only(monkeypatch)
    _assert_within_one(texdecode.decode_raw("BC3", data, W, H), ref)


def test_bc3_colour_block_is_always_four_colour(monkeypatch: pytest.MonkeyPatch) -> None:
    # One block: opaque alpha (a0=a1=255, indices 0), then c0 < c1 with every index = 3.
    # In 3-colour mode index 3 would be transparent black; BC3 must give (p0 + 2*p1) / 3.
    c0, c1 = 0x0000, 0xFFFF  # black, white
    block = bytes([255, 255, 0, 0, 0, 0, 0, 0]) + c0.to_bytes(2, "little")
    block += c1.to_bytes(2, "little") + b"\xff\xff\xff\xff"
    _numpy_only(monkeypatch)
    out = texdecode.decode_raw("BC3", block, 4, 4)
    assert out.shape == (4, 4, 4)
    assert out[..., 3].min() == 255
    assert out[..., 0].min() >= 169 and out[..., 0].max() <= 171  # 2/3 of the way to white


# ---------------------------------------------------------------- BC4 / BC5


def test_bc4_numpy_matches_reference_red_channel() -> None:
    # The module has no compiled path for BC4; the reference writes R only, the module
    # replicates R into G and B (a grey image) so only channel 0 is comparable.
    data = _random_blocks(6, 8).tobytes()
    ref = _bgra_to_rgba(t2d.decode_bc4(data, W, H), W, H)
    out = texdecode.decode_raw("BC4", data, W, H)
    _assert_within_one(out[..., 0], ref[..., 0])
    np.testing.assert_array_equal(out[..., 0], out[..., 1])
    np.testing.assert_array_equal(out[..., 0], out[..., 2])
    assert out[..., 3].min() == 255


def test_bc5_numpy_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _random_blocks(7, 16).tobytes()  # two BC4 planes: R then G
    ref = _bgra_to_rgba(t2d.decode_bc5(data, W, H), W, H)
    _numpy_only(monkeypatch)
    out = texdecode.decode_raw("BC5", data, W, H)
    _assert_within_one(out[..., :2], ref[..., :2])
    assert not out[..., 2].any()  # no blue plane in BC5
    assert out[..., 3].min() == 255


def test_bc_alpha_plane_eight_and_six_step_palettes() -> None:
    # a0 > a1: 8-step ramp. a0 <= a1: 6-step ramp with index 6 = 0 and index 7 = 255.
    # Index bits are 3 per texel, LSB first, across the 6 bytes after a0/a1.
    def block(a0: int, a1: int, index: int) -> bytes:
        bits = 0
        for t in range(16):
            bits |= index << (3 * t)
        return bytes([a0, a1]) + bits.to_bytes(6, "little")

    assert texdecode._bc_alpha_plane(block(200, 100, 0), 4, 4, 8, 0).min() == 200
    assert texdecode._bc_alpha_plane(block(200, 100, 1), 4, 4, 8, 0).min() == 100
    assert texdecode._bc_alpha_plane(block(200, 100, 7), 4, 4, 8, 0).min() == 114  # 1/7 of the way
    assert texdecode._bc_alpha_plane(block(100, 200, 6), 4, 4, 8, 0).max() == 0
    assert texdecode._bc_alpha_plane(block(100, 200, 7), 4, 4, 8, 0).min() == 255
    assert texdecode._bc_alpha_plane(block(100, 200, 5), 4, 4, 8, 0).min() == 180  # (a0+4a1)/5


# ---------------------------------------------------------------- BC7


def test_bc7_decodes_a_block_from_the_pure_python_encoder() -> None:
    # 6x6 is not a multiple of 4: the encoder pads to 8x8 (4 blocks), the decoder crops back.
    y, x = np.mgrid[0:6, 0:6]
    src = np.stack([x * 30 + 20, np.full_like(x, 90), np.full_like(x, 160), 255 - y * 25], -1)
    src = src.astype(np.uint8)
    blocks = bc7enc.encode(src)
    assert len(blocks) == 2 * 2 * 16
    out = texdecode.decode_raw("BC7", blocks, 6, 6)
    assert out.shape == (6, 6, 4)
    assert np.abs(out.astype(int) - src.astype(int)).max() <= 4


def test_bc7_without_the_compiled_decoder_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _numpy_only(monkeypatch)
    with pytest.raises(RuntimeError, match="texture2ddecoder"):
        texdecode.decode_raw("BC7", bytes(16), 4, 4)
    with pytest.raises(RuntimeError, match="BC6H"):
        texdecode.decode_raw("BC6H", bytes(16), 4, 4)


# ---------------------------------------------------------------- uncompressed


def test_b8g8r8a8_round_trip() -> None:
    rng = np.random.default_rng(8)
    rgba = rng.integers(0, 256, (3, 5, 4), dtype=np.uint8)
    stored = rgba[..., [2, 1, 0, 3]].tobytes()  # file order is B, G, R, A
    np.testing.assert_array_equal(texdecode.decode_raw("B8G8R8A8", stored, 5, 3), rgba)


def test_b8g8r8x8_forces_alpha_to_255() -> None:
    rng = np.random.default_rng(9)
    rgba = rng.integers(0, 256, (3, 5, 4), dtype=np.uint8)
    rgba[..., 3] = 7  # whatever sits in X must be ignored
    out = texdecode.decode_raw("B8G8R8X8", rgba[..., [2, 1, 0, 3]].tobytes(), 5, 3)
    np.testing.assert_array_equal(out[..., :3], rgba[..., :3])
    assert out[..., 3].min() == 255


def test_l8_replicates_luminance_with_opaque_alpha() -> None:
    lum = np.arange(12, dtype=np.uint8).reshape(3, 4) * 20
    out = texdecode.decode_raw("L8", lum.tobytes(), 4, 3)
    for ch in range(3):
        np.testing.assert_array_equal(out[..., ch], lum)
    assert out[..., 3].min() == 255


def test_a8_keeps_alpha_and_zeroes_rgb() -> None:
    a = np.arange(12, dtype=np.uint8).reshape(3, 4) * 20 + 1
    out = texdecode.decode_raw("A8", a.tobytes(), 4, 3)
    np.testing.assert_array_equal(out[..., 3], a)
    assert not out[..., :3].any()


def test_b5g5r5a1_bit_replicates_so_white_is_255() -> None:
    # u16 LE: B bits 0-4, G bits 5-9, R bits 10-14, A bit 15.
    def px(r: int, g: int, b: int, a: int) -> int:
        return (a << 15) | (r << 10) | (g << 5) | b

    pixels = np.array([px(31, 31, 31, 1), px(0, 0, 0, 0), px(16, 8, 1, 1), px(31, 0, 0, 0)], "<u2")
    out = texdecode.decode_raw("B5G5R5A1", pixels.tobytes(), 4, 1)
    np.testing.assert_array_equal(out[0, 0], [255, 255, 255, 255])  # not 248
    np.testing.assert_array_equal(out[0, 1], [0, 0, 0, 0])
    np.testing.assert_array_equal(out[0, 2], [132, 66, 8, 255])  # (v << 3) | (v >> 2)
    np.testing.assert_array_equal(out[0, 3], [255, 0, 0, 0])


def test_b4g4r4a4_scales_nibbles_by_17() -> None:
    # u16 LE: B bits 0-3, G bits 4-7, R bits 8-11, A bits 12-15.
    pixels = np.array([0xFFFF, 0x0000, 0xF000, 0x0F00, 0x00F0, 0x000F, 0x1234], "<u2")
    out = texdecode.decode_raw("B4G4R4A4", pixels.tobytes(), 7, 1)
    expected = [
        [255, 255, 255, 255],
        [0, 0, 0, 0],
        [0, 0, 0, 255],
        [255, 0, 0, 0],
        [0, 255, 0, 0],
        [0, 0, 255, 0],
        [2 * 17, 3 * 17, 4 * 17, 1 * 17],
    ]
    np.testing.assert_array_equal(out[0], expected)


@pytest.mark.parametrize("fmt", ["R32F", "R16G16B16A16F", "D24S8", "0xbeef"])
def test_unknown_format_raises_instead_of_returning_garbage(fmt: str) -> None:
    with pytest.raises(NotImplementedError, match=fmt):
        texdecode.decode_raw(fmt, bytes(64), 4, 4)
