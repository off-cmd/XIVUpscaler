"""``.tex`` header parsing and the one mip-dimension rule.

Regression: ``TexHeader.surface_size`` rounded mip dimensions *up* while the decoder rounded
*down*, so for non-power-of-two textures the byte count and the pixel count described
different surfaces. Direct3D and Lumina both floor and clamp at 1.
"""

import struct

import numpy as np
import pytest

from clarity.ffxiv import tex, texdecode, texwrite


def _header(w: int, h: int, fmt_code: int, mips: int) -> tex.TexHeader:
    hdr = bytearray(texwrite.HDR)
    struct.pack_into("<IIHHHBB", hdr, 0, texwrite.ATTR_2D, fmt_code, w, h, 1, mips, 1)
    return tex.TexHeader(bytes(hdr))


@pytest.mark.parametrize(
    ("w", "h", "mip", "expected"),
    [
        (12, 12, 0, (12, 12)),
        (12, 12, 1, (6, 6)),
        (12, 12, 2, (3, 3)),
        (12, 12, 3, (1, 1)),  # ceil would have said 2
        (1024, 512, 10, (1, 1)),
        (5, 3, 1, (2, 1)),
    ],
)
def test_mip_dimensions_floor_and_clamp(w, h, mip, expected) -> None:
    hdr = _header(w, h, 0x1450, 8)  # B8G8R8A8
    assert hdr.mip_dimensions(mip)[:2] == expected


def test_surface_size_uses_the_same_dimensions_as_the_decoder() -> None:
    hdr = _header(12, 12, 0x6432, 4)  # BC7: block-compressed, padded to 4
    # mip 3 is 1x1 -> padded to 4x4 -> one 16-byte block. The old ceil rule gave 2x2,
    # which also pads to 4x4, so the *size* hid the disagreement; the pixel count did not.
    assert hdr.mip_dimensions(3)[:2] == (1, 1)
    assert hdr.surface_size(3) == 16


def test_decode_refuses_a_missing_mip() -> None:
    src = np.zeros((8, 8, 4), np.uint8)
    data = texwrite.write(src, mips=1)
    with pytest.raises(ValueError, match="mip 1 not present"):
        texdecode.decode(data, mip=1)
    with pytest.raises(ValueError):
        texdecode.decode(data, mip=-1)


def test_decode_round_trips_uncompressed_rgba() -> None:
    rng = np.random.default_rng(0)
    src = rng.integers(0, 256, (6, 10, 4), dtype=np.uint8)
    data = texwrite.write(src, mips=1)
    out = texdecode.decode(data, 0)
    assert out.shape == (6, 10, 4)
    np.testing.assert_array_equal(out, src)
