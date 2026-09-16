"""The ``clarity.ffxiv`` package exposes every module it owns, and the pure-Python BC7
encoder is reachable from the texture writer.

Regression: ``texwrite.write_bc7`` and ``texio``'s BC7 fallback did ``import bc7enc`` as a
top-level name, and ``mdlstrings`` did ``import mdlpatch``. Neither resolves from inside a
package, and ``ffxiv/__init__.py`` swallowed the ImportError, so ``kb.mdlstrings`` was
silently absent and every non-Windows BC7 encode raised ``ModuleNotFoundError``.
"""

import numpy as np
import pytest

from clarity import ffxiv as kb
from clarity.ffxiv import bc7enc, tex, texdecode, texwrite

EXPECTED_MODULES = (
    "bc7enc",
    "exd",
    "imcfile",
    "mdlpatch",
    "mdlstrings",
    "mtrl",
    "sqpack",
    "tex",
    "texdecode",
    "texwrite",
)


@pytest.mark.parametrize("name", EXPECTED_MODULES)
def test_module_is_exported(name: str) -> None:
    assert hasattr(kb, name), f"clarity.ffxiv.{name} is not importable"
    assert name in kb.__all__


def test_legacy_aliases_point_at_canonical_modules() -> None:
    assert kb.exdfile is kb.exd
    assert kb.mtrlfile is kb.mtrl
    assert kb.texfile is kb.tex


def _checker(size: int = 16) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    rgb = np.where(((x // 4 + y // 4) % 2)[..., None] == 0, [220, 40, 40], [30, 30, 200])
    a = np.full((size, size, 1), 255)
    return np.concatenate([rgb, a], axis=2).astype(np.uint8)


def test_bc7_encoder_round_trips_through_the_reference_decoder() -> None:
    src = _checker()
    blocks = bc7enc.encode(src)
    assert len(blocks) == (16 // 4) * (16 // 4) * 16
    back = texdecode.decode_raw("BC7", blocks, 16, 16)
    err = np.abs(back.astype(int) - src.astype(int))
    # Mode 6 on flat 4x4 cells is near-lossless; anything above a few counts is a packing bug.
    assert err.max() <= 4


def test_write_bc7_produces_a_valid_tex_header() -> None:
    src = _checker()
    data = texwrite.write_bc7(src, mips=1)
    hdr = tex.TexHeader(data)
    assert (hdr.width, hdr.height) == (16, 16)
    assert hdr.format_name == "BC7"
    assert hdr.mip_count == 1
    assert len(data) == texwrite.HDR + 16 * 16
