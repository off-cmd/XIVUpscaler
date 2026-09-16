"""Top-level dispatch for role-based texture processing."""

import numpy as np
import numpy.typing as npt

from . import engine as eng
from .color import do_color
from .masks import do_mask
from .normals import do_normal
from .ui import do_ui, do_ui_batch
from .utils import Upscaler, _u8

TIER_SCALE: dict[str, int] = {"4x": 4, "2x": 2, "native": 1}

# family -> (top tier, max source edge that still gets the top tier)
POLICY: dict[str, tuple[str | None, int]] = {
    "equipment": ("4x", 1024),
    "accessory": ("4x", 1024),
    "weapon": ("4x", 1024),
    "monster": ("2x", 2048),
    "demihuman": ("2x", 2048),
    "human-body": ("2x", 2048),
    "human-face": ("2x", 2048),
    "human-tail": ("2x", 2048),
    "human-zear": ("2x", 2048),
    "human-hair": (None, 0),  # Hair Defined 2 owns hair
    "bg": ("2x", 1024),
    "bgcommon": ("2x", 1024),
    "bg-hou": ("2x", 2048),
    "bg-ind": ("2x", 2048),
    "bgcommon-hou": ("2x", 2048),
    "bgcommon-mji": ("2x", 1024),
    "ui-icon": ("4x", 512),
    "ui-uld": ("2x", 2048),
    "ui-other": ("2x", 2048),
    "vfx": (None, 0),
}
MAX_EDGE_OUT: int = 4096


def top_tier(family: str, w: int, h: int, override: str | None = None) -> str | None:
    """Calculate the maximum upscaling tier for a given texture family and dimensions.

    Args:
        family: The resource family name.
        w: The width of the texture.
        h: The height of the texture.
        override: Optional explicit tier override (e.g., "4x", "2x").

    Returns:
        The target tier string, or None if skipped.
    """
    tier, cap = POLICY.get(family, (None, 0))
    if tier is None:
        # A family with no tier is excluded (human-hair: Hair Defined 2 owns it; vfx), and
        # skip_reason() marks its rows at plan time. An override selects a tier, it does
        # not resurrect a family, or `run --family human-hair --top 2x` would produce
        # products the policy says never to make.
        return None
    if override:
        tier = override
    s = TIER_SCALE[tier]
    while s > 1 and (max(w, h) > cap or max(w, h) * s > MAX_EDGE_OUT):
        s //= 2
    return {4: "4x", 2: "2x", 1: "native"}[s]


def tiers_below(top: str) -> list[str]:
    """Return all downsampled mip tiers below a given top tier.

    Args:
        top: The starting tier (e.g., "4x").

    Returns:
        A list of tier strings, starting with the `top` tier.
    """
    order = ["4x", "2x", "native"]
    return order[order.index(top) :]


ROLE_FN = {
    "normal": do_normal,
    "mask": do_mask,
    "color": do_color,
    "icon": do_ui,
    "ui": do_ui,
}


def process_top_batch(
    engine: Upscaler,
    role: str,
    family: str,
    rgbas: list[npt.NDArray[np.uint8]],
    top: str,
) -> list[npt.NDArray[np.uint8]]:
    """Process a batch of same-sized images at the top tier.

    Only the UI roles have a dedicated batched path; others fall back to one-at-a-time processing.

    Args:
        engine: The neural network engine.
        role: The role of the textures.
        family: The family name.
        rgbas: A list of source image arrays as uint8.
        top: The target top tier string.

    Returns:
        A list of upscaled image arrays as uint8.
    """
    scale = TIER_SCALE[top]
    if role in ("icon", "ui"):
        return [_u8(o) for o in do_ui_batch(engine, rgbas, scale)]
    return [process_top(engine, role, family, r, None, top) for r in rgbas]


def process_top(
    engine: Upscaler,
    role: str,
    family: str,
    rgba: npt.NDArray[np.uint8],
    src_fmt: str | None,
    top: str,
) -> npt.NDArray[np.uint8]:
    """Process a single image at its top tier.

    Args:
        engine: The neural network engine.
        role: The role of the texture.
        family: The family name.
        rgba: The source image array as uint8.
        src_fmt: The original BC format (e.g., "BC1", "BC7").
        top: The target top tier string.

    Returns:
        The upscaled image array as uint8.
    """
    scale = TIER_SCALE[top]
    return _u8(ROLE_FN[role](engine, rgba, scale, src_fmt, family))


def process(
    engine: Upscaler,
    role: str,
    family: str,
    rgba: npt.NDArray[np.uint8],
    src_fmt: str | None,
    top: str,
) -> dict[str, npt.NDArray[np.uint8]]:
    """Process an image at all tiers down to native (legacy path).

    Args:
        engine: The neural network engine.
        role: The role of the texture.
        family: The family name.
        rgba: The source image array as uint8.
        src_fmt: The original BC format.
        top: The target top tier string.

    Returns:
        A dictionary mapping tier names to the processed image arrays.
    """
    out = process_top(engine, role, family, rgba, src_fmt, top).astype(np.float32) / 255.0
    result = {top: _u8(out)}
    cur = out
    for tier in tiers_below(top)[1:]:
        cur = eng.box_down(cur, 2)
        result[tier] = _u8(cur)
    return result
