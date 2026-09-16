"""Processing logic for color (albedo/diffuse) maps."""

import numpy as np
import numpy.typing as npt

from . import engine as eng
from .utils import Upscaler, _f, average_color_fix, gray, has_alpha


def do_color(
    engine: Upscaler,
    rgba: npt.NDArray[np.uint8],
    scale: int,
    src_fmt: str | None,
    family: str,
) -> npt.NDArray[np.float32]:
    """Upscale an FFXIV color (albedo) map.

    Applies the primary color upscaling models (e.g., PBRify, face/skin specializations)
    and applies a low-frequency average color fix to prevent neural network color shifting.

    Args:
        engine: The neural network processing engine.
        rgba: The source image array as uint8.
        scale: The integer scaling factor.
        src_fmt: The original BC compression format.
        family: The resource family.

    Returns:
        The upscaled color map as float32 [0.0, 1.0].
    """
    x = _f(rgba)
    rgb = x[..., :3]

    if family == "human-face":
        slot = "face"
    elif family == "monster" and src_fmt == "BC7" and engine.has("color_v3"):
        slot = "color_v3"
    else:
        slot = "color"

    if src_fmt == "BC1" and scale > 1 and engine.enabled("bc1clean"):
        rgb = engine.run("bc1clean", rgb, 1)
    if family in (
        "human-body",
        "human-face",
        "human-tail",
        "human-zear",
    ) and engine.has("skin"):
        rgb = engine.run("skin", rgb, 1)
    if scale > 1:
        out = engine.run(slot, rgb, scale)
        out = average_color_fix(out, eng.lanczos(rgb, scale))
    else:
        out = rgb
    a = gray(engine, x[..., 3], scale, slot="ui") if has_alpha(rgba) else np.ones_like(out[..., 0])
    return np.dstack([out, a]).astype(np.float32)
