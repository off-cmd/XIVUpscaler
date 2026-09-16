"""Processing logic for material control masks."""

import numpy as np
import numpy.typing as npt

from .utils import Upscaler, _f, has_alpha


def do_mask(
    engine: Upscaler,
    rgba: npt.NDArray[np.uint8],
    scale: int,
    src_fmt: str | None,
    family: str,  # noqa: ARG001 - uniform ROLE_FN signature
) -> npt.NDArray[np.float32]:
    """Upscale an FFXIV material mask.

    A mask is typically three independent scalars (specular power, roughness, AO).
    Each channel must be run independently through the model so it does not bleed
    into its neighbors. This is executed as a batch of three model calls.

    Args:
        engine: The neural network processing engine.
        rgba: The source image array as uint8.
        scale: The integer scaling factor (e.g., 2, 4).
        src_fmt: The original BC compression format (e.g., "BC1", "BC7").
        family: The resource family (e.g., "equipment", "monster").

    Returns:
        The upscaled mask as float32 [0.0, 1.0].
    """
    x = _f(rgba)
    planes = [np.repeat(x[..., c][..., None], 3, 2) for c in range(3)]
    if has_alpha(rgba):
        planes.append(np.repeat(x[..., 3][..., None], 3, 2))
    if src_fmt == "BC1" and scale > 1 and engine.enabled("bc1clean"):
        planes = engine.run_batch("bc1clean", planes, 1)
    if scale > 1:
        planes = engine.run_batch("mask", planes, scale)
    chans = [p.mean(2) for p in planes]
    if len(chans) == 3:
        chans.append(np.ones_like(chans[0]))
    return np.dstack(chans).astype(np.float32)
