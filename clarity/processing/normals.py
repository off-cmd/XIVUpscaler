"""Processing logic for tangent-space normal maps."""

import numpy as np
import numpy.typing as npt

from .utils import Upscaler, _f, gray, has_alpha, nearest


def do_normal(
    engine: Upscaler,
    rgba: npt.NDArray[np.uint8],
    scale: int,
    src_fmt: str | None,
    family: str,
) -> npt.NDArray[np.float32]:
    """Upscale and restore an FFXIV normal map.

    FFXIV normal maps store tangent-space X and Y in the Red and Green channels.
    The Z vector must be mathematically reconstructed, normalized, and stored back.
    The Blue channel is typically an opacity or skin influence scalar.
    The Alpha channel is typically a color-set row index or wetness scalar.

    Args:
        engine: The neural network processing engine.
        rgba: The source image array as uint8.
        scale: The integer scaling factor (e.g., 2, 4).
        src_fmt: The original BC compression format (e.g., "BC1", "BC7").
        family: The resource family (e.g., "human-body", "equipment").

    Returns:
        The upscaled and mathematically renormalized normal map as float32 [0.0, 1.0].
    """
    x = _f(rgba)
    rg = x[..., :2]
    # The RG0 models come in a BC1-trained and a BC7-trained variant: use the one that matches the
    # source's compression, and only run the generic BC1 cleaner in front when no BC1-trained
    # normal model is available (the cleaner is a colour model; on RG0 normals it is a mismatch).
    # RunDevelopment specifically recommends using the BC1 model for BC7 textures that contain alpha.
    needs_bc1_model = (src_fmt == "BC1") or (src_fmt == "BC7" and has_alpha(rgba))
    slot = "normal_bc1" if needs_bc1_model else "normal"
    if (
        src_fmt == "BC1"
        and scale > 1
        and engine.enabled("bc1clean")
        and not engine.has("normal_bc1")
    ):
        cleaned = engine.run("bc1clean", np.dstack([rg, np.zeros_like(rg[..., :1])]), 1)
        rg = cleaned[..., :2]
    if scale > 1:
        y = engine.run(slot, np.dstack([rg, np.zeros_like(rg[..., :1])]), scale)
        rg = y[..., :2]
    nx, ny = rg[..., 0] * 2 - 1, rg[..., 1] * 2 - 1
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
    n = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    rg = np.dstack([(nx / n + 1) / 2, (ny / n + 1) / 2])
    b = gray(engine, x[..., 2], scale)  # opacity / skin influence
    if has_alpha(rgba):
        a = (
            gray(engine, x[..., 3], scale)
            if family.startswith("human-")
            else _f(nearest(rgba[..., 3], scale))
        )
    else:
        a = np.ones_like(b)
    return np.dstack([rg, b, a]).astype(np.float32)
