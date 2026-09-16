"""Processing logic for UI elements (icons, ULD sheets)."""

import numpy as np
import numpy.typing as npt

from .utils import Upscaler, _f, gray, has_alpha


def do_ui(
    engine: Upscaler,
    rgba: npt.NDArray[np.uint8],
    scale: int,
    src_fmt: str | None,  # noqa: ARG001 - uniform ROLE_FN signature
    family: str,  # noqa: ARG001 - uniform ROLE_FN signature
) -> npt.NDArray[np.float32]:
    """Upscale UI textures using alpha-premultiplication to avoid dark fringing.

    Args:
        engine: The neural network processing engine.
        rgba: The source image array as uint8.
        scale: The integer scaling factor.
        src_fmt: The original BC compression format (often None for UI).
        family: The resource family (e.g., "ui-icon", "ui-uld").

    Returns:
        The upscaled UI element as float32 [0.0, 1.0].
    """
    x = _f(rgba)
    a = x[..., 3:4]
    if not has_alpha(rgba):
        rgb = engine.run("ui", x[..., :3], scale) if scale > 1 else x[..., :3]
        return np.dstack([np.clip(rgb, 0, 1), np.ones_like(rgb[..., :1])]).astype(np.float32)
    pre = x[..., :3] * a
    if scale > 1:
        pre_up = engine.run("ui", pre, scale)
        straight_up = engine.run("ui", x[..., :3], scale)  # colour the artist left under alpha 0
        a_up = gray(engine, a[..., 0], scale, slot="ui")[..., None]
    else:
        pre_up, straight_up, a_up = pre, x[..., :3], a
    # edge texels: un-premultiplied (no dark fringe); transparent texels: keep the straight colour so
    # bilinear filtering against them does not darken either
    rgb = np.where(a_up > 0.02, pre_up / np.maximum(a_up, 0.02), straight_up)
    return np.dstack([np.clip(rgb, 0, 1), a_up]).astype(np.float32)


def do_ui_batch(
    engine: Upscaler, rgbas: list[npt.NDArray[np.uint8]], scale: int
) -> list[npt.NDArray[np.float32]]:
    """Upscale a batch of UI textures simultaneously.

    This dramatically speeds up processing for many small UI icons by stacking
    them into a single tensor batch for the neural network.

    Args:
        engine: The neural network processing engine.
        rgbas: A list of source image arrays as uint8.
        scale: The integer scaling factor.

    Returns:
        A list of upscaled UI elements as float32 [0.0, 1.0].
    """
    if not rgbas:
        return []
    if scale <= 1:
        return [do_ui(engine, r, scale, None, "") for r in rgbas]
    xs = [_f(r) for r in rgbas]
    alpha = [has_alpha(r) for r in rgbas]

    jobs, index = [], []  # jobs: images to run; index: (kind, i) per job
    for i, x in enumerate(xs):
        if alpha[i]:
            jobs.append(x[..., :3] * x[..., 3:4])
            index.append(("pre", i))
            jobs.append(x[..., :3])
            index.append(("straight", i))
            jobs.append(np.repeat(x[..., 3:4], 3, 2))
            index.append(("alpha", i))
        else:
            jobs.append(x[..., :3])
            index.append(("rgb", i))
    ups = engine.run_batch("ui", jobs, scale)

    got = {}
    for (kind, i), u in zip(index, ups, strict=True):
        got[(kind, i)] = u
    out = []
    for i in range(len(xs)):
        if not alpha[i]:
            rgb = np.clip(got[("rgb", i)], 0, 1)
            out.append(np.dstack([rgb, np.ones_like(rgb[..., :1])]).astype(np.float32))
            continue
        pre_up = got[("pre", i)]
        straight_up = got[("straight", i)]
        a_up = got[("alpha", i)].mean(2)[..., None]
        rgb = np.where(a_up > 0.02, pre_up / np.maximum(a_up, 0.02), straight_up)
        out.append(np.dstack([np.clip(rgb, 0, 1), a_up]).astype(np.float32))
    return out
