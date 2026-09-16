"""Utility functions for processing FFXIV textures."""

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from . import engine as eng


class Upscaler(Protocol):
    """What a role function needs from the inference engine.

    ``engine.Engine`` is the real one. The role modules only ever call these four methods,
    so this is the whole contract -- and the reason a test can stand in a numpy fake.
    """

    def has(self, slot: str) -> bool:
        """Whether ``slot`` has a loadable weight file (and torch is present)."""

    def enabled(self, slot: str) -> bool:
        """Whether ``slot`` is switched on in the registry (``null`` means deliberately off)."""

    def run(self, slot: str, img: npt.NDArray[np.floating], scale: int) -> npt.NDArray[np.float32]:
        """Upscale one float32 ``(H, W, C)`` image in ``[0, 1]`` by ``scale``."""

    def run_batch(
        self, slot: str, imgs: list[npt.NDArray[np.floating]], scale: int
    ) -> list[npt.NDArray[np.float32]]:
        """``run`` over a list, batched on the device when the shapes allow."""


def _f(u8: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Convert a uint8 image to a float32 image in the [0.0, 1.0] range."""
    return u8.astype(np.float32) / 255.0


def _u8(f: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """Convert a float32 image in the [0.0, 1.0] range to a uint8 image in [0, 255]."""
    return np.clip(np.rint(f * 255.0), 0, 255).astype(np.uint8)


def nearest(ch: npt.NDArray[Any], scale: int) -> npt.NDArray[Any]:
    """Perform nearest-neighbor upscaling on a single channel."""
    return np.repeat(np.repeat(ch, scale, 0), scale, 1)


def gray(
    engine: Upscaler, ch: npt.NDArray[np.float32], scale: int, slot: str = "mask"
) -> npt.NDArray[np.float32]:
    """Upscale a single scalar channel by passing it through a model as replicated RGB."""
    if scale == 1:
        return ch
    out = engine.run(slot, np.repeat(ch[..., None], 3, 2), scale)
    return out.mean(2)


def average_color_fix(
    out: npt.NDArray[np.float32], src_up: npt.NDArray[np.float32], factor: int = 8
) -> npt.NDArray[np.float32]:
    """Match the low-frequency color of the output image to the source image."""
    H, W, C = out.shape
    f = max(1, min(factor, H // 4, W // 4))
    if f <= 1:
        return out
    ref = eng.box_down(src_up, f)
    low = eng.box_down(out, f)
    corr = eng.lanczos(ref - low + 0.5, f) - 0.5  # lanczos works on [0,1]; shift the difference
    corr = corr[:H, :W]
    if corr.shape[:2] != (H, W):
        pad = np.zeros((H, W, C), np.float32)
        pad[: corr.shape[0], : corr.shape[1]] = corr
        corr = pad
    return np.clip(out + corr, 0, 1)


def has_alpha(rgba: npt.NDArray[np.number]) -> bool:
    """Determine if an RGBA image contains any transparency."""
    return int(rgba[..., 3].min()) < 255
