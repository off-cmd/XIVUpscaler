"""The per-role image maths, run against a recording fake engine.

The engine is the only thing in ``clarity.processing`` that needs torch; everything else is
numpy and is the part worth pinning: which model slot a texture is routed to, and what is
done to the pixels before and after the model. The fake engine upscales by nearest
neighbour and records every call, so a test can assert both the routing and the maths.
"""

from __future__ import annotations

import numpy as np
import pytest

from clarity.processing import color, engine, masks, normals, roles, ui, utils

# ---------------------------------------------------------------------------
# A fake engine: same surface as processing.engine.Engine, no torch.
# ---------------------------------------------------------------------------


class FakeEngine(utils.Upscaler):
    """Nearest-neighbour "model" that records what it was asked to do.

    ``present`` is the set of slots that have a weight file; ``disabled`` are slots mapped to
    null in the registry (deliberately off, no fallback).
    """

    def __init__(self, present: set[str] | None = None, disabled: set[str] | None = None) -> None:
        self.present = present if present is not None else set(engine.DEFAULT_REGISTRY)
        self.disabled = disabled or set()
        self.calls: list[tuple[str, tuple[int, ...], int]] = []
        self.missing: set[str] = set()

    # -- the Engine API the role functions use
    def has(self, slot: str) -> bool:
        return slot in self.present and slot not in self.disabled

    def enabled(self, slot: str) -> bool:
        return slot not in self.disabled

    def run(self, slot: str, img: np.ndarray, scale: int) -> np.ndarray:
        self.calls.append((slot, img.shape, scale))
        if not self.has(slot):
            self.missing.add(slot)
        return utils.nearest(img, scale).astype(np.float32)

    def run_batch(self, slot: str, imgs: list[np.ndarray], scale: int) -> list[np.ndarray]:
        return [self.run(slot, i, scale) for i in imgs]

    # -- module helpers the role functions reach through `eng`
    @staticmethod
    def lanczos(img: np.ndarray, scale: float) -> np.ndarray:
        return engine.lanczos(img, scale)

    @staticmethod
    def box_down(img: np.ndarray, factor: int) -> np.ndarray:
        return engine.box_down(img, factor)


def _u8(*chans: np.ndarray) -> np.ndarray:
    return np.dstack(chans).astype(np.uint8)


def _flat(h: int, w: int, r: int, g: int, b: int, a: int = 255) -> np.ndarray:
    return np.full((h, w, 4), (r, g, b, a), np.uint8)


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------


def test_f_and_u8_round_trip_every_value() -> None:
    ramp = np.arange(256, dtype=np.uint8).reshape(16, 16, 1)
    assert utils._f(ramp).dtype == np.float32
    np.testing.assert_array_equal(utils._u8(utils._f(ramp)), ramp)


def test_u8_rounds_half_up_and_clips() -> None:
    f = np.array([[[-0.1, 0.0, 0.5 / 255, 1.0, 1.7]]], np.float32)
    assert utils._u8(f).tolist() == [[[0, 0, 0, 255, 255]]]
    assert utils._u8(np.array([[[0.5]]], np.float32)).item() in (127, 128)  # np.rint, banker's


def test_nearest_repeats_pixels_and_keeps_dtype() -> None:
    a = np.array([[1, 2], [3, 4]], np.uint8)
    up = utils.nearest(a, 2)
    assert up.dtype == np.uint8
    assert up.tolist() == [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]


def test_has_alpha_means_any_texel_below_255() -> None:
    assert not utils.has_alpha(_flat(4, 4, 0, 0, 0, 255))
    assert utils.has_alpha(_flat(4, 4, 0, 0, 0, 254))
    img = _flat(4, 4, 0, 0, 0, 255)
    img[3, 3, 3] = 0
    assert utils.has_alpha(img)


def test_gray_replicates_to_rgb_runs_the_mask_slot_and_collapses_back() -> None:
    eng = FakeEngine()
    ch = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    out = utils.gray(eng, ch, 2)
    assert eng.calls == [("mask", (4, 4, 3), 2)]
    assert out.shape == (8, 8)
    np.testing.assert_allclose(out, utils.nearest(ch, 2), atol=1e-6)


def test_average_color_fix_restores_the_reference_low_frequency_colour() -> None:
    rng = np.random.default_rng(0)
    ref = np.full((16, 16, 3), 0.25, np.float32)
    # A model output with the right texture but the wrong overall tint.
    shifted = np.clip(ref + 0.2 + rng.normal(0, 0.01, ref.shape).astype(np.float32), 0, 1)
    fixed = utils.average_color_fix(shifted, ref)
    assert abs(float(fixed.mean()) - 0.25) < 0.02
    assert abs(float(shifted.mean()) - 0.45) < 0.02


# ---------------------------------------------------------------------------
# normals
# ---------------------------------------------------------------------------


def test_normal_feeds_the_model_rg_only_with_b_zeroed() -> None:
    eng = FakeEngine()
    src = _flat(4, 4, 128, 128, 200, 255)
    normals.do_normal(eng, src, 2, "BC7", "equipment")
    slot, shape, scale = eng.calls[0]
    assert (slot, shape, scale) == ("normal", (4, 4, 3), 2)


def test_normal_slot_follows_the_source_compression() -> None:
    for fmt, alpha, expected in [
        ("BC7", 255, "normal"),
        ("BC1", 255, "normal_bc1"),
        ("BC7", 200, "normal_bc1"),  # BC7 with alpha: RunDevelopment's advice
    ]:
        eng = FakeEngine()
        normals.do_normal(eng, _flat(4, 4, 128, 128, 0, alpha), 2, fmt, "equipment")
        assert eng.calls[0][0] == expected, (fmt, alpha)


def test_normal_bc1_cleaner_only_runs_when_no_bc1_normal_model_exists() -> None:
    with_model = FakeEngine()
    normals.do_normal(with_model, _flat(4, 4, 128, 128, 0), 2, "BC1", "equipment")
    assert next(c[0] for c in with_model.calls) == "normal_bc1"
    assert "bc1clean" not in [c[0] for c in with_model.calls]

    without = FakeEngine(present={"normal", "bc1clean", "mask"})
    normals.do_normal(without, _flat(4, 4, 128, 128, 0), 2, "BC1", "equipment")
    assert [c[0] for c in without.calls][:2] == ["bc1clean", "normal_bc1"]


def test_normal_output_is_unit_length_even_when_the_model_overshoots() -> None:
    class Overshoot(FakeEngine):
        def run(self, slot: str, img: np.ndarray, scale: int) -> np.ndarray:
            out = super().run(slot, img, scale)
            if slot.startswith("normal"):
                out[..., :2] = 1.0  # x = y = +1: impossible for a unit normal
            return out

    out = normals.do_normal(Overshoot(), _flat(4, 4, 128, 128, 0), 2, "BC7", "equipment")
    nx, ny = out[..., 0] * 2 - 1, out[..., 1] * 2 - 1
    length_xy = np.sqrt(nx * nx + ny * ny)
    # Projected back onto the unit circle (nz = 0), not left at sqrt(2).
    np.testing.assert_allclose(length_xy, 1.0, atol=1e-3)


def test_normal_alpha_is_nearest_for_gear_and_model_for_humans() -> None:
    # Alpha carries a colour-set row index on gear: it must stay exact. On humans it is a
    # continuous value and goes through the mask model like B.
    src = _flat(4, 4, 128, 128, 0, 255)
    src[0, 0, 3] = 17
    src[0, 1, 3] = 34

    gear = FakeEngine()
    out = normals.do_normal(gear, src, 2, "BC7", "equipment")
    assert [c[0] for c in gear.calls] == ["normal_bc1", "mask"]  # B via gray(); no alpha call
    np.testing.assert_array_equal(utils._u8(out[..., 3]), utils.nearest(src[..., 3], 2))

    human = FakeEngine()
    normals.do_normal(human, src, 2, "BC7", "human-body")
    assert [c[0] for c in human.calls] == ["normal_bc1", "mask", "mask"]  # B, then A


def test_normal_without_alpha_writes_opaque_alpha() -> None:
    out = normals.do_normal(FakeEngine(), _flat(4, 4, 128, 128, 0), 2, "BC7", "equipment")
    np.testing.assert_array_equal(out[..., 3], 1.0)


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------


def test_mask_runs_each_channel_as_its_own_grey_image() -> None:
    eng = FakeEngine()
    src = _u8(np.full((4, 4), 10), np.full((4, 4), 20), np.full((4, 4), 30), np.full((4, 4), 255))
    masks.do_mask(eng, src, 2, "BC7", "equipment")
    assert eng.calls == [("mask", (4, 4, 3), 2)] * 3


def test_mask_channels_do_not_bleed_into_each_other() -> None:
    class Tinting(FakeEngine):
        # A model that adds chroma: it brightens R and darkens B of whatever it is given.
        def run(self, slot: str, img: np.ndarray, scale: int) -> np.ndarray:
            out = super().run(slot, img, scale)
            out[..., 0] = np.clip(out[..., 0] + 0.2, 0, 1)
            out[..., 2] = np.clip(out[..., 2] - 0.2, 0, 1)
            return out

    src = _u8(
        np.full((4, 4), 128), np.full((4, 4), 128), np.full((4, 4), 128), np.full((4, 4), 255)
    )
    out = masks.do_mask(Tinting(), src, 2, "BC7", "equipment")
    # Replicate-to-RGB then mean(2) cancels the invented chroma: every scalar stays 0.5.
    np.testing.assert_allclose(out[..., :3], 128 / 255, atol=1e-6)


def test_mask_alpha_is_processed_when_present_and_filled_when_not() -> None:
    src = _u8(
        np.full((4, 4), 128), np.full((4, 4), 128), np.full((4, 4), 128), np.full((4, 4), 200)
    )
    eng = FakeEngine()
    out = masks.do_mask(eng, src, 2, "BC7", "equipment")
    assert len(eng.calls) == 4
    np.testing.assert_allclose(out[..., 3], 200 / 255, atol=1e-6)

    opaque = src.copy()
    opaque[..., 3] = 255
    out = masks.do_mask(FakeEngine(), opaque, 2, "BC7", "equipment")
    np.testing.assert_array_equal(out[..., 3], 1.0)


# ---------------------------------------------------------------------------
# ui
# ---------------------------------------------------------------------------


def test_ui_runs_premultiplied_straight_and_alpha_passes() -> None:
    eng = FakeEngine()
    src = _flat(4, 4, 200, 100, 50, 128)
    ui.do_ui(eng, src, 2, "BC3", "ui-icon")
    assert [c[0] for c in eng.calls] == ["ui", "ui", "ui"]


def test_ui_transparent_texels_keep_the_straight_colour() -> None:
    # A red icon on fully transparent texels whose stored colour is green: after upscaling,
    # the transparent area must still be green (so bilinear filtering does not darken the
    # edge), not the black that premultiplication would leave.
    src = _flat(4, 4, 0, 255, 0, 0)
    src[1:3, 1:3] = (255, 0, 0, 255)
    out = ui.do_ui(FakeEngine(), src, 2, "BC3", "ui-icon")
    np.testing.assert_allclose(out[0, 0, :3], [0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(out[3, 3, :3], [1, 0, 0], atol=1e-6)
    assert out[0, 0, 3] == 0 and out[3, 3, 3] == 1


def test_ui_edge_texels_are_unpremultiplied() -> None:
    src = _flat(4, 4, 200, 100, 50, 128)  # half-transparent everywhere
    out = ui.do_ui(FakeEngine(), src, 2, "BC3", "ui-icon")
    expected = np.broadcast_to([200 / 255, 100 / 255, 50 / 255], out[..., :3].shape)
    np.testing.assert_allclose(out[..., :3], expected, atol=1 / 255 + 1e-6)
    np.testing.assert_allclose(out[..., 3], 128 / 255, atol=1e-6)


def test_ui_batch_matches_single_for_every_input() -> None:
    rng = np.random.default_rng(1)
    imgs = [rng.integers(0, 256, (4, 4, 4), dtype=np.uint8) for _ in range(3)]
    imgs[1][..., 3] = 255  # one opaque icon in the batch
    single = [ui.do_ui(FakeEngine(), i, 2, "BC3", "ui-icon") for i in imgs]
    batch = ui.do_ui_batch(FakeEngine(), imgs, 2)
    for s, b in zip(single, batch, strict=True):
        np.testing.assert_allclose(s, b, atol=1e-6)


# ---------------------------------------------------------------------------
# color
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "fmt", "present", "expected"),
    [
        ("equipment", "BC7", None, "color"),
        ("human-face", "BC7", None, "face"),
        ("monster", "BC7", {"color", "color_v3", "mask", "ui"}, "color_v3"),
        ("monster", "BC7", {"color", "mask", "ui"}, "color"),  # v3 absent -> generic
        ("monster", "BC1", None, "color"),  # v3 only for BC7 sources
    ],
)
def test_color_slot_routing(family, fmt, present, expected) -> None:
    eng = FakeEngine(present=present)
    color.do_color(eng, _flat(8, 8, 90, 60, 30), 2, fmt, family)
    upscale_calls = [c for c in eng.calls if c[2] == 2 and c[0] != "ui"]
    assert upscale_calls[0][0] == expected


def test_color_bc1_sources_are_cleaned_first_and_humans_get_the_skin_pass() -> None:
    eng = FakeEngine()
    color.do_color(eng, _flat(8, 8, 90, 60, 30), 2, "BC1", "human-body")
    assert [c[0] for c in eng.calls][:3] == ["bc1clean", "skin", "color"]


def test_color_alpha_goes_through_the_ui_model() -> None:
    eng = FakeEngine()
    color.do_color(eng, _flat(8, 8, 90, 60, 30, 128), 2, "BC7", "equipment")
    assert eng.calls[-1][0] == "ui"


def test_color_output_keeps_the_source_tint() -> None:
    class Tinting(FakeEngine):
        def run(self, slot: str, img: np.ndarray, scale: int) -> np.ndarray:
            out = super().run(slot, img, scale)
            return np.clip(out + 0.2, 0, 1) if slot == "color" else out

    src = _flat(16, 16, 64, 128, 192)
    out = color.do_color(Tinting(), src, 2, "BC7", "equipment")
    np.testing.assert_allclose(
        out[..., :3].mean((0, 1)), [64 / 255, 128 / 255, 192 / 255], atol=0.02
    )


# ---------------------------------------------------------------------------
# the dispatch table
# ---------------------------------------------------------------------------


def test_process_top_returns_uint8_of_the_right_size() -> None:
    src = _flat(4, 4, 128, 128, 0)
    for role in ("normal", "mask", "color", "icon", "ui"):
        out = roles.process_top(FakeEngine(), role, "equipment", src, "BC7", "2x")
        assert out.dtype == np.uint8 and out.shape == (8, 8, 4), role


def test_process_derives_lower_tiers_by_box_down() -> None:
    src = _flat(8, 8, 128, 128, 0)
    tiers = roles.process(FakeEngine(), "color", "equipment", src, "BC7", "4x")
    assert set(tiers) == {"4x", "2x", "native"}
    assert tiers["4x"].shape[:2] == (32, 32)
    assert tiers["2x"].shape[:2] == (16, 16)
    assert tiers["native"].shape[:2] == (8, 8)
