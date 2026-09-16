"""`Engine._tiled`: batching a texture's tiles must not change a single pixel.

The engine is the one part of ``clarity.processing`` that needs torch, so these tests import it
and skip when it is absent. They deliberately do not use a convolution as the stand-in model: on
CPU torch selects different kernels for different batch sizes, so a convolution is not bitwise
stable across the batch dimension and a test built on one would be measuring oneDNN rather than
the tiling. Elementwise arithmetic is bitwise stable, and upsampling by repeat keeps every output
pixel traceable to exactly one input pixel -- so a tile read from the wrong window, or written to
the wrong place, changes the result and the test catches it.
"""

from __future__ import annotations

import pytest

from clarity.processing.engine import Engine, _pad8

torch = pytest.importorskip("torch")


class FakeModel:
    """Deterministic, elementwise, ``scale``-times nearest upsample."""

    input_channels = 3
    output_channels = 3

    def __init__(self, scale: int = 2) -> None:
        self.scale = scale
        self.batches: list[int] = []

    def __call__(self, x):
        self.batches.append(x.shape[0])
        y = x * 3.0 - 0.25
        return torch.repeat_interleave(torch.repeat_interleave(y, self.scale, 2), self.scale, 3)


class OOMOnce(FakeModel):
    """Raises the CUDA out-of-memory RuntimeError the first time it sees a batch."""

    def __init__(self, scale: int = 2) -> None:
        super().__init__(scale)
        self.raised = False

    def __call__(self, x):
        if not self.raised and x.shape[0] > 1:
            self.raised = True
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return super().__call__(x)


def reference(m, t, ms, tile, pad):
    """The tile loop written out plainly, one forward pass per tile."""
    n, _c, h, w = t.shape
    out = torch.zeros((n, m.output_channels, h * ms, w * ms), dtype=t.dtype)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
            py0, px0 = max(0, y0 - pad), max(0, x0 - pad)
            py1, px1 = min(h, y1 + pad), min(w, x1 + pad)
            o = m(_pad8(t[:, :, py0:py1, px0:px1]))
            oy, ox = (y0 - py0) * ms, (x0 - px0) * ms
            out[:, :, y0 * ms : y1 * ms, x0 * ms : x1 * ms] = o[
                :, :, oy : oy + (y1 - y0) * ms, ox : ox + (x1 - x0) * ms
            ]
    return out


def engine(tile_batch, tile=16, pad=4):
    eng = Engine(None, device="cpu", tile=tile, pad=pad, fp16=False, tile_batch=tile_batch)
    assert eng.torch is not None, "torch imported but Engine did not pick it up"
    return eng


def source(n=1, h=100, w=100):
    torch.manual_seed(7)
    return torch.rand((n, 3, h, w), dtype=torch.float32)


@pytest.mark.parametrize("tile_batch", [1, 2, 4, 8, 64])
@pytest.mark.parametrize("n", [1, 2])
def test_batched_tiles_are_bitwise_identical(tile_batch, n):
    t = source(n)
    want = reference(FakeModel(), t, 2, tile=16, pad=4)
    got = engine(tile_batch)._tiled(FakeModel(), t, 2)
    assert torch.equal(got, want)


def test_batching_actually_reduces_forward_passes():
    t = source()
    one, many = FakeModel(), FakeModel()
    engine(1)._tiled(one, t, 2)
    engine(64)._tiled(many, t, 2)
    # 100 at tile 16 is 7 columns and 7 rows; the padded window is narrower only at the two
    # edges, so the 49 tiles fall into 3 x 3 = 9 distinct shapes and 9 forward passes.
    assert len(one.batches) == 49
    assert len(many.batches) == 9
    assert sum(many.batches) == 49


def test_tile_batch_divided_by_the_incoming_batch():
    """Two images at ``tile_batch`` 8 send four tiles at a time, not eight."""
    t = source(n=2)
    m = FakeModel()
    engine(8)._tiled(m, t, 2)
    assert max(m.batches) == 8  # 4 tiles x 2 images


def test_tile_batch_of_one_is_the_sequential_loop():
    t = source()
    m = FakeModel()
    engine(1)._tiled(m, t, 2)
    assert set(m.batches) == {1}


def test_oom_halves_the_cap_and_keeps_the_result(monkeypatch):
    t = source()
    m = OOMOnce()
    eng = engine(8)
    monkeypatch.setattr(torch, "cuda", type("_", (), {"empty_cache": staticmethod(lambda: None)})())
    got = eng._tiled(m, t, 2)
    assert m.raised
    # The cap is halved from the batch that actually failed, not from the cap: the group
    # that OOMs here holds 5 tiles, fewer than the cap of 8, so the new cap is 2 and one
    # halving is enough. Halving the cap instead would have taken two rounds of OOM.
    assert eng.tile_batch == 2
    assert torch.equal(got, reference(FakeModel(), t, 2, tile=16, pad=4))


def test_non_oom_runtime_errors_propagate():
    class Boom(FakeModel):
        def __call__(self, x):
            raise RuntimeError("something else entirely")

    with pytest.raises(RuntimeError, match="something else entirely"):
        engine(8)._tiled(Boom(), source(), 2)


def test_small_source_still_takes_the_single_pass_path():
    """Smaller than tile + 2 * pad in both axes: one call, no tiling, no grouping."""
    m = FakeModel()
    t = source(h=20, w=20)
    got = engine(8, tile=16, pad=4)._tiled(m, t, 2)
    assert m.batches == [1]
    assert got.shape == (1, 3, 40, 40)


class OOMOnceEvenAlone(FakeModel):
    """Raises CUDA out-of-memory on its first call at ANY batch size, then behaves."""

    def __init__(self, scale: int = 2) -> None:
        super().__init__(scale)
        self.raised = False

    def __call__(self, x):
        if not self.raised:
            self.raised = True
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return super().__call__(x)


class OOMAlways(FakeModel):
    def __call__(self, x):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


def _fake_cuda(monkeypatch):
    """`empty_cache` is a no-op on this CPU-only engine; the rest of torch.cuda stays real
    (torch.manual_seed reads torch.cuda._is_in_bad_fork)."""
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def test_tile_batching_is_off_by_default():
    """A 1024 colour texture at 4 tiles a pass wanted ~32 GB on a 12 GB card; off is the default."""
    eng = Engine(None, device="cpu", fp16=False)
    assert eng.tile_batch == 1


def test_single_tile_oom_evicts_the_other_models_and_retries(monkeypatch):
    """One tile that does not fit: the other slots' models are released, the tile is retried."""
    _fake_cuda(monkeypatch)
    t = source()
    m = OOMOnceEvenAlone()
    eng = engine(1)
    eng._models = {"color": m, "ui": FakeModel(), "face": FakeModel()}
    got = eng._tiled(m, t, 2)
    assert m.raised
    assert list(eng._models) == ["color"], "everything but the model in use is released"
    assert torch.equal(got, reference(FakeModel(), t, 2, tile=16, pad=4))


def test_single_tile_oom_with_nothing_to_evict_raises(monkeypatch):
    """No other model resident, one tile still does not fit: that is the texture's failure."""
    _fake_cuda(monkeypatch)
    m = OOMOnceEvenAlone()
    eng = engine(1)
    eng._models = {"color": m}
    with pytest.raises(RuntimeError, match="out of memory"):
        eng._tiled(m, source(), 2)


def test_single_tile_oom_after_eviction_raises(monkeypatch):
    """Eviction is tried once. If the tile still does not fit, it does not fit."""
    _fake_cuda(monkeypatch)
    m = OOMAlways()
    eng = engine(1)
    eng._models = {"color": m, "ui": FakeModel()}
    with pytest.raises(RuntimeError, match="out of memory"):
        eng._tiled(m, source(), 2)
    assert list(eng._models) == ["color"]


def test_batched_oom_halves_first_then_evicts_at_one(monkeypatch):
    """Cap 8 -> the failing batch's half -> ... -> 1 -> evict -> retry, in that order."""
    _fake_cuda(monkeypatch)

    eng = engine(8)

    class OOMUntilAloneAndEvicted(FakeModel):
        def __call__(self, x):
            if x.shape[0] > 1 or len(eng._models) > 1:
                raise RuntimeError("CUDA out of memory")
            return super().__call__(x)

    t = source()
    m = OOMUntilAloneAndEvicted()
    eng._models = {"color": m, "ui": FakeModel()}
    got = eng._tiled(m, t, 2)
    assert eng.tile_batch == 1
    assert list(eng._models) == ["color"]
    assert torch.equal(got, reference(FakeModel(), t, 2, tile=16, pad=4))


def test_single_pass_path_evicts_and_retries_too(monkeypatch):
    """The no-tiling path (source smaller than tile + 2 * pad) gets the same recovery."""
    _fake_cuda(monkeypatch)
    m = OOMOnceEvenAlone()
    eng = engine(1, tile=64, pad=4)
    eng._models = {"color": m, "ui": FakeModel()}
    t = source(h=20, w=20)
    got = eng._tiled(m, t, 2)
    assert m.raised
    assert list(eng._models) == ["color"]
    assert torch.equal(got, FakeModel()(t))


def test_evict_others_reports_whether_it_did_anything():
    eng = engine(1)
    keep = FakeModel()
    eng._models = {"a": keep}
    assert eng.evict_others(keep) is False
    eng._models = {"a": keep, "b": FakeModel()}
    assert eng.evict_others(keep) is True
    assert list(eng._models) == ["a"]
