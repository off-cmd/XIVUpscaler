"""The batched encoder: one texconv invocation for many textures, one result per texture.

``_texconv_many`` is exercised against a stand-in for texconv.exe that behaves the way the real
one does with several inputs -- converts each file it can, skips the ones it cannot, and exits
non-zero at the end if any was skipped -- so the part this module owns (inputs in, outputs out,
a missing output is the failure, retry that one alone) is pinned without the binary.
``encode_tiers_many`` is pinned on both paths: per item on the numpy path, and through the batched
call on a simulated texconv path, where its bytes must equal what ``encode_tiers`` writes.
"""

from __future__ import annotations

import os
import pathlib
import struct

import numpy as np
import pytest

from clarity import texio
from clarity.ffxiv import bc7enc, texwrite


def _noise(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 4), dtype=np.uint8)


# ---------------------------------------------------------------------------- _texconv_many


class FakeTexconv:
    """Stands in for subprocess.run on a texconv command line.

    Reads every input path after ``-o <dir>``, writes ``<dir>/<name>`` for each it "converts",
    skips any whose pixels are all zero (the "file texconv cannot convert"), and returns rc=1 if
    it skipped one -- after converting the rest, as texconv does.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **_kw):
        self.calls.append(list(cmd))
        out_dir = cmd[cmd.index("-o") + 1]
        inputs = [c for c in cmd[cmd.index("-gpu") + 2 :] if not c.startswith("-") and c != "x"]
        rc = 0
        for src in inputs:
            raw = pathlib.Path(src).read_bytes()
            if not any(raw[128:]):
                rc = 1
                continue
            pathlib.Path(out_dir, os.path.basename(src)).write_bytes(b"ENC" + raw)

        class R:
            returncode = rc
            stdout = ""
            stderr = "skipped one" if rc else ""

        return R()


@pytest.fixture
def fake_texconv(monkeypatch, tmp_path):
    fake = FakeTexconv()
    monkeypatch.setattr(texio.subprocess, "run", fake)
    monkeypatch.setattr(texio.paths, "SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setattr(texio, "TEXCONV", "texconv.exe")
    return fake


def test_many_is_one_invocation_with_every_input(fake_texconv) -> None:
    imgs = [_noise(4, 4, s) for s in range(5)]
    out = texio._texconv_many(imgs, "BC7_UNORM", 0)
    assert len(fake_texconv.calls) == 1
    assert len(out) == 5
    for got, img in zip(out, imgs, strict=True):
        assert isinstance(got, bytes)
        assert got == b"ENC" + texio._dds_rgba(img)


def test_many_of_one_is_the_single_path(fake_texconv) -> None:
    img = _noise(4, 4)
    out = texio._texconv_many([img], "BC7_UNORM", 0)
    assert out == [b"ENC" + texio._dds_rgba(img)]
    # the single path names its input in.dds and overwrites it in place
    assert fake_texconv.calls[0][-1].endswith("in.dds")


def test_one_bad_file_fails_alone_and_is_retried_alone(fake_texconv) -> None:
    good = [_noise(4, 4, 1), _noise(4, 4, 2)]
    bad = np.zeros((4, 4, 4), np.uint8)
    out = texio._texconv_many([good[0], bad, good[1]], "BC7_UNORM", 0)
    assert out[0] == b"ENC" + texio._dds_rgba(good[0])
    assert out[2] == b"ENC" + texio._dds_rgba(good[1])
    assert isinstance(out[1], RuntimeError)
    assert "texconv failed after" in str(out[1])
    # one batched call, then TEXCONV_TRIES single retries of the bad one and nothing else
    assert len(fake_texconv.calls) == 1 + texio.TEXCONV_TRIES
    for call in fake_texconv.calls[1:]:
        assert call[-1].endswith("in.dds")


def test_empty_batch(fake_texconv) -> None:
    assert texio._texconv_many([], "BC7_UNORM", 0) == []
    assert fake_texconv.calls == []


def test_inputs_and_outputs_are_separate_directories(fake_texconv) -> None:
    """A failed conversion leaves the input where it was; only a real output counts."""
    texio._texconv_many([_noise(4, 4, 1), _noise(4, 4, 2)], "BC1_UNORM", 3)
    cmd = fake_texconv.calls[0]
    out_dir = cmd[cmd.index("-o") + 1]
    inputs = cmd[cmd.index("-gpu") + 2 :]
    assert all(os.path.dirname(i) != out_dir for i in inputs)
    assert cmd[cmd.index("-m") + 1] == "3"
    assert "-bc" not in cmd  # the 3-subset switch is BC7 only


# ---------------------------------------------------------------------------- encode_tiers_many


def test_many_equals_encode_tiers_per_item_on_the_numpy_path() -> None:
    items = [
        (_noise(8, 8, 1), texio.BC7, None, (0, 1), True),
        (_noise(8, 4, 2), texio.BGRA8, texwrite.ATTR_2D, (0,), False),
        (_noise(4, 8, 3), texio.BC3, None, (0, 1, 2), True),
    ]
    got = texio.encode_tiers_many(items)
    assert len(got) == 3
    for res, (rgba, fmt, attr, offsets, keep) in zip(got, items, strict=True):
        assert res == texio.encode_tiers(rgba, fmt, attr, offsets=offsets, keep_mips=keep)


def test_one_bad_item_does_not_fail_the_batch() -> None:
    items = [
        (_noise(4, 4, 1), texio.BGRA8, None, (0,), True),
        (_noise(4, 4, 2), 0x9999, None, (0,), True),
        (_noise(4, 4, 3), texio.BGRA8, None, (0,), True),
    ]
    got = texio.encode_tiers_many(items)
    assert isinstance(got[0], dict) and isinstance(got[2], dict)
    assert isinstance(got[1], ValueError)


def _dx10_dds(levels: list[bytes], w: int, h: int) -> bytes:
    """What texconv hands back: a DX10-header DDS whose payload is the block chain."""
    hdr = bytearray(124)
    struct.pack_into("<3I", hdr, 0, 124, 0x1 | 0x2 | 0x4 | 0x1000, h)
    struct.pack_into("<I", hdr, 12, w)
    hdr[80:84] = b"DX10"  # pixel format fourcc lives at header offset 80 (file offset 84)
    return b"DDS " + bytes(hdr) + bytes(20) + b"".join(levels)


def test_texconv_path_cuts_the_same_tiers_as_encode_tiers(monkeypatch) -> None:
    """With texconv 'available', one batched call per format and bytes equal to the per-item path."""
    calls: list[tuple[str, int]] = []

    def fake_many(rgbas, fmt_name, mips):
        calls.append((fmt_name, len(rgbas)))
        assert fmt_name == "BC7_UNORM" and mips == 0
        out = []
        for a in rgbas:
            h, w = a.shape[:2]
            chain = texio.float_chain(a, texio.full_mips(w, h))
            out.append(_dx10_dds([bc7enc.encode(L) for L in chain], w, h))
        return out

    items = [
        (_noise(8, 8, 1), texio.BC7, None, (0, 1), True),
        (_noise(8, 8, 2), texio.BC7, None, (0,), False),
        (_noise(4, 4, 3), texio.BGRA8, None, (0, 1), True),
    ]
    want = [texio.encode_tiers(r, f, at, offsets=o, keep_mips=k) for r, f, at, o, k in items]
    monkeypatch.setattr(texio, "use_texconv", lambda: True)
    monkeypatch.setattr(texio, "_texconv_many", fake_many)
    got = texio.encode_tiers_many(items)
    assert calls == [("BC7_UNORM", 2)]  # the two BC7 items in one call; BGRA8 never goes there
    assert got == want
