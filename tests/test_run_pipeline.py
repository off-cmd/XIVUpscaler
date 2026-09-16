"""`clarity run` end to end against a stubbed game: the encode batching and its bookkeeping.

No game install, no models, no texconv: the game handle is a stub that serves small B8G8R8A8
textures, the engine falls back to Lanczos, and the encoder is the numpy path. What this pins
is the run loop around them -- textures come off the model into a staging batch, a full batch
goes to the encoder as one call, and every row is recorded on its own: done with its tiers on
disk, or failed with its own reason, never taken down by a neighbour in the same batch.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from clarity import cli, manifest, texio
from clarity.ffxiv import texwrite
from clarity.processing import roles

PATHS = [f"chara/equipment/e{i:04d}/texture/v01_c0101e{i:04d}_top_d.tex" for i in range(1, 6)]


def _src(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (8, 8, 4), dtype=np.uint8)


class StubGame:
    """The two things the run loop asks the game for: bytes of a path, and where it lives."""

    sqpack = "stub"

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def read(self, path: str) -> bytes:
        return self.files[path]

    def locate(self, _path: str):
        return None  # no fingerprint: the row is stamped with nothing, and that is allowed


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A manifest of five planned equipment/color rows, a stub game serving them, and a spy on
    the batched encoder that records every batch size.
    """
    srcs = {p: _src(i) for i, p in enumerate(PATHS)}
    game = StubGame({p: texwrite.write(s, mips=4) for p, s in srcs.items()})
    monkeypatch.setattr(cli.kb, "game", lambda *_a, **_k: game)
    monkeypatch.setattr(cli.kb.sqpack, "game_version", lambda _sq: "stub-version")
    monkeypatch.setattr(cli, "check_encoder", lambda _a: True)
    monkeypatch.setattr(cli, "check_disk", lambda _a, _m, _f: True)
    db = str(tmp_path / "manifest.sqlite")
    man = manifest.Manifest(db)
    for p in PATHS:
        hdr, _ = texio.read(game.read(p))
        man.add(p, "equipment", "top", "color", hdr)
    man.commit()
    man.db.close()
    sizes: list[int] = []
    real = texio.encode_tiers_many

    def spy(items):
        sizes.append(len(items))
        return real(items)

    monkeypatch.setattr(texio, "encode_tiers_many", spy)
    return {"db": db, "out": str(tmp_path / "out"), "srcs": srcs, "sizes": sizes}


def _run(world, *extra: str) -> int:
    return cli.main(
        [
            "--db",
            world["db"],
            "run",
            "--family",
            "equipment",
            "--profile",
            "legacy",
            "--top",
            "native",
            "--models",
            str(world["out"]) + "-no-models",
            "--out",
            world["out"],
            *extra,
        ]
    )


def _statuses(db: str) -> dict[str, tuple[str, str, str]]:
    man = manifest.Manifest(db)
    try:
        return {
            p: (s, t, n) for p, s, t, n in man.db.execute("SELECT path,status,tiers,note FROM tex")
        }
    finally:
        man.db.close()


def test_batches_fill_to_the_cap_and_every_row_is_done(world, capsys) -> None:
    assert _run(world, "--encode-batch", "2") == 0
    assert world["sizes"] == [2, 2, 1]
    st = _statuses(world["db"])
    assert all(v[0] == "done" and v[1] == "native" for v in st.values()), st
    mod = cli.packer.mod_for_family("equipment")
    for p, src in world["srcs"].items():
        out = (world["out"] + "/" + mod + "/" + cli.packer.file_rel("native", p)).replace("\\", "/")
        _hdr, got = texio.read(pathlib.Path(out).read_bytes())
        np.testing.assert_array_equal(got, src)  # native tier, uncompressed: the same pixels back
    out = capsys.readouterr().out
    assert "(batch of 2)" in out and "finished: 5 done, 0 failed" in out


def test_encode_batch_of_one_is_the_old_per_texture_path(world) -> None:
    assert _run(world, "--encode-batch", "1") == 0
    assert world["sizes"] == [1, 1, 1, 1, 1]
    assert all(v[0] == "done" for v in _statuses(world["db"]).values())


def test_byte_cap_splits_a_batch(world, monkeypatch) -> None:
    """The cap is measured on the UPSCALED image, so stand in a 1 MiB result for each source."""
    big = np.zeros((512, 512, 4), np.uint8)
    monkeypatch.setattr(cli.roles, "process_top", lambda *_a, **_k: big)
    assert _run(world, "--encode-batch", "32", "--encode-mb", "1") == 0
    # 1 MiB each against a 1 MiB cap: every texture flushes on its own
    assert world["sizes"] == [1, 1, 1, 1, 1]


def test_one_failed_encode_does_not_take_its_batch_down(world, monkeypatch) -> None:
    victim = PATHS[2]
    real = texio.encode_tiers_many

    def sabotage(items):
        out = real(items)
        for i, (rgba, *_r) in enumerate(items):
            if np.array_equal(rgba, world["srcs"][victim]):
                out[i] = RuntimeError("texconv said no")
        return out

    monkeypatch.setattr(texio, "encode_tiers_many", sabotage)
    assert _run(world, "--encode-batch", "5") == 0
    st = _statuses(world["db"])
    assert st[victim][0] == "failed" and "texconv said no" in st[victim][2]
    assert [st[p][0] for p in PATHS if p != victim] == ["done"] * 4


def test_a_batch_that_raises_records_the_reason_on_every_row(world, monkeypatch) -> None:
    monkeypatch.setattr(
        texio,
        "encode_tiers_many",
        lambda _items: (_ for _ in ()).throw(RuntimeError("scratch is gone")),
    )
    assert _run(world, "--encode-batch", "2") == 0
    st = _statuses(world["db"])
    assert all(v[0] == "failed" and "scratch is gone" in v[2] for v in st.values())


def test_budget_drains_what_is_through_the_model(world, monkeypatch) -> None:
    """A run that stops on --budget still encodes and records the textures already upscaled --
    including the ones sitting in a half-full batch -- and a rerun picks up the rest.
    """
    import time

    real = roles.process_top

    def slow(*args, **kw):
        time.sleep(0.05)
        return real(*args, **kw)

    monkeypatch.setattr(cli.roles, "process_top", slow)
    assert _run(world, "--encode-batch", "32", "--budget", "0.12") == 3
    st = _statuses(world["db"])
    done = [p for p, v in st.items() if v[0] == "done"]
    assert 1 <= len(done) < len(PATHS), st
    assert all(v[0] in ("done", "planned") for v in st.values())
    # a batch of 32 never filled: what was staged went out as one batch at the stop
    assert world["sizes"] == [len(done)]
    monkeypatch.setattr(cli.roles, "process_top", real)
    assert _run(world, "--encode-batch", "32") == 0
    assert all(v[0] == "done" for v in _statuses(world["db"]).values())
    assert world["sizes"] == [len(done), len(PATHS) - len(done)]
