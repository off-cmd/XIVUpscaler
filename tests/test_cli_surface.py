"""The argparse surface of ``clarity`` and the handlers that need only a manifest.

Pins that every subcommand's parser constructs (``--help`` exits 0 for each of them, which is
the cheapest proof that no ``add_argument`` call is broken), that ``main`` dispatches and
returns the handler's exit code, and the manifest-only handlers: ``reclassify`` (dry run
reports and writes nothing; a real run rewrites family/part/role and nothing else, and leaves
header-skipped rows alone), ``audit``, ``estimate`` and ``where``. ``run``, ``qa``, ``probe``,
``modup`` and ``fingerprint`` need a game install, a GPU or texconv and are only parsed here.
"""

import argparse
import re

import pytest

from clarity import cli, manifest, paths

SUBCOMMANDS = [
    "plan",
    "estimate",
    "run",
    "pack",
    "probe",
    "qa",
    "requeue",
    "reclassify",
    "audit",
    "fingerprint",
    "where",
    "modup",
    "release",
]


# --------------------------------------------------------------------------------- parser
@pytest.mark.parametrize("cmd", SUBCOMMANDS)
def test_every_subcommand_parses_help(cmd, capsys) -> None:
    with pytest.raises(SystemExit) as e:
        cli.main([cmd, "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith(f"usage: clarity {cmd}")


def test_top_level_help(capsys) -> None:
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for cmd in SUBCOMMANDS:
        assert cmd in out, cmd


def test_a_subcommand_is_required(capsys) -> None:
    with pytest.raises(SystemExit) as e:
        cli.main([])
    assert e.value.code == 2
    assert "required" in capsys.readouterr().err


def test_an_unknown_subcommand_is_rejected(capsys) -> None:
    with pytest.raises(SystemExit) as e:
        cli.main(["frobnicate"])
    assert e.value.code == 2


def test_db_default_is_the_resolved_layout() -> None:
    # `--db` sits on the top-level parser, before the subcommand.
    with pytest.raises(SystemExit):
        cli.main(["--db", "x.sqlite", "audit", "--help"])
    with pytest.raises(SystemExit) as e:
        cli.main(["audit", "--db", "x.sqlite"])
    assert e.value.code == 2


def test_modup_requires_a_mod(capsys) -> None:
    with pytest.raises(SystemExit) as e:
        cli.main(["modup"])
    assert e.value.code == 2
    assert "--mod" in capsys.readouterr().err


def test_run_profile_choices(capsys) -> None:
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "--profile", "sometimes"])
    assert e.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


# ------------------------------------------------------------------------------- manifest
def _insert(man, path, family, part, role, w=64, h=64, fmt="BC7", ttype="2D", status="planned"):
    man.db.execute(
        "INSERT INTO tex(path,family,part,role,w,h,fmt,mips,bytes,ttype,status,tiers,note,updated)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (path, family, part, role, w, h, fmt, 1, w * h, ttype, status, "", "", 0.0),
    )


HOU = "bg/ffxiv/sea_s1/hou/s1h1/texture/s1h1_wall_d.tex"
BIG_ICON = "ui/icon/070000/070001_hr1.tex"
LOADING = "common/graphics/texture/-nowloading_base01.tex"
FINE = "chara/equipment/e0001/material/v0001/mt_c0101e0001_top_n.tex"
ARRAY = "chara/equipment/e0002/material/v0001/mt_c0101e0002_top_n.tex"


def _stale_manifest(tmp_path) -> manifest.Manifest:
    """Rows whose stored classification predates the current classifier."""
    man = manifest.Manifest(str(tmp_path / "m.sqlite"))
    # The old housing split: family bg, part hou, role other -- now bg-hou / s1h1 / color.
    _insert(man, HOU, "bg", "hou", "other")
    # A 1024-wide icon scanned before the `width > 512 -> ui` rule.
    _insert(man, BIG_ICON, "ui-icon", "", "icon", w=1024, h=1024)
    # A loading screen filed as `other` before the -nowloading rule; also marked done.
    _insert(man, LOADING, "common", "", "other", w=1920, h=1080, fmt="BC1", status="done")
    # Already correct.
    _insert(man, FINE, "equipment", "", "normal")
    # Header-skipped, with a deliberately wrong family: reclassify must not touch role=skip rows.
    _insert(man, ARRAY, "weapon", "", "skip", ttype="2DArray")
    man.db.execute("UPDATE tex SET tiers='native', note='models:x' WHERE path=?", (LOADING,))
    man.commit()
    return man


def _classification(man) -> dict[str, tuple]:
    return {
        r[0]: r[1:]
        for r in man.db.execute(
            "SELECT path, family, part, role, status, tiers, note FROM tex ORDER BY path"
        )
    }


def _reclassify_args(**kw) -> argparse.Namespace:
    base = {
        "db": "unused",
        "family": None,
        "path_like": None,
        "sync_status": False,
        "dry_run": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


def test_reclassify_dry_run_reports_and_writes_nothing(tmp_path, monkeypatch, capsys) -> None:
    man = _stale_manifest(tmp_path)
    monkeypatch.setattr(manifest, "Manifest", lambda _db: man)
    before = _classification(man)
    assert cli.cmd_reclassify(_reclassify_args(dry_run=True)) == 0
    out = capsys.readouterr().out
    assert "3 row(s)" in out
    assert "dry run, nothing written" in out
    assert "bg[hou]/other -> bg-hou[s1h1]/color" in out
    assert "ui-icon[-]/icon -> ui-icon[-]/ui" in out
    assert "common[-]/other -> common[-]/ui" in out
    assert _classification(man) == before


def test_reclassify_rewrites_family_part_role_and_nothing_else(
    tmp_path, monkeypatch, capsys
) -> None:
    man = _stale_manifest(tmp_path)
    monkeypatch.setattr(manifest, "Manifest", lambda _db: man)
    assert cli.cmd_reclassify(_reclassify_args()) == 0
    assert "written" in capsys.readouterr().out
    after = _classification(man)
    assert after[HOU] == ("bg-hou", "s1h1", "color", "planned", "", "")
    assert after[BIG_ICON] == ("ui-icon", "", "ui", "planned", "", "")
    # Status, tiers and note survive: a done row stays done.
    assert after[LOADING] == ("common", "", "ui", "done", "native", "models:x")
    assert after[FINE] == ("equipment", "", "normal", "planned", "", "")
    # The header verdict is not re-derivable from the path, so the row is left as it was.
    assert after[ARRAY] == ("weapon", "", "skip", "planned", "", "")
    # The write is committed.
    assert _classification(manifest.Manifest(man.path)) == after


def test_reclassify_is_idempotent(tmp_path, monkeypatch, capsys) -> None:
    man = _stale_manifest(tmp_path)
    monkeypatch.setattr(manifest, "Manifest", lambda _db: man)
    cli.cmd_reclassify(_reclassify_args())
    capsys.readouterr()
    assert cli.cmd_reclassify(_reclassify_args()) == 0
    assert "nothing to reclassify" in capsys.readouterr().out


def test_reclassify_family_and_path_filters(tmp_path, monkeypatch, capsys) -> None:
    man = _stale_manifest(tmp_path)
    monkeypatch.setattr(manifest, "Manifest", lambda _db: man)
    assert cli.cmd_reclassify(_reclassify_args(family=["ui-icon"])) == 0
    assert "1 row(s)" in capsys.readouterr().out
    after = _classification(man)
    assert after[BIG_ICON][2] == "ui" and after[HOU][0] == "bg"
    assert cli.cmd_reclassify(_reclassify_args(path_like="bg/%")) == 0
    assert "1 row(s)" in capsys.readouterr().out
    assert _classification(man)[HOU][0] == "bg-hou"
    assert _classification(man)[LOADING][2] == "other"


def test_reclassify_sync_status_marks_rows_with_no_path(tmp_path, monkeypatch, capsys) -> None:
    man = _stale_manifest(tmp_path)
    _insert(man, "chara/equipment/e0003/material/v0001/mt_x_id.tex", "equipment", "", "id")
    man.commit()
    monkeypatch.setattr(manifest, "Manifest", lambda _db: man)
    assert cli.cmd_reclassify(_reclassify_args(sync_status=True)) == 0
    out = capsys.readouterr().out
    assert "status sync: 2 planned row(s) marked skipped" in out
    assert "index map" in out
    after = _classification(man)
    assert after["chara/equipment/e0003/material/v0001/mt_x_id.tex"][3] == "skipped"
    assert after[ARRAY][3] == "skipped"  # role skip, planned: flipped by the sync
    assert after[BIG_ICON][3] == "planned"  # ui-icon/ui has a path
    assert after[HOU][3] == "planned"
    assert after[LOADING][3] == "done"  # never touched: it is not planned


# ---------------------------------------------------------------------------------- audit
def _audit_manifest(tmp_path) -> manifest.Manifest:
    man = manifest.Manifest(str(tmp_path / "m.sqlite"))
    _insert(man, FINE, "equipment", "", "normal", status="done")
    _insert(man, "chara/x_m.tex", "equipment", "", "mask", w=4, h=4)  # below the 8 px pad
    _insert(man, "chara/y_d.tex", "equipment", "", "color", w=6, h=10)  # not a multiple of 4
    _insert(man, "chara/z_d.tex", "equipment", "", "color", fmt="R16G16F")  # no decoder
    _insert(man, "chara/big_d.tex", "monster", "", "color", w=2048, h=2048)  # at the cap
    _insert(man, ARRAY, "equipment", "", "skip", ttype="2DArray")
    man.db.execute("UPDATE tex SET note='models:normal=x' WHERE path=?", (FINE,))
    man.commit()
    return man


def test_audit_reports_every_section(tmp_path, monkeypatch, capsys) -> None:
    man = _audit_manifest(tmp_path)
    monkeypatch.setattr(manifest, "Manifest", lambda _db: man)
    assert cli.cmd_audit(argparse.Namespace(db="unused")) == 0
    out = capsys.readouterr().out
    assert "status: {'done': 1, 'planned': 5}" in out
    assert re.search(r"R16G16F\s+equipment\s+color\s+1\s+<-- would fail", out)
    assert re.search(r"^\s+4x4\s+BC7\s+mask\s+1$", out, re.M)
    assert re.search(r"^\s+6x10\s+color\s+1$", out, re.M)
    assert re.search(r"^\s+2DArray\s+1$", out, re.M)
    assert "mips=1  6 of 6" in out
    assert "at or above the 4096 output cap" in out
    assert re.search(r"^\s+2048x2048\s+1$", out, re.M)
    assert re.search(r"^\s+registry recipe\s+1$", out, re.M)


def test_audit_on_an_empty_manifest(tmp_path, monkeypatch, capsys) -> None:
    man = manifest.Manifest(str(tmp_path / "m.sqlite"))
    monkeypatch.setattr(manifest, "Manifest", lambda _db: man)
    assert cli.cmd_audit(argparse.Namespace(db="unused")) == 0
    out = capsys.readouterr().out
    assert "status: {}" in out and "  none" in out and "mips=1  0 of 0" in out


# ------------------------------------------------------------------------------- estimate
def _estimate_manifest(tmp_path) -> manifest.Manifest:
    man = manifest.Manifest(str(tmp_path / "m.sqlite"))
    _insert(man, FINE, "equipment", "", "normal", w=1024, h=1024)
    _insert(man, "chara/a_id.tex", "equipment", "", "id", w=1024, h=1024)
    _insert(man, "chara/h_n.tex", "human-hair", "hair", "normal", w=1024, h=1024)
    _insert(man, "ui/icon/000000/000001.tex", "ui-icon", "", "icon", w=40, h=40, fmt="B8G8R8A8")
    man.commit()
    return man


def _lines(out: str) -> dict[str, list[str]]:
    return {ln.split()[0]: ln.split() for ln in out.splitlines() if ln.strip()}


def test_estimate_prints_a_table(tmp_path, monkeypatch, capsys) -> None:
    man = _estimate_manifest(tmp_path)
    monkeypatch.setattr(manifest, "Manifest", lambda _db: man)
    cli.cmd_estimate(argparse.Namespace(db="unused", top=None))
    out = capsys.readouterr().out
    rows = _lines(out)
    assert rows["family"] == [
        "family",
        "role",
        "status",
        "n",
        "src",
        "MB",
        "native",
        "MB",
        "2x",
        "MB",
        "4x",
        "MB",
    ]
    # equipment/normal 1024^2 BC7: 4x -> native 1.4 MB, 2x 5.6 MB, 4x 22.3 MB (x1.33, 1 bpp).
    assert rows["equipment"][:4] == ["equipment", "normal", "planned", "1"]
    assert rows["equipment"][5:8] == ["1", "6", "22"]
    # id rows are listed with dashes; a family without a tier contributes nothing.
    assert "id" in out and out.count("-") >= 3
    id_line = next(ln for ln in out.splitlines() if " id " in ln)
    assert id_line.split()[5:] == ["-", "-", "-"]
    assert rows["human-hair"][5:8] == ["0", "0", "0"]
    assert rows["TOTAL"][0] == "TOTAL" and "each tier is a full separate set" in out


def test_estimate_honours_the_top_override(tmp_path, monkeypatch, capsys) -> None:
    man = _estimate_manifest(tmp_path)
    monkeypatch.setattr(manifest, "Manifest", lambda _db: man)
    cli.cmd_estimate(argparse.Namespace(db="unused", top="2x"))
    rows = _lines(capsys.readouterr().out)
    assert rows["equipment"][5:8] == ["1", "6", "0"]
    assert rows["ui-icon"][7] == "0"


# ------------------------------------------------------------------------------- dispatch
def test_main_dispatches_estimate_against_the_db_flag(tmp_path, capsys) -> None:
    db = tmp_path / "m.sqlite"
    man = manifest.Manifest(str(db))
    _insert(man, FINE, "equipment", "", "normal", w=512, h=512)
    man.commit()
    assert cli.main(["--db", str(db), "estimate"]) == 0
    out = capsys.readouterr().out
    assert "equipment      normal  planned        1" in out


def test_main_returns_the_handlers_exit_code(tmp_path, capsys) -> None:
    db = str(tmp_path / "m.sqlite")
    assert cli.main(["--db", db, "requeue"]) == 2  # nothing selected
    assert "nothing selected" in capsys.readouterr().out
    assert cli.main(["--db", db, "requeue", "--failed", "--dry-run"]) == 0
    assert cli.main(["--db", db, "audit"]) == 0
    assert cli.main(["--db", db, "reclassify", "--dry-run"]) == 0
    assert "nothing to reclassify" in capsys.readouterr().out


def test_where_prints_the_resolved_layout(capsys) -> None:
    assert cli.main(["where"]) == 0
    out = capsys.readouterr().out
    assert out.strip() == paths.describe().strip()
    for key in ("project", "db", "models", "registry", "texconv", "scratch", "fingerprints"):
        assert key in out, key
    assert paths.PROJECT in out
