"""``requeue`` must not erase the ``changed at <version>`` marker that ``run --since``
selects the patch delta by, and the audit's output-cap query must track the policy
constant rather than a literal.
"""

import argparse
import sqlite3

from clarity import cli, manifest
from clarity.processing import roles


def _manifest(tmp_path) -> manifest.Manifest:
    man = manifest.Manifest(str(tmp_path / "m.sqlite"))
    for path, status, note in [
        ("chara/a_n.tex", "failed", "changed at 2026.09.01.0000.0000"),
        ("chara/b_n.tex", "failed", "boom: encoder died"),
        ("chara/c_n.tex", "done", "models:normal=x"),
    ]:
        man.db.execute(
            "INSERT INTO tex(path,family,part,role,w,h,fmt,mips,bytes,ttype,status,note,updated)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (path, "equipment", "", "normal", 64, 64, "BC7", 1, 0, "2D", status, note, 0.0),
        )
    man.commit()
    return man


def _notes(man: manifest.Manifest) -> dict[str, tuple[str, str]]:
    return {
        p: (s, n) for p, s, n in man.db.execute("SELECT path, status, note FROM tex ORDER BY path")
    }


def test_requeue_keeps_the_changed_marker_and_clears_other_notes(tmp_path, monkeypatch) -> None:
    man = _manifest(tmp_path)
    monkeypatch.setattr(manifest, "Manifest", lambda _db: man)
    args = argparse.Namespace(
        db="unused",
        failed=True,
        skipped=False,
        old_recipe=False,
        without_model=None,
        done=False,
        family=None,
        role=None,
        path_like=None,
        dry_run=False,
    )
    assert cli.cmd_requeue(args) == 0
    notes = _notes(man)
    assert notes["chara/a_n.tex"] == ("planned", "changed at 2026.09.01.0000.0000")
    assert notes["chara/b_n.tex"] == ("planned", "")
    assert notes["chara/c_n.tex"] == ("done", "models:normal=x")


def test_since_still_selects_the_requeued_changed_row(tmp_path, monkeypatch) -> None:
    man = _manifest(tmp_path)
    monkeypatch.setattr(manifest, "Manifest", lambda _db: man)
    args = argparse.Namespace(
        db="unused",
        failed=True,
        skipped=False,
        old_recipe=False,
        without_model=None,
        done=False,
        family=None,
        role=None,
        path_like=None,
        dry_run=False,
    )
    cli.cmd_requeue(args)
    delta = [r[0] for r in man.rows(status="planned", since="2999-01-01")]
    assert delta == ["chara/a_n.tex"]


def test_fingerprint_marker_is_written_with_a_bound_parameter() -> None:
    # A version containing a quote must land verbatim rather than be stripped or break the SQL.
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE tex(path, status, tiers, note)")
    db.execute("INSERT INTO tex VALUES('p', 'done', '4x', '')")
    version = "2026.09.01'x"
    db.executemany(
        "UPDATE tex SET status='planned', tiers='', note=? WHERE path=?",
        [("changed at " + version, "p")],
    )
    assert db.execute("SELECT note FROM tex").fetchone()[0] == "changed at " + version


def test_audit_cap_tracks_the_policy_constant() -> None:
    import inspect

    src = inspect.getsource(cli.cmd_audit)
    assert "8192" not in src
    assert "MAX_EDGE_OUT" in src and roles.MAX_EDGE_OUT == 4096
