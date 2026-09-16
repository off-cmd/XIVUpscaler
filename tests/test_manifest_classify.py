"""The classifier, the skip policy and the manifest's own read/write surface.

Pins: ``classify(path, hdr)`` -> (family, part, role) for every family prefix and every
suffix rule, including the housing split inside ``bg/`` and ``bgcommon/`` (an earlier version
tested ``family in ("bg", "bgcommon")`` and sent every ``bg-hou`` texture to role ``other``),
the ``common/-nowloading*`` loading-screen rule and the header overrides; ``skip_reason`` and
``sync_skipped`` as the one written-down policy for rows the run never walks; and
``Manifest.add/rows/set_status/set_hash/summary/snapshot`` against a real sqlite file.
"""

import struct
import time

import pytest

from clarity import manifest
from clarity.ffxiv import tex, texwrite
from clarity.manifest import Manifest, StoredHeader, classify, skip_reason, sync_skipped

BC7, R32F, BGRA8 = 0x6432, 0x2150, 0x1450


def _tex_header(
    w: int = 64, h: int = 64, fmt: int = BC7, mips: int = 1, depth: int = 1, attr: int | None = None
) -> tex.TexHeader:
    """A real TexHeader from real header bytes, so Manifest.add sees what a scan sees."""
    hdr = bytearray(texwrite.HDR)
    struct.pack_into(
        "<IIHHHBB",
        hdr,
        0,
        texwrite.ATTR_2D if attr is None else attr,
        fmt,
        w,
        h,
        depth,
        mips,
        1,
    )
    return tex.TexHeader(bytes(hdr))


# ----------------------------------------------------------------------------- classify
EQ = "chara/equipment/e0001/material/v0001/mt_c0101e0001_top_{}.tex"

PATH_TABLE = [
    # family prefixes
    (EQ.format("n"), ("equipment", "", "normal")),
    (
        "chara/accessory/a0001/material/v0001/mt_c0101a0001_ear_d.tex",
        ("accessory", "", "color"),
    ),
    (
        "chara/weapon/w0001/obj/body/b0001/material/v0001/mt_w0001b0001_a_m.tex",
        ("weapon", "", "mask"),
    ),
    (
        "chara/monster/m0001/obj/body/b0001/material/v0001/mt_m0001b0001_a_id.tex",
        ("monster", "", "id"),
    ),
    (
        "chara/demihuman/d0001/obj/equipment/e0001/material/v0001/mt_d0001e0001_top_s.tex",
        ("demihuman", "", "color"),
    ),
    # human: the part refines the family and is stored in `part`
    (
        "chara/human/c0101/obj/body/b0001/material/v0001/mt_c0101b0001_a_n.tex",
        ("human-body", "body", "normal"),
    ),
    (
        "chara/human/c0101/obj/face/f0001/material/mt_c0101f0001_fac_d.tex",
        ("human-face", "face", "color"),
    ),
    (
        "chara/human/c0101/obj/hair/h0001/material/v0001/mt_c0101h0001_hir_n.tex",
        ("human-hair", "hair", "normal"),
    ),
    (
        "chara/human/c0101/obj/tail/t0001/material/v0001/mt_c0101t0001_a_m.tex",
        ("human-tail", "tail", "mask"),
    ),
    (
        "chara/human/c0101/obj/zear/z0001/material/v0001/mt_c0101z0001_a_id.tex",
        ("human-zear", "zear", "id"),
    ),
    ("chara/human/c0101/texture/foo_d.tex", ("human-other", "", "color")),
    # bg: segment 3 is the zone kind; hou/ind split out, everything else keeps `bg`
    ("bg/ffxiv/sea_s1/fld/s1f1/texture/s1f1_a0_rock1_d.tex", ("bg", "fld", "color")),
    ("bg/ffxiv/sea_s1/twn/s1t1/texture/s1t1_a0_wall_n.tex", ("bg", "twn", "normal")),
    ("bg/ffxiv/sea_s1/hou/s1h1/texture/s1h1_wall_n.tex", ("bg-hou", "s1h1", "normal")),
    ("bg/ffxiv/fst_f1/ind/f1i1/texture/f1i1_floor_m.tex", ("bg-ind", "f1i1", "mask")),
    ("bg/x.tex", ("bg", "", "color")),
    # bgcommon: segment 1 is the group; hou and mji split out
    ("bgcommon/nature/grass/texture/grass_d.tex", ("bgcommon", "grass", "color")),
    (
        "bgcommon/hou/indoor/general/0001/texture/fun_b_m_d.tex",
        ("bgcommon-hou", "indoor", "color"),
    ),
    ("bgcommon/hou/outdoor/general/0001/texture/gar_n.tex", ("bgcommon-hou", "outdoor", "normal")),
    ("bgcommon/mji/farm/texture/mji_crop_n.tex", ("bgcommon-mji", "farm", "normal")),
    # ui: three families, two roles; the suffix is never consulted
    ("ui/icon/000000/000001.tex", ("ui-icon", "", "icon")),
    ("ui/icon/070000/en/070001_hr1.tex", ("ui-icon", "", "icon")),
    ("ui/uld/parameter_gauge.tex", ("ui-uld", "", "ui")),
    ("ui/map/s1f1/00/s1f1_m.tex", ("ui-other", "", "ui")),
    # vfx is always skip; common is `other` except the loading screens
    ("vfx/common/texture/foo_n.tex", ("vfx", "", "skip")),
    ("common/graphics/texture/-nowloading_base01.tex", ("common", "", "ui")),
    ("common/graphics/texture/-fresnel.tex", ("common", "", "other")),
    ("common/font/axis_12.tex", ("common", "", "other")),
    # no known prefix at all
    ("shader/foo_d.tex", ("other", "", "color")),
    ("shader/foo.tex", ("other", "", "other")),
]


@pytest.mark.parametrize(("path", "expected"), PATH_TABLE, ids=[p for p, _ in PATH_TABLE])
def test_classify_by_path(path, expected) -> None:
    assert classify(path, None) == expected
    # A plain 2D header adds nothing the path did not already decide.
    assert classify(path, StoredHeader(64, 64, "BC7", "2D")) == expected


@pytest.mark.parametrize(("suffix", "role"), sorted(manifest.SUFFIX_ROLE.items()))
def test_every_suffix_rule(suffix, role) -> None:
    assert classify(EQ.format(suffix), None) == ("equipment", "", role)


@pytest.mark.parametrize(
    "path",
    [
        "bg/ffxiv/sea_s1/hou/s1h1/texture/wall.tex",
        "bg/ffxiv/fst_f1/ind/f1i1/texture/floor.tex",
        "bgcommon/hou/indoor/general/0001/texture/fun.tex",
        "bgcommon/mji/farm/texture/crop.tex",
        "bg/ffxiv/sea_s1/fld/s1f1/texture/rock.tex",
        "bgcommon/nature/grass/texture/grass.tex",
    ],
)
def test_world_paths_without_a_known_suffix_default_to_color(path) -> None:
    # Regression: `family in ("bg", "bgcommon")` demoted every housing texture to `other`.
    assert classify(path, None)[2] == "color"


def test_family_split_does_not_change_the_role() -> None:
    plain = classify("bg/ffxiv/sea_s1/fld/s1f1/texture/a_n.tex", None)
    housing = classify("bg/ffxiv/sea_s1/hou/s1h1/texture/a_n.tex", None)
    assert plain[0] == "bg" and housing[0] == "bg-hou"
    assert plain[2] == housing[2] == "normal"


def test_unknown_suffix_outside_the_world_is_other() -> None:
    assert classify(EQ.format("catc"), None) == ("equipment", "", "other")
    assert classify(EQ.format("mult"), None) == ("equipment", "", "other")


@pytest.mark.parametrize(
    "hdr",
    [
        StoredHeader(64, 64, "BC7", "2DArray"),
        StoredHeader(64, 64, "BC7", "Cube"),
        StoredHeader(64, 64, "BC7", "3D"),
        StoredHeader(64, 64, "R32F", "2D"),
        StoredHeader(64, 64, "BC6H", "2D"),
        StoredHeader(64, 64, "BC4", "2D"),
        StoredHeader(64, 64, "D24S8", "2D"),
        _tex_header(64, 64, BC7, depth=2),
    ],
    ids=["2DArray", "Cube", "3D", "R32F", "BC6H", "BC4", "D24S8", "depth=2"],
)
def test_header_overrides_the_path_to_skip(hdr) -> None:
    assert classify(EQ.format("n"), hdr)[2] == "skip"
    assert classify("ui/icon/000000/000001.tex", hdr)[2] == "skip"
    assert classify("bg/ffxiv/sea_s1/hou/s1h1/texture/wall_d.tex", hdr)[2] == "skip"


@pytest.mark.parametrize("fmt", sorted(manifest.SKIP_FORMATS))
def test_every_skip_format_skips(fmt) -> None:
    assert classify(EQ.format("d"), StoredHeader(64, 64, fmt, "2D"))[2] == "skip"


def test_a_plain_2d_tex_header_does_not_skip() -> None:
    assert classify(EQ.format("n"), _tex_header())[2] == "normal"
    assert _tex_header().texture_type == "2D"


def test_big_icons_become_ui() -> None:
    p = "ui/icon/070000/070001_hr1.tex"
    assert classify(p, StoredHeader(512, 512, "BC7", "2D")) == ("ui-icon", "", "icon")
    assert classify(p, StoredHeader(513, 512, "BC7", "2D")) == ("ui-icon", "", "ui")
    assert classify(p, StoredHeader(1024, 1024, "BC1", "2D")) == ("ui-icon", "", "ui")
    # Only the width is consulted.
    assert classify(p, StoredHeader(512, 1024, "BC7", "2D")) == ("ui-icon", "", "icon")


def test_stored_header_defaults() -> None:
    h = StoredHeader(None, None, None, None)
    assert (h.width, h.height, h.format_name, h.texture_type, h.depth) == (0, 0, "", "2D", 1)


# -------------------------------------------------------------------------- skip_reason
@pytest.mark.parametrize(
    ("family", "role", "ttype", "fragment"),
    [
        ("equipment", "id", "2D", "index map"),
        ("equipment", "skip", "2D", "not a 2D colour texture"),
        ("equipment", "other", "2D", "no suffix rule"),
        ("equipment", "weird", "2D", "role 'weird' has no processing function"),
        ("equipment", "normal", "2DArray", "texture type 2DArray"),
        ("ui-icon", "icon", "Cube", "texture type Cube"),
        ("human-hair", "normal", "2D", "family 'human-hair' has no tier"),
        ("vfx", "color", "2D", "family 'vfx' has no tier"),
        ("common", "ui", "2D", "family 'common' has no tier"),
        ("no-such-family", "color", None, "family 'no-such-family' has no tier"),
    ],
)
def test_skip_reason_names_the_reason(family, role, ttype, fragment) -> None:
    r = skip_reason(family, role, ttype)
    assert isinstance(r, str) and fragment in r


@pytest.mark.parametrize(
    ("family", "role"),
    [
        ("equipment", "normal"),
        ("equipment", "mask"),
        ("equipment", "color"),
        ("ui-icon", "icon"),
        ("ui-uld", "ui"),
        ("bg-hou", "color"),
        ("bgcommon-mji", "normal"),
        ("human-face", "normal"),
    ],
)
def test_skip_reason_is_none_for_processed_rows(family, role) -> None:
    assert skip_reason(family, role, "2D") is None
    assert skip_reason(family, role) is None


def test_role_gate_wins_over_family_gate() -> None:
    # A row that fails both is reported for its role, which is what `note` should say.
    assert skip_reason("vfx", "skip", "2D").startswith("not a 2D colour texture")


# ------------------------------------------------------------------------- sync_skipped
def _raw_insert(man: Manifest, path: str, family: str, role: str, status: str, ttype="2D") -> None:
    man.db.execute(
        "INSERT INTO tex(path,family,part,role,w,h,fmt,mips,bytes,ttype,status,note,updated)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (path, family, "", role, 64, 64, "BC7", 1, 0, ttype, status, "", 0.0),
    )


def _status_note(man: Manifest) -> dict[str, tuple[str, str]]:
    return {
        p: (s, n) for p, s, n in man.db.execute("SELECT path, status, note FROM tex ORDER BY path")
    }


def _mixed_manifest(tmp_path) -> Manifest:
    man = Manifest(str(tmp_path / "m.sqlite"))
    _raw_insert(man, "a_keep_n.tex", "equipment", "normal", "planned")
    _raw_insert(man, "b_id.tex", "equipment", "id", "planned")
    _raw_insert(man, "c_vfx.tex", "vfx", "skip", "planned")
    _raw_insert(man, "d_hair_n.tex", "human-hair", "normal", "planned")
    _raw_insert(man, "e_cube_n.tex", "equipment", "normal", "planned", ttype="Cube")
    _raw_insert(man, "f_done_id.tex", "equipment", "id", "done")
    _raw_insert(man, "g_failed_other.tex", "equipment", "other", "failed")
    man.commit()
    return man


def test_sync_skipped_flips_exactly_the_planned_rows_with_no_path(tmp_path) -> None:
    man = _mixed_manifest(tmp_path)
    n, why = sync_skipped(man)
    assert n == 4
    assert dict(why) == {
        "index map": 1,
        "not a 2D colour texture the role functions can operate on (array, cube, depth, float format, or vfx)": 1,
        "family 'human-hair' has no tier in roles.POLICY": 1,
        "texture type Cube": 1,
    }
    got = _status_note(man)
    assert got["a_keep_n.tex"] == ("planned", "")
    assert got["b_id.tex"][0] == "skipped" and got["b_id.tex"][1].startswith("index map")
    assert got["c_vfx.tex"][0] == "skipped"
    assert got["d_hair_n.tex"] == ("skipped", "family 'human-hair' has no tier in roles.POLICY")
    assert got["e_cube_n.tex"] == (
        "skipped",
        "texture type Cube: the role functions are 2D operations",
    )
    # done / failed rows describe something that happened and are not touched.
    assert got["f_done_id.tex"] == ("done", "")
    assert got["g_failed_other.tex"] == ("failed", "")
    # The write is committed: a second connection sees it.
    other = Manifest(man.path)
    assert other.count(status="skipped") == 4


def test_sync_skipped_dry_run_counts_but_writes_nothing(tmp_path) -> None:
    man = _mixed_manifest(tmp_path)
    before = _status_note(man)
    n, why = sync_skipped(man, dry_run=True)
    assert n == 4 and sum(why.values()) == 4
    assert _status_note(man) == before


def test_sync_skipped_is_idempotent(tmp_path) -> None:
    man = _mixed_manifest(tmp_path)
    sync_skipped(man)
    assert sync_skipped(man) == (0, {})


# ------------------------------------------------------------------------------ Manifest
def test_add_inserts_once_and_keeps_the_first_verdict(tmp_path) -> None:
    man = Manifest(str(tmp_path / "m.sqlite"))
    hdr = _tex_header(128, 64, BC7, mips=3)
    man.add("p.tex", "equipment", "", "normal", hdr)
    man.add("p.tex", "weapon", "x", "color", hdr)  # INSERT OR IGNORE: a no-op
    man.commit()
    assert man.count() == 1
    (row,) = man.rows()
    assert row == ("p.tex", "equipment", "", "normal", 128, 64, "BC7", 3, "planned", "")
    (b, tt, note) = man.db.execute("SELECT bytes, ttype, note FROM tex").fetchone()
    assert b == hdr.total_size() and tt == "2D" and note == ""


def test_add_writes_the_skip_verdict_at_insert_time(tmp_path) -> None:
    man = Manifest(str(tmp_path / "m.sqlite"))
    man.add("id.tex", "equipment", "", "id", _tex_header())
    man.add("hair.tex", "human-hair", "hair", "normal", _tex_header())
    man.add("arr.tex", "equipment", "", "normal", _tex_header(attr=1 << 28))  # 2DArray
    man.add("ok.tex", "equipment", "", "normal", _tex_header())
    man.commit()
    got = _status_note(man)
    assert got["id.tex"][0] == "skipped" and got["id.tex"][1].startswith("index map")
    assert got["hair.tex"] == ("skipped", "family 'human-hair' has no tier in roles.POLICY")
    assert got["arr.tex"] == (
        "skipped",
        "texture type 2DArray: the role functions are 2D operations",
    )
    assert got["ok.tex"] == ("planned", "")
    # The note column is capped at 400 characters, so nothing here can exceed it.
    assert all(len(n) <= 400 for _s, n in got.values())


def test_set_status_writes_tiers_note_and_updated(tmp_path) -> None:
    man = Manifest(str(tmp_path / "m.sqlite"))
    man.add("p.tex", "equipment", "", "normal", _tex_header())
    man.db.execute("UPDATE tex SET updated=0, srchash='old', srcver='v0'")
    t0 = time.time()
    man.set_status("p.tex", "done", tiers="4x,2x,native", note="models:normal=x")
    man.commit()
    status, tiers, note, updated, srchash, srcver = man.db.execute(
        "SELECT status, tiers, note, updated, srchash, srcver FROM tex"
    ).fetchone()
    assert (status, tiers, note) == ("done", "4x,2x,native", "models:normal=x")
    assert updated >= t0
    # srchash=None leaves the stamp alone.
    assert (srchash, srcver) == ("old", "v0")


def test_set_status_with_srchash_stamps_in_the_same_write(tmp_path) -> None:
    man = Manifest(str(tmp_path / "m.sqlite"))
    man.add("p.tex", "equipment", "", "normal", _tex_header())
    man.set_status("p.tex", "done", tiers="native", srchash="abc123", srcver="2026.09.01")
    assert man.db.execute("SELECT status, srchash, srcver FROM tex").fetchone() == (
        "done",
        "abc123",
        "2026.09.01",
    )
    # An empty hash is a value, not "leave it": it clears the stamp.
    man.set_status("p.tex", "planned", srchash="", srcver="")
    assert man.db.execute("SELECT srchash, srcver FROM tex").fetchone() == ("", "")


def test_set_hash_only_touches_the_stamp(tmp_path) -> None:
    man = Manifest(str(tmp_path / "m.sqlite"))
    man.add("p.tex", "equipment", "", "normal", _tex_header())
    man.set_hash("p.tex", "h1", "v1")
    assert man.db.execute("SELECT status, srchash, srcver FROM tex").fetchone() == (
        "planned",
        "h1",
        "v1",
    )


def test_set_status_on_an_unknown_path_is_a_no_op(tmp_path) -> None:
    man = Manifest(str(tmp_path / "m.sqlite"))
    man.set_status("nope.tex", "done")
    assert man.count() == 0


def _rows_manifest(tmp_path) -> Manifest:
    man = Manifest(str(tmp_path / "m.sqlite"))
    for path, fam, role, status, updated, note in [
        ("a.tex", "equipment", "normal", "planned", 100.0, ""),
        ("b.tex", "equipment", "color", "done", 100.0, ""),
        ("c.tex", "weapon", "normal", "planned", 300.0, ""),
        ("d.tex", "weapon", "normal", "failed", 100.0, "changed at 2026.09.01.0000.0000"),
        ("e.tex", "ui-icon", "icon", "planned", 200.0, ""),
    ]:
        man.db.execute(
            "INSERT INTO tex(path,family,part,role,w,h,fmt,mips,bytes,ttype,status,note,updated)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (path, fam, "", role, 64, 64, "BC7", 1, 1000, "2D", status, note, updated),
        )
    man.commit()
    return man


def _paths(rows) -> list[str]:
    return [r[0] for r in rows]


def test_rows_filters(tmp_path) -> None:
    man = _rows_manifest(tmp_path)
    assert _paths(man.rows()) == ["a.tex", "b.tex", "c.tex", "d.tex", "e.tex"]
    assert _paths(man.rows(family="equipment")) == ["a.tex", "b.tex"]
    assert _paths(man.rows(role="normal")) == ["a.tex", "c.tex", "d.tex"]
    assert _paths(man.rows(status="planned")) == ["a.tex", "c.tex", "e.tex"]
    assert _paths(man.rows(family="weapon", role="normal", status="planned")) == ["c.tex"]
    assert _paths(man.rows(limit=2)) == ["a.tex", "b.tex"]
    assert _paths(man.rows(family="nope")) == []


def test_rows_since_is_the_patch_delta(tmp_path) -> None:
    man = _rows_manifest(tmp_path)
    # updated >= since, OR a `changed at` note regardless of when the row was inserted.
    assert _paths(man.rows(since=200.0)) == ["c.tex", "d.tex", "e.tex"]
    assert _paths(man.rows(since=250.0)) == ["c.tex", "d.tex"]
    assert _paths(man.rows(since=1e9)) == ["d.tex"]
    assert _paths(man.rows(status="planned", since=250.0)) == ["c.tex"]


def test_rows_shape(tmp_path) -> None:
    man = _rows_manifest(tmp_path)
    row = man.rows(limit=1)[0]
    assert len(row) == 10
    assert row == ("a.tex", "equipment", "", "normal", 64, 64, "BC7", 1, "planned", "")


def test_count_with_keyword_filters(tmp_path) -> None:
    man = _rows_manifest(tmp_path)
    assert man.count() == 5
    assert man.count(family="weapon") == 2
    assert man.count(family="weapon", status="failed") == 1


def test_summary_groups_by_family_role_status(tmp_path) -> None:
    man = _rows_manifest(tmp_path)
    assert man.summary() == [
        ("equipment", "color", "done", 1, 1000),
        ("equipment", "normal", "planned", 1, 1000),
        ("ui-icon", "icon", "planned", 1, 1000),
        ("weapon", "normal", "failed", 1, 1000),
        ("weapon", "normal", "planned", 1, 1000),
    ]


def test_snapshot_round_trip(tmp_path) -> None:
    man = Manifest(str(tmp_path / "m.sqlite"))
    assert man.snapshots() == []
    t0 = time.time()
    man.snapshot("2026.09.01.0000.0000", "stamp", 10, 0, 0, 0)
    man.snapshot("2026.09.15.0000.0000", "check", 10, 2, 1, 0, note="requeued")
    rows = man.snapshots()
    assert len(rows) == 2
    # Newest first, and every column comes back as it went in.
    ts, ver, mode, nh, nc, ng, nn, note = rows[0]
    assert ts >= t0
    assert (ver, mode, nh, nc, ng, nn, note) == (
        "2026.09.15.0000.0000",
        "check",
        10,
        2,
        1,
        0,
        "requeued",
    )
    assert rows[1][1:] == ("2026.09.01.0000.0000", "stamp", 10, 0, 0, 0, "")
    assert len(man.snapshots(limit=1)) == 1
    # snapshot() commits on its own: a fresh connection sees both rows.
    assert len(Manifest(man.path).snapshots()) == 2


def test_reopening_a_manifest_replays_the_migrations_harmlessly(tmp_path) -> None:
    p = str(tmp_path / "m.sqlite")
    first = Manifest(p)
    first.add("p.tex", "equipment", "", "normal", _tex_header())
    first.commit()
    man = Manifest(p)  # every ALTER TABLE now hits "duplicate column name" and is swallowed
    cols = {r[1] for r in man.db.execute("PRAGMA table_info(tex)")}
    assert {"recipe", "srchash", "srcver"} <= cols
    assert man.db.execute("SELECT recipe, srchash, srcver FROM tex").fetchone() == ("", "", "")


def test_formats_and_roles_agree_with_the_module_tables() -> None:
    # The manifest's own suffix table must only name roles the pipeline knows about.
    assert set(manifest.SUFFIX_ROLE.values()) <= {*manifest.PROCESSED_ROLES, "id", "skip"}
    # Every skip format is a real .tex format name (or the decoder's own name for one).
    assert set(tex.FORMATS.values()) >= manifest.SKIP_FORMATS
    assert BGRA8 == texwrite.B8G8R8A8 and R32F in tex.FORMATS
