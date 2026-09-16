"""Penumbra packaging: the single-select Tier group, the meta.json shape, ``pack`` and the
legacy config merge.

Pins the rule in the comment above ``options_for``: the Tier group is SINGLE-SELECT, so an
option is the whole mod, and every option must carry each texture at the highest tier that
texture actually produced at or below the option's own. Also pins that ``write_mod_json``
writes a FileVersion-4 meta.json with backslash paths and removes the pre-v4 group files,
that ``pack`` only lists files that exist and never deletes a mod it did not write, and that
``merge_penumbra`` migrates the em-dash mod names rather than orphaning them.
"""

import json
import os
import pathlib
import zipfile

import pytest

from clarity import manifest
from clarity.packaging import penumbra
from clarity.processing import roles

A, B, C = "chara/a_d.tex", "chara/b_n.tex", "chara/c_m.tex"


def _rel(tier: str, path: str) -> str:
    return f"files/{tier}/{path}"


def _tiers(**by_tier: list[str]) -> dict[str, dict[str, str]]:
    """{tier: [paths]} -> {tier: {path: rel}} the way pack() builds it."""
    return {tier: {p: _rel(tier, p) for p in paths} for tier, paths in by_tier.items()}


# ---------------------------------------------------------------------------- constants
def test_tier_tables_agree() -> None:
    assert penumbra.TIER_ORDER == ["native", "2x", "4x"]
    assert set(penumbra.TIER_LABEL) == set(penumbra.TIER_ORDER) == set(roles.TIER_SCALE)
    assert penumbra.file_rel("4x", A) == "files/4x/chara/a_d.tex"


def test_legacy_names_map_the_em_dash_to_the_ascii_name() -> None:
    assert len(penumbra.LEGACY_NAMES) == len(penumbra.MODS)
    for legacy, current in penumbra.LEGACY_NAMES.items():
        assert "—" in legacy and legacy not in penumbra.MODS
        assert current in penumbra.MODS
        assert legacy.replace("—", "-") == current


def test_mod_priorities_are_unique_and_in_the_080_range() -> None:
    prios = [p for _fams, p in penumbra.MODS.values()]
    assert len(set(prios)) == len(prios)
    assert all(80 <= p <= 89 for p in prios)


def test_mod_for_family_covers_every_family_with_a_tier() -> None:
    for family, (tier, _cap) in roles.POLICY.items():
        mod = penumbra.mod_for_family(family)
        if tier is None:
            assert mod is None, family
        else:
            assert mod in penumbra.MODS and family in penumbra.MODS[mod][0], family
    assert penumbra.mod_for_family("common") == "Clarity - UI & HUD"
    assert penumbra.mod_for_family("human-hair") is None
    assert penumbra.mod_for_family("vfx") is None
    assert penumbra.mod_for_family("no-such-family") is None


def test_each_family_belongs_to_exactly_one_mod() -> None:
    seen = [f for fams, _p in penumbra.MODS.values() for f in fams]
    assert len(seen) == len(set(seen))


# -------------------------------------------------------------------------- options_for
def test_options_native_only() -> None:
    out = penumbra.options_for(_tiers(native=[A, B]))
    assert list(out) == ["native"]
    assert out["native"] == {A: _rel("native", A), B: _rel("native", B)}


def test_options_2x_falls_back_to_native_per_texture() -> None:
    out = penumbra.options_for(_tiers(native=[A, B], **{"2x": [A]}))
    assert list(out) == ["native", "2x"]
    assert out["native"] == {A: _rel("native", A), B: _rel("native", B)}
    # B never reached 2x, so the 2x option still carries it -- at native.
    assert out["2x"] == {A: _rel("2x", A), B: _rel("native", B)}


def test_options_every_tier_carries_every_texture() -> None:
    out = penumbra.options_for(_tiers(native=[A, B, C], **{"2x": [A, B], "4x": [A]}))
    assert list(out) == ["native", "2x", "4x"]
    for tier, files in out.items():
        assert set(files) == {A, B, C}, tier
    assert out["4x"] == {A: _rel("4x", A), B: _rel("2x", B), C: _rel("native", C)}
    assert out["2x"] == {A: _rel("2x", A), B: _rel("2x", B), C: _rel("native", C)}


def test_options_skip_a_tier_nothing_reached() -> None:
    # No 2x at all: that option would duplicate native, so it is not emitted; 4x still
    # falls through to native for the texture that stopped there.
    out = penumbra.options_for(_tiers(native=[A, B], **{"4x": [A]}))
    assert list(out) == ["native", "4x"]
    assert out["4x"] == {A: _rel("4x", A), B: _rel("native", B)}


def test_options_empty() -> None:
    assert penumbra.options_for({}) == {}
    assert penumbra.options_for({"2x": {}, "native": {}}) == {}


def test_options_do_not_mutate_the_input() -> None:
    src = _tiers(native=[A], **{"2x": [A]})
    snapshot = json.dumps(src, sort_keys=True)
    penumbra.options_for(src)
    assert json.dumps(src, sort_keys=True) == snapshot


# ----------------------------------------------------------------------- write_mod_json
def _meta(out: pathlib.Path, name: str) -> dict:
    return json.loads((out / name / "meta.json").read_text(encoding="utf-8"))


def test_write_mod_json_shape(tmp_path) -> None:
    n = penumbra.write_mod_json(
        str(tmp_path), "Clarity - Gear", _tiers(native=[A, B], **{"2x": [A, B], "4x": [A]})
    )
    assert n == 3
    meta = _meta(tmp_path, "Clarity - Gear")
    assert meta["FileVersion"] == 4
    assert meta["Name"] == "Clarity - Gear"
    assert meta["DefaultData"] == {"Files": {}, "FileSwaps": {}, "Manipulations": []}
    (group,) = meta["Groups"]
    assert group["Name"] == "Tier" and group["Type"] == "Single"
    assert group["DefaultSettings"] == 2  # the highest tier, last in the list
    names = [o["Name"] for o in group["Options"]]
    assert names == [penumbra.TIER_LABEL[t] for t in ("native", "2x", "4x")]
    for opt in group["Options"]:
        assert opt["Description"] == "2 textures"
        assert opt["FileSwaps"] == {} and opt["Manipulations"] == []
        assert list(opt["Files"]) == sorted(opt["Files"])
        for gp, rel in opt["Files"].items():
            assert "/" in gp and "/" not in rel and "\\" in rel, (gp, rel)
    assert group["Options"][2]["Files"] == {
        A: "files\\4x\\chara\\a_d.tex",
        B: "files\\2x\\chara\\b_n.tex",
    }


def test_write_mod_json_default_is_the_only_option_when_native_only(tmp_path) -> None:
    assert penumbra.write_mod_json(str(tmp_path), "M", _tiers(native=[A])) == 1
    (group,) = _meta(tmp_path, "M")["Groups"]
    assert group["DefaultSettings"] == 0
    assert [o["Name"] for o in group["Options"]] == [penumbra.TIER_LABEL["native"]]


def test_write_mod_json_with_nothing_writes_no_group(tmp_path) -> None:
    assert penumbra.write_mod_json(str(tmp_path), "M", {}) == 0
    assert _meta(tmp_path, "M")["Groups"] == []


def test_write_mod_json_removes_legacy_group_files(tmp_path) -> None:
    d = tmp_path / "M"
    d.mkdir()
    for old in ("default_mod.json", "group_001_tier.json"):
        (d / old).write_text("{}", encoding="utf-8")
    (d / "unrelated.json").write_text("{}", encoding="utf-8")
    penumbra.write_mod_json(str(tmp_path), "M", _tiers(native=[A]))
    assert not (d / "default_mod.json").exists()
    assert not (d / "group_001_tier.json").exists()
    assert (d / "unrelated.json").exists()
    assert (d / "meta.json").exists()


def test_write_mod_json_version_and_description(tmp_path) -> None:
    penumbra.write_mod_json(str(tmp_path), "M", {}, version="7.56.0", description="hello")
    meta = _meta(tmp_path, "M")
    assert meta["Version"] == "7.56.0" and meta["Description"] == "hello"
    penumbra.write_mod_json(str(tmp_path), "M2", {})
    assert _meta(tmp_path, "M2")["Version"] == "0.1.0"
    assert "upscale" in _meta(tmp_path, "M2")["Description"]
    assert _meta(tmp_path, "M2")["ModTags"] == ["clarity", "upscale"]


# --------------------------------------------------------------------------------- pack
def _row(man: manifest.Manifest, path: str, family: str, status: str, tiers: str) -> None:
    man.db.execute(
        "INSERT INTO tex(path,family,part,role,w,h,fmt,mips,bytes,ttype,status,tiers,updated)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (path, family, "", "color", 64, 64, "BC7", 1, 0, "2D", status, tiers, 0.0),
    )


def _touch(out: pathlib.Path, mod: str, tier: str, path: str) -> None:
    f = out / mod / _rel(tier, path)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"tex")


def _pack_fixture(tmp_path) -> tuple[manifest.Manifest, pathlib.Path]:
    man = manifest.Manifest(str(tmp_path / "m.sqlite"))
    out = tmp_path / "out"
    gear, world = "Clarity - Gear", "Clarity - World"
    # A: every tier on disk.  B: manifest says 4x,2x,native but only 2x and native exist.
    _row(man, A, "equipment", "done", "4x,2x,native")
    for t in ("4x", "2x", "native"):
        _touch(out, gear, t, A)
    _row(man, B, "weapon", "done", "4x,2x,native")
    for t in ("2x", "native"):
        _touch(out, gear, t, B)
    # C: done, native only, in the World mod.
    _row(man, C, "bg-hou", "done", "native")
    _touch(out, world, "native", C)
    # Not packed: a planned row with a file, a reserved path, a family with no mod, a done
    # row whose file was deleted, and a done row with an empty tiers column.
    _row(man, "chara/p_d.tex", "equipment", "planned", "")
    _touch(out, gear, "native", "chara/p_d.tex")
    reserved = "chara/common/texture/white.tex"
    assert reserved in penumbra.RESERVED_PATHS
    _row(man, reserved, "equipment", "done", "native")
    _touch(out, gear, "native", reserved)
    _row(man, "chara/hair_n.tex", "human-hair", "done", "native")
    _touch(out, "Clarity - Gear", "native", "chara/hair_n.tex")
    _row(man, "chara/gone_d.tex", "equipment", "done", "native")
    _row(man, "chara/empty_d.tex", "equipment", "done", "")
    _touch(out, gear, "native", "chara/empty_d.tex")
    man.commit()
    return man, out


def test_pack_writes_one_mod_per_family_group(tmp_path) -> None:
    man, out = _pack_fixture(tmp_path)
    log = []
    written = penumbra.pack(man, str(out), log=log.append)
    assert written == {
        "Clarity - Gear": {"native": 2, "2x": 2, "4x": 2},
        "Clarity - World": {"native": 1},
    }
    gear = _meta(out, "Clarity - Gear")
    (group,) = gear["Groups"]
    by_name = {o["Name"]: o["Files"] for o in group["Options"]}
    assert by_name[penumbra.TIER_LABEL["4x"]] == {
        A: "files\\4x\\chara\\a_d.tex",
        B: "files\\2x\\chara\\b_n.tex",  # B's 4x file is missing: fall back, silently
    }
    assert by_name[penumbra.TIER_LABEL["native"]] == {
        A: "files\\native\\chara\\a_d.tex",
        B: "files\\native\\chara\\b_n.tex",
    }
    # Nothing else leaked in.
    listed = {gp for files in by_name.values() for gp in files}
    assert listed == {A, B}
    (world_group,) = _meta(out, "Clarity - World")["Groups"]
    assert [o["Files"] for o in world_group["Options"]] == [{C: "files\\native\\chara\\c_m.tex"}]
    # No mod was emptied, so the log is quiet.
    assert log == []


def test_pack_zips_each_written_mod(tmp_path) -> None:
    man, out = _pack_fixture(tmp_path)
    penumbra.pack(man, str(out), log=None)
    for mod in ("Clarity - Gear", "Clarity - World"):
        pmp = out / f"{mod}.pmp"
        assert pmp.is_file() and not (out / f"{mod}.zip").exists()
        with zipfile.ZipFile(pmp) as z:
            names = set(z.namelist())
        assert "meta.json" in names
        assert not any(n.endswith(".pmp") for n in names)
    with zipfile.ZipFile(out / "Clarity - Gear.pmp") as z:
        assert "files/4x/chara/a_d.tex" in z.namelist()
    # Rerunning replaces the archive rather than failing on the existing one.
    penumbra.pack(man, str(out), log=None)
    assert (out / "Clarity - Gear.pmp").is_file()


def test_pack_ignores_families_with_no_mod_and_reserved_paths(tmp_path) -> None:
    man, out = _pack_fixture(tmp_path)
    penumbra.pack(man, str(out), log=None)
    assert not (out / "Clarity - Human (skin, faces, tails, ears)").exists()
    text = (out / "Clarity - Gear" / "meta.json").read_text(encoding="utf-8")
    assert "white.tex" not in text and "hair_n" not in text
    assert "p_d.tex" not in text and "gone_d" not in text and "empty_d" not in text


def test_pack_with_nothing_done_writes_nothing(tmp_path) -> None:
    man = manifest.Manifest(str(tmp_path / "m.sqlite"))
    _row(man, A, "equipment", "planned", "")
    man.commit()
    out = tmp_path / "out"
    assert penumbra.pack(man, str(out), log=None) == {}
    assert not out.exists()


def test_pack_refuses_to_delete_an_emptied_mod_and_says_so(tmp_path) -> None:
    man, out = _pack_fixture(tmp_path)
    # The Human mod exists on disk from an earlier pack but has no done rows any more.
    stale = out / "Clarity - Human (skin, faces, tails, ears)"
    stale.mkdir(parents=True)
    # NB: pack() looks for the pre-v4 `group_001_tier.json`, which write_mod_json now deletes;
    # a mod packed by the current code leaves only meta.json, which this check does not see.
    (stale / "group_001_tier.json").write_text("{}", encoding="utf-8")
    (stale / "meta.json").write_text("{}", encoding="utf-8")
    log = []
    penumbra.pack(man, str(out), log=log.append)
    assert (stale / "group_001_tier.json").exists()
    assert (stale / "meta.json").exists()
    assert not (out / "Clarity - Human (skin, faces, tails, ears).pmp").exists()
    assert len(log) == 1
    assert "no finished rows" in log[0] and str(stale) in log[0]
    # And with the log switched off the mod is still left alone.
    penumbra.pack(man, str(out), log=None)
    assert (stale / "group_001_tier.json").exists()


# ------------------------------------------------------------------------ merge_penumbra
GEAR, WORLD = "Clarity - Gear", "Clarity - World"
GUID = "b615f2fe-afef-4cb7-91d3-4f353501f64c"


def _read(p: pathlib.Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def test_merge_refuses_a_litedb_config(tmp_path) -> None:
    (tmp_path / "mod_data.db").write_bytes(b"")
    with pytest.raises(RuntimeError, match="LiteDB"):
        penumbra.merge_penumbra(str(tmp_path), [GEAR])
    assert not (tmp_path / "sort_order.json").exists()


def test_merge_creates_sort_order_from_nothing(tmp_path) -> None:
    so_path = penumbra.merge_penumbra(str(tmp_path), [GEAR, WORLD])
    assert so_path == os.path.join(str(tmp_path), "sort_order.json")
    so = _read(pathlib.Path(so_path))
    assert so["EmptyFolders"] == [] and so["LockedPaths"] == []
    assert so["Data"] == {
        GEAR: f"{penumbra.FOLDER}/080 {GEAR}",
        WORLD: f"{penumbra.FOLDER}/083 {WORLD}",
    }
    # No collection file: nothing is created for it.
    assert not (tmp_path / "collections").exists()


def test_merge_keeps_other_sort_entries_and_drops_legacy_ones(tmp_path) -> None:
    legacy_gear = next(k for k, v in penumbra.LEGACY_NAMES.items() if v == GEAR)
    (tmp_path / "sort_order.json").write_text(
        json.dumps(
            {
                "Data": {"Someone Else": "9 Misc/Someone Else", legacy_gear: "old/place"},
                "EmptyFolders": ["x"],
                "LockedPaths": [],
            }
        ),
        encoding="utf-8",
    )
    so = _read(pathlib.Path(penumbra.merge_penumbra(str(tmp_path), [GEAR])))
    assert so["Data"] == {
        "Someone Else": "9 Misc/Someone Else",
        GEAR: f"{penumbra.FOLDER}/080 {GEAR}",
    }
    assert so["EmptyFolders"] == ["x"]


def test_merge_enables_mods_in_the_default_collection(tmp_path) -> None:
    coll_dir = tmp_path / "collections"
    coll_dir.mkdir()
    coll = coll_dir / f"{GUID}.json"
    coll.write_text(
        json.dumps(
            {"Name": "Default", "Settings": {"Other Mod": {"Priority": 5, "Enabled": False}}}
        ),
        encoding="utf-8",
    )
    penumbra.merge_penumbra(str(tmp_path), [GEAR, WORLD])
    j = _read(coll)
    assert j["Name"] == "Default"
    assert j["Settings"]["Other Mod"] == {"Priority": 5, "Enabled": False}
    assert j["Settings"][GEAR] == {"Priority": 80, "Enabled": True}
    assert j["Settings"][WORLD] == {"Priority": 83, "Enabled": True}


def test_merge_does_not_override_an_existing_user_setting(tmp_path) -> None:
    coll_dir = tmp_path / "collections"
    coll_dir.mkdir()
    coll = coll_dir / f"{GUID}.json"
    mine = {"Priority": 200, "Enabled": False, "Settings": {"Tier": 1}}
    coll.write_text(json.dumps({"Settings": {GEAR: mine}}), encoding="utf-8")
    penumbra.merge_penumbra(str(tmp_path), [GEAR])
    assert _read(coll)["Settings"][GEAR] == mine


def test_merge_migrates_a_legacy_collection_entry(tmp_path) -> None:
    legacy_gear = next(k for k, v in penumbra.LEGACY_NAMES.items() if v == GEAR)
    coll_dir = tmp_path / "collections"
    coll_dir.mkdir()
    coll = coll_dir / f"{GUID}.json"
    old = {"Priority": 99, "Enabled": False, "Settings": {"Tier": 2}}
    coll.write_text(json.dumps({"Settings": {legacy_gear: old}}), encoding="utf-8")
    penumbra.merge_penumbra(str(tmp_path), [GEAR])
    settings = _read(coll)["Settings"]
    assert legacy_gear not in settings
    assert settings[GEAR] == old  # the user's choices ride over to the new name


def test_merge_honours_a_custom_collection_guid(tmp_path) -> None:
    coll_dir = tmp_path / "collections"
    coll_dir.mkdir()
    (coll_dir / "custom.json").write_text(json.dumps({"Settings": {}}), encoding="utf-8")
    penumbra.merge_penumbra(str(tmp_path), [WORLD], default_guid="custom")
    assert _read(coll_dir / "custom.json")["Settings"][WORLD] == {"Priority": 83, "Enabled": True}


def test_merge_with_an_unknown_mod_name_raises(tmp_path) -> None:
    with pytest.raises(KeyError):
        penumbra.merge_penumbra(str(tmp_path), ["Not A Clarity Mod"])
