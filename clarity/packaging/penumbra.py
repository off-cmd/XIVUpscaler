"""Penumbra packaging: one mod per family group, a single-select "Tier" group whose options are

the tiers that were produced. Output textures are written by `run` straight into
`<out>/<mod>/files/<tier>/<game path>`, so packing only writes the JSON.

Also merges the new mods into the XIVPenumbra layout: `sort_order.json` paths under
`0 Base upscales (everyone)/Clarity tiers (G4)/` and enabled entries (priority 080–089) in the
Default collection.
"""

import os
from typing import Any

from ..jsonio import read_json, write_json

# ASCII names only. The em-dash these used to carry (U+2014, bytes E2 80 94) is read by the game's
# native file loader in the system ANSI codepage, so every redirect came back as
# "Clarity â€” Gear/..." -- a folder that does not exist. Every Clarity texture then failed to
# load, every material that referenced one failed, and with the Gear mod in the Default collection
# that was every character in view: invisible, and a crash on the disable-redraw. Found 2026-09-08
# in dalamud.log ("Failed to synchronously load resource C:/.../Clarity â€” Gear/files/4x/...").
# Penumbra's own guidance is ASCII mod paths; this is why.
MODS = {
    "Clarity - Gear": (["equipment", "accessory", "weapon"], 80),
    "Clarity - Monsters & Demihumans": (["monster", "demihuman"], 81),
    "Clarity - Human (skin, faces, tails, ears)": (
        ["human-body", "human-face", "human-tail", "human-zear"],
        82,
    ),
    # bg/ and bgcommon/ are separate sqpack archives and stay separate families, but they ship as
    # one mod: a user who wants better scenery wants all of it, and splitting the housing families
    # into their own mod would make "turn the world up" a four-checkbox operation.
    "Clarity - World": (
        ["bg", "bg-hou", "bg-ind", "bgcommon", "bgcommon-hou", "bgcommon-mji"],
        83,
    ),
    # `common` rides with the UI mod because the only thing in it that ever reaches here is the
    # loading screens (manifest.classify sends -nowloading* to role `ui`; everything else in the
    # family stays role `other` and PROCESSED_ROLES will not touch it). They belong with the HUD
    # rather than the world: full-screen 2D art, same model, same reason to turn the mod off.
    "Clarity - UI & HUD": (["ui-icon", "ui-uld", "ui-other", "common"], 84),
}

# The pre-fix names, so an existing Penumbra config is migrated rather than orphaned.
LEGACY_NAMES = {name.replace("Clarity - ", "Clarity \u2014 "): name for name in MODS}
TIER_LABEL = {"native": "Native (BC7 re-encode)", "2x": "2×", "4x": "4×"}
TIER_ORDER = ["native", "2x", "4x"]
FOLDER = "0 Base upscales (everyone)/Clarity tiers (G4)"


def mod_for_family(family):
    for name, (fams, _) in MODS.items():
        if family in fams:
            return name
    return None


def file_rel(tier, game_path):
    return f"files/{tier}/{game_path}"


def options_for(tiers_files):
    """{tier: {game_path: rel}} -> the same, with each option carrying EVERY texture it can.

    THE TIER GROUP IS SINGLE-SELECT, SO AN OPTION IS THE WHOLE MOD, NOT A LAYER. Selecting "4×"
    applies exactly that option's Files and nothing else -- so listing only the textures that
    literally reached 4× meant the best-looking option had the WORST coverage.

    It bites wherever a family's cap is below its sources. `roles.top_tier` halves the scale while
    the source edge exceeds the cap, so with equipment capped at 1024 a 1024px texture produces
    4x/2x/native and a 2048px one produces native ONLY. Picking 4× -- which is the default, being
    last -- then left every 2048px gear texture pointing at vanilla, while the "Native" option
    upscaled it. Same for a 2048px wall under `bg` (cap 1024) in the World mod.

    So each option now falls back per texture to the highest tier that texture actually produced at
    or below the one asked for. "4×" means "the best I made for each texture, capped at 4×", which
    is what a single-select list of qualities has to mean.
    """
    out = {}
    for i, tier in enumerate(TIER_ORDER):
        if not tiers_files.get(tier):
            continue  # nothing reached this tier: the option would duplicate the one below
        merged = {}
        for lower in TIER_ORDER[: i + 1]:  # ascending, so a higher tier overwrites a lower one
            merged.update(tiers_files.get(lower, {}))
        out[tier] = merged
    return out


RESERVED_PATHS = {
    "common/graphics/texture/dummy.tex",
    "chara/common/texture/white.tex",
    "chara/common/texture/black.tex",
    "chara/common/texture/id_16.tex",
    "chara/common/texture/red.tex",
    "chara/common/texture/green.tex",
    "chara/common/texture/blue.tex",
    "chara/common/texture/null_normal.tex",
    "chara/common/texture/skin_mask.tex",
}


def write_mod_json(out, name, tiers_files, version="0.1.0", description=""):
    """tiers_files: {tier: {game_path: rel_path}} -- the tiers each texture actually produced."""
    tiers_files = options_for(tiers_files)
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)

    options = []
    for tier in TIER_ORDER:
        files = tiers_files.get(tier)
        if not files:
            continue
        options.append(
            {
                "Name": TIER_LABEL[tier],
                "Description": "%d textures" % len(files),
                "Priority": 0,
                "Files": {gp: rel.replace("/", "\\") for gp, rel in sorted(files.items())},
                "FileSwaps": {},
                "Manipulations": [],
            }
        )

    groups = []
    if options:
        groups.append(
            {
                "Name": "Tier",
                "Description": "Highest tier last; choose what the GPU affords.",
                "Priority": 0,
                "Type": "Single",
                "DefaultSettings": max(0, len(options) - 1),
                "Options": options,
            }
        )

    meta = {
        "FileVersion": 4,
        "Name": name,
        "Author": "clarity-upscale (Arisa)",
        "Version": version,
        "Description": description
        or "Material-aware upscale of vanilla textures, authored at the top tier and downsampled per tier. "
        "Normals: BC1-clean + normal model + renormalise; masks: per-channel scalars; colour: model + average-colour fix; "
        "index maps untouched. Pick the tier the card affords.",
        "Website": "",
        "ModTags": ["clarity", "upscale"],
        "DefaultData": {"Files": {}, "FileSwaps": {}, "Manipulations": []},
        "Groups": groups,
    }
    write_json(os.path.join(d, "meta.json"), meta)

    # Remove old files if they exist
    for old_f in ["default_mod.json", "group_001_tier.json"]:
        p = os.path.join(d, old_f)
        if os.path.isfile(p):
            os.remove(p)

    return len(options)


def pack(manifest, out, log=print):
    """Walk done rows, collect files per mod/tier from what exists on disk, write JSON."""
    per = {}
    for path, family, _part, _role, _w, _h, _fmt, _mips, _status, tiers in manifest.rows(
        status="done"
    ):
        if path in RESERVED_PATHS:
            continue
        mod = mod_for_family(family)
        if not mod:
            continue
        for tier in (tiers or "").split(","):
            if not tier:
                continue
            rel = file_rel(tier, path)
            if os.path.isfile(os.path.join(out, mod, rel)):
                per.setdefault(mod, {}).setdefault(tier, {})[path] = rel

    written = {}
    for mod, tiers_files in per.items():
        write_mod_json(out, mod, tiers_files)
        written[mod] = {t: len(f) for t, f in options_for(tiers_files).items()}

    # A MOD WITH NO ROWS LEFT KEEPS ITS OLD JSON, AND SAYING SO IS THE HONEST OPTION.
    # Only mods that still have qualifying rows are rewritten, so if a family is requeued wholesale,
    # reclassified out, or has its files deleted, the previous meta.json stays on disk and
    # Penumbra keeps redirecting to it while this function's summary no longer mentions it. Deleting
    # it automatically would be worse: packing halfway through a rebuild is normal, and an empty
    # option list is indistinguishable from "not finished yet". So warn, and let the human decide.
    import shutil

    if log:
        for mod in MODS:
            if mod in per:
                continue
            if os.path.isfile(os.path.join(out, mod, "meta.json")):
                log(
                    f"  NOTE: {mod} has a package on disk but no finished rows; its old redirections are"
                    f" still live. Delete {os.path.join(out, mod)} if it is obsolete."
                )

    # Zip up written mods into .pmp files
    for mod in written:
        mod_dir = os.path.join(out, mod)
        if os.path.isdir(mod_dir):
            shutil.make_archive(mod_dir, "zip", mod_dir)
            if os.path.exists(mod_dir + ".pmp"):
                os.remove(mod_dir + ".pmp")
            os.rename(mod_dir + ".zip", mod_dir + ".pmp")

    return written


def merge_penumbra(config_dir, mods_present, default_guid="b615f2fe-afef-4cb7-91d3-4f353501f64c"):
    """Add the Clarity mods to sort_order.json (created or merged) and to the Default collection."""
    if os.path.isfile(os.path.join(config_dir, "mod_data.db")):
        raise RuntimeError(
            "Modern Penumbra uses LiteDB: use the release installer, not legacy pack --penumbra-config"
        )
    so_path = os.path.join(config_dir, "sort_order.json")
    so: dict[str, Any] = {"Data": {}, "EmptyFolders": [], "LockedPaths": []}
    if os.path.isfile(so_path):
        so = read_json(so_path)
    # Migrate the em-dash names: drop their sort entries (the folders are gone) so they do not
    # linger as missing mods.
    for legacy in LEGACY_NAMES:
        so["Data"].pop(legacy, None)
    for mod in mods_present:
        prio = MODS[mod][1]
        so["Data"][mod] = "%s/%03d %s" % (FOLDER, prio, mod)
    write_json(so_path, so, indent=4)
    coll = os.path.join(config_dir, "collections", default_guid + ".json")
    if os.path.isfile(coll):
        j = read_json(coll)
        # Carry the old entry's enabled/priority over to the new name, then drop the old key.
        for legacy, current in LEGACY_NAMES.items():
            if legacy in j["Settings"]:
                j["Settings"].setdefault(current, j["Settings"][legacy])
                del j["Settings"][legacy]
        for mod in mods_present:
            j["Settings"].setdefault(mod, {"Priority": MODS[mod][1], "Enabled": True})
        write_json(coll, j, indent=4)
    return so_path


def merge_icon_twins(
    config_dir, twins, interface_guid="6e1d5b0a-4c2f-4a1e-9a7b-2f8e3c9d1a01", bump=40
):
    """twins: {original mod dir name: twin dir name}. The twin goes to

    `9 Interface/Icons - upscaled (G6)/<prio+bump> <name> (upscaled)` and is enabled in the
    Interface collection at the original's priority + bump, so it wins over the original.
    """
    if os.path.isfile(os.path.join(config_dir, "mod_data.db")):
        raise RuntimeError(
            "Modern Penumbra uses LiteDB: use the release installer, not legacy pack --penumbra-config"
        )
    so_path = os.path.join(config_dir, "sort_order.json")
    so: dict[str, Any] = {"Data": {}, "EmptyFolders": [], "LockedPaths": []}
    if os.path.isfile(so_path):
        so = read_json(so_path)
    coll_p = os.path.join(config_dir, "collections", interface_guid + ".json")
    coll = read_json(coll_p) if os.path.isfile(coll_p) else None
    for orig, twin in twins.items():
        prio = 900
        if coll and orig in coll["Settings"]:
            prio = coll["Settings"][orig].get("Priority", 900)
        label = os.path.basename(twin)
        so["Data"][label] = "9 Interface/Icons - upscaled (G6)/%03d %s" % (
            prio + bump,
            label,
        )
        if coll is not None:
            entry = dict(coll["Settings"].get(orig, {}))
            entry.update({"Priority": prio + bump, "Enabled": True})
            coll["Settings"][label] = entry
    write_json(so_path, so, indent=4)
    if coll is not None:
        write_json(coll_p, coll, indent=4)
    return so_path
