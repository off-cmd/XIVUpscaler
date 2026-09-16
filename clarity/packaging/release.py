"""Versioned family/profile exports. Working products are never moved or requeued."""

import hashlib
import json
import os
import pathlib
import re
import shutil
import sqlite3
import time
from typing import Any

from .. import paths
from ..processing import roles
from . import penumbra as pack

CATALOG = [
    (
        "Human",
        [
            ("human-face", "Faces"),
            ("human-body", "Bodies"),
            ("human-zear", "Ears"),
            ("human-tail", "Tails"),
        ],
    ),
    (
        "Gear",
        [
            ("equipment", "Clothing and Armour"),
            ("accessory", "Accessories"),
            ("weapon", "Weapons"),
        ],
    ),
    ("Monsters and Demihumans", [("monster", "Monsters"), ("demihuman", "Demihumans")]),
    (
        "World",
        [
            ("bg", "General Environments"),
            ("bgcommon", "Shared Environments"),
            ("bg-hou", "Housing Environments"),
            ("bg-ind", "Interiors"),
            ("bgcommon-hou", "Shared Housing"),
            ("bgcommon-mji", "Island Sanctuary"),
        ],
    ),
    (
        "UI and HUD",
        [
            ("ui-icon", "Icons"),
            ("ui-uld", "Interface Layouts"),
            ("ui-other", "Other Interface Textures"),
            ("common", "Loading Screens"),
        ],
    ),
]
POLICY_VERSION = "everyday-1"


def version_key(version):
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:-preview\.(\d+))?", version)
    if not m:
        raise ValueError("Invalid Clarity release version: " + version)
    major, minor, revision, preview = m.groups()
    return (int(major), int(minor), int(revision), preview is None, int(preview or 0))


def version_for(game, revision, preview=None):
    m = re.fullmatch(r"(\d+)\.(\d{1,2})(?:h(\d+))?", game)
    if not m or revision < 0:
        raise ValueError("Expected game patch such as 7.56 or 7.56h1 and nonnegative revision")
    if preview is not None and preview < 1:
        raise ValueError("Preview number must be a positive integer")
    major, minor, _hotfix = m.groups()
    version = f"{int(major)}.{int(minor.ljust(2, '0'))}.{revision}"
    if preview is not None:
        version += f"-preview.{preview}"
    return version


def requested_tier(family, role, path, profile="everyday"):
    if profile in ("native", "2x", "4x"):
        return profile
    # These are the existing classifier's conventional specular suffixes. Exact
    # material exceptions must be recorded separately, not guessed from actor names.
    specular = pathlib.PurePosixPath(path).stem.endswith(("_s", "_spec", "_specular"))
    if family in ("equipment", "accessory", "weapon") and (role == "normal" or specular):
        return "native"
    if family.startswith("bg") and role == "normal":
        return "native"
    return "2x"


def generation_tier(family, role, path, w, h, override=None, profile="everyday"):
    if override or profile == "legacy":
        return roles.top_tier(family, w, h, override)
    policy_tier, cap = roles.POLICY.get(family, (None, 0))
    if policy_tier is None:
        # No tier means "never process" (human-hair, vfx, and any family the policy does
        # not name). top_tier() and skip_reason() both return None here; this must too, or
        # `run --family human-hair` walks past the `top is None` guard and produces
        # native-tier products for a family the policy excludes.
        return None
    target = requested_tier(family, role, path, profile)
    scale = roles.TIER_SCALE[target]
    while scale > 1 and (max(w, h) > cap or max(w, h) * scale > roles.MAX_EDGE_OUT):
        scale //= 2
    return {1: "native", 2: "2x", 4: "4x"}[scale]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, data):
    pathlib.Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def select_product(row, root, profile):
    target = requested_tier(row["family"], row["role"], row["path"], profile)
    available = set((row["tiers"] or "").split(","))
    for tier in reversed(["native", "2x", "4x"][: ["native", "2x", "4x"].index(target) + 1]):
        p = root / pack.mod_for_family(row["family"]) / pack.file_rel(tier, row["path"])
        if tier in available and p.is_file():
            return tier, p
    return None, None


# The placeholder textures every family shares; never shipped in a mod. One
# definition, owned by the packer.
RESERVED_PATHS = pack.RESERVED_PATHS


def export(a):
    version = version_for(a.game, a.revision, a.preview)
    final = pathlib.Path(a.destination) / version
    if final.exists():
        raise ValueError(f"Immutable export already exists: {final}")
    # Everything is written under a staging name and renamed into place only once
    # release.json exists. An export that fails part-way (stale sources, a bad
    # copy) therefore never leaves a directory that blocks the next attempt at the
    # same version, and a directory named "<version>" is complete by construction.
    dest = final.with_name(final.name + ".building")
    if dest.exists():
        shutil.rmtree(dest)
    for previous in pathlib.Path(a.destination).glob("*/release.json"):
        old = json.loads(previous.read_text(encoding="utf-8"))["version"]
        if version_key(version) <= version_key(old):
            raise ValueError(f"Release must increase beyond {old}")
    source = pathlib.Path(a.source)
    db = sqlite3.connect(pathlib.Path(a.db).resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("BEGIN")
    rows = [
        dict(r)
        for r in db.execute("SELECT * FROM tex ORDER BY path")
        if r["path"] not in RESERVED_PATHS
    ]
    report = {
        "version": version,
        "target_game": a.game,
        "policy": POLICY_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "variants": [],
        "unresolved_rules": "Kartoffels exact-resource rules are pending 7.56 source/material verification; no historical overrides are enabled.",
    }
    planned = []
    for gi, (group, families) in enumerate(CATALOG):
        for fi, (family, title) in enumerate(families):
            eligible = [r for r in rows if r["family"] == family and r["status"] != "skipped"]
            for profile in a.profile:
                if profile == "4x" and roles.POLICY.get(family, (None, 0))[0] != "4x":
                    continue
                products = []
                missing = []
                for row in eligible:
                    tier, p = (
                        select_product(row, source, profile)
                        if row["status"] == "done"
                        else (None, None)
                    )
                    if p is None:
                        missing.append(
                            {
                                "path": row["path"],
                                "reason": "not-done"
                                if row["status"] != "done"
                                else "output-missing",
                            }
                        )
                    else:
                        products.append((row, tier, p))
                label = (
                    "Everyday (Up to 2x)"
                    if profile == "everyday"
                    else {"native": "Native", "2x": "Up to 2x", "4x": "Up to 4x"}[profile]
                )
                ident = f"clarity-{family}-{profile}"
                entry: dict[str, Any] = {
                    "id": ident,
                    "package": "Clarity - " + group,
                    "family": family,
                    "title": title + " - " + label,
                    "order": [gi, fi],
                    "profile": profile,
                    "eligible": len(eligible),
                    "available": len(products),
                    "missing": missing,
                    "default_enabled": profile == "everyday"
                    and family not in ("monster", "demihuman"),
                    "path": str(pathlib.Path("Clarity - " + group) / (title + " - " + profile)),
                }
                report["variants"].append(entry)
                if products:
                    planned.append((entry, products))
    report["total_missing"] = sum(len(v["missing"]) for v in report["variants"])
    if report["total_missing"] and not a.preview:
        raise ValueError(
            "Incomplete family products: use a preview release; no stable release created"
        )
    if a.dry_run:
        print(
            json.dumps(
                {
                    **report,
                    "variants": [
                        {k: v for k, v in e.items() if k != "missing"} for e in report["variants"]
                    ],
                },
                indent=2,
            )
        )
        return 0
    dest.mkdir(parents=True)
    for group, _ in CATALOG:
        (dest / ("Clarity - " + group)).mkdir(exist_ok=True)
    backup = sqlite3.connect(dest / "source-manifest.sqlite")
    db.backup(backup)
    backup.close()
    db.close()
    # Validate current source identity using the existing game reader; missing stamps
    # are not silently adopted as current.
    from .. import ffxiv as kb
    from ..cli import src_fingerprint

    gd = kb.game()
    source_version = kb.sqpack.game_version(gd.sqpack)
    report["client_build"] = source_version
    checked = {}
    copied = {}
    objects = pathlib.Path(a.destination).parent / "release-objects"
    objects.mkdir(exist_ok=True)
    dump(dest / "release-progress.json", {"state": "building", "version": version})
    for entry, products in planned:
        folder = dest / entry["path"]
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        provenance = []
        stale = []
        for row, tier, p in products:
            if row["path"] not in checked:
                checked[row["path"]] = src_fingerprint(row["path"], gd)
            if not row["srchash"] or checked[row["path"]] != row["srchash"]:
                stale.append({"path": row["path"], "reason": "source-unverified-or-changed"})
                continue
            rel = "files/" + row["path"]
            output = folder / rel
            output.parent.mkdir(parents=True, exist_ok=True)
            if str(p) not in copied:
                sha = digest(p)
                obj = objects / sha
                if not obj.exists():
                    shutil.copyfile(p, obj)
                if digest(obj) != sha:
                    raise RuntimeError("Object copy mismatch: " + str(p))
                copied[str(p)] = (sha, obj)
            sha, obj = copied[str(p)]
            if not output.exists():
                os.link(obj, output)
            files[row["path"]] = rel.replace("/", "\\")
            provenance.append(
                {
                    "path": row["path"],
                    "tier": tier,
                    "sha256": sha,
                    "bytes": output.stat().st_size,
                    "source_hash": row["srchash"],
                    "source_version": row["srcver"],
                    "recipe": row.get("recipe") or row["note"],
                    "origin": str(p),
                    "reuse": "existing processed product; native may derive from a higher-tier mip",
                }
            )
        entry["missing"] += stale
        entry["exported"] = len(files)
        if stale and not a.preview:
            raise RuntimeError("Source changes found; stable release not finalized")
        dump(
            folder / "meta.json",
            {
                "FileVersion": 4,
                "Name": entry["package"] + " - " + entry["title"],
                "Author": "clarity-upscale",
                "Version": version,
                "Description": f"{entry['exported']}/{entry['eligible']} eligible resources. {entry['profile']} profile. Gear: 2x color/masks, native normals/specular. World: 2x color, native normals. Actual scales may be capped. Existing native products may be downsampled model results. Preview gaps retain vanilla. No verified Kartoffels exceptions applied.",
                "Website": "",
                "ModTags": ["clarity", "upscale"],
                "DefaultData": {"Files": files, "FileSwaps": {}, "Manipulations": []},
                "Groups": [],
            },
        )
        dump(folder / "resources.json", provenance)
        print(
            entry["id"],
            len(files),
            "exported;",
            len(entry["missing"]),
            "missing",
            flush=True,
        )
        # Package into a .pmp file for manual import
        shutil.make_archive(str(folder), "zip", str(folder))
        if os.path.exists(str(folder) + ".pmp"):
            os.remove(str(folder) + ".pmp")
        os.rename(str(folder) + ".zip", str(folder) + ".pmp")

    report["total_missing"] = sum(len(v["missing"]) for v in report["variants"])
    dump(dest / "release.json", report)
    dump(dest / "release-progress.json", {"state": "complete", "version": version})
    dest.rename(final)
    print("Release:", final)
    return 0


def add_parser(sub):
    p = sub.add_parser("release", help="Export immutable family/profile releases without inference")
    p.add_argument("--game", required=True)
    p.add_argument("--revision", type=int, required=True)
    p.add_argument("--source", required=True)
    p.add_argument(
        "--destination",
        default=str(pathlib.Path(paths.PROJECT) / "build-output/releases"),
    )
    p.add_argument("--profile", action="append", choices=["everyday", "native", "2x", "4x"])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--preview",
        type=int,
        default=None,
        metavar="N",
        help="build a preview release (version X.Y.Z-preview.N) that may ship with "
        "missing or source-changed products; a stable release refuses both",
    )

    def run(a):
        a.profile = a.profile or ["everyday"]
        return export(a)

    p.set_defaults(fn=run)
