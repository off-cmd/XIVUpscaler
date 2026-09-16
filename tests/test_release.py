import argparse
import json
import pathlib
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from clarity.packaging.release import (
    CATALOG,
    generation_tier,
    requested_tier,
    select_product,
    version_for,
    version_key,
)


class ReleaseTests(unittest.TestCase):
    def test_game_order(self):
        labels = [
            "2.0",
            "2.1",
            "2.15",
            "2.16",
            "2.2",
            "2.21",
            "2.25",
            "2.28",
            "2.3",
            "2.35",
            "2.38",
            "2.4",
            "2.45",
            "2.5",
            "2.51",
            "2.55",
            "3.0",
            "3.01",
            "3.05",
            "3.1",
            "3.15",
            "3.2",
            "3.25",
            "3.26",
            "3.3",
            "3.35",
            "3.38",
            "3.4",
            "3.45",
            "3.5",
            "3.55",
            "3.56",
            "3.57",
            "4.0",
            "4.01",
            "4.05",
            "4.06",
            "4.1",
            "4.11",
            "4.15",
            "4.18",
            "4.2",
            "4.25",
            "4.3",
            "4.31",
            "4.35",
            "4.36",
            "4.4",
            "4.41",
            "4.45",
            "4.5",
            "4.55",
            "4.56",
            "4.57",
            "4.58",
            "5.0",
            "5.01",
            "5.05",
            "5.08",
            "5.1",
            "5.11",
            "5.15",
            "5.18",
            "5.2",
            "5.21",
            "5.25",
            "5.3",
            "5.31",
            "5.35",
            "5.4",
            "5.41",
            "5.45",
            "5.5",
            "5.55",
            "5.57",
            "5.58",
            "6.0",
            "6.01",
            "6.05",
            "6.08",
            "6.1",
            "6.11",
            "6.15",
            "6.18",
            "6.2",
            "6.21",
            "6.25",
            "6.28",
            "6.3",
            "6.31",
            "6.35",
            "6.38",
            "6.4",
            "6.41",
            "6.45",
            "6.48",
            "6.5",
            "6.51",
            "6.55",
            "6.57",
            "6.58",
            "7.0",
            "7.01",
            "7.05",
            "7.1",
        ]
        versions = [tuple(map(int, version_for(x, 0).split("."))) for x in labels]
        self.assertEqual(versions, sorted(set(versions)))
        self.assertEqual(version_for("2.5h2", 4), "2.50.4")
        for x in ["7", "2.555", "two", "2.5-foo"]:
            with self.assertRaises(ValueError):
                version_for(x, 0)

    def test_everyday_roles(self):
        self.assertEqual(requested_tier("equipment", "normal", "gear_n.tex"), "native")
        self.assertEqual(requested_tier("weapon", "color", "weapon_s.tex"), "native")
        self.assertEqual(requested_tier("equipment", "mask", "gear_m.tex"), "2x")
        self.assertEqual(requested_tier("bg-hou", "normal", "wall_n.tex"), "native")
        self.assertEqual(requested_tier("human-face", "normal", "face_norm.tex"), "2x")
        self.assertEqual(generation_tier("equipment", "color", "gear_d.tex", 2048, 2048), "native")
        self.assertEqual(
            generation_tier("equipment", "color", "gear_d.tex", 512, 512, profile="legacy"),
            "4x",
        )

    def test_product_no_upward_fallback(self):
        with tempfile.TemporaryDirectory() as t:
            root = pathlib.Path(t)
            p = root / "Clarity - Gear/files/4x/chara/a_d.tex"
            p.parent.mkdir(parents=True)
            p.write_bytes(b"test")
            row = {
                "family": "equipment",
                "role": "color",
                "path": "chara/a_d.tex",
                "tiers": "native,2x,4x",
            }
            self.assertEqual(select_product(row, root, "everyday"), (None, None))
            p = root / "Clarity - Gear/files/native/chara/a_d.tex"
            p.parent.mkdir(parents=True)
            p.write_bytes(b"low")
            self.assertEqual(select_product(row, root, "everyday")[0], "native")

    def test_catalog(self):
        ids = [f for _, fs in CATALOG for f, _ in fs]
        self.assertEqual(len(CATALOG), 5)
        self.assertEqual(len(ids), 19)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn("human-hair", ids)

    def test_preview_order(self):
        versions = [
            "2.5.0-preview.1",
            "2.5.0-preview.2",
            "2.5.0",
            "2.5.1",
            "2.8.0",
            "3.0.0",
        ]
        self.assertEqual(sorted(versions, key=version_key), versions)

    def test_export_is_self_contained_and_source_guarded(self):
        from clarity import ffxiv as kb
        from clarity.packaging import release

        with tempfile.TemporaryDirectory() as t:
            root = pathlib.Path(t)
            source = root / "source"
            product = source / "Clarity - World/files/native/bg/example_n.tex"
            product.parent.mkdir(parents=True)
            product.write_bytes(b"processed texture")
            db = sqlite3.connect(root / "manifest.sqlite")
            db.execute("CREATE TABLE tex(path,family,role,status,tiers,srchash,srcver,note)")
            db.execute(
                "INSERT INTO tex VALUES(?,?,?,?,?,?,?,?)",
                (
                    "bg/example_n.tex",
                    "bg",
                    "normal",
                    "done",
                    "native",
                    "verified",
                    "build",
                    "models:test",
                ),
            )
            db.commit()
            db.close()
            args = argparse.Namespace(
                game="2.5",
                revision=0,
                preview=1,
                destination=str(root / "releases"),
                source=str(source),
                db=str(root / "manifest.sqlite"),
                profile=["everyday"],
                dry_run=False,
            )
            fakegame = argparse.Namespace(sqpack=None)
            with (
                patch("clarity.cli.src_fingerprint", return_value="verified"),
                patch.object(kb, "game", return_value=fakegame),
                patch.object(kb.sqpack, "game_version", return_value="build"),
            ):
                release.export(args)
            folder = (
                root / "releases/2.50.0-preview.1/Clarity - World/General Environments - everyday"
            )
            meta_data = json.loads((folder / "meta.json").read_text())
            mappings = meta_data["DefaultData"]["Files"]
            self.assertEqual(len(mappings), 1)
            for relative in mappings.values():
                self.assertEqual(
                    (folder / relative.replace("\\", "/")).read_bytes(),
                    b"processed texture",
                )
            # An immutable version cannot be overwritten, and changed game resources
            # cannot be silently carried into a stable release.
            args.game = "2.51"
            args.preview = None
            with (
                patch("clarity.cli.src_fingerprint", return_value="changed"),
                patch.object(kb, "game", return_value=fakegame),
                patch.object(kb.sqpack, "game_version", return_value="build"),
                self.assertRaises(RuntimeError),
            ):
                release.export(args)
            # A failed export leaves no directory under the final version name, so
            # the same version can be retried once the sources are verified again.
            self.assertFalse((root / "releases/2.51.0").exists())
            with (
                patch("clarity.cli.src_fingerprint", return_value="verified"),
                patch.object(kb, "game", return_value=fakegame),
                patch.object(kb.sqpack, "game_version", return_value="build"),
            ):
                release.export(args)
            self.assertTrue((root / "releases/2.51.0/release.json").exists())
            self.assertFalse((root / "releases/2.51.0.building").exists())


if __name__ == "__main__":
    unittest.main()
