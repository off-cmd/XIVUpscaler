"""Every path Penumbra is handed must be ASCII.

The game's loader reads mod paths in the system ANSI code page: the em dash in
"Clarity — Gear" became "Clarity â€” Gear", every redirect resolved to a folder that did
not exist, and every character in view went invisible. The header of penumbra.py records
the incident. Option *labels* live inside UTF-8 JSON and are display-only, so "2×" there
is fine; mod names, folder names and sort-order paths are not.
"""

import json
import pathlib

from clarity.packaging import penumbra


def test_mod_names_are_ascii() -> None:
    for name in penumbra.MODS:
        assert name.isascii(), name


def test_sort_order_folder_is_ascii() -> None:
    assert penumbra.FOLDER.isascii()


def test_merge_penumbra_writes_ascii_paths(tmp_path) -> None:
    path = penumbra.merge_penumbra(str(tmp_path), list(penumbra.MODS))
    so = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    for mod, folder in so["Data"].items():
        assert mod.isascii() and folder.isascii(), (mod, folder)


def test_merge_icon_twins_writes_ascii_paths(tmp_path) -> None:
    # Regression: this format string carried an em dash in the same file whose header
    # documents that an em dash in a Penumbra path made every character invisible.
    path = penumbra.merge_icon_twins(str(tmp_path), {"Some Icons": "Some Icons (upscaled)"})
    so = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    folder = so["Data"]["Some Icons (upscaled)"]
    assert folder.isascii(), folder
    assert folder.startswith("9 Interface/Icons - upscaled (G6)/940 ")
