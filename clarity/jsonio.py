"""JSON files the way Penumbra and this project write them.

Penumbra's own files are UTF-8, sometimes with a BOM (``utf-8-sig`` reads both), and they
carry non-ASCII mod names, so writes keep ``ensure_ascii=False``. Every JSON read or write
in the package goes through these two functions so that the encoding decision is made once.
"""

from __future__ import annotations

import json
import os
from typing import Any

__all__ = ["read_json", "write_json"]


def read_json(path: str | os.PathLike[str]) -> Any:
    """Parse the JSON file at ``path``, accepting a UTF-8 BOM."""
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: str | os.PathLike[str], data: Any, *, indent: int = 2) -> None:
    """Write ``data`` as UTF-8 JSON with a trailing newline, non-ASCII preserved."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
        f.write("\n")
