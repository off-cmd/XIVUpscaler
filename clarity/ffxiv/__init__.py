"""Shared FINAL FANTASY XIV binary file-access layer.

One copy of the readers and writers that more than one tool needs: the sqpack index/dat
reader, the ``.tex`` texture format, ``.mtrl`` materials, ``.mdl`` model string tables and
vertex patching, ``.imc`` variant tables and the ``.exd`` Excel sheets.

Aliases (``exdfile``, ``mtrlfile``, ``texfile``) are kept for the names these modules had
when they lived in ``ffxiv-kbtools``; the canonical names are the module names.
"""

from . import bc7enc, exd, imcfile, mdlpatch, mdlstrings, mtrl, sqpack, tex, texdecode, texwrite

exdfile = exd
mtrlfile = mtrl
texfile = tex

__all__ = [
    "GameData",
    "bc7enc",
    "exd",
    "exdfile",
    "game",
    "imcfile",
    "mdlpatch",
    "mdlstrings",
    "mtrl",
    "mtrlfile",
    "sqpack",
    "tex",
    "texdecode",
    "texfile",
    "texwrite",
]


def GameData(path: str) -> sqpack.GameData:  # noqa: N802 - kept for callers that use the old name
    """Open the game's ``sqpack`` directory. Thin alias for :class:`sqpack.GameData`."""
    return sqpack.GameData(path)


_GD: sqpack.GameData | None = None


def game(path: str | None = None) -> sqpack.GameData:
    """Return the process-wide game handle, locating the install on first use.

    Args:
        path: An explicit ``sqpack`` directory. When omitted, :func:`sqpack.find_game`
            searches the usual install locations and the ``FFXIV_SQPACK`` /
            ``FFXIV_PATH`` environment variables.
    """
    global _GD  # noqa: PLW0603 - the one process-wide game handle
    if _GD is None:
        _GD = sqpack.GameData(sqpack.find_game(path))
    return _GD
