"""Where things live.

Every location below is a default, not a rule: the ``CLARITY_*`` variable named beside it
wins when set. The defaults describe one project directory::

    <project>/
    ├── build-output/manifest.sqlite          the working database (ignored, hash-guarded)
    ├── build-output/out/                     packed Penumbra mods
    ├── analysis-specimens/models/            model weights, as obtained
    ├── analysis-specimens/reslogger/         CurrentPathList-<date>.gz from rl2.perchbird.dev
    ├── vendor-tools/texconv/texconv.exe      a *release* DirectXTex texconv
    ├── hash-manifests/texture-fingerprints.tsv
    └── temp-scratch/texconv/                 texconv's per-texture scratch (see SCRATCH)

The project directory is found in this order:

1. ``CLARITY_PROJECT``, when set.
2. The repository root, when this package is imported from a source checkout (the
   directory above the package contains ``pyproject.toml``).
3. The current working directory. This is the case for an installed wheel: ``cd`` into the
   directory you want the pipeline to use and run ``clarity where`` to see what resolved.

The tracked model registry ships inside the package (``clarity/models/registry.json``) so
that an installed copy has a working default; ``CLARITY_REGISTRY`` overrides it.
"""

from __future__ import annotations

import glob
import os

__all__ = [
    "DB",
    "FINGERPRINTS",
    "MODELS",
    "PACKAGE",
    "PROJECT",
    "PROJECTS",
    "REGISTRY",
    "RESLOGGER_DIR",
    "SCRATCH",
    "TEXCONV",
    "TOOL_ROOT",
    "describe",
    "newest_pathlist",
]


def _env(name: str, default: str) -> str:
    """The environment variable ``name``, or ``default`` when unset or empty."""
    value = os.environ.get(name)
    return value if value else default


def _project_root() -> str:
    explicit = os.environ.get("CLARITY_PROJECT")
    if explicit:
        return os.path.abspath(explicit)
    checkout = os.path.normpath(os.path.join(PACKAGE, ".."))
    if os.path.isfile(os.path.join(checkout, "pyproject.toml")):
        return checkout
    return os.getcwd()


PACKAGE: str = os.path.dirname(os.path.abspath(__file__))
"""The ``clarity`` package directory."""

PROJECT: str = _project_root()
"""The project directory every default below hangs off. See the module docstring."""

TOOL_ROOT: str = PROJECT
"""Alias for :data:`PROJECT`, kept for the benchmarking scripts."""

PROJECTS: str = os.path.normpath(os.path.join(PROJECT, ".."))
"""The directory holding sibling projects (``P:\\workspaces`` on the archive drive)."""

# Where texconv's per-texture scratch files go.
#
# This is NOT cosmetic. `_texconv` hands the encoder an *uncompressed* RGBA DDS of the already
# upscaled image, and Python's tempfile defaults to %TEMP%, which on Windows is on C:. A 4x pass
# over a 2048-square source writes 8192x8192x4 = 256 MiB of input plus ~85 MiB of BC7 output, per
# texture, to the system drive -- so the upscale step churns C: even when the project, the models,
# the database and the output folder are all on P:. That is why the low-space warning arrives during
# upscaling and never during packing: packing writes to --out, and cmd_run's disk advisory measures
# --out, so neither of them is looking at the drive actually filling up.
#
# The default keeps it inside the project. Point CLARITY_SCRATCH at a RAM disk or a scratch SSD if
# the write volume matters: the files are written once, read once, and deleted, so this is pure
# churn -- at roughly 340 MiB a texture it dwarfs the ~404 GB of real output.
SCRATCH: str = _env("CLARITY_SCRATCH", os.path.join(PROJECT, "temp-scratch", "texconv"))

DB: str = _env("CLARITY_DB", os.path.join(PROJECT, "build-output", "manifest.sqlite"))
MODELS: str = _env("CLARITY_MODELS", os.path.join(PROJECT, "analysis-specimens", "models"))
REGISTRY: str = _env("CLARITY_REGISTRY", os.path.join(PACKAGE, "models", "registry.json"))
TEXCONV: str = _env(
    "CLARITY_TEXCONV", os.path.join(PROJECT, "vendor-tools", "texconv", "texconv.exe")
)
FINGERPRINTS: str = _env(
    "CLARITY_FINGERPRINTS",
    os.path.join(PROJECT, "hash-manifests", "texture-fingerprints.tsv"),
)
RESLOGGER_DIR: str = os.path.join(PROJECT, "analysis-specimens", "reslogger")


def newest_pathlist() -> str | None:
    """The most recent ResLogger list in ``analysis-specimens/reslogger``, or ``None``.

    ``CLARITY_PATHLIST`` names one explicitly.
    """
    explicit = os.environ.get("CLARITY_PATHLIST")
    if explicit:
        return explicit
    candidates = sorted(glob.glob(os.path.join(RESLOGGER_DIR, "CurrentPathList*.gz")))
    return candidates[-1] if candidates else None


def describe() -> str:
    """The resolved layout as a table, marking anything that does not exist. ``clarity where``."""
    rows = [
        ("project", PROJECT),
        ("db", DB),
        ("models", MODELS),
        ("registry", REGISTRY),
        ("texconv", TEXCONV),
        ("scratch", SCRATCH),
        ("fingerprints", FINGERPRINTS),
        ("path list", newest_pathlist() or "(none)"),
    ]
    return "\n".join(
        f"  {key:<12} {value}{'' if os.path.exists(value) else '   (missing)'}"
        for key, value in rows
    )
