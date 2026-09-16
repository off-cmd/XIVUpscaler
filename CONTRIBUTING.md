# Contributing

XIVUpscaler is a [uv](https://docs.astral.sh/uv/) project. Every command below runs from a
checkout with `uv` installed; nothing else needs to be on the machine.

## Set up

```
uv sync
```

That creates `.venv/` with the runtime dependencies (including the CUDA 12.8 torch wheels
pinned in `pyproject.toml`) and the `dev` group: ruff, ty, pytest, pytest-cov, build, twine.
Python 3.12 is pinned in `.python-version`; 3.11 also works. 3.13 does not, because
`texture2ddecoder` publishes no wheels for it.

On a machine without an NVIDIA GPU, or in a container, swap in the CPU wheels once:

```
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

Everything except `clarity run` and `clarity modup` is numpy-only and works either way.

## The quality gate

Four commands; CI runs exactly these.

```
uv run ruff check .            # lint
uv run ruff format --check .   # formatting (drop --check to apply)
uv run ty check                # types
uv run pytest --cov            # tests with branch coverage
```

`ruff check --fix` handles the mechanical findings. The rule set and every ignore, with its
reason, is in `pyproject.toml` under `[tool.ruff]`; if you think a rule is wrong for this
codebase, change the config with a comment rather than sprinkling `noqa`.

The tests need no game install, no GPU and no model weights. They build synthetic `.tex`,
`.imc`, `.exd` and `.mdl` files in memory and run the per-role maths against a numpy fake of
the inference engine (`tests/test_processing_roles.py`). Anything that genuinely needs the
GPU is exercised by `clarity probe` and `clarity run` on a real machine, and
`clarity/processing/engine.py` is omitted from coverage for that reason.

## Build and install

```
uv build                              # dist/*.tar.gz and dist/*.whl
uvx twine check dist/*                # metadata is valid for an index
uv pip install dist/*.whl             # into the current venv, to try the installed layout
```

An installed copy resolves its project directory from `CLARITY_PROJECT`, else the current
directory. `clarity where` prints what resolved.

## How the code is organised

```
clarity/ffxiv/       the game's binary formats: sqpack, .tex, .mtrl, .mdl, .imc, .exd
clarity/processing/  inference (engine.py) and one module per texture role
clarity/packaging/   Penumbra mod folders and immutable releases
clarity/manifest.py  the SQLite manifest: enumeration, classification, row lifecycle
clarity/texio.py     reading and writing .tex, texconv integration, the BC7 fallback
clarity/cli.py       `clarity <command>`; the runbook is in README.md
scripts/             model benchmarking, not part of the package
```

Within a module: docstring, imports, constants, types, helpers, public functions, then any
`__main__` block. Role functions take an `Upscaler` (a `Protocol` in
`clarity/processing/utils.py`), not the concrete `Engine`, which is what keeps them testable.

## Things that will bite

* **Every path Penumbra is handed must be ASCII.** The game's loader reads mod paths in
  the system ANSI code page; an em dash in a mod name once made every character in view
  invisible. `tests/test_penumbra_names.py` guards it.
* **A texture is committed only after every tier is on disk.** Interrupting a run leaves
  the current row `planned`, never `done` and never `failed`.
* **`requeue` must keep the `changed at <version>` note.** `run --since` selects the
  patch delta by it.
* **Do not add a second copy of a constant.** `PROCESSED_ROLES` lives in `manifest.py`,
  `RESERVED_PATHS` in `packaging/penumbra.py`; `tests/test_single_sources.py` will tell you.

## Commit messages

Imperative subject under 72 characters with a conventional prefix (`fix:`, `feat:`,
`refactor:`, `test:`, `style:`, `docs:`, `ci:`), then a body that says what was wrong and
why the change is the fix, not what the diff does. The history is written to be read.
