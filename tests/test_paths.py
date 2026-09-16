"""Project-root resolution and the environment overrides."""

import importlib
import os
import sys

import pytest


def _reload(monkeypatch, **env):
    for k in list(os.environ):
        if k.startswith("CLARITY_"):
            monkeypatch.delenv(k)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    sys.modules.pop("clarity.paths", None)
    return importlib.import_module("clarity.paths")


def test_checkout_resolves_to_the_repository_root(monkeypatch) -> None:
    paths = _reload(monkeypatch)
    assert os.path.isfile(os.path.join(paths.PROJECT, "pyproject.toml"))
    assert paths.TOOL_ROOT == paths.PROJECT


def test_clarity_project_wins(monkeypatch, tmp_path) -> None:
    paths = _reload(monkeypatch, CLARITY_PROJECT=str(tmp_path))
    assert str(tmp_path) == paths.PROJECT
    assert os.path.join(str(tmp_path), "build-output", "manifest.sqlite") == paths.DB


def test_registry_default_is_package_data(monkeypatch) -> None:
    paths = _reload(monkeypatch)
    assert paths.REGISTRY.endswith(os.path.join("clarity", "models", "registry.json"))
    assert os.path.isfile(paths.REGISTRY)


@pytest.mark.parametrize(
    ("var", "attr"),
    [
        ("CLARITY_DB", "DB"),
        ("CLARITY_MODELS", "MODELS"),
        ("CLARITY_REGISTRY", "REGISTRY"),
        ("CLARITY_TEXCONV", "TEXCONV"),
        ("CLARITY_SCRATCH", "SCRATCH"),
        ("CLARITY_FINGERPRINTS", "FINGERPRINTS"),
    ],
)
def test_every_override_is_honoured(monkeypatch, var, attr) -> None:
    paths = _reload(monkeypatch, **{var: "/somewhere/else"})
    assert getattr(paths, attr) == "/somewhere/else"


def test_empty_override_means_unset(monkeypatch) -> None:
    paths = _reload(monkeypatch, CLARITY_DB="")
    assert paths.DB.endswith("manifest.sqlite")


def test_pathlist_override(monkeypatch) -> None:
    paths = _reload(monkeypatch, CLARITY_PATHLIST="/x/CurrentPathList.gz")
    assert paths.newest_pathlist() == "/x/CurrentPathList.gz"


def test_describe_marks_missing_entries(monkeypatch, tmp_path) -> None:
    paths = _reload(monkeypatch, CLARITY_PROJECT=str(tmp_path))
    text = paths.describe()
    assert "(missing)" in text
    assert "registry" in text and "project" in text
