"""The ``release`` subcommand's argparse surface and two crashes it used to have.

Regressions:
  * ``export()`` read ``a.preview`` but no parser defined ``--preview``, so every real
    ``clarity release`` invocation raised ``AttributeError``. The test suite hid it by
    hand-building a Namespace.
  * ``--profile 4x`` indexed ``roles.POLICY[family]`` for every catalog family, and
    ``common`` (Loading Screens) has no policy entry, so it raised ``KeyError``.
"""

import argparse

import pytest

from clarity.packaging import release
from clarity.processing import roles


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="unused.sqlite")
    release.add_parser(ap.add_subparsers(dest="cmd", required=True))
    return ap


def test_preview_is_a_real_flag() -> None:
    a = _parser().parse_args(
        ["release", "--game", "7.56", "--revision", "0", "--source", "s", "--preview", "3"]
    )
    assert a.preview == 3


def test_preview_defaults_to_none_meaning_stable() -> None:
    a = _parser().parse_args(["release", "--game", "7.56", "--revision", "0", "--source", "s"])
    assert a.preview is None


@pytest.mark.parametrize(
    ("game", "revision", "preview", "expected"),
    [
        ("7.56", 0, None, "7.56.0"),
        ("7.5", 2, None, "7.50.2"),
        ("7.56h1", 1, None, "7.56.1"),
        ("7.56", 0, 1, "7.56.0-preview.1"),
        ("7.56", 4, 12, "7.56.4-preview.12"),
    ],
)
def test_version_for_includes_the_preview_segment(game, revision, preview, expected) -> None:
    assert release.version_for(game, revision, preview) == expected


def test_preview_orders_below_its_stable_release() -> None:
    assert release.version_key("7.56.0-preview.9") < release.version_key("7.56.0")


def test_preview_number_must_be_positive() -> None:
    with pytest.raises(ValueError):
        release.version_for("7.56", 0, 0)


def test_every_catalog_family_survives_a_4x_policy_lookup() -> None:
    families = [fam for _group, fams in release.CATALOG for fam, _title in fams]
    assert "common" in families and "common" not in roles.POLICY
    for fam in families:
        # The lookup export() performs for --profile 4x; must never KeyError.
        roles.POLICY.get(fam, (None, 0))[0]


@pytest.mark.parametrize("family", ["human-hair", "vfx", "common"])
def test_generation_tier_is_none_for_families_the_policy_excludes(family) -> None:
    # top_tier() and skip_reason() both say "never"; generation_tier used to say "native".
    assert roles.top_tier(family, 512, 512) is None
    assert release.generation_tier(family, "color", "x.tex", 512, 512) is None


def test_generation_tier_still_caps_processed_families() -> None:
    assert release.generation_tier("equipment", "color", "x_d.tex", 512, 512) == "2x"
    assert release.generation_tier("equipment", "normal", "x_n.tex", 512, 512) == "native"
