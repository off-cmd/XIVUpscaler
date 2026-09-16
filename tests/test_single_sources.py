"""Constants that used to be defined twice, and the invariant tying them together."""

from clarity import cli, manifest
from clarity.packaging import penumbra, release
from clarity.processing import roles


def test_processed_roles_has_one_definition() -> None:
    assert not hasattr(cli, "PROCESSED_ROLES")
    assert manifest.PROCESSED_ROLES == ("normal", "mask", "color", "icon", "ui")


def test_processed_roles_are_exactly_the_roles_with_a_processing_function() -> None:
    assert set(manifest.PROCESSED_ROLES) == set(roles.ROLE_FN)


def test_reserved_paths_has_one_definition() -> None:
    assert release.RESERVED_PATHS is penumbra.RESERVED_PATHS
    assert "common/graphics/texture/dummy.tex" in penumbra.RESERVED_PATHS
