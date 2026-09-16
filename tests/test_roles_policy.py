"""The tier policy: ``roles.POLICY``, ``top_tier`` and ``tiers_below``.

Pins the two halving rules in ``top_tier`` exactly as written -- a source edge *above* the
family cap sends the texture to ``native`` outright (the cap test does not involve the scale,
so it halves until nothing is left), and an output edge *above* ``MAX_EDGE_OUT`` (4096) drops
one tier at a time, so an output of exactly 4096 is allowed -- plus what an override does on
a family whose policy tier is None (it is honoured, and the zero cap then halves it all the
way to ``native``).
No model, no torch: ``roles`` is imported as a plain module.
"""

import pytest

from clarity.processing import roles

# ---------------------------------------------------------------------------- constants


def test_tier_scale() -> None:
    assert roles.TIER_SCALE == {"4x": 4, "2x": 2, "native": 1}
    assert roles.MAX_EDGE_OUT == 4096


def test_policy_shape() -> None:
    for family, (tier, cap) in roles.POLICY.items():
        assert isinstance(family, str) and isinstance(cap, int), family
        if tier is None:
            assert cap == 0, family
        else:
            assert tier in roles.TIER_SCALE and cap > 0, family


def test_policy_names_the_families_the_classifier_produces() -> None:
    assert roles.POLICY["human-hair"] == (None, 0)
    assert roles.POLICY["vfx"] == (None, 0)
    assert "common" not in roles.POLICY  # loading screens are gated by the pack catalog instead
    for fam in ("bg", "bg-hou", "bg-ind", "bgcommon", "bgcommon-hou", "bgcommon-mji"):
        assert roles.POLICY[fam][0] == "2x", fam
    for fam in ("equipment", "accessory", "weapon", "ui-icon"):
        assert roles.POLICY[fam][0] == "4x", fam


def test_role_fn_table() -> None:
    assert set(roles.ROLE_FN) == {"normal", "mask", "color", "icon", "ui"}
    assert all(callable(fn) for fn in roles.ROLE_FN.values())
    assert roles.ROLE_FN["icon"] is roles.ROLE_FN["ui"]


# ----------------------------------------------------------------------------- top_tier
TOP_TIER_TABLE = [
    # equipment: top 4x, cap 1024
    ("equipment", 512, 512, None, "4x"),
    ("equipment", 1024, 1024, None, "4x"),  # 1024 is not > 1024, and 1024*4 == 4096 is allowed
    ("equipment", 1024, 256, None, "4x"),  # the larger edge decides
    ("equipment", 2048, 2048, None, "native"),  # cap rule: > 1024 halves twice
    ("equipment", 2048, 64, None, "native"),
    ("accessory", 512, 512, None, "4x"),
    ("weapon", 2048, 2048, None, "native"),
    # monster: top 2x, cap 2048 -- 2048*2 == 4096 is exactly the output cap, and allowed
    ("monster", 2048, 2048, None, "2x"),
    ("monster", 4096, 4096, None, "native"),
    ("demihuman", 1024, 1024, None, "2x"),
    # ui-icon: top 4x, cap 512
    ("ui-icon", 512, 512, None, "4x"),
    ("ui-icon", 1024, 1024, None, "native"),  # over the cap: native only, never a middle tier
    ("ui-icon", 40, 40, None, "4x"),
    # world: bg capped at 1024, housing at 2048
    ("bg", 1024, 1024, None, "2x"),
    ("bg", 2048, 2048, None, "native"),
    ("bg-hou", 2048, 2048, None, "2x"),
    ("bgcommon-mji", 2048, 2048, None, "native"),
    # families with no tier
    ("human-hair", 512, 512, None, None),
    ("vfx", 64, 64, None, None),
    ("common", 1920, 1080, None, None),
    ("no-such-family", 64, 64, None, None),
    # overrides
    ("equipment", 512, 512, "2x", "2x"),
    ("equipment", 512, 512, "native", "native"),
    ("equipment", 2048, 2048, "4x", "native"),  # an override does not lift the cap
    ("monster", 1024, 1024, "4x", "4x"),  # 1024*4 == 4096: allowed
    ("monster", 2048, 2048, "4x", "2x"),  # 2048*4 > 4096 halves once; 2048*2 == 4096 stays
    ("ui-icon", 1024, 1024, "4x", "native"),
]


@pytest.mark.parametrize(
    ("family", "w", "h", "override", "expected"),
    TOP_TIER_TABLE,
    ids=[f"{f}-{w}x{h}-{o or 'policy'}" for f, w, h, o, _ in TOP_TIER_TABLE],
)
def test_top_tier(family, w, h, override, expected) -> None:
    assert roles.top_tier(family, w, h, override) == expected


def test_output_cap_is_exclusive() -> None:
    # The rule is `max(w, h) * s > MAX_EDGE_OUT`: an output edge equal to the cap is fine, one
    # pixel over is not. Exercised through an override so the family cap is not what decides.
    assert roles.top_tier("monster", 1024, 1024, "4x") == "4x"
    assert roles.top_tier("monster", 1025, 1025, "4x") == "2x"
    assert roles.top_tier("monster", 2048, 2048, "2x") == "2x"
    assert roles.top_tier("monster", 2049, 2048, "2x") == "native"


def test_family_cap_is_exclusive_and_all_or_nothing() -> None:
    # `max(w, h) > cap` does not involve the scale, so once a source is over its family cap the
    # loop halves all the way to native: there is no "one tier down" for an oversized source.
    assert roles.top_tier("equipment", 1024, 1024) == "4x"
    assert roles.top_tier("equipment", 1025, 1024) == "native"
    assert roles.top_tier("ui-icon", 512, 512) == "4x"
    assert roles.top_tier("ui-icon", 513, 512) == "native"
    assert roles.top_tier("bg", 1025, 1025) == "native"


@pytest.mark.parametrize("family", ["human-hair", "vfx", "no-such-family"])
def test_override_cannot_resurrect_an_excluded_family(family) -> None:
    # Regression: POLICY gives (None, 0) and the override used to replace the None tier, so
    # `run --family human-hair --top 2x` produced native-tier products for a family the
    # policy excludes. An override picks a tier; it does not pick which families exist.
    assert roles.top_tier(family, 512, 512, "2x") is None
    assert roles.top_tier(family, 512, 512, "4x") is None
    assert roles.top_tier(family, 0, 0, "2x") is None


def test_unknown_override_raises() -> None:
    with pytest.raises(KeyError):
        roles.top_tier("equipment", 512, 512, "8x")


# -------------------------------------------------------------------------- tiers_below
@pytest.mark.parametrize(
    ("top", "expected"),
    [
        ("4x", ["4x", "2x", "native"]),
        ("2x", ["2x", "native"]),
        ("native", ["native"]),
    ],
)
def test_tiers_below_starts_with_the_top_tier(top, expected) -> None:
    # The name says "below", the docstring says "starting with `top`"; the code and every
    # caller (`process()` slices [1:], `run` enumerates offsets from 0) rely on the latter.
    assert roles.tiers_below(top) == expected


def test_tiers_below_rejects_an_unknown_tier() -> None:
    with pytest.raises(ValueError):
        roles.tiers_below("8x")


def test_every_policy_top_has_a_chain_ending_in_native() -> None:
    for family, (tier, _cap) in roles.POLICY.items():
        if tier is not None:
            chain = roles.tiers_below(tier)
            assert chain[0] == tier and chain[-1] == "native", family
            assert [roles.TIER_SCALE[t] for t in chain] == sorted(
                (roles.TIER_SCALE[t] for t in chain), reverse=True
            )
