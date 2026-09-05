"""The four gates that stand between a search result and an enforced limit.

Each gate gets a test that fails for that gate and no other reason, so a
regression names itself.  The threat model is the point: a search result is
untrusted text, and the dangerous direction is *loosening* a safety limit.
"""

from __future__ import annotations

from typing import Any

from crepidinem.mcp_tools.harness import PhysicsLimits
from crepidinem.research.hardware import ENVELOPE, LimitFinding, _judge, _normalise

REAL_URL = "https://www.universal-robots.com/manuals/ur5e/tech_spec_sheet.htm"
CORPUS_TEXT = (
    "UR5e technical specifications. Maximum payload | 5 kg / 11 lb. "
    "Speed | Joints: Max 180 deg/s. Tool: Approx. 1 m/s. Reach | 850 mm."
)
CORPUS = _normalise(CORPUS_TEXT)
KNOWN = frozenset({REAL_URL})


def judge(
    raw: dict[str, Any],
    *,
    trust: str = "envelope",
    limits: PhysicsLimits | None = None,
) -> LimitFinding | None:
    """Run one proposed finding through the gates."""
    return _judge(
        raw,
        limits=limits or PhysicsLimits(),
        known_urls=KNOWN,
        corpus=CORPUS,
        trust=trust,
    )


# ----------------------------------------------------------------- gate 1


def test_gate1_only_whitelisted_fields_are_researchable() -> None:
    """Cell geometry describes this bench. No web page may move it."""
    for field in ("table_height", "obstacles", "workspace_radius", "__class__"):
        assert (
            judge({"field": field, "value": 0.5, "source_url": REAL_URL, "quote": CORPUS_TEXT})
            is None
        ), f"{field} must not be researchable"

    # The four capability limits are the only way in.
    assert set(ENVELOPE) == {
        "max_velocity",
        "max_acceleration",
        "max_payload_kg",
        "max_gripper_force_n",
    }


# ----------------------------------------------------------------- gate 2


def test_gate2_an_implausible_value_is_refused_however_well_cited() -> None:
    """A poisoned page claiming 400 m/s is refused on physics, not on trust."""
    finding = judge(
        {
            "field": "max_velocity",
            "value": 400.0,
            "source_url": REAL_URL,
            "quote": CORPUS_TEXT,
        }
    )
    assert finding is not None
    assert finding.accepted is False
    assert "plausible range" in finding.reason


# ----------------------------------------------------------------- gate 3


def test_gate3_a_fabricated_citation_is_refused() -> None:
    """Catches hallucinated *provenance*, not just hallucinated numbers."""
    finding = judge(
        {
            "field": "max_velocity",
            "value": 1.0,
            "source_url": "https://totally-real-specs.example.com/ur5e",
            "quote": CORPUS_TEXT,
        }
    )
    assert finding is not None
    assert finding.accepted is False
    assert "did not return" in finding.reason


# ----------------------------------------------------------------- gate 4


def test_gate4_loosening_requires_a_verbatim_quote() -> None:
    """Widening is the dangerous direction, so it must be quoted to be believed."""
    base = PhysicsLimits(max_velocity=0.5)

    unquoted = judge(
        {"field": "max_velocity", "value": 1.0, "source_url": REAL_URL, "quote": ""},
        limits=base,
    )
    assert unquoted is not None
    assert unquoted.accepted is False
    assert "not quoted" in unquoted.reason

    quoted = judge(
        {
            "field": "max_velocity",
            "value": 1.0,
            "source_url": REAL_URL,
            "quote": "Speed | Joints: Max 180 deg/s. Tool: Approx. 1 m/s.",
        },
        limits=base,
    )
    assert quoted is not None
    assert quoted.accepted is True
    assert quoted.previous == 0.5


def test_gate4_tightening_needs_no_evidence_at_all() -> None:
    """The asymmetry is the design: evidence to relax, none to restrict.

    An over-tight limit rejects a script. An over-loose one breaks an arm.
    """
    finding = judge(
        {"field": "max_velocity", "value": 0.2, "source_url": REAL_URL, "quote": ""},
        limits=PhysicsLimits(max_velocity=0.5),
    )
    assert finding is not None
    assert finding.accepted is True
    assert "tightens" in finding.reason

    # narrow-only refuses every loosening outright, whatever the evidence.
    refused = judge(
        {
            "field": "max_velocity",
            "value": 1.0,
            "source_url": REAL_URL,
            "quote": "Tool: Approx. 1 m/s.",
        },
        limits=PhysicsLimits(max_velocity=0.5),
        trust="narrow-only",
    )
    assert refused is not None
    assert refused.accepted is False
    assert "narrow-only" in refused.reason
