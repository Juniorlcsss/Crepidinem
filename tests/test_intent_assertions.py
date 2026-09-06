"""IntentSpec: the typed goal-assertion vocabulary the PI emits."""

from __future__ import annotations

import pytest

from crepidinem.intent.assertions import (
    CountEvent,
    ForbiddenZone,
    IntentSpec,
    IntentSpecError,
    OrderedVisits,
    PayloadCarriedBetween,
    PickAt,
    PlaceAt,
    validate_against_skeleton,
)


def test_round_trips_through_dict() -> None:
    spec = IntentSpec(
        assertions=(
            PickAt(label="A", point=(0.2, -0.2, 0.15)),
            PlaceAt(label="B", point=(0.2, 0.2, 0.15), tol=0.05),
            PayloadCarriedBetween(pick_label="A", place_label="B"),
            OrderedVisits(points=((0.2, -0.2, 0.15), (0.2, 0.2, 0.15)), labels=("A", "B")),
            ForbiddenZone(label="rack", box=(-0.3, -0.1, 0.0, -0.1, 0.1, 0.15)),
            CountEvent(event="grip", n=1),
        )
    )
    restored = IntentSpec.from_dict(spec.to_dict())
    assert restored == spec


def test_brief_lists_every_assertion_as_a_line() -> None:
    spec = IntentSpec(
        assertions=(
            PickAt(label="A", point=(0.2, -0.2, 0.15)),
            PlaceAt(label="B", point=(0.2, 0.2, 0.15)),
        )
    )
    brief = spec.brief()
    assert "pick" in brief.lower()
    assert "A" in brief and "B" in brief
    assert brief.count("\n") >= 1


def test_from_dict_rejects_an_unknown_kind() -> None:
    with pytest.raises(IntentSpecError):
        IntentSpec.from_dict({"assertions": [{"kind": "teleport", "label": "x"}]})


def test_from_dict_rejects_a_malformed_point() -> None:
    with pytest.raises(IntentSpecError):
        IntentSpec.from_dict(
            {"assertions": [{"kind": "pick_at", "label": "A", "point": [0.2, -0.2]}]}
        )


def test_from_dict_rejects_a_non_numeric_tol() -> None:
    with pytest.raises(IntentSpecError):
        IntentSpec.from_dict(
            {"assertions": [{"kind": "pick_at", "label": "A", "point": [0, 0, 0.1], "tol": "abc"}]}
        )


def test_from_dict_rejects_non_list_labels_on_ordered_visits() -> None:
    with pytest.raises(IntentSpecError):
        IntentSpec.from_dict(
            {"assertions": [{"kind": "ordered_visits", "points": [[0, 0, 0.1]], "labels": 5}]}
        )


def test_skeleton_validation_flags_a_missing_required_kind() -> None:
    spec = IntentSpec(assertions=(PickAt(label="A", point=(0.0, 0.0, 0.1)),))
    with pytest.raises(ValueError, match="place_at"):
        validate_against_skeleton(spec, frozenset({"pick_at", "place_at"}))


def test_skeleton_validation_passes_when_all_required_kinds_present() -> None:
    spec = IntentSpec(
        assertions=(
            PickAt(label="A", point=(0.0, 0.0, 0.1)),
            PlaceAt(label="B", point=(0.1, 0.0, 0.1)),
        )
    )
    validate_against_skeleton(spec, frozenset({"pick_at", "place_at"}))
