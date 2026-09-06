"""The typed, machine checkable goal assertions the Principal Investigator emits"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

__all__ = [
    "Assertion",
    "CountEvent",
    "DwellAt",
    "ForbiddenZone",
    "IntentSpec",
    "IntentSpecError",
    "OrderedVisits",
    "PayloadCarriedBetween",
    "PickAt",
    "PlaceAt",
    "Vec3",
    "validate_against_skeleton",
]

Vec3 = tuple[float, float, float]
Box = tuple[float, float, float, float, float, float]

_DEFAULT_TOL = 0.03
_COUNTABLE = frozenset({"grip", "release", "move_to"})
_VEC3_LEN = 3
_BOX_LEN = 6


class IntentSpecError(ValueError):
    ...


def _as_vec3(value: object, *, context: str) -> Vec3:
    if not isinstance(value, (list, tuple)) or len(value) != _VEC3_LEN:
        msg = f"{context}: expected [x, y, z], got {value!r}"
        raise IntentSpecError(msg)
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as exc:
        msg = f"{context}: coordinates must be numbers ({value!r})"
        raise IntentSpecError(msg) from exc


def _as_box(value: object, *, context: str) -> Box:
    if not isinstance(value, (list, tuple)) or len(value) != _BOX_LEN:
        msg = f"{context}: expected [x0, y0, z0, x1, y1, z1], got {value!r}"
        raise IntentSpecError(msg)
    try:
        x0, y0, z0, x1, y1, z1 = (float(v) for v in value)
    except (TypeError, ValueError) as exc:
        msg = f"{context}: box bounds must be numbers ({value!r})"
        raise IntentSpecError(msg) from exc
    return (min(x0, x1), min(y0, y1), min(z0, z1), max(x0, x1), max(y0, y1), max(z0, z1))


def _as_str(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        msg = f"{context}: expected a non-empty string, got {value!r}"
        raise IntentSpecError(msg)
    return value.strip()


def _as_float(value: object, default: float, *, context: str, field: str = "tol") -> float:
    if value is None:
        return default
    
    if not isinstance(value, (int, float, str)):
        msg = f"{context}: {field!r} must be a number"
        raise IntentSpecError(msg)
    
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        msg = f"{context}: {field!r} must be a number"
        raise IntentSpecError(msg) from exc


@dataclass(frozen=True, slots=True)
class PickAt:

    label: str
    point: Vec3
    tol: float = _DEFAULT_TOL
    kind: ClassVar[str] = "pick_at"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "label": self.label, "point": list(self.point), "tol": self.tol}

    def brief(self) -> str:
        return f"- pick_at {self.label}: grip within {self.tol} m of {self.point}"


@dataclass(frozen=True, slots=True)
class PlaceAt:

    label: str
    point: Vec3
    tol: float = _DEFAULT_TOL
    kind: ClassVar[str] = "place_at"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "label": self.label, "point": list(self.point), "tol": self.tol}

    def brief(self) -> str:
        return f"- place_at {self.label}: release within {self.tol} m of {self.point}"


@dataclass(frozen=True, slots=True)
class OrderedVisits:

    points: tuple[Vec3, ...]
    tol: float = _DEFAULT_TOL
    labels: tuple[str, ...] = ()
    kind: ClassVar[str] = "ordered_visits"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "points": [list(p) for p in self.points],
            "tol": self.tol,
            "labels": list(self.labels),
        }

    def brief(self) -> str:
        names = ", ".join(self.labels) if self.labels else f"{len(self.points)} points"
        return f"- ordered_visits: reach {names} in that order (tol {self.tol} m)"


@dataclass(frozen=True, slots=True)
class PayloadCarriedBetween:

    pick_label: str
    place_label: str
    kind: ClassVar[str] = "payload_carried_between"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "pick_label": self.pick_label, "place_label": self.place_label}

    def brief(self) -> str:
        return (
            f"- payload_carried_between {self.pick_label} -> {self.place_label}: "
            f"gripper stays closed the whole way"
        )


@dataclass(frozen=True, slots=True)
class ForbiddenZone:

    label: str
    box: Box
    kind: ClassVar[str] = "forbidden_zone"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "label": self.label, "box": list(self.box)}

    def brief(self) -> str:
        return f"- forbidden_zone {self.label}: never enter {self.box}"


@dataclass(frozen=True, slots=True)
class DwellAt:

    label: str
    point: Vec3
    min_seconds: float
    tol: float = _DEFAULT_TOL
    kind: ClassVar[str] = "dwell_at"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "point": list(self.point),
            "min_seconds": self.min_seconds,
            "tol": self.tol,
        }

    def brief(self) -> str:
        return f"- dwell_at {self.label}: hold {self.point} for >= {self.min_seconds} s"


@dataclass(frozen=True, slots=True)
class CountEvent:

    event: str
    n: int
    kind: ClassVar[str] = "count"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "event": self.event, "n": self.n}

    def brief(self) -> str:
        """return human readable form for planner/coder brief"""
        return f"- count: exactly {self.n} {self.event} event(s)"


Assertion = (
    PickAt | PlaceAt | OrderedVisits | PayloadCarriedBetween | ForbiddenZone | DwellAt | CountEvent
)


def _pick_at_from_dict(data: Mapping[str, Any], *, context: str) -> PickAt:
    return PickAt(
        label=_as_str(data.get("label"), context=context),
        point=_as_vec3(data.get("point"), context=context),
        tol=_as_float(data.get("tol"), _DEFAULT_TOL, context=context),
    )


def _place_at_from_dict(data: Mapping[str, Any], *, context: str) -> PlaceAt:
    return PlaceAt(
        label=_as_str(data.get("label"), context=context),
        point=_as_vec3(data.get("point"), context=context),
        tol=_as_float(data.get("tol"), _DEFAULT_TOL, context=context),
    )


def _ordered_visits_from_dict(data: Mapping[str, Any], *, context: str) -> OrderedVisits:
    raw = data.get("points")
    if not isinstance(raw, (list, tuple)) or not raw:
        msg = f"{context}: 'points' must be a non-empty list"
        raise IntentSpecError(msg)
    labels = data.get("labels") or ()

    if labels and not isinstance(labels, (list, tuple)):
        msg = f"{context}: 'labels' must be a list of strings"
        raise IntentSpecError(msg)
    
    return OrderedVisits(
        points=tuple(_as_vec3(p, context=context) for p in raw),
        tol=_as_float(data.get("tol"), _DEFAULT_TOL, context=context),
        labels=tuple(str(x) for x in labels),
    )


def _payload_from_dict(data: Mapping[str, Any], *, context: str) -> PayloadCarriedBetween:
    return PayloadCarriedBetween(
        pick_label=_as_str(data.get("pick_label"), context=context),
        place_label=_as_str(data.get("place_label"), context=context),
    )


def _forbidden_zone_from_dict(data: Mapping[str, Any], *, context: str) -> ForbiddenZone:
    return ForbiddenZone(
        label=_as_str(data.get("label"), context=context),
        box=_as_box(data.get("box"), context=context),
    )


def _dwell_at_from_dict(data: Mapping[str, Any], *, context: str) -> DwellAt:
    raw = data.get("min_seconds")
    if raw is None:
        msg = f"{context}: 'min_seconds' must be a number"
        raise IntentSpecError(msg)
    min_seconds = _as_float(raw, 0.0, context=context, field="min_seconds")
    return DwellAt(
        label=_as_str(data.get("label"), context=context),
        point=_as_vec3(data.get("point"), context=context),
        min_seconds=min_seconds,
        tol=_as_float(data.get("tol"), _DEFAULT_TOL, context=context),
    )


def _count_from_dict(data: Mapping[str, Any], *, context: str) -> CountEvent:
    event = _as_str(data.get("event"), context=context)
    if event not in _COUNTABLE:
        msg = f"{context}: 'event' must be one of {sorted(_COUNTABLE)}"
        raise IntentSpecError(msg)
    try:
        n = int(data["n"])
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"{context}: 'n' must be an integer"
        raise IntentSpecError(msg) from exc
    return CountEvent(event=event, n=n)


_BUILDERS: dict[str, Callable[..., Assertion]] = {
    "pick_at": _pick_at_from_dict,
    "place_at": _place_at_from_dict,
    "ordered_visits": _ordered_visits_from_dict,
    "payload_carried_between": _payload_from_dict,
    "forbidden_zone": _forbidden_zone_from_dict,
    "dwell_at": _dwell_at_from_dict,
    "count": _count_from_dict,
}


def _assertion_from_dict(data: Mapping[str, Any]) -> Assertion:
    kind = data.get("kind")
    builder = _BUILDERS.get(kind) if isinstance(kind, str) else None
    if builder is None:
        msg = f"unknown assertion kind {kind!r}; expected one of {sorted(_BUILDERS)}"
        raise IntentSpecError(msg)
    
    return builder(data, context=f"assertion {kind}")


@dataclass(frozen=True, slots=True)
class IntentSpec:
    """goal assertions to check against trajectory"""

    assertions: tuple[Assertion, ...] = field(default_factory=tuple)

    def kinds(self) -> frozenset[str]:
        """Return the set of kind strings in spec"""
        return frozenset(a.kind for a in self.assertions)

    def brief(self) -> str:
        """Return requirements block for the coder brief"""
        if not self.assertions:
            return "INTENT REQUIREMENTS: none."
        
        lines = ["INTENT REQUIREMENTS (the run must satisfy every line):"]
        lines.extend(a.brief() for a in self.assertions)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"assertions": [a.to_dict() for a in self.assertions]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IntentSpec:
        raw = data.get("assertions")
        if raw is None and isinstance(data, Mapping) and "kind" in data:
            raw = [data]

        if not isinstance(raw, (list, tuple)):
            msg = "intent_spec must have an 'assertions' list"
            raise IntentSpecError(msg)
        
        parsed: list[Assertion] = []
        for entry in raw:
            if not isinstance(entry, Mapping):
                msg = f"each assertion must be an object, got {entry!r}"
                raise IntentSpecError(msg)
            
            parsed.append(_assertion_from_dict(entry))
        return cls(assertions=tuple(parsed))


def validate_against_skeleton(spec: IntentSpec, required: frozenset[str]) -> None:
    missing = sorted(required - spec.kinds())
    if missing:
        msg = f"intent_spec is missing required assertion kind(s): {', '.join(missing)}"
        raise ValueError(msg)
