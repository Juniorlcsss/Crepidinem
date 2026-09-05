"""Fold a run's event stream into something renderable"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crepidinem.orchestrator.events import RunEvent

__all__ = ["AttemptView", "DashboardState"]

_MAX_STREAM_CHARS = 60_000

#cartesian point
_POINT_DIMS = 3


@dataclass(slots=True)
class AttemptView:
    """Everything the dashboard knows about one attempt."""

    index: int
    code: str = ""
    streamed: str = ""
    reasoning: str = ""
    feedback: str = ""
    verdict: str = ""
    error: str = ""
    passed: bool | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)
    waypoints: list[dict[str, Any]] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    generation_s: float = 0.0

    @property
    def settled(self) -> bool:
        return self.passed is not None

    @property
    def badge(self) -> str:
        if self.passed is None:
            return "running"
        return "certified" if self.passed else "blocked"

    @property
    def failure_point(self) -> tuple[float, float, float] | None:
        """Where the violation happened, when the harness pinpointed it."""
        point = self.detail.get("point")
        if isinstance(point, list) and len(point) == _POINT_DIMS:
            try:
                return (float(point[0]), float(point[1]), float(point[2]))
            except (TypeError, ValueError):
                return None
        return None

    @property
    def display_code(self) -> str:
        """The parsed script once there is one, the raw stream until then.

        Mid-stream the model has usually emitted an opening ``` fence but not
        the closing one, so the fence markers are trimmed here rather than
        shown as code. Everything else is left exactly as the model wrote it -
        the point of this pane is to show what was actually generated, not a
        cleaned-up version of it.
        """
        if self.code:
            return self.code
        return _strip_fences(self.streamed)


@dataclass(slots=True)
class DashboardState:
    """Accumulated view of one run."""

    goal: str = ""
    status: list[str] = field(default_factory=list)
    pi_reasoning: str = ""
    pi_answer: str = ""
    plan: str = ""
    plan_steps: list[str] = field(default_factory=list)
    limits: dict[str, Any] = field(default_factory=dict)
    planner_model: str = ""
    coder_model: str = ""
    backend: str = ""
    research: dict[str, Any] = field(default_factory=dict)
    attempts: list[AttemptView] = field(default_factory=list)
    finished: bool = False
    certified: bool = False
    certified_code: str = ""
    certified_waypoints: list[dict[str, Any]] = field(default_factory=list)
    certified_telemetry: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    #accessors

    def attempt(self, index: int) -> AttemptView:
        """Get or create the view for attempt index"""
        for view in self.attempts:
            if view.index == index:
                return view
        view = AttemptView(index=index)
        self.attempts.append(view)
        self.attempts.sort(key=lambda a: a.index)
        return view

    @property
    def current(self) -> AttemptView | None:
        """attempt in flight, or the last one to have run"""
        return self.attempts[-1] if self.attempts else None

    @property
    def violations(self) -> list[str]:
        return [a.error for a in self.attempts if a.passed is False and a.error]

    @property
    def plot_waypoints(self) -> list[dict[str, Any]]:
        """Best trajectory to draw right now"""
        if self.certified_waypoints:
            return self.certified_waypoints
        for view in reversed(self.attempts):
            if view.waypoints:
                return view.waypoints
        return []

    #folding

    def apply(self, event: RunEvent) -> None:
        """Fold one event into this state."""
        handler = _HANDLERS.get(event.kind)
        if handler is not None:
            handler(self, event)

    def apply_all(self, events: list[RunEvent]) -> None:
        for event in events:
            self.apply(event)


def _strip_fences(text: str) -> str:
    """drop markdown code fences from a partial stream."""
    stripped = text.lstrip()
    if stripped.startswith("```"):
        newline = stripped.find("\n")
        stripped = stripped[newline + 1 :] if newline != -1 else ""
    return stripped.removesuffix("```").rstrip("`")


def _clip(text: str) -> str:
    """bound a streamed buffer"""
    return text if len(text) <= _MAX_STREAM_CHARS else text[-_MAX_STREAM_CHARS:]


def _on_status(state: DashboardState, event: RunEvent) -> None:
    state.status.append(event.text)


def _on_research(state: DashboardState, event: RunEvent) -> None:
    state.research = {"summary": event.text, **event.payload}


def _on_limits(state: DashboardState, event: RunEvent) -> None:
    payload = event.payload
    limits = payload.get("limits")
    state.limits = limits if isinstance(limits, dict) else {}
    state.planner_model = str(payload.get("planner_model") or "")
    state.coder_model = str(payload.get("coder_model") or "")
    state.backend = str(payload.get("backend") or "")


def _on_pi_token(state: DashboardState, event: RunEvent) -> None:
    if event.channel == "reasoning":
        state.pi_reasoning = _clip(state.pi_reasoning + event.text)
    else:
        state.pi_answer = _clip(state.pi_answer + event.text)


def _on_plan(state: DashboardState, event: RunEvent) -> None:
    state.plan = event.text
    steps = event.payload.get("steps")
    state.plan_steps = [str(s) for s in steps] if isinstance(steps, list) else []


def _on_attempt(state: DashboardState, event: RunEvent) -> None:
    view = state.attempt(event.attempt)
    view.feedback = str(event.payload.get("feedback") or "")


def _on_coder_token(state: DashboardState, event: RunEvent) -> None:
    view = state.attempt(event.attempt)
    if event.channel == "reasoning":
        view.reasoning = _clip(view.reasoning + event.text)
    else:
        view.streamed = _clip(view.streamed + event.text)


def _on_code(state: DashboardState, event: RunEvent) -> None:
    view = state.attempt(event.attempt)
    view.code = event.text
    view.generation_s = float(event.payload.get("generation_s") or 0.0)


def _on_verdict(state: DashboardState, event: RunEvent) -> None:
    view = state.attempt(event.attempt)
    view.verdict = event.text
    view.passed = bool(event.payload.get("passed"))
    view.error = str(event.payload.get("error") or "")
    view.duration_s = float(event.payload.get("duration_s") or 0.0)
    for name in ("telemetry", "detail"):
        value = event.payload.get(name)
        setattr(view, name, value if isinstance(value, dict) else {})
    waypoints = event.payload.get("waypoints")
    view.waypoints = waypoints if isinstance(waypoints, list) else []


def _on_done(state: DashboardState, event: RunEvent) -> None:
    state.finished = True
    state.certified = bool(event.payload.get("certified"))
    state.certified_code = str(event.payload.get("code") or "")
    waypoints = event.payload.get("waypoints")
    state.certified_waypoints = waypoints if isinstance(waypoints, list) else []
    telemetry = event.payload.get("telemetry")
    state.certified_telemetry = telemetry if isinstance(telemetry, dict) else {}


def _on_error(state: DashboardState, event: RunEvent) -> None:
    state.error = event.text
    state.finished = True


_HANDLERS = {
    "status": _on_status,
    "research": _on_research,
    "limits": _on_limits,
    "pi_token": _on_pi_token,
    "plan": _on_plan,
    "attempt": _on_attempt,
    "coder_token": _on_coder_token,
    "code": _on_code,
    "verdict": _on_verdict,
    "done": _on_done,
    "error": _on_error,
}
