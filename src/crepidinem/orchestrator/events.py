"""events emitted by a run, so a UI can watch it happen"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from crepidinem.llm.nebius_client import TokenSink
from crepidinem.logging_setup import get_logger

__all__ = [
    "EventKind",
    "EventSink",
    "RunEvent",
    "TokenSink",
    "safe_sink",
]

log = get_logger(__name__)

EventKind = Literal[
    "status",  #a human readable progress line
    "research",  #the hardware dossier is ready
    "limits",  #the enforced limits for this run
    "pi_token",  #one streamed token from the Principal Investigator
    "plan",  #the parsed plan
    "attempt",  #a new attempt has begun
    "coder_token",  #one streamed token from the Code Smith
    "code",  #the complete proposed script
    "verdict",  #the guardrail's answer for an attempt
    "done",  #the run finished, certified or not
    "error",  #the run could not complete
]


@dataclass(frozen=True, slots=True)
class RunEvent:
    """One thing that happened during a run"""

    kind: EventKind
    text: str = ""
    attempt: int = 0
    channel: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


EventSink = Callable[[RunEvent], None]


def safe_sink(sink: EventSink | None) -> EventSink:
    """Wrap sink so a failing consumer cannot break the run"""
    if sink is None:
        return _discard

    def emit(event: RunEvent) -> None:
        try:
            sink(event)
        except Exception:
            log.exception("Event sink raised on a %s event; continuing", event.kind)

    return emit


def _discard(event: RunEvent) -> None:
    """Default sink: events go nowhere."""
    del event
