"""Run the loop off the Streamlit thread"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import Any

from crepidinem.config import Settings
from crepidinem.exceptions import CrepidinemError
from crepidinem.logging_setup import get_logger
from crepidinem.mcp_tools.harness import PhysicsLimits
from crepidinem.orchestrator.core import ExperimentReport, Orchestrator
from crepidinem.orchestrator.events import RunEvent

__all__ = ["BackgroundRun"]

log = get_logger(__name__)

_DRAIN_BUDGET = 2000


class BackgroundRun:
    """One experiment, executing on a worker thread.

    The object is created on the UI thread, started once, and then polled with
    :meth:`drain` on each rerun until :attr:`finished` is true.
    """

    def __init__(
        self,
        goal: str,
        *,
        settings: Settings,
        limits: PhysicsLimits,
        max_attempts: int,
    ) -> None:
        self.goal = goal
        self._settings = settings
        self._limits = limits
        self._max_attempts = max_attempts
        self._queue: queue.Queue[RunEvent] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        self._report: ExperimentReport | None = None
        self._error: str = ""

    #lifecycle

    def start(self) -> None:
        """Begin the run. Idempotent; a second call is ignored."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="crepidinem-run", daemon=True)
        self._thread.start()
        log.info("Background run started for goal: %s", self.goal)

    def _run(self) -> None:
        """Thread entry point: own event loop, own orchestrator."""
        try:
            asyncio.run(self._run_async())

        except CrepidinemError as exc:
            self._error = str(exc)
            log.exception("Background run failed")

        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            log.exception("Background run crashed")

        finally:
            self._done.set()

    async def _run_async(self) -> None:
        orchestrator = Orchestrator(
            settings=self._settings,
            limits=self._limits,
            max_attempts=self._max_attempts,
            print_transcript=False,
            events=self._queue.put,
        )
        self._report = await orchestrator.run(self.goal)

    #polling

    def drain(self) -> list[RunEvent]:
        """take every event queued so far without blocking."""
        events: list[RunEvent] = []
        for _ in range(_DRAIN_BUDGET):
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events

    @property
    def started(self) -> bool:
        return self._thread is not None

    @property
    def finished(self) -> bool:
        """true once the worker has stopped"""
        return self._done.is_set()

    @property
    def running(self) -> bool:
        return self.started and not self.finished

    @property
    def report(self) -> ExperimentReport | None:
        """The finished report, or ``None`` while the run is in flight."""
        return self._report if self.finished else None

    @property
    def error(self) -> str:
        """Why the run could not complete, or an empty string."""
        return self._error if self.finished else ""

    def snapshot(self) -> dict[str, Any]:
        """A small dict for debugging the UI's view of this run."""
        return {
            "goal": self.goal,
            "running": self.running,
            "finished": self.finished,
            "error": self._error,
            "certified": bool(self._report and self._report.success),
        }
