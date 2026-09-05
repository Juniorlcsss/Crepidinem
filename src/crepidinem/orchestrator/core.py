"""The core execution loop.

    goal -> research -> PI (Ultra) -> plan -> CS (Nano) -> guardrail
"""

from __future__ import annotations

import asyncio
import dataclasses
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any

from crepidinem.agents.base import (
    CodeSmith,
    Conversational,
    ExperimentPlan,
    PrincipalInvestigator,
    StreamingAgent,
)
from crepidinem.agents.factory import AgentTeam, build_agents
from crepidinem.config import Settings, load_settings
from crepidinem.exceptions import AgentError, CrepidinemError
from crepidinem.llm.nebius_client import NebiusLLMClient
from crepidinem.logging_setup import get_logger
from crepidinem.mcp_tools.harness import PhysicsLimits
from crepidinem.mcp_tools.physics_sandbox import (
    GuardrailReport,
    format_feedback,
    no_code_report,
    test_code_in_physics_sandbox,
)
from crepidinem.orchestrator.events import EventKind, EventSink, RunEvent, safe_sink
from crepidinem.research.cache import CacheStats
from crepidinem.research.hardware import HardwareDossier, dossier_brief, research_hardware
from crepidinem.research.search import LiteratureSearch, build_search
from crepidinem.sandbox.base import SandboxBackend
from crepidinem.sandbox.factory import build_backend

__all__ = [
    "DEFAULT_GOAL",
    "DEFAULT_REPLAN_AFTER",
    "Attempt",
    "ExperimentReport",
    "Orchestrator",
    "run_experiment",
    "run_experiment_sync",
]

log = get_logger(__name__)

DEFAULT_GOAL = (
    "Move the robotic arm from sample station A to station B to transfer a vial, "
    "without hitting the table."
)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_REPLAN_AFTER = 2

_RULE = "=" * 78
_THIN = "-" * 78


@dataclass(frozen=True, slots=True)
class Attempt:
    """oneloop"""

    index: int
    code: str
    report: GuardrailReport
    duration_s: float
    generation_s: float = 0.0
    plan_revision: int = 0

    @property
    def passed(self) -> bool:
        return self.report["success"]

    @property
    def waypoints(self) -> list[dict[str, Any]]:
        """The trajectory the harness observed, certified or not."""
        return self.report["waypoints"]


@dataclass(slots=True)
class ExperimentReport:
    goal: str
    plan: ExperimentPlan
    attempts: list[Attempt] = field(default_factory=list)
    certified_code: str | None = None
    planner_model: str = ""
    coder_model: str = ""
    limits: PhysicsLimits = field(default_factory=PhysicsLimits)
    dossier: HardwareDossier | None = None
    revisions: list[ExperimentPlan] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.certified_code is not None

    @property
    def violations(self) -> list[str]:
        """physics violations the guardrail caught"""
        return [
            a.report["violation_code"]
            for a in self.attempts
            if not a.passed and a.report["violation_code"] is not None
        ]

    @property
    def certified_telemetry(self) -> dict[str, Any]:
        """sandbox telemetry from the certified run"""
        for attempt in self.attempts:
            if attempt.passed:
                return dict(attempt.report["telemetry"])
        return {}

    @property
    def certified_waypoints(self) -> list[dict[str, Any]]:
        """validated trajectory"""
        for attempt in self.attempts:
            if attempt.passed:
                return attempt.waypoints
        return []


class Orchestrator:
    """runs research/plan/generate/verify/repair loop"""

    def __init__(
        self,
        *,
        principal_investigator: PrincipalInvestigator | None = None,
        code_smith: CodeSmith | None = None,
        settings: Settings | None = None,
        backend: SandboxBackend | None = None,
        limits: PhysicsLimits | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        replan_after: int = DEFAULT_REPLAN_AFTER,
        print_transcript: bool = True,
        events: EventSink | None = None,
        search: LiteratureSearch | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.limits = limits or PhysicsLimits()
        self.max_attempts = max(1, max_attempts)
        self.replan_after = max(0, replan_after)
        self.print_transcript = print_transcript
        self._backend = backend
        self._owns_backend = backend is None
        self._emit_event = safe_sink(events)
        self._streaming = events is not None
        self._search = search
        self._owns_search = search is None
        self._injected_pi = principal_investigator
        self._injected_smith = code_smith
        self._team: AgentTeam | None = None

    #output
    def _say(self, text: str = "") -> None:
        if self.print_transcript:
            print(text)

    def _show_code(self, code: str) -> None:
        if not self.print_transcript:
            return
        for line in code.rstrip().splitlines():
            print(f"    | {line}")

    def _event(
        self,
        kind: EventKind,
        text: str = "",
        *,
        attempt: int = 0,
        channel: str = "",
        **payload: Any,
    ) -> None:
        self._emit_event(
            RunEvent(kind=kind, text=text, attempt=attempt, channel=channel, payload=payload)
        )

    def _status(self, text: str) -> None:
        self._say(text)
        self._event("status", text)

    #agents
    @property
    def _builds_own_team(self) -> bool:
        return self._injected_pi is None and self._injected_smith is None

    def _resolve_team(self) -> AgentTeam:
        """pick the agents for this run"""
        if self._injected_pi is not None and self._injected_smith is not None:
            return AgentTeam(
                principal_investigator=self._injected_pi,
                code_smith=self._injected_smith,
                planner_model="injected",
                coder_model="injected",
            )
        team = build_agents(self.settings, self.limits)
        if self._injected_pi is not None:
            team.principal_investigator = self._injected_pi

        if self._injected_smith is not None:
            team.code_smith = self._injected_smith

        return team

    def _stream_to(self, agent: object, kind: EventKind, attempt: int = 0) -> None:
        """route agents deltas into the event stream"""
        if self._streaming and isinstance(agent, StreamingAgent):
            agent.set_token_sink(
                lambda channel, text: self._event(kind, text, attempt=attempt, channel=channel)
            )

    @staticmethod
    def _stop_streaming(agent: object) -> None:
        if isinstance(agent, StreamingAgent):
            agent.set_token_sink(None)

    #loop
    async def run(self, goal: str = DEFAULT_GOAL) -> ExperimentReport:
        """execute loop until the guardrail certifies a script"""
        self._say(_RULE)
        self._say("CREPIDINEM :: physics-verified control loop")
        self._say(_RULE)
        self._say(f"GOAL:     {goal}")
        self._event("status", f"Goal: {goal}")

        dossier = await self._research_for(goal)
        if dossier is not None:
            self.limits = dossier.limits

        team = self._resolve_team()
        self._team = team
        owns_team = team.llm_client is not None

        self._say(f"PLANNING: {team.planner_model}")
        self._say(f"CODING:   {team.coder_model}")
        self._say(f"SANDBOX:  {self.settings.sandbox_backend}")
        self._say()
        self._event(
            "limits",
            f"Enforcing max_velocity={self.limits.max_velocity} m/s",
            limits=_limits_payload(self.limits),
            planner_model=team.planner_model,
            coder_model=team.coder_model,
            backend=self.settings.sandbox_backend,
        )

        try:
            report = await self._run_with_team(goal, team, dossier)
        except CrepidinemError as exc:
            self._event("error", str(exc))
            raise
        finally:
            if owns_team:
                await team.aclose()

        self._render_summary(report)
        self._event(
            "done",
            "CERTIFIED" if report.success else "NOT CERTIFIED",
            certified=report.success,
            attempts=len(report.attempts),
            violations=report.violations,
            waypoints=report.certified_waypoints,
            telemetry=report.certified_telemetry,
            code=report.certified_code or "",
        )
        return report

    async def _research_for(self, goal: str) -> HardwareDossier | None:
        """run research for goal"""

        if not self._builds_own_team:
            return None
        if not self.settings.research_available:
            log.info("Hardware research skipped (disabled or no TAVILY_API_KEY)")
            return None

        self._status("[Research] looking up the hardware's published limits...")
        client = NebiusLLMClient.from_settings(self.settings)
        search = self._search or build_search(self.settings)
        try:
            dossier = await research_hardware(
                goal,
                client=client,
                settings=self.settings,
                search=search,
                limits=self.limits,
            )
        finally:
            if self._owns_search:
                await search.aclose()
            await client.aclose()

        self._status(f"[Research] {dossier.summary()}")
        for finding in dossier.findings:
            self._say(f"    {finding.describe()}")

        #cached answer must never pass for a live one
        stats = getattr(search, "stats", None)
        search_cost = stats.describe() if isinstance(stats, CacheStats) else ""
        if search_cost:
            self._status(f"[Research] {search_cost}")

        self._event(
            "research",
            dossier.summary(),
            search_cost=search_cost,
            hardware=list(dossier.hardware),
            queries=list(dossier.queries),
            sources=[{"title": h.title, "url": h.url} for h in dossier.hits],
            findings=[
                {
                    "field": f.field,
                    "value": f.value,
                    "unit": f.unit,
                    "previous": f.previous,
                    "accepted": f.accepted,
                    "reason": f.reason,
                    "source_url": f.source_url,
                    "quote": f.quote,
                }
                for f in dossier.findings
            ],
            notes=list(dossier.notes),
        )
        return dossier

    async def _plan(
        self,
        team: AgentTeam,
        goal: str,
        brief: str,
        citations: tuple[str, ...],
        *,
        revision: int,
        feedback: str | None = None,
    ) -> ExperimentPlan:
        """Ask the Principal Investigator for a plan, or for a better one.

        ``feedback`` is the guardrail's verdict on the last script. Passing it
        is what turns a second call into a replan rather than a re-roll: the
        planner is told exactly which constraint the previous steps could not
        be coded to satisfy.
        """
        started = time.perf_counter()
        if revision:
            self._status(
                f"[Principal Investigator] the plan itself looks unsatisfiable; "
                f"replanning (revision {revision})..."
            )
        else:
            self._status("[Principal Investigator] designing the experiment...")

        self._stream_to(team.principal_investigator, "pi_token")
        try:
            plan = await team.principal_investigator.design_experiment(
                goal if not brief else f"{goal}\n\n{brief}",
                feedback=feedback,
            )
        finally:
            self._stop_streaming(team.principal_investigator)

        if citations:
            plan = dataclasses.replace(plan, citations=citations)

        label = "plan ready" if not revision else f"revised plan ready (revision {revision})"
        self._say(f"[Principal Investigator] {label} ({time.perf_counter() - started:.1f}s):")
        self._say(textwrap.indent(plan.as_prompt(), "  "))
        self._say()
        self._event(
            "plan",
            plan.as_prompt(),
            steps=list(plan.steps),
            revision=revision,
            citations=list(plan.citations),
        )
        return plan

    async def _run_with_team(
        self,
        goal: str,
        team: AgentTeam,
        dossier: HardwareDossier | None,
    ) -> ExperimentReport:
        brief = dossier_brief(dossier) if dossier is not None else ""
        citations = dossier.citations if dossier is not None else ()
        plan = await self._plan(team, goal, brief, citations, revision=0)

        report = ExperimentReport(
            goal=goal,
            plan=plan,
            planner_model=team.planner_model,
            coder_model=team.coder_model,
            limits=self.limits,
            dossier=dossier,
        )
        backend = self._backend or build_backend(self.settings)
        feedback: str | None = None
        revision = 0
        since_replan = 0

        try:
            for index in range(1, self.max_attempts + 1):
                self._say(_THIN)
                self._say(f"ATTEMPT {index} of {self.max_attempts}")
                self._say(_THIN)
                self._event(
                    "attempt",
                    f"Attempt {index} of {self.max_attempts}",
                    attempt=index,
                    feedback=feedback or "",
                )

                drafted = time.perf_counter()
                self._stream_to(team.code_smith, "coder_token", index)
                try:
                    code = await team.code_smith.write_control_script(
                        plan, attempt=index, feedback=feedback
                    )
                except AgentError as exc:
                    log.warning("Attempt %d produced no usable script: %s", index, exc)
                    self._say(f"[Code Smith] FAILED to produce usable code: {exc}")
                    self._say()
                    report.attempts.append(
                        Attempt(
                            index=index,
                            code="",
                            report=no_code_report(str(exc)),
                            duration_s=0.0,
                            generation_s=time.perf_counter() - drafted,
                            plan_revision=revision,
                        )
                    )
                    self._event(
                        "verdict",
                        f"No usable code: {exc}",
                        attempt=index,
                        passed=False,
                        error="AGENT_OUTPUT_UNUSABLE",
                    )
                    feedback = (
                        f"FAIL [AGENT_OUTPUT_UNUSABLE]: {exc}\n"
                        "Reply with one ```python block containing the whole script."
                    )
                    continue

                finally:
                    self._stop_streaming(team.code_smith)
                generation_s = time.perf_counter() - drafted

                self._say(f"[Code Smith] proposed control script ({generation_s:.1f}s):")
                self._show_code(code)
                self._say()
                self._event("code", code, attempt=index, generation_s=round(generation_s, 3))

                verified = time.perf_counter()
                self._say("[MCP] test_code_in_physics_sandbox -> sandbox")
                guardrail = await test_code_in_physics_sandbox(
                    code,
                    backend=backend,
                    limits=self.limits,
                    settings=self.settings,
                )
                elapsed = time.perf_counter() - verified

                attempt = Attempt(
                    index=index,
                    code=code,
                    report=guardrail,
                    duration_s=elapsed,
                    generation_s=generation_s,
                    plan_revision=revision,
                )
                report.attempts.append(attempt)
                self._event(
                    "verdict",
                    guardrail["message"],
                    attempt=index,
                    passed=attempt.passed,
                    error=guardrail["error"] or "",
                    detail=guardrail["detail"],
                    telemetry=guardrail["telemetry"],
                    waypoints=guardrail["waypoints"],
                    duration_s=round(elapsed, 3),
                )

                if attempt.passed:
                    self._say(f"[Guardrail] PASS  ({elapsed:.2f}s)  {guardrail['message']}")
                    self._say()
                    report.certified_code = code
                    break

                self._say(f"[Guardrail] BLOCKED ({elapsed:.2f}s)  {guardrail['error']}")
                self._say(textwrap.indent(guardrail["message"].strip(), "    "))
                self._say()
                feedback = format_feedback(guardrail)
                log.warning("Attempt %d blocked by the guardrail: %s", index, guardrail["error"])

                since_replan += 1
                if self._should_replan(since_replan, index):
                    revised = await self._try_replan(
                        team, goal, brief, citations, revision + 1, feedback
                    )
                    if revised is not None:
                        plan, revision = revised, revision + 1
                        report.revisions.append(plan)
                        since_replan = 0
                        if isinstance(team.code_smith, Conversational):
                            team.code_smith.reset()
                        feedback = None
        finally:
            if self._owns_backend:
                await backend.aclose()

        return report

    def _should_replan(self, since_replan: int, index: int) -> bool:
        if not self.replan_after or since_replan < self.replan_after:
            return False
        return index < self.max_attempts

    async def _try_replan(
        self,
        team: AgentTeam,
        goal: str,
        brief: str,
        citations: tuple[str, ...],
        revision: int,
        feedback: str,
    ) -> ExperimentPlan | None:
        """Replan, or carry on with the current plan if the planner fails.

        A replan is an optimisation, not a requirement. If the planner errors
        or returns something unparseable, the run keeps its existing plan and
        spends its remaining attempts on rewrites, which is strictly better
        than aborting a run that might still succeed.
        """
        try:
            return await self._plan(
                team, goal, brief, citations, revision=revision, feedback=feedback
            )
        except AgentError as exc:
            log.warning("Replanning failed, keeping the current plan: %s", exc)
            self._say(f"[Principal Investigator] could not replan ({exc}); keeping the plan.")
            self._say()
            return None

    #summary
    def _render_summary(self, report: ExperimentReport) -> None:
        self._say(_RULE)
        self._say("SUMMARY")
        self._say(_RULE)
        for attempt in report.attempts:
            status = "PASS   " if attempt.passed else "BLOCKED"
            reason = "" if attempt.passed else f"  <- {attempt.report['error']}"
            plan_note = f"  [plan rev {attempt.plan_revision}]" if report.revisions else ""
            self._say(f"  attempt {attempt.index}: {status}{reason}{plan_note}")
        self._say()

        if not report.success:
            self._say(
                f"NOT CERTIFIED after {len(report.attempts)} attempt(s). "
                f"Nothing is released to hardware."
            )
            self._say(_RULE)
            return

        self._say(
            f"CERTIFIED after {len(report.attempts)} attempt(s). "
            f"{len(report.violations)} violation(s) caught before hardware."
        )
        self._say()
        self._say("VERIFIED CONTROL SCRIPT")
        self._say(_THIN)
        self._show_code(report.certified_code or "")
        self._say()
        self._say("SANDBOX TELEMETRY")
        self._say(_THIN)
        for key, value in report.certified_telemetry.items():
            self._say(f"  {key:<18} {value}")
        self._say(f"  {'waypoints':<18} {len(report.certified_waypoints)}")
        self._say(_RULE)


def _limits_payload(limits: PhysicsLimits) -> dict[str, Any]:
    """limits, flattened for an event payload"""
    return {
        "max_velocity": limits.max_velocity,
        "max_acceleration": limits.max_acceleration,
        "max_payload_kg": limits.max_payload_kg,
        "max_gripper_force_n": limits.max_gripper_force_n,
        "workspace_x": list(limits.workspace_x),
        "workspace_y": list(limits.workspace_y),
        "workspace_z": list(limits.workspace_z),
        "table_height": limits.table_height,
        "table_clearance": limits.table_clearance,
        "obstacles": [list(box) for box in limits.obstacles],
    }


async def run_experiment(
    goal: str = DEFAULT_GOAL,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    replan_after: int = DEFAULT_REPLAN_AFTER,
    settings: Settings | None = None,
    backend: SandboxBackend | None = None,
    limits: PhysicsLimits | None = None,
    print_transcript: bool = True,
    events: EventSink | None = None,
) -> ExperimentReport:
    """Convenience wrapper: build a default Orchestrator and run it."""
    orchestrator = Orchestrator(
        settings=settings,
        backend=backend,
        limits=limits,
        max_attempts=max_attempts,
        replan_after=replan_after,
        print_transcript=print_transcript,
        events=events,
    )
    return await orchestrator.run(goal)


def run_experiment_sync(
    goal: str = DEFAULT_GOAL,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    replan_after: int = DEFAULT_REPLAN_AFTER,
    settings: Settings | None = None,
    limits: PhysicsLimits | None = None,
    print_transcript: bool = True,
    events: EventSink | None = None,
) -> ExperimentReport:
    """Synchronous entry point for the CLI."""
    return asyncio.run(
        run_experiment(
            goal,
            max_attempts=max_attempts,
            replan_after=replan_after,
            settings=settings,
            limits=limits,
            print_transcript=print_transcript,
            events=events,
        )
    )
