"""The repair loop, proven without touching the network.

The agents are local doubles with scripted output, so the *model* is mocked.
The sandbox is the real local backend running the real harness, so every
verdict asserted below was computed by the physics engine rather than written
into the test.  That is the half worth not faking: the claim under test is
"unsafe code does not get certified", and mocking the judge would prove nothing.
"""

from __future__ import annotations

import dataclasses

import pytest

from crepidinem.agents.base import ExperimentPlan
from crepidinem.agents.factory import AgentTeam
from crepidinem.config import Settings
from crepidinem.exceptions import AgentError
from crepidinem.mcp_tools.harness import PhysicsLimits
from crepidinem.orchestrator.core import Orchestrator
from crepidinem.sandbox.base import SandboxBackend
from crepidinem.sandbox.models import ExecutionResult

PLAN = ExperimentPlan(
    goal="Transfer the vial",
    rationale="Lift, translate, place.",
    steps=("approach", "grip", "lift", "translate", "place"),
)

TOO_FAST = """\
arm.home()
arm.set_velocity(2.0)
arm.move_to(0.2, -0.2, 0.15)
"""

SAFE = """\
arm.home()
arm.set_velocity(0.3)
arm.move_to(0.2, -0.2, 0.15)
arm.grip(force_n=10.0, payload_kg=0.2)
arm.move_to(0.2, -0.2, 0.25)
arm.move_to(0.2, 0.2, 0.25)
arm.move_to(0.2, 0.2, 0.15)
arm.release()
"""


class ScriptedPI:
    """Returns the same plan every time; the PI is not what is under test."""

    def __init__(self) -> None:
        self.calls = 0

    async def design_experiment(self, goal: str, *, feedback: str | None = None) -> ExperimentPlan:
        self.calls += 1
        return dataclasses.replace(PLAN, goal=goal)


class ScriptedCodeSmith:
    """Emits a fixed script per attempt and records the feedback it was given."""

    def __init__(self, scripts: list[str]) -> None:
        self._scripts = scripts
        self.feedback_seen: list[str | None] = []

    async def write_control_script(
        self,
        plan: ExperimentPlan,
        *,
        attempt: int = 1,
        feedback: str | None = None,
    ) -> str:
        self.feedback_seen.append(feedback)
        index = min(attempt, len(self._scripts)) - 1
        return self._scripts[index]


class BrokenBackend:
    """A sandbox that cannot run anything, to prove the loop fails closed."""

    async def create_sandbox(self, image: str) -> str:
        return "broken-1"

    async def write_file(self, sandbox_id: str, path: str, content: str) -> None:
        return

    async def execute_code(self, sandbox_id: str, code: str, timeout: int) -> ExecutionResult:
        # No verdict marker at all: the guardrail must not read this as a pass.
        return ExecutionResult(stdout="", stderr="segfault", exit_code=139)

    async def destroy_sandbox(self, sandbox_id: str) -> None:
        return

    async def aclose(self) -> None:
        return


def build(
    settings: Settings,
    smith: ScriptedCodeSmith,
    *,
    max_attempts: int = 3,
    backend: SandboxBackend | None = None,
) -> Orchestrator:
    return Orchestrator(
        principal_investigator=ScriptedPI(),
        code_smith=smith,
        settings=settings,
        backend=backend,
        limits=PhysicsLimits(max_velocity=0.5),
        max_attempts=max_attempts,
        print_transcript=False,
    )


# --------------------------------------------------------------- the loop


@pytest.mark.asyncio
async def test_a_violation_is_fed_back_and_the_rewrite_is_certified(
    settings: Settings,
) -> None:
    """Attempt 1 breaks the speed limit; attempt 2, told exactly that, passes."""
    smith = ScriptedCodeSmith([TOO_FAST, SAFE])
    report = await build(settings, smith).run("Transfer the vial")

    assert len(report.attempts) == 2
    assert report.attempts[0].passed is False
    assert report.attempts[1].passed is True
    assert report.success is True
    assert report.violations == ["VELOCITY_LIMIT_EXCEEDED"]

    # The first attempt is asked cold; the second carries the verbatim verdict.
    assert smith.feedback_seen[0] is None
    retry_prompt = smith.feedback_seen[1] or ""
    assert "VELOCITY_LIMIT_EXCEEDED" in retry_prompt, (
        "the Code Smith must be told which limit it broke, or the rewrite is blind"
    )


@pytest.mark.asyncio
async def test_the_loop_gives_up_rather_than_certifying_unsafe_code(
    settings: Settings,
) -> None:
    """Code that never improves burns its attempts and certifies nothing."""
    smith = ScriptedCodeSmith([TOO_FAST])
    report = await build(settings, smith, max_attempts=3).run("Transfer the vial")

    assert len(report.attempts) == 3
    assert report.success is False
    assert report.certified_code is None
    assert report.violations == ["VELOCITY_LIMIT_EXCEEDED"] * 3


@pytest.mark.asyncio
async def test_a_sandbox_that_cannot_run_fails_closed(settings: Settings) -> None:
    """No verdict is not a pass. An unusable sandbox certifies nothing."""
    smith = ScriptedCodeSmith([SAFE])
    report = await build(settings, smith, max_attempts=2, backend=BrokenBackend()).run(
        "Transfer the vial"
    )

    assert report.success is False
    assert report.certified_code is None
    assert all(a.passed is False for a in report.attempts)


@pytest.mark.asyncio
async def test_safe_code_is_certified_on_the_first_attempt(settings: Settings) -> None:
    """The loop must not manufacture failures for code that is already fine."""
    smith = ScriptedCodeSmith([SAFE])
    report = await build(settings, smith).run("Transfer the vial")

    assert len(report.attempts) == 1
    assert report.success is True
    assert report.certified_code is not None
    assert report.attempts[0].waypoints, "a certified run should report its trajectory"


class MuteCodeSmith:
    """A model that answers with prose instead of a code block, every time."""

    async def write_control_script(
        self,
        plan: ExperimentPlan,
        *,
        attempt: int = 1,
        feedback: str | None = None,
    ) -> str:
        msg = "no ```python block in the reply"
        raise AgentError(msg)


@pytest.mark.asyncio
async def test_an_attempt_that_produced_no_code_is_still_recorded(
    settings: Settings,
) -> None:
    """A retry burned on unusable output must be visible, not silently dropped.

    Without this the JSON report carries `success: false` and an empty attempts
    list, so a caller reading only the machine-readable output sees a failure
    with no reason attached anywhere.
    """
    orchestrator = Orchestrator(
        principal_investigator=ScriptedPI(),
        code_smith=MuteCodeSmith(),
        settings=settings,
        limits=PhysicsLimits(max_velocity=0.5),
        max_attempts=3,
        print_transcript=False,
    )
    report = await orchestrator.run("Transfer the vial")

    assert len(report.attempts) == 3, "every burned retry must appear in the report"
    assert report.success is False
    assert all(a.report["error"] == "AGENT_OUTPUT_UNUSABLE" for a in report.attempts)
    assert all("python block" in a.report["message"] for a in report.attempts)
    # No code ever ran, so there is no physics violation to claim.
    assert report.violations == [], "a model that wrote no code broke no limit"


class ReplanningPI:
    """Emits a different plan once it is told why the last one failed."""

    def __init__(self) -> None:
        self.calls = 0
        self.feedback_seen: list[str | None] = []

    async def design_experiment(self, goal: str, *, feedback: str | None = None) -> ExperimentPlan:
        self.calls += 1
        self.feedback_seen.append(feedback)
        if feedback is None:
            return dataclasses.replace(PLAN, goal=goal, rationale="first pass")
        return dataclasses.replace(PLAN, goal=goal, rationale="revised: keep under the speed limit")


class PlanSensitiveCodeSmith:
    """Writes unsafe code until the plan changes, then writes safe code.

    Stands in for the real failure this feature exists for: the code is a
    faithful implementation of the plan, and the *plan* is what cannot be
    satisfied. No amount of rewriting fixes that.
    """

    def __init__(self) -> None:
        self.resets = 0
        self.plans_seen: list[str] = []

    def reset(self) -> None:
        self.resets += 1

    async def write_control_script(
        self,
        plan: ExperimentPlan,
        *,
        attempt: int = 1,
        feedback: str | None = None,
    ) -> str:
        self.plans_seen.append(plan.rationale)
        return SAFE if plan.rationale.startswith("revised") else TOO_FAST


@pytest.mark.asyncio
async def test_a_plan_that_cannot_be_coded_is_replanned(settings: Settings) -> None:
    """After two rejections the loop suspects the plan, not the code."""
    pi, smith = ReplanningPI(), PlanSensitiveCodeSmith()
    orchestrator = Orchestrator(
        principal_investigator=pi,
        code_smith=smith,
        settings=settings,
        limits=PhysicsLimits(max_velocity=0.5),
        max_attempts=4,
        replan_after=2,
        print_transcript=False,
    )
    report = await orchestrator.run("Transfer the vial")

    assert pi.calls == 2, "the planner must be asked again, not just the coder"
    assert pi.feedback_seen[0] is None
    assert "VELOCITY_LIMIT_EXCEEDED" in (pi.feedback_seen[1] or ""), (
        "the replan must say which constraint the old plan could not meet"
    )

    assert smith.resets == 1, "history arguing for the old plan must be dropped"
    assert report.success is True
    assert len(report.revisions) == 1
    # Attempts 1-2 ran against the original plan, attempt 3 against the revision.
    assert [a.plan_revision for a in report.attempts] == [0, 0, 1]


@pytest.mark.asyncio
async def test_replanning_can_be_switched_off(settings: Settings) -> None:
    """replan_after=0 keeps the old behaviour: rewrite until the budget is gone."""
    pi, smith = ReplanningPI(), PlanSensitiveCodeSmith()
    report = await Orchestrator(
        principal_investigator=pi,
        code_smith=smith,
        settings=settings,
        limits=PhysicsLimits(max_velocity=0.5),
        max_attempts=3,
        replan_after=0,
        print_transcript=False,
    ).run("Transfer the vial")

    assert pi.calls == 1
    assert smith.resets == 0
    assert report.success is False
    assert report.revisions == []


@pytest.mark.asyncio
async def test_the_final_attempt_is_never_spent_on_a_replan(settings: Settings) -> None:
    """A new plan nobody will code against is a wasted call."""
    pi, smith = ReplanningPI(), PlanSensitiveCodeSmith()
    report = await Orchestrator(
        principal_investigator=pi,
        code_smith=smith,
        settings=settings,
        limits=PhysicsLimits(max_velocity=0.5),
        max_attempts=2,
        replan_after=2,
        print_transcript=False,
    ).run("Transfer the vial")

    assert pi.calls == 1, "no budget left to code a revision, so do not ask for one"
    assert report.success is False


@pytest.mark.asyncio
async def test_research_citations_are_attached_to_the_plan(settings: Settings) -> None:
    """The planner never sees the dossier, so the loop must attach provenance.

    Without this the sources behind an enforced limit exist only in a log line,
    and a certified script cannot be traced back to the datasheet that widened
    its speed limit.
    """
    source = "https://example.invalid/ur5e-datasheet"
    orchestrator = Orchestrator(
        principal_investigator=ScriptedPI(),
        code_smith=ScriptedCodeSmith([SAFE]),
        settings=settings,
        print_transcript=False,
    )
    team = AgentTeam(
        principal_investigator=ScriptedPI(),
        code_smith=ScriptedCodeSmith([SAFE]),
        planner_model="double",
        coder_model="double",
    )

    plan = await orchestrator._plan(team, "g", "", (source,), revision=0)

    assert plan.citations == (source,)
    rendered = plan.as_prompt()
    assert "LIMITS SOURCED FROM:" in rendered, "the coder must see where limits came from"
    assert source in rendered
