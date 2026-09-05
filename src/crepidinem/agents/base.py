"""contracts every agent is held to and the plan they pass between them"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from crepidinem.llm.nebius_client import TokenSink

__all__ = [
    "CodeSmith",
    "Conversational",
    "ExperimentPlan",
    "PrincipalInvestigator",
    "StreamingAgent",
]


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    """principle investigator in coder form"""

    goal: str
    rationale: str
    steps: tuple[str, ...]
    constraints: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()
    citations: tuple[str, ...] = field(default_factory=tuple)

    def as_prompt(self) -> str:
        """Render the plan as the brief handed to the coder."""
        blocks = [f"GOAL: {self.goal}", f"RATIONALE: {self.rationale}", "STEPS:"]
        blocks.extend(f"  {i}. {step}" for i, step in enumerate(self.steps, start=1))
        if self.constraints:
            blocks.append("HARD CONSTRAINTS:")
            blocks.extend(f"  - {c}" for c in self.constraints)
        if self.success_criteria:
            blocks.append("SUCCESS CRITERIA:")
            blocks.extend(f"  - {c}" for c in self.success_criteria)
        if self.citations:
            blocks.append("LIMITS SOURCED FROM:")
            blocks.extend(f"  - {c}" for c in self.citations)
        return "\n".join(blocks)


@runtime_checkable
class PrincipalInvestigator(Protocol):
    """Turns a scientific goal into a plan, and revises it when told why."""

    async def design_experiment(self, goal: str, *, feedback: str | None = None) -> ExperimentPlan:
        """Produce a plan for ``goal``.

        Args:
            goal: The scientific objective.
            feedback: A guardrail verdict from a previous attempt. When given,
                the planner is being asked to *replan* — the last plan proved
                impossible to satisfy, so returning the same steps again is a
                wasted call.
        """
        ...


@runtime_checkable
class CodeSmith(Protocol):
    """Writes control code for a plan, and repairs it against a verdict."""

    async def write_control_script(
        self, plan: ExperimentPlan, *, attempt: int, feedback: str | None
    ) -> str:
        """Produce a complete control script.

        Args:
            plan: The brief to implement.
            attempt: 1-based index of this attempt, for logs and labels.
            feedback: The guardrail's verbatim verdict on the previous script,
                or ``None`` for a first draft. This is what makes the loop a
                repair rather than a resample.
        """
        ...


@runtime_checkable
class StreamingAgent(Protocol):
    """agent can narrate itself"""

    def set_token_sink(self, sink: TokenSink | None) -> None:
        """Route this agent's output deltas to ``sink``, or stop routing."""
        ...


@runtime_checkable
class Conversational(Protocol):
    """agent carries context between calls"""

    def reset(self) -> None:
        """Drop the accumulated conversation.

        The loop calls this when the plan is replaced: a rewrite history built
        against the old plan is actively misleading once the steps change.
        """
        ...
