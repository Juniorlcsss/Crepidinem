"""The Principal Investigator"""

from __future__ import annotations

from crepidinem.agents.base import ExperimentPlan
from crepidinem.agents.prompts import (
    PI_SYSTEM,
    UNBRIEFED_NOTE,
    cell_briefing,
    plan_schema,
)
from crepidinem.config import Settings
from crepidinem.exceptions import AgentParseError
from crepidinem.llm.nebius_client import ChatMessage, NebiusLLMClient, TokenSink
from crepidinem.llm.parsing import extract_json_object
from crepidinem.logging_setup import get_logger
from crepidinem.mcp_tools.harness import PhysicsLimits

__all__ = ["NemotronPrincipalInvestigator"]

log = get_logger(__name__)

_REQUIRED_KEYS = ("goal", "rationale", "steps")


def _as_str_tuple(value: object) -> tuple[str, ...]:
    """Coerce whatever the model returned into a tuple of strings."""
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())

            elif isinstance(item, dict):
                parts = [str(v).strip() for v in item.values() if isinstance(v, str | int | float)]

                if parts:
                    out.append(" - ".join(p for p in parts if p))
        return tuple(out)
    return ()


class NemotronPrincipalInvestigator:
    """Implements PI against nemotron 3 ultra."""

    def __init__(
        self,
        client: NebiusLLMClient,
        settings: Settings,
        *,
        limits: PhysicsLimits | None = None,
        temperature: float = 0.3,
        parse_retries: int = 1,
        brief_limits: bool = True,
    ) -> None:
        self._client = client
        self._settings = settings
        self._limits = limits or PhysicsLimits()
        self._brief_limits = brief_limits
        self._temperature = temperature
        self._parse_retries = max(0, parse_retries)
        self._token_sink: TokenSink | None = None

    @property
    def model(self) -> str:
        return self._settings.planner_model

    def set_token_sink(self, sink: TokenSink | None) -> None:
        """stream this agents reasoning and answer to ``sink``."""
        self._token_sink = sink

    async def design_experiment(self, goal: str, *, feedback: str | None = None) -> ExperimentPlan:
        """turn goal into structured plan"""

        log.info("Principal Investigator (%s) planning: %s", self.model, goal)

        user = [
            f"SCIENTIFIC GOAL:\n{goal}",
            cell_briefing(self._limits) if self._brief_limits else UNBRIEFED_NOTE,
            plan_schema(),
        ]
        if feedback:
            user.append(
                "A previous attempt to execute a plan for this goal was blocked "
                "by the physics guardrail:\n"
                f"{feedback}\n"
                "Design a plan that makes that failure impossible."
            )

        messages: list[ChatMessage] = [
            {"role": "system", "content": PI_SYSTEM},
            {"role": "user", "content": "\n\n".join(user)},
        ]

        last_error: AgentParseError | None = None
        for attempt in range(self._parse_retries + 1):
            response = await self._client.complete(
                self.model,
                messages,
                temperature=self._temperature,
                max_tokens=self._settings.planner_max_tokens,
                label="Principal Investigator",
                on_token=self._token_sink,
            )
            try:
                return self._to_plan(goal, response.content)
            except AgentParseError as exc:
                last_error = exc
                log.warning(
                    "Principal Investigator returned unparseable output (attempt %d): %s",
                    attempt + 1,
                    exc,
                )
                if attempt == self._parse_retries:
                    break
                
                messages.extend(
                    [
                        {"role": "assistant", "content": response.content},
                        {
                            "role": "user",
                            "content": (
                                f"That could not be parsed: {exc}\n"
                                "Reply with the JSON object only. No markdown fence, "
                                "no explanation, no text before or after it."
                            ),
                        },
                    ]
                )

        assert last_error is not None
        raise last_error

    def _to_plan(self, goal: str, content: str) -> ExperimentPlan:
        """Validate the model's JSON and build an :class:`ExperimentPlan`."""
        payload = extract_json_object(content)

        missing = [key for key in _REQUIRED_KEYS if not payload.get(key)]
        if missing:
            msg = f"Plan JSON is missing required key(s): {', '.join(missing)}."
            raise AgentParseError(msg, raw=content)

        steps = _as_str_tuple(payload.get("steps"))
        if not steps:
            msg = "Plan JSON contains no usable steps."
            raise AgentParseError(msg, raw=content)

        plan = ExperimentPlan(
            goal=str(payload.get("goal") or goal),
            rationale=str(payload.get("rationale") or ""),
            steps=steps,
            constraints=_as_str_tuple(payload.get("constraints")),
            success_criteria=_as_str_tuple(payload.get("success_criteria")),
        )
        log.info(
            "Plan accepted: %d steps, %d constraints, %d success criteria",
            len(plan.steps),
            len(plan.constraints),
            len(plan.success_criteria),
        )
        return plan
