"""The Code Smith"""

from __future__ import annotations

from crepidinem.agents.base import ExperimentPlan
from crepidinem.agents.prompts import (
    ARM_API,
    CODE_SMITH_SYSTEM,
    UNBRIEFED_NOTE,
    cell_briefing,
)
from crepidinem.config import Settings
from crepidinem.exceptions import AgentParseError
from crepidinem.llm.nebius_client import ChatMessage, NebiusLLMClient, TokenSink
from crepidinem.llm.parsing import extract_python_code
from crepidinem.logging_setup import get_logger
from crepidinem.mcp_tools.harness import PhysicsLimits

__all__ = ["NemotronCodeSmith"]

log = get_logger(__name__)


class NemotronCodeSmith:
    """implements CodeSmith against nemotron nano"""

    def __init__(
        self,
        client: NebiusLLMClient,
        settings: Settings,
        *,
        limits: PhysicsLimits | None = None,
        temperature: float = 0.2,
        parse_retries: int = 1,
        brief_limits: bool = True,
    ) -> None:
        self._client = client
        self._settings = settings
        self._limits = limits or PhysicsLimits()
        self._brief_limits = brief_limits
        self._temperature = temperature
        self._parse_retries = max(0, parse_retries)
        self._history: list[ChatMessage] = []
        self._token_sink: TokenSink | None = None

    @property
    def model(self) -> str:
        return self._settings.coder_model

    def set_token_sink(self, sink: TokenSink | None) -> None:
        """stream this agents reasoning and answer to sink"""
        self._token_sink = sink

    def reset(self) -> None:
        """forget conversation"""
        self._history = []

    def _opening_prompt(self, plan: ExperimentPlan) -> str:
        blocks = ["Write the control script for this plan.", plan.as_prompt()]
        if self._brief_limits:
            blocks.append(cell_briefing(self._limits))
        else:
            blocks.append(UNBRIEFED_NOTE)
        blocks += [ARM_API, "Output one ```python block containing the complete script."]
        return "\n\n".join(blocks)

    @staticmethod
    def _retry_prompt(feedback: str) -> str:
        return (
            "The physics sandbox REJECTED that script. Verbatim verdict:\n\n"
            f"{feedback}\n\n"
            "Fix the specific cause above. Keep everything that was already "
            "safe, change only what the violation requires, and do not "
            "introduce a new violation of a different limit. "
            "Output the complete corrected script in one ```python block."
        )

    async def write_control_script(
        self, plan: ExperimentPlan, *, attempt: int, feedback: str | None
    ) -> str:
        """Draft or repair the control script.

        Raises:
            AgentParseError: The model never returned compilable Python.
        """
        if not self._history or feedback is None:
            self._history = [
                {"role": "system", "content": CODE_SMITH_SYSTEM},
                {"role": "user", "content": self._opening_prompt(plan)},
            ]
            log.info("Code Smith (%s) drafting attempt %d", self.model, attempt)
        else:
            self._history.append({"role": "user", "content": self._retry_prompt(feedback)})
            log.info(
                "Code Smith (%s) rewriting for attempt %d after: %s",
                self.model,
                attempt,
                feedback.splitlines()[0],
            )

        last_error: AgentParseError | None = None
        for parse_attempt in range(self._parse_retries + 1):
            response = await self._client.complete(
                self.model,
                self._history,
                temperature=self._temperature,
                max_tokens=self._settings.coder_max_tokens,
                label=f"Code Smith #{attempt}",
                on_token=self._token_sink,
            )
            try:
                code = extract_python_code(response.content)
            except AgentParseError as exc:
                last_error = exc
                log.warning(
                    "Code Smith returned no usable Python (parse attempt %d): %s",
                    parse_attempt + 1,
                    exc,
                )
                if parse_attempt == self._parse_retries:
                    break
                self._history.extend(
                    [
                        {"role": "assistant", "content": response.content},
                        {
                            "role": "user",
                            "content": (
                                f"{exc} Reply with the complete script in a single "
                                "```python fenced block and nothing else."
                            ),
                        },
                    ]
                )
                continue

            self._history.append({"role": "assistant", "content": f"```python\n{code}\n```"})
            log.info("Code Smith produced %d bytes of control code", len(code))
            return code

        assert last_error is not None
        raise last_error
