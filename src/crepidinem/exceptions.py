"""Exception hierarchy for Crepidinem"""

from __future__ import annotations

__all__ = [
    "AgentError",
    "AgentParseError",
    "ConfigurationError",
    "CrepidinemError",
    "LLMError",
    "LLMTransportError",
    "PhysicsViolationError",
    "ReasoningBudgetExhaustedError",
    "ResearchError",
    "SandboxAPIError",
    "SandboxCreationError",
    "SandboxError",
    "SandboxExecutionError",
    "SandboxTimeoutError",
]


class CrepidinemError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(CrepidinemError):
    """Required configuration is missing or malformed."""


class AgentError(CrepidinemError):
    """An agent failed to produce a usable artefact (plan, code, ...)."""


class AgentParseError(AgentError):
    """An agent answered, but its output could not be parsed.

    Carries the raw text so the caller can show the model what it produced
    when asking for a correction.
    """

    def __init__(self, message: str, *, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


class LLMError(CrepidinemError):
    """The inference API answered, but not usefully."""


class LLMTransportError(LLMError):
    """The inference API could not be reached, or refused the request."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ReasoningBudgetExhaustedError(LLMError):
    """A reasoning model spent its whole token budget before answering.

    This arrives as an HTTP 200 with ``finish_reason == "length"`` and empty
    ``content``, which reads like "the model had nothing to say" but means "the
    model never reached its answer".  It is separate from :class:`LLMError`
    because the remedy is specific and mechanical: raise ``max_tokens`` for that
    agent and try again.
    """


class ResearchError(CrepidinemError):
    """Literature search could not be completed.

    Never fatal to a run: research only refines the limits, so the caller
    falls back to the cell defaults and says so.
    """


class SandboxError(CrepidinemError):
    """Base class for sandbox lifecycle and execution failures."""


class SandboxAPIError(SandboxError):
    """The sandbox API returned an unexpected status or payload."""

    def __init__(self, message: str, *, status_code: int | None = None, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class SandboxCreationError(SandboxError):
    """A sandbox could not be provisioned."""


class SandboxExecutionError(SandboxError):
    """Code was submitted but the sandbox could not run it to completion.

    This is an *infrastructure* failure.  Code that runs and then fails a
    physics check is not this - that is a :class:`PhysicsViolationError`.
    """

    def __init__(
        self,
        message: str,
        *,
        stdout: str = "",
        stderr: str = "",
        exit_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


class SandboxTimeoutError(SandboxExecutionError):
    """The submitted code exceeded its wall-clock budget."""


class PhysicsViolationError(CrepidinemError):
    """Generated control code broke a hard physical constraint.

    ``code`` is a stable machine-readable identifier (for example
    ``VELOCITY_LIMIT_EXCEEDED``) that the Code Smith is fed back verbatim so
    that its rewrite is targeted rather than a blind retry.
    """

    def __init__(self, code: str, message: str, *, detail: dict[str, object] | None = None) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.detail: dict[str, object] = detail or {}
