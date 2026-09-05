"""Crepidinem an autonomous physical-science agent with a physics guardrail."""

from __future__ import annotations

from crepidinem.exceptions import (
    AgentError,
    ConfigurationError,
    CrepidinemError,
    PhysicsViolationError,
    SandboxError,
    SandboxExecutionError,
)

__version__ = "0.1.0"

__all__ = [
    "AgentError",
    "ConfigurationError",
    "CrepidinemError",
    "PhysicsViolationError",
    "SandboxError",
    "SandboxExecutionError",
    "__version__",
]
