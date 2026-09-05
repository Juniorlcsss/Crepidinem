"""sandbox execution backends"""

from __future__ import annotations

from crepidinem.sandbox.base import SandboxBackend, ephemeral_sandbox
from crepidinem.sandbox.client import NebiusSandboxClient
from crepidinem.sandbox.docker import DockerSandbox
from crepidinem.sandbox.factory import build_backend
from crepidinem.sandbox.local import LocalSubprocessSandbox
from crepidinem.sandbox.models import ExecutionResult

__all__ = [
    "DockerSandbox",
    "ExecutionResult",
    "LocalSubprocessSandbox",
    "NebiusSandboxClient",
    "SandboxBackend",
    "build_backend",
    "ephemeral_sandbox",
]
