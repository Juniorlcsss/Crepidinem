"""sandbox contract"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol, runtime_checkable

from crepidinem.sandbox.models import ExecutionResult

__all__ = ["SandboxBackend", "ephemeral_sandbox"]


@runtime_checkable
class SandboxBackend(Protocol):

    async def create_sandbox(self, image: str) -> str:
        """provision a sandbox and return identifier"""
        ...

    async def write_file(self, sandbox_id: str, path: str, content: str) -> None:
        """Write content to path inside the sandbox"""
        ...

    async def execute_code(self, sandbox_id: str, code: str, timeout: int) -> ExecutionResult:
        """Run code as a python script and capture its result"""
        ...

    async def destroy_sandbox(self, sandbox_id: str) -> None:
        """kill sandbox """
        ...

    async def aclose(self) -> None:
        """release backend level resources"""
        ...


@asynccontextmanager
async def ephemeral_sandbox(backend: SandboxBackend, image: str) -> AsyncIterator[str]:
    """create a sandbox, yield its id, and guarantee teardown"""
    sandbox_id = await backend.create_sandbox(image)
    try:
        yield sandbox_id
    finally:
        await backend.destroy_sandbox(sandbox_id)
