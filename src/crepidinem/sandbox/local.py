"""Local subprocess backend"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import TracebackType
from typing import Self

from crepidinem.exceptions import SandboxCreationError, SandboxExecutionError
from crepidinem.logging_setup import get_logger
from crepidinem.sandbox.models import ExecutionResult

__all__ = ["LocalSubprocessSandbox"]

log = get_logger(__name__)

_ENTRYPOINT = "__crepidinem_main__.py"


class LocalSubprocessSandbox:
    """runs harness code in a throwaway directory on this machine"""

    def __init__(self, *, root: Path | None = None, python: str | None = None) -> None:
        self._root = root or Path(tempfile.gettempdir()) / "crepidinem-sandboxes"
        self._python = python or sys.executable
        self._live: dict[str, Path] = {}

    def _dir_for(self, sandbox_id: str) -> Path:
        try:
            return self._live[sandbox_id]
        except KeyError as exc:
            msg = f"Unknown sandbox id {sandbox_id!r} (already destroyed?)"
            raise SandboxExecutionError(msg) from exc

    #SandboxBackend

    async def create_sandbox(self, image: str) -> str:
        """create a temp directory standing in for a container"""
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            path = Path(tempfile.mkdtemp(prefix="sbx-", dir=self._root))
        except OSError as exc:
            msg = f"Could not create a local sandbox under {self._root}: {exc}"
            raise SandboxCreationError(msg) from exc

        sandbox_id = path.name
        self._live[sandbox_id] = path
        log.info(
            "Created local sandbox %s at %s (image=%s ignored locally)",
            sandbox_id,
            path,
            image,
        )
        return sandbox_id

    async def write_file(self, sandbox_id: str, path: str, content: str) -> None:
        """write content to a path"""
        base = self._dir_for(sandbox_id)
        target = (base / path.lstrip("/\\")).resolve()

        if not target.is_relative_to(base.resolve()):
            msg = f"Refusing to write outside the sandbox: {path!r}"
            raise SandboxExecutionError(msg)
        
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        log.debug("Wrote %d bytes to %s", len(content), target)

    async def execute_code(self, sandbox_id: str, code: str, timeout: int) -> ExecutionResult:
        """run code as a script with the sandbox directory"""
        base = self._dir_for(sandbox_id)
        entry = base / _ENTRYPOINT
        entry.write_text(code, encoding="utf-8")

        log.info(
            "Executing %d bytes in local sandbox %s (timeout=%ds)", len(code), sandbox_id, timeout
        )
        started = time.perf_counter()
        try:
            process = await asyncio.create_subprocess_exec(
                self._python,
                "-I",
                str(entry),
                cwd=str(base),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            msg = f"Could not start the local interpreter {self._python!r}: {exc}"
            raise SandboxExecutionError(msg) from exc

        timed_out = False
        try:
            raw_out, raw_err = await asyncio.wait_for(process.communicate(), timeout=timeout)

        except TimeoutError:
            timed_out = True
            process.kill()
            raw_out, raw_err = await process.communicate()

        result = ExecutionResult(
            stdout=raw_out.decode("utf-8", errors="replace"),
            stderr=raw_err.decode("utf-8", errors="replace"),
            exit_code=process.returncode if process.returncode is not None else -1,
            duration_s=time.perf_counter() - started,
            timed_out=timed_out,
            sandbox_id=sandbox_id,
        )
        log.info("Local sandbox %s execution finished: %s", sandbox_id, result.summary())
        return result

    async def destroy_sandbox(self, sandbox_id: str) -> None:
        """remove the sandbox directory"""
        path = self._live.pop(sandbox_id, None)
        if path is None:
            log.debug("Local sandbox %s was already destroyed", sandbox_id)
            return
        shutil.rmtree(path, ignore_errors=True)
        log.info("Destroyed local sandbox %s", sandbox_id)

    async def aclose(self) -> None:
        """destroy any sandbox this backend still owns."""
        for sandbox_id in list(self._live):
            await self.destroy_sandbox(sandbox_id)

    #context management

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()
