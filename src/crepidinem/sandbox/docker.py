"""Docker backend: real isolation, run locally"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from pathlib import Path
from types import TracebackType
from typing import Self

from crepidinem.exceptions import SandboxCreationError, SandboxExecutionError
from crepidinem.logging_setup import get_logger
from crepidinem.sandbox.models import ExecutionResult

__all__ = ["DockerSandbox"]

log = get_logger(__name__)

_ENTRYPOINT = "__crepidinem_main__.py"
_WORKDIR = "/sandbox"


class DockerSandbox:
    """runs the harness inside a locked down throwaway container"""

    def __init__(
        self,
        *,
        docker: str | None = None,
        memory: str = "256m",
        pids_limit: int = 128,
        cpus: str = "1.0",
    ) -> None:
        self._docker = docker or shutil.which("docker") or "docker"
        self._memory = memory
        self._pids_limit = pids_limit
        self._cpus = cpus
        self._live: dict[str, tuple[Path, str]] = {}

    def _entry_for(self, sandbox_id: str) -> tuple[Path, str]:
        try:
            return self._live[sandbox_id]
        except KeyError as exc:
            msg = f"Unknown sandbox id {sandbox_id!r} (already destroyed?)"
            raise SandboxExecutionError(msg) from exc

    async def _run(self, *args: str, timeout: float) -> tuple[int, str, str]:
        """Invoke the docker CLI and capture its output."""
        try:
            process = await asyncio.create_subprocess_exec(
                self._docker,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            msg = f"Could not run {self._docker!r}. Is Docker installed and on PATH? ({exc})"
            raise SandboxCreationError(msg) from exc

        try:
            raw_out, raw_err = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            raw_out, raw_err = await process.communicate()
            return -1, raw_out.decode("utf-8", "replace"), "docker command timed out"

        return (
            process.returncode if process.returncode is not None else -1,
            raw_out.decode("utf-8", "replace"),
            raw_err.decode("utf-8", "replace"),
        )

    #SandboxBackend

    async def create_sandbox(self, image: str) -> str:
        """prepare a host directory and confirm the image is available"""

        code, _, stderr = await self._run("image", "inspect", image, timeout=30)

        if code != 0:
            log.info("Image %s not present locally, pulling", image)
            code, _, stderr = await self._run("pull", image, timeout=600)

            if code != 0:
                msg = f"Could not pull image {image!r}: {stderr.strip()[:400]}"
                raise SandboxCreationError(msg)

        root = Path(tempfile.gettempdir()) / "crepidinem-docker"

        try:
            root.mkdir(parents=True, exist_ok=True)
            path = Path(tempfile.mkdtemp(prefix="sbx-", dir=root))

        except OSError as exc:
            msg = f"Could not create a sandbox workdir under {root}: {exc}"
            raise SandboxCreationError(msg) from exc

        sandbox_id = path.name
        self._live[sandbox_id] = (path, image)
        log.info("Created docker sandbox %s at %s (image=%s)", sandbox_id, path, image)
        return sandbox_id

    async def write_file(self, sandbox_id: str, path: str, content: str) -> None:
        """write into the host directory"""

        base, _ = self._entry_for(sandbox_id)
        target = (base / path.lstrip("/\\")).resolve()
        if not target.is_relative_to(base.resolve()):
            msg = f"Refusing to write outside the sandbox: {path!r}"
            raise SandboxExecutionError(msg)
        
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        log.debug("Wrote %d bytes to %s", len(content), target)

    async def execute_code(self, sandbox_id: str, code: str, timeout: int) -> ExecutionResult:
        """Run code in a fresh networkless read-only container."""

        base, image = self._entry_for(sandbox_id)
        (base / _ENTRYPOINT).write_text(code, encoding="utf-8")

        log.info(
            "Executing %d bytes in docker sandbox %s (image=%s timeout=%ds)",
            len(code),
            sandbox_id,
            image,
            timeout,
        )
        started = time.perf_counter()
        exit_code, stdout, stderr = await self._run(
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            f"--memory={self._memory}",
            f"--pids-limit={self._pids_limit}",
            f"--cpus={self._cpus}",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
            "-v",
            f"{base}:{_WORKDIR}:ro",
            "-w",
            _WORKDIR,
            image,
            "python",
            "-I",
            f"{_WORKDIR}/{_ENTRYPOINT}",
            timeout=timeout + 20,
        )
        duration = time.perf_counter() - started
        timed_out = exit_code == -1 and "timed out" in stderr

        result = ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_s=duration,
            timed_out=timed_out,
            sandbox_id=sandbox_id,
            metadata={"image": image, "isolation": "docker"},
        )
        log.info("Docker sandbox %s execution finished: %s", sandbox_id, result.summary())
        return result

    async def destroy_sandbox(self, sandbox_id: str) -> None:
        """remove the host workdir"""
        entry = self._live.pop(sandbox_id, None)
        if entry is None:
            log.debug("Docker sandbox %s was already destroyed", sandbox_id)
            return
        shutil.rmtree(entry[0], ignore_errors=True)
        log.info("Destroyed docker sandbox %s", sandbox_id)

    async def aclose(self) -> None:
        """destroy any sandbox"""
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
