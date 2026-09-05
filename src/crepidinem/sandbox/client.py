"""REST client for the Nebius Token Factory Sandbox API"""

from __future__ import annotations

import asyncio
import time
from types import TracebackType
from typing import Any, Final, Self

import httpx

from crepidinem.exceptions import (
    SandboxAPIError,
    SandboxCreationError,
    SandboxExecutionError,
    SandboxTimeoutError,
)
from crepidinem.logging_setup import get_logger
from crepidinem.sandbox.models import ExecutionResult

__all__ = ["NebiusSandboxClient"]

log = get_logger(__name__)

_ID_KEYS: Final = ("id", "sandbox_id", "sandboxId", "name", "uuid")
_STDOUT_KEYS: Final = ("stdout", "output", "stdOut")
_STDERR_KEYS: Final = ("stderr", "error", "stdErr")
_EXIT_KEYS: Final = ("exit_code", "exitCode", "returncode", "status")
_RETRY_STATUS: Final = frozenset({408, 425, 429, 500, 502, 503, 504})
_BODY_SNIPPET: Final = 2000


def _pick_str(payload: dict[str, Any], keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return default


def _pick_int(payload: dict[str, Any], keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.lstrip("-").isdigit():
            return int(value)
    return default


class NebiusSandboxClient:
    """lifecycle and execution client"""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        http_timeout: float = 60.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._max_retries = max(1, max_retries)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(http_timeout),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "crepidinem/0.1",
            },
        )

    #HTTP

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """issue one API call"""
        url = f"{self._base_url}{path}"
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._client.request(method, url, json=json, timeout=timeout)

            except httpx.TimeoutException as exc:
                last_error = exc
                log.warning("%s %s timed out (attempt %d)", method, url, attempt)

            except httpx.HTTPError as exc:
                last_error = exc
                log.warning("%s %s failed: %s (attempt %d)", method, url, exc, attempt)

            else:
                retryable = response.status_code in _RETRY_STATUS and attempt < self._max_retries
                if not retryable:
                    if response.status_code >= httpx.codes.BAD_REQUEST:
                        msg = f"{method} {url} returned HTTP {response.status_code}"
                        raise SandboxAPIError(
                            msg,
                            status_code=response.status_code,
                            body=response.text[:_BODY_SNIPPET],
                        )
                    return response
                log.warning(
                    "%s %s -> HTTP %d, retrying (attempt %d)",
                    method,
                    url,
                    response.status_code,
                    attempt,
                )

            if attempt < self._max_retries:
                await asyncio.sleep(0.5 * 2 ** (attempt - 1))

        msg = f"{method} {url} failed after {self._max_retries} attempts"
        raise SandboxAPIError(msg) from last_error

    @staticmethod
    def _json_object(response: httpx.Response) -> dict[str, Any]:
        try:
            payload: Any = response.json()
        except ValueError as exc:
            msg = "Sandbox API returned a non-JSON body"
            raise SandboxAPIError(
                msg, status_code=response.status_code, body=response.text[:_BODY_SNIPPET]
            ) from exc
        if not isinstance(payload, dict):
            msg = f"Sandbox API returned {type(payload).__name__}, expected an object"
            raise SandboxAPIError(
                msg, status_code=response.status_code, body=response.text[:_BODY_SNIPPET]
            )
        return payload

    # SandboxBackend

    async def create_sandbox(self, image: str) -> str:
        """provision an sandbox running image"""

        log.info("Creating Nebius sandbox (image=%s)", image)
        response = await self._request("POST", "", json={"image": image})
        payload = self._json_object(response)
        sandbox_id = _pick_str(payload, _ID_KEYS)
        if not sandbox_id:
            msg = f"Sandbox creation response contained no id field: {payload!r}"
            raise SandboxCreationError(msg)
        log.info("Sandbox created: %s", sandbox_id)
        return sandbox_id

    async def write_file(self, sandbox_id: str, path: str, content: str) -> None:
        """Write ``content`` to ``path`` inside the sandbox."""
        log.debug("Writing %d bytes to %s:%s", len(content), sandbox_id, path)
        await self._request(
            "POST",
            f"/{sandbox_id}/files",
            json={"path": path, "content": content},
        )

    async def execute_code(self, sandbox_id: str, code: str, timeout: int) -> ExecutionResult:
        """execute code inside the sandbox and return its result"""

        log.info("Executing %d bytes in sandbox %s (timeout=%ds)", len(code), sandbox_id, timeout)
        started = time.perf_counter()
        try:
            response = await self._request(
                "POST",
                f"/{sandbox_id}/execute",
                json={"code": code, "timeout": timeout},
                timeout=timeout + 15,
            )
        except SandboxAPIError as exc:
            msg = f"Sandbox {sandbox_id} could not execute the submitted code: {exc}"
            raise SandboxExecutionError(msg) from exc
        duration = time.perf_counter() - started

        payload = self._json_object(response)
        result = ExecutionResult(
            stdout=_pick_str(payload, _STDOUT_KEYS),
            stderr=_pick_str(payload, _STDERR_KEYS),
            exit_code=_pick_int(payload, _EXIT_KEYS, default=0),
            duration_s=duration,
            timed_out=bool(payload.get("timed_out")),
            sandbox_id=sandbox_id,
        )
        log.info("Sandbox %s execution finished: %s", sandbox_id, result.summary())
        if result.timed_out:
            msg = f"Execution in sandbox {sandbox_id} exceeded its {timeout}s budget"
            raise SandboxTimeoutError(
                msg,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
            )
        return result

    async def destroy_sandbox(self, sandbox_id: str) -> None:
        """delete the sandbox"""

        log.info("Destroying sandbox %s", sandbox_id)
        try:
            await self._request("DELETE", f"/{sandbox_id}")
        except SandboxAPIError as exc:
            if exc.status_code == httpx.codes.NOT_FOUND:
                log.debug("Sandbox %s was already gone", sandbox_id)
                return

            log.exception("Failed to destroy sandbox %s", sandbox_id)
            raise

    async def aclose(self) -> None:
        """Close the underlying connection pool if this client owns it."""
        if self._owns_client:
            await self._client.aclose()

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
