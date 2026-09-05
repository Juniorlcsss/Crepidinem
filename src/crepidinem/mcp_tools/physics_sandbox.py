"""physics guardrail MCP tool"""

from __future__ import annotations

import json
from typing import Any, Final, TypedDict

from crepidinem.config import Settings, load_settings
from crepidinem.exceptions import (
    SandboxError,
    SandboxExecutionError,
    SandboxTimeoutError,
)
from crepidinem.logging_setup import get_logger
from crepidinem.mcp_tools.harness import (
    EXIT_PHYSICS_VIOLATION,
    RESULT_MARKER,
    VIOLATION_CODES,
    VIOLATION_MARKER,
    PhysicsLimits,
    build_harness,
    new_nonce,
)
from crepidinem.sandbox.base import SandboxBackend, ephemeral_sandbox
from crepidinem.sandbox.factory import build_backend

__all__ = [
    "GuardrailReport",
    "format_feedback",
    "no_code_report",
    "test_code_in_physics_sandbox",
]

log = get_logger(__name__)

_HARNESS_PATH: Final = "/workspace/harness.py"
_MAX_STREAM_CHARS: Final = 8000


class GuardrailReport(TypedDict):
    """structured verdict returned to the agent loop"""

    success: bool
    error: str | None
    message: str
    violation_code: str | None
    detail: dict[str, Any]
    telemetry: dict[str, Any]
    command_log: list[dict[str, Any]]
    waypoints: list[dict[str, Any]]
    exit_code: int | None
    stdout: str
    stderr: str
    sandbox_id: str
    duration_s: float


def _report(
    *,
    success: bool,
    error: str | None,
    message: str,
    violation_code: str | None = None,
    detail: dict[str, Any] | None = None,
    telemetry: dict[str, Any] | None = None,
    command_log: list[dict[str, Any]] | None = None,
    waypoints: list[dict[str, Any]] | None = None,
    exit_code: int | None = None,
    stdout: str = "",
    stderr: str = "",
    sandbox_id: str = "",
    duration_s: float = 0.0,
) -> GuardrailReport:
    return GuardrailReport(
        success=success,
        error=error,
        message=message,
        violation_code=violation_code,
        detail=detail or {},
        telemetry=telemetry or {},
        command_log=command_log or [],
        waypoints=waypoints or [],
        exit_code=exit_code,
        stdout=stdout[-_MAX_STREAM_CHARS:],
        stderr=stderr[-_MAX_STREAM_CHARS:],
        sandbox_id=sandbox_id,
        duration_s=duration_s,
    )


def _extract_harness_payload(stdout: str, nonce: str) -> dict[str, Any] | None:
    """pull last authentic verdict line out of stdout and parse it"""

    payload: dict[str, Any] | None = None
    for line in stdout.splitlines():
        if not line.startswith(RESULT_MARKER):
            continue

        try:
            parsed: Any = json.loads(line[len(RESULT_MARKER) :])

        except json.JSONDecodeError:
            log.warning("Discarding malformed harness result line")
            continue

        if not isinstance(parsed, dict):
            continue

        if parsed.get("nonce") != nonce:
            log.warning("Discarding a verdict line that is not stamped by this harness")
            continue

        payload = parsed
    return payload


def _announced_violation(stderr: str) -> str | None:
    """return the code the harness announced on stderr if it announced one."""
    for line in stderr.splitlines():
        if not line.startswith(VIOLATION_MARKER):
            continue

        code = line.split(":", 1)[-1].strip()
        if code in VIOLATION_CODES:
            return code
    return None


def _scan_for_violation_code(*streams: str) -> str | None:
    """fallback"""
    blob = "\n".join(streams)
    for code in sorted(VIOLATION_CODES):
        if code in blob:
            return code
    return None


def _verdict_from_payload(
    payload: dict[str, Any],
    *,
    exit_code: int,
    stdout: str,
    stderr: str,
    sandbox_id: str,
    duration_s: float,
) -> GuardrailReport:
    status = payload.get("status")
    raw_telemetry = payload.get("telemetry")
    telemetry: dict[str, Any] = raw_telemetry if isinstance(raw_telemetry, dict) else {}
    raw_log = payload.get("command_log")
    command_log = (
        [entry for entry in raw_log if isinstance(entry, dict)] if isinstance(raw_log, list) else []
    )
    raw_waypoints = payload.get("waypoints")
    waypoints = (
        [point for point in raw_waypoints if isinstance(point, dict)]
        if isinstance(raw_waypoints, list)
        else []
    )
    common: dict[str, Any] = {
        "telemetry": telemetry,
        "command_log": command_log,
        "waypoints": waypoints,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "sandbox_id": sandbox_id,
        "duration_s": duration_s,
    }

    if status == "violation":
        raw = payload.get("violation")
        violation: dict[str, Any] = raw if isinstance(raw, dict) else {}
        code = str(violation.get("code") or "UNKNOWN_VIOLATION")
        message = str(violation.get("message") or "The control script broke a physical constraint.")
        detail = violation.get("detail")
        return _report(
            success=False,
            error=code,
            message=message,
            violation_code=code,
            detail=detail if isinstance(detail, dict) else {},
            **common,
        )

    if status == "error":
        message = str(payload.get("error") or "The control script raised an exception.")
        traceback_text = payload.get("traceback")
        if isinstance(traceback_text, str) and traceback_text.strip():
            message = f"{message}\n{traceback_text.strip()}"
        return _report(
            success=False,
            error="RUNTIME_ERROR",
            message=message,
            violation_code=None,
            **common,
        )

    if status == "ok":
        announced = _announced_violation(stderr)
        if announced is not None:
            return _report(
                success=False,
                error=announced,
                message=(
                    f"The harness announced {announced} during the run, then reported "
                    f"success. The control script suppressed a violation; refusing to "
                    f"certify."
                ),
                violation_code=announced,
                **common,
            )
        commands = telemetry.get("commands", 0)
        return _report(
            success=True,
            error=None,
            message=(
                f"Passed every physics check: {commands} commands, "
                f"{telemetry.get('path_length_m', 0.0)} m travelled, "
                f"{telemetry.get('sim_time_s', 0.0)} s simulated."
            ),
            **common,
        )

    return _report(
        success=False,
        error="HARNESS_PROTOCOL_ERROR",
        message=f"Harness reported an unrecognised status {status!r}; treating as failure.",
        **common,
    )


def no_code_report(message: str) -> GuardrailReport:
    """verdict for an attempt that never produced code to test"""
    return _report(success=False, error="AGENT_OUTPUT_UNUSABLE", message=message)


async def test_code_in_physics_sandbox(
    code: str,
    *,
    backend: SandboxBackend | None = None,
    limits: PhysicsLimits | None = None,
    settings: Settings | None = None,
    timeout: int | None = None,
) -> GuardrailReport:
    """Run generated control code against the deterministic physics harness.

    Args:
        code: Raw Python produced by the Code Smith.
        backend: Sandbox to run in. When omitted, one is built from
            ``settings`` and torn down before returning.
        limits: Physical constraints to enforce. Defaults to
            :class:`~crepidinem.mcp_tools.harness.PhysicsLimits`.
        settings: Configuration. Loaded from the environment when omitted.
        timeout: Wall-clock budget for the run, in seconds.

    Returns:
        A :class:`GuardrailReport`. ``success`` is ``True`` only when the
        harness explicitly certified the run.
    """
    resolved_settings = settings or load_settings()
    resolved_limits = limits or PhysicsLimits()
    resolved_timeout = timeout if timeout is not None else resolved_settings.exec_timeout

    log.info(
        "MCP tool call: test_code_in_physics_sandbox (code=%d bytes, backend=%s, timeout=%ds)",
        len(code),
        resolved_settings.sandbox_backend,
        resolved_timeout,
    )

    #cheapest check first
    try:
        compile(code, "control_script.py", "exec")
    except SyntaxError as exc:
        message = f"Control script does not parse: {exc.msg} (line {exc.lineno})."
        log.warning("Guardrail rejected code before execution: %s", message)
        return _report(
            success=False,
            error="SYNTAX_ERROR",
            message=message,
            detail={"line": exc.lineno, "offset": exc.offset, "text": (exc.text or "").strip()},
        )

    nonce = new_nonce()
    harness_source = build_harness(code, resolved_limits, nonce=nonce)
    owns_backend = backend is None
    resolved_backend = backend or build_backend(resolved_settings)

    try:
        async with ephemeral_sandbox(resolved_backend, resolved_settings.sandbox_image) as sid:
            await resolved_backend.write_file(sid, _HARNESS_PATH, harness_source)
            try:
                result = await resolved_backend.execute_code(sid, harness_source, resolved_timeout)
            
            except SandboxTimeoutError as exc:
                log.warning("Guardrail run timed out in sandbox %s", sid)
                return _report(
                    success=False,
                    error="EXECUTION_TIMEOUT",
                    message=(
                        f"Control script exceeded its {resolved_timeout}s budget and was "
                        f"killed. Suspect an unbounded loop or a blocking call."
                    ),
                    exit_code=exc.exit_code,
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                    sandbox_id=sid,
                )
            
            except SandboxExecutionError as exc:
                log.exception("Sandbox %s could not execute the harness", sid)
                return _report(
                    success=False,
                    error="SANDBOX_EXECUTION_ERROR",
                    message=f"The sandbox could not run the harness: {exc}",
                    exit_code=exc.exit_code,
                    stdout=exc.stdout,
                    stderr=exc.stderr,
                    sandbox_id=sid,
                )

            if result.timed_out:
                log.warning("Guardrail run hit its time budget in sandbox %s", sid)
                return _report(
                    success=False,
                    error="EXECUTION_TIMEOUT",
                    message=(
                        f"Control script exceeded its {resolved_timeout}s budget and was "
                        f"killed. Suspect an unbounded loop or a blocking call."
                    ),
                    exit_code=result.exit_code,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    sandbox_id=sid,
                    duration_s=result.duration_s,
                )

            payload = _extract_harness_payload(result.stdout, nonce)
            if payload is not None:
                report = _verdict_from_payload(
                    payload,
                    exit_code=result.exit_code,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    sandbox_id=sid,
                    duration_s=result.duration_s,
                )
            else:
                report = _fallback_verdict(
                    result.stdout,
                    result.stderr,
                    result.exit_code,
                    sid,
                    result.duration_s,
                )

    except SandboxError as exc:
        log.exception("Guardrail could not complete")
        return _report(
            success=False,
            error="SANDBOX_ERROR",
            message=f"Sandbox lifecycle failed, so the code is unverified: {exc}",
        )
    finally:
        if owns_backend:
            await resolved_backend.aclose()

    log.info(
        "MCP tool result: success=%s error=%s (%s)",
        report["success"],
        report["error"],
        report["message"].splitlines()[0] if report["message"] else "",
    )
    return report


def _fallback_verdict(
    stdout: str, stderr: str, exit_code: int, sandbox_id: str, duration_s: float
) -> GuardrailReport:
    """Decide without a harness payload. Always conservative."""
    log.warning(
        "Harness emitted no result line in sandbox %s (exit=%d); falling back to "
        "exit code and marker scan",
        sandbox_id,
        exit_code,
    )
    scanned = _scan_for_violation_code(stdout, stderr)
    if scanned is not None:
        return _report(
            success=False,
            error=scanned,
            message=(
                f"Detected {scanned} in sandbox output, but the harness produced no "
                f"structured result. Treating as a violation."
            ),
            violation_code=scanned,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            sandbox_id=sandbox_id,
            duration_s=duration_s,
        )

    if exit_code == EXIT_PHYSICS_VIOLATION:
        return _report(
            success=False,
            error="PHYSICS_VIOLATION",
            message="Harness exited with the physics-violation status but reported no detail.",
            violation_code="PHYSICS_VIOLATION",
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            sandbox_id=sandbox_id,
            duration_s=duration_s,
        )

    tail = (stderr or stdout).strip()[-1200:]
    return _report(
        success=False,
        error="NO_HARNESS_RESULT" if exit_code == 0 else "NON_ZERO_EXIT",
        message=(
            f"The harness produced no verdict (exit={exit_code}). Code is unverified and "
            f"must not reach hardware.\n{tail}"
        ),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        sandbox_id=sandbox_id,
        duration_s=duration_s,
    )


def format_feedback(report: GuardrailReport) -> str:
    """render a report as the correction prompt handed back to the Code Smith"""
    if report["success"]:
        return f"PASS: {report['message']}"
    lines = [f"FAIL [{report['error']}]: {report['message']}"]
    if report["detail"]:
        lines.append(f"Detail: {json.dumps(report['detail'], sort_keys=True)}")
    if report["telemetry"]:
        lines.append(f"State at failure: {json.dumps(report['telemetry'], sort_keys=True)}")
    lines.append("Rewrite the control script so this constraint is respected.")
    return "\n".join(lines)
