"""MCP server exposing the physics guardrail"""

from __future__ import annotations

import json
from typing import Any

from crepidinem.config import load_settings
from crepidinem.logging_setup import configure_logging, get_logger
from crepidinem.mcp_tools.harness import PhysicsLimits
from crepidinem.mcp_tools.physics_sandbox import test_code_in_physics_sandbox

__all__ = ["build_server", "main"]

log = get_logger(__name__)

_DEFAULT_LIMITS = PhysicsLimits()

_TOOL_DESCRIPTION = """\
Validate generated robot control code against a deterministic physics harness
inside an ephemeral sandbox. Returns a JSON verdict with `success`, `error`
(a stable violation code such as VELOCITY_LIMIT_EXCEEDED or COLLISION_DETECTED)
and `message`. Code that does not pass this tool must never reach hardware.

The control script is executed with an `arm` object in scope, offering
set_velocity(v), move_to(x, y, z, velocity=None), move_by(dx, dy, dz),
grip(force_n, payload_kg), release(), wait(seconds) and home().
"""


def build_server() -> Any:
    """Construct the MCP server. Imported lazily so ``mcp`` stays optional."""
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as exc:
        msg = "The MCP server needs the 'mcp' extra (SDK 2.x): uv sync --extra mcp"
        raise RuntimeError(msg) from exc

    server = MCPServer("crepidinem-physics-guardrail")
    settings = load_settings()

    @server.tool(description=_TOOL_DESCRIPTION)
    async def test_code_in_physics_sandbox_tool(
        code: str,
        max_velocity: float = _DEFAULT_LIMITS.max_velocity,
        timeout: int = 0,
    ) -> str:
        """Run ``code`` against the physics harness and return a JSON verdict."""
        log.info("MCP stdio request received (%d bytes)", len(code))
        limits = PhysicsLimits(max_velocity=max_velocity)
        report = await test_code_in_physics_sandbox(
            code,
            limits=limits,
            settings=settings,
            timeout=timeout or None,
        )
        
        trimmed = {k: v for k, v in report.items() if k not in {"stdout", "stderr"}}
        return json.dumps(trimmed, indent=2, sort_keys=True)

    return server


def main() -> None:
    """Serve the guardrail over stdio."""
    configure_logging(load_settings().log_level)
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
