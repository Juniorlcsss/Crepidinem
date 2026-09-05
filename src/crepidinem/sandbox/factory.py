"""Backend selection"""

from __future__ import annotations

from crepidinem.config import Settings
from crepidinem.logging_setup import get_logger
from crepidinem.sandbox.base import SandboxBackend
from crepidinem.sandbox.client import NebiusSandboxClient
from crepidinem.sandbox.docker import DockerSandbox
from crepidinem.sandbox.local import LocalSubprocessSandbox

__all__ = ["build_backend"]

log = get_logger(__name__)


def build_backend(settings: Settings) -> SandboxBackend:
    """construct backend named by settings.sandbox_backend"""

    if settings.sandbox_backend == "docker":
        log.info("Using docker sandbox backend (image=%s)", settings.sandbox_image)
        return DockerSandbox()

    if settings.sandbox_backend == "nebius":
        log.info("Using Nebius sandbox backend at %s", settings.nebius_sandbox_api_url)
        return NebiusSandboxClient(
            api_key=settings.require_nebius_key(),
            base_url=settings.nebius_sandbox_api_url,
            http_timeout=settings.http_timeout,
        )

    log.info("Using local subprocess sandbox backend (not an isolation boundary)")
    return LocalSubprocessSandbox()
