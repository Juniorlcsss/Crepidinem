"""Shared fixtures.

Every test here is hermetic: no network, no API key, no Docker.  Agents are
local doubles and the sandbox is the real local subprocess backend, so the
physics verdicts under test are computed rather than asserted into existence.
"""

from __future__ import annotations

import dataclasses

import pytest

from crepidinem.config import Settings, load_settings


@pytest.fixture
def settings() -> Settings:
    """Configuration for a hermetic run: local sandbox, no research."""
    return dataclasses.replace(
        load_settings(use_dotenv=False),
        nebius_api_key=None,
        tavily_api_key=None,
        sandbox_backend="local",
        sandbox_image="python:3.12-slim",
        research_enabled=False,
        exec_timeout=30,
        log_level="WARNING",
    )
