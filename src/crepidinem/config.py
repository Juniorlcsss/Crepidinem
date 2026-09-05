"""Environment-backed configuration"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, cast, get_args

from dotenv import load_dotenv

from crepidinem.exceptions import ConfigurationError

__all__ = ["BackendName", "ResearchTrust", "Settings", "load_settings"]

BackendName = Literal["local", "docker", "nebius"]

ResearchTrust = Literal["envelope", "narrow-only"]

_DEFAULT_SANDBOX_URL = "https://api.studio.nebius.com/v1/sandboxes"
_DEFAULT_IMAGE = "python:3.12-slim"

_DEFAULT_LLM_BASE_URL = "https://api.tokenfactory.nebius.com/v1"

NEMOTRON_ULTRA = "nvidia/Nemotron-3-Ultra-550b-a55b"
NEMOTRON_SUPER = "nvidia/nemotron-3-super-120b-a12b"
NEMOTRON_NANO = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved runtime configuration."""

    nebius_api_key: str | None
    nebius_sandbox_api_url: str
    tavily_api_key: str | None
    sandbox_backend: BackendName
    sandbox_image: str
    exec_timeout: int
    http_timeout: float
    log_level: str

    #LLM routing
    llm_base_url: str
    planner_model: str
    coder_model: str
    llm_timeout: float
    planner_max_tokens: int
    coder_max_tokens: int
    brief_limits: bool

    #literature research
    research_enabled: bool
    research_trust: ResearchTrust
    research_max_results: int


    research_cache_enabled: bool
    research_cache_refresh: bool
    research_cache_ttl_hours: float
    research_cache_dir: str | None

    @property
    def research_available(self) -> bool:
        """Whether research can run: it is switched on and there is a key.

        The extraction step also needs an inference key, but that is not
        checked here: research is an enhancement, and a missing Nebius key
        should fail once, loudly, where it is actually required.
        """
        return self.research_enabled and bool(self.tavily_api_key)

    def require_tavily_key(self) -> str:
        """Return the Tavily key or fail with an actionable message."""
        if not self.tavily_api_key:
            msg = (
                "TAVILY_API_KEY is not set, so hardware research cannot run. "
                "Add it to .env, or pass --no-research to use the cell defaults."
            )
            raise ConfigurationError(msg)
        return self.tavily_api_key

    def require_nebius_key(self) -> str:
        """Return the Nebius key or fail with an actionable message."""
        if not self.nebius_api_key:
            msg = (
                "NEBIUS_API_KEY is not set. Copy .env.example to .env and fill it in, "
                "or run with CREPIDINEM_SANDBOX_BACKEND=local."
            )
            raise ConfigurationError(msg)
        return self.nebius_api_key


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        msg = f"{name} must be an integer, got {raw!r}"
        raise ConfigurationError(msg) from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        msg = f"{name} must be a number, got {raw!r}"
        raise ConfigurationError(msg) from exc


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    msg = f"{name} must be a boolean (true/false), got {raw!r}"
    raise ConfigurationError(msg)


def _env_choice[ChoiceT: (BackendName, ResearchTrust)](
    name: str, default: ChoiceT, allowed: tuple[str, ...]
) -> ChoiceT:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value not in allowed:
        msg = f"{name} must be one of {allowed}, got {raw!r}"
        raise ConfigurationError(msg)
    return cast("ChoiceT", value)


def _env_str(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def load_settings(*, use_dotenv: bool = True) -> Settings:
    """Read configuration from the process environment (and ``.env``)."""
    if use_dotenv:
        load_dotenv(override=False)

    return Settings(
        nebius_api_key=_env_str("NEBIUS_API_KEY"),
        nebius_sandbox_api_url=(_env_str("NEBIUS_SANDBOX_API_URL") or _DEFAULT_SANDBOX_URL).rstrip(
            "/"
        ),
        tavily_api_key=_env_str("TAVILY_API_KEY"),
        sandbox_backend=_env_choice("CREPIDINEM_SANDBOX_BACKEND", "local", get_args(BackendName)),
        sandbox_image=_env_str("CREPIDINEM_SANDBOX_IMAGE") or _DEFAULT_IMAGE,
        exec_timeout=_env_int("CREPIDINEM_EXEC_TIMEOUT", 30),
        http_timeout=_env_float("CREPIDINEM_HTTP_TIMEOUT", 60.0),
        log_level=(_env_str("CREPIDINEM_LOG_LEVEL") or "INFO").upper(),
        llm_base_url=(_env_str("NEBIUS_BASE_URL") or _DEFAULT_LLM_BASE_URL).rstrip("/"),
        planner_model=_env_str("CREPIDINEM_PLANNER_MODEL") or NEMOTRON_ULTRA,
        coder_model=_env_str("CREPIDINEM_CODER_MODEL") or NEMOTRON_NANO,
        llm_timeout=_env_float("CREPIDINEM_LLM_TIMEOUT", 300.0),
        planner_max_tokens=_env_int("CREPIDINEM_PLANNER_MAX_TOKENS", 4096),
        coder_max_tokens=_env_int("CREPIDINEM_CODER_MAX_TOKENS", 4096),
        brief_limits=_env_bool("CREPIDINEM_BRIEF_LIMITS", default=True),
        research_enabled=_env_bool("CREPIDINEM_RESEARCH", default=True),
        research_trust=_env_choice(
            "CREPIDINEM_RESEARCH_TRUST", "envelope", get_args(ResearchTrust)
        ),
        research_max_results=_env_int("CREPIDINEM_RESEARCH_MAX_RESULTS", 5),
        research_cache_enabled=_env_bool("CREPIDINEM_RESEARCH_CACHE", default=True),
        research_cache_refresh=_env_bool("CREPIDINEM_RESEARCH_REFRESH", default=False),
        research_cache_ttl_hours=_env_float("CREPIDINEM_RESEARCH_CACHE_TTL_HOURS", 24 * 14),
        research_cache_dir=_env_str("CREPIDINEM_RESEARCH_CACHE_DIR"),
    )
