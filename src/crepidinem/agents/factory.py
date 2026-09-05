"""agent selection and routing"""

from __future__ import annotations

from dataclasses import dataclass

from crepidinem.agents.base import CodeSmith, PrincipalInvestigator
from crepidinem.agents.code_smith_agent import NemotronCodeSmith
from crepidinem.agents.pi_agent import NemotronPrincipalInvestigator
from crepidinem.config import Settings
from crepidinem.llm.nebius_client import NebiusLLMClient
from crepidinem.logging_setup import get_logger
from crepidinem.mcp_tools.harness import PhysicsLimits

__all__ = ["AgentTeam", "build_agents"]

log = get_logger(__name__)


@dataclass(slots=True)
class AgentTeam:
    """the two agents plus whatever resources they own"""

    principal_investigator: PrincipalInvestigator
    code_smith: CodeSmith
    planner_model: str
    coder_model: str
    llm_client: NebiusLLMClient | None = None

    async def aclose(self) -> None:
        """release the shared inference connection pool"""
        if self.llm_client is not None:
            await self.llm_client.aclose()


def build_agents(settings: Settings, limits: PhysicsLimits | None = None) -> AgentTeam:
    """construct agent team"""
    
    client = NebiusLLMClient.from_settings(settings)
    log.info(
        "Routing: planning -> %s, coding -> %s (via %s), limits briefed to coder: %s",
        settings.planner_model,
        settings.coder_model,
        settings.llm_base_url,
        settings.brief_limits,
    )
    return AgentTeam(
        principal_investigator=NemotronPrincipalInvestigator(
            client, settings, limits=limits, brief_limits=settings.brief_limits
        ),
        code_smith=NemotronCodeSmith(
            client, settings, limits=limits, brief_limits=settings.brief_limits
        ),
        planner_model=settings.planner_model,
        coder_model=settings.coder_model,
        llm_client=client,
    )
