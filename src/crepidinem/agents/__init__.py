"""agent definitions and routing"""

from __future__ import annotations

from crepidinem.agents.base import CodeSmith, ExperimentPlan, PrincipalInvestigator
from crepidinem.agents.code_smith_agent import NemotronCodeSmith
from crepidinem.agents.factory import AgentTeam, build_agents
from crepidinem.agents.pi_agent import NemotronPrincipalInvestigator

__all__ = [
    "AgentTeam",
    "CodeSmith",
    "ExperimentPlan",
    "NemotronCodeSmith",
    "NemotronPrincipalInvestigator",
    "PrincipalInvestigator",
    "build_agents",
]
