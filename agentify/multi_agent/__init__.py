from agentify.multi_agent.tool_wrapper import AgentTool
from agentify.multi_agent.team import Team
from agentify.multi_agent.pipeline import SequentialPipeline
from agentify.multi_agent.hierarchical import HierarchicalTeam

# Alias for convenience
Pipeline = SequentialPipeline

__all__ = ["AgentTool", "Team", "Pipeline", "SequentialPipeline", "HierarchicalTeam"]
