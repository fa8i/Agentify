from typing import Dict, List, Union, Generator, AsyncGenerator
from agentify.core.agent import BaseAgent
from agentify.memory.interfaces import MemoryAddress
from agentify.multi_agent.tool_wrapper import AgentTool, FlowTool, Flow


class HierarchicalTeam:
    """Orchestrates a hierarchy of agents (Tree structure).

    - Root agent is the entry point.
    - Parents delegate to children via tools.
    - Communication is strictly Top-Down.
    """

    def __init__(
        self,
        root: BaseAgent,
        hierarchy: Dict[BaseAgent, List[Union[BaseAgent, Flow]]],
    ):
        """
        Args:
            root: The top-level agent.
            hierarchy: A dictionary mapping parent agents to their list of children.
        """
        self.root = root
        self.hierarchy = hierarchy

    def run(
        self,
        user_input: str,
        session_id: str = "default_session",
        user_id: str = "default_user",
    ) -> Union[str, Generator[str, None, None]]:
        """Run the hierarchical flow."""

        # 1. Setup Root Address
        root_addr = MemoryAddress(
            user_id=user_id,
            conversation_id=session_id,
            agent_id=self.root.config.name,
        )

        # 2. Register hierarchy tools for this session
        self._register_hierarchy_tools(session_id, user_id)

        # 3. Run Root
        return self.root.run(user_input=user_input, addr=root_addr)

    async def arun(
        self,
        user_input: str,
        session_id: str = "default_session",
        user_id: str = "default_user",
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Async version of run(). Uses root agent's arun() for async execution."""

        # 1. Setup Root Address
        root_addr = MemoryAddress(
            user_id=user_id,
            conversation_id=session_id,
            agent_id=self.root.config.name,
        )

        # 2. Register hierarchy tools for this session
        self._register_hierarchy_tools(session_id, user_id)

        # 3. Run Root asynchronously
        return await self.root.arun(user_input=user_input, addr=root_addr)

    def _register_hierarchy_tools(self, session_id: str, user_id: str) -> None:
        """Registers children as tools for their parents based on the current session."""

        for parent, children in self.hierarchy.items():
            parent_addr = MemoryAddress(
                user_id=user_id,
                conversation_id=session_id,
                agent_id=parent.config.name,
            )

            for child in children:
                # Check if child is a BaseAgent or a Flow
                if isinstance(child, BaseAgent):
                    tool_wrapper = AgentTool(agent=child, parent_addr=parent_addr)
                else:
                    # Handle Flows (Team, Pipeline, etc)
                    child_name = getattr(child, "name", f"Team_{id(child)}")
                    if hasattr(child, "config"):
                        child_name = child.config.name

                    child_desc = getattr(
                        child, "description", f"Delegate to {child_name}"
                    )

                    tool_wrapper = FlowTool(
                        flow=child,
                        name=child_name,
                        description=child_desc,
                        parent_addr=parent_addr,
                    )

                # Register with parent
                parent.register_tool(tool_wrapper)

