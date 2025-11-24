from typing import Dict, List, Union, Generator
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

        # Validate that root is in the hierarchy if it has children,
        # or at least that the structure makes sense.
        # (We assume the user constructs the dict correctly for now)

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

        # 2. Recursively register children as tools for parents
        # We need to do this for every parent in the hierarchy map.
        # But we must ensure the 'parent_addr' is correct.
        # Actually, in a static hierarchy, the parent's address might need to be dynamic
        # if we want to support multiple sessions.

        # However, AgentTool needs 'parent_addr' at construction time to know where to link memory?
        # Let's look at AgentTool again.
        # AgentTool takes 'parent_addr'. When run_agent is called, it creates 'child_addr' using parent's IDs.

        # Issue: We don't know the parent's address for DEEP nodes until runtime?
        # Wait, the user requirement says:
        # "Para cada padre: parent_addr = MemoryAddress(..., agent_id=parent.config.name)"
        # This implies we can pre-calculate addresses based on the session_id passed to `run`.

        self._register_hierarchy_tools(session_id, user_id)

        # 3. Run Root
        return self.root.respond(user_input=user_input, addr=root_addr)

    def _register_hierarchy_tools(self, session_id: str, user_id: str) -> None:
        """Registers children as tools for their parents based on the current session."""

        for parent, children in self.hierarchy.items():
            # Define parent's address for this session
            parent_addr = MemoryAddress(
                user_id=user_id,
                conversation_id=session_id,
                agent_id=parent.config.name,
            )

            for child in children:
                # Check if tool already exists to avoid duplication?
                # The user said "avoid registering duplicate tools" is a nice to have for Team,
                # but here we should probably check too.

                # Check if child is a BaseAgent or a Flow
                if isinstance(child, BaseAgent):
                    tool_wrapper = AgentTool(agent=child, parent_addr=parent_addr)
                else:
                    # It's a Flow (Team, Pipeline, etc)
                    # We need a name and description for the tool
                    # We try to get it from the object, or generate a default
                    child_name = getattr(child, "name", f"Team_{id(child)}")
                    if hasattr(child, "config"):
                        child_name = child.config.name
                    
                    child_desc = getattr(child, "description", f"Delegate to {child_name}")
                    
                    tool_wrapper = FlowTool(
                        flow=child,
                        name=child_name,
                        description=child_desc,
                        parent_addr=parent_addr
                    )

                # Register with parent
                parent.register_tool(tool_wrapper)
