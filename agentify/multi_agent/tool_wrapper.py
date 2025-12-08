from typing import Any, Dict, Optional
from agentify.core.agent import BaseAgent
from agentify.core.tool import Tool
from agentify.memory.interfaces import MemoryAddress


class AgentTool(Tool):
    """Wraps a BaseAgent as a Tool so it can be called by another agent.

    When the tool is invoked:
    1. It receives 'instructions' from the caller.
    2. It triggers the wrapped agent's `respond` method.
    3. It returns the agent's final answer as the tool output.
    """

    def __init__(
        self,
        agent: BaseAgent,
        parent_addr: MemoryAddress,
        description_override: Optional[str] = None,
    ):
        self.agent = agent
        self.parent_addr = parent_addr

        # Define the schema for the LLM to understand how to call this agent
        schema = {
            "name": f"call_{agent.config.name.lower().replace(' ', '_')}",
            "description": (
                description_override
                or (
                    f"Delegate a task to {agent.config.name}. "
                    f"Capabilities: {agent.config.system_prompt}"
                )
            )[:1024],
            "parameters": {
                "type": "object",
                "properties": {
                    "instructions": {
                        "type": "string",
                        "description": "The specific task or question for the agent.",
                    }
                },
                "required": ["instructions"],
            },
        }

        super().__init__(schema, self._run_agent)

    def _run_agent(self, instructions: str) -> Dict[str, Any]:
        """The actual function that runs when the tool is called."""

        # Create unique address for child, linked to parent's session
        child_addr = MemoryAddress(
            user_id=self.parent_addr.user_id,
            conversation_id=self.parent_addr.conversation_id,
            agent_id=self.agent.config.name,
        )

        # Run the agent
        response = self.agent.run(user_input=instructions, addr=child_addr)

        # Consume generator if needed
        if hasattr(response, "__iter__") and not isinstance(response, str):
            response = "".join(list(response))

        return {"response": response}


class Flow(Any):
    """Protocol for any multi-agent flow (Team, Pipeline, etc)."""

    def run(
        self,
        user_input: str,
        session_id: str = "default_session",
        user_id: str = "default_user",
    ) -> Any: ...


class FlowTool(Tool):
    """Wraps a Flow (Team, Pipeline, HierarchicalTeam) as a Tool."""

    def __init__(
        self,
        flow: Any,
        name: str,
        description: str,
        parent_addr: MemoryAddress,
    ):
        self.flow = flow
        self.parent_addr = parent_addr

        schema = {
            "name": f"call_{name.lower().replace(' ', '_')}",
            "description": description[:1024],
            "parameters": {
                "type": "object",
                "properties": {
                    "instructions": {
                        "type": "string",
                        "description": "The specific task or instructions for this team/pipeline.",
                    }
                },
                "required": ["instructions"],
            },
        }

        super().__init__(schema, self._run_flow)

    def _run_flow(self, instructions: str) -> Dict[str, Any]:
        """Runs the wrapped flow."""

        # Maintain context continuity
        response = self.flow.run(
            user_input=instructions,
            session_id=self.parent_addr.conversation_id,
            user_id=self.parent_addr.user_id,
        )

        # Consume generator if needed
        if hasattr(response, "__iter__") and not isinstance(response, str):
            response = "".join(list(response))

        return {"response": response}
