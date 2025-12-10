from typing import Any, Dict, Optional
import asyncio
from agentify.core.agent import BaseAgent
from agentify.core.tool import Tool
from agentify.memory.interfaces import MemoryAddress


class AgentTool(Tool):
    """Wraps a BaseAgent as a Tool so it can be called by another agent.

    When the tool is invoked:
    1. It receives 'instructions' from the caller.
    2. It triggers the wrapped agent's `run` or `arun` method.
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
        # Store async func for detection by BaseAgent
        self.async_func = self._arun_agent

    def _run_agent(self, instructions: str) -> Dict[str, Any]:
        """The actual function that runs when the tool is called synchronously."""

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

    async def _arun_agent(self, instructions: str) -> Dict[str, Any]:
        """Async version: runs the wrapped agent using arun()."""

        child_addr = MemoryAddress(
            user_id=self.parent_addr.user_id,
            conversation_id=self.parent_addr.conversation_id,
            agent_id=self.agent.config.name,
        )

        # Run the agent asynchronously
        response = await self.agent.arun(user_input=instructions, addr=child_addr)

        # Consume async generator if needed
        if hasattr(response, "__aiter__"):
            parts = []
            async for chunk in response:
                parts.append(chunk)
            response = "".join(parts)
        elif hasattr(response, "__iter__") and not isinstance(response, str):
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

    async def arun(
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
        # Store async func for detection by BaseAgent
        self.async_func = self._arun_flow

    def _run_flow(self, instructions: str) -> Dict[str, Any]:
        """Runs the wrapped flow synchronously."""

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

    async def _arun_flow(self, instructions: str) -> Dict[str, Any]:
        """Async version: runs the wrapped flow using arun() if available."""

        if hasattr(self.flow, "arun"):
            response = await self.flow.arun(
                user_input=instructions,
                session_id=self.parent_addr.conversation_id,
                user_id=self.parent_addr.user_id,
            )
        else:
            # Fallback to sync if no arun available
            response = self.flow.run(
                user_input=instructions,
                session_id=self.parent_addr.conversation_id,
                user_id=self.parent_addr.user_id,
            )

        # Consume async generator if needed
        if hasattr(response, "__aiter__"):
            parts = []
            async for chunk in response:
                parts.append(chunk)
            response = "".join(parts)
        elif hasattr(response, "__iter__") and not isinstance(response, str):
            response = "".join(list(response))

        return {"response": response}


class SpawnAgentTool(Tool):
    """Tool to dynamically spawn a transient sub-agent for a specific task."""

    def __init__(
        self,
        base_config: Any,  # AgentConfig type ideally, but Any to avoid circular imports context
        memory_service: Any, # MemoryService
        parent_addr: MemoryAddress,
        client_factory: Optional[Any] = None,
    ):
        self.base_config = base_config
        self.memory_service = memory_service
        self.parent_addr = parent_addr
        self.client_factory = client_factory

        schema = {
            "name": "spawn_subagent",
            "description": "Spawn a temporary specialized sub-agent to handle a complex sub-task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "role_name": {
                        "type": "string",
                        "description": "Name of the sub-agent (e.g., 'ResearchAssistant').",
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Specific task instructions for the sub-agent.",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "System prompt defining the sub-agent's persona and constraints.",
                    }
                },
                "required": ["role_name", "instructions"],
            },
        }
        super().__init__(schema, self._spawn_and_run)
        # Store async func for detection by BaseAgent
        self.async_func = self._aspawn_and_run

    def _spawn_and_run(
        self, 
        role_name: str, 
        instructions: str, 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates and runs a new agent instance synchronously."""
        from agentify.core.agent import BaseAgent
        from agentify.core.config import AgentConfig
        import copy

        # Clone config but override name and system prompt
        new_config = copy.deepcopy(self.base_config)
        new_config.name = f"{self.base_config.name}.{role_name}"
        if system_prompt:
            new_config.system_prompt = system_prompt
        
        # Create a unique address for this interaction
        child_addr = MemoryAddress(
            user_id=self.parent_addr.user_id,
            conversation_id=f"{self.parent_addr.conversation_id}_{role_name}_{instructions[:10]}",
            agent_id=new_config.name,
        )

        # Create the agent
        sub_agent = BaseAgent(
            config=new_config,
            memory=self.memory_service,
            memory_address=child_addr,
            client_factory=self.client_factory
        )

        response = sub_agent.run(user_input=instructions)
        
        # Consume generator if needed
        if hasattr(response, "__iter__") and not isinstance(response, str):
            response = "".join(list(response))

        return {
            "subagent": role_name,
            "status": "finished",
            "response": response
        }

    async def _aspawn_and_run(
        self, 
        role_name: str, 
        instructions: str, 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Async version: creates and runs a new agent instance asynchronously."""
        from agentify.core.agent import BaseAgent
        from agentify.core.config import AgentConfig
        import copy

        new_config = copy.deepcopy(self.base_config)
        new_config.name = f"{self.base_config.name}.{role_name}"
        if system_prompt:
            new_config.system_prompt = system_prompt
        
        child_addr = MemoryAddress(
            user_id=self.parent_addr.user_id,
            conversation_id=f"{self.parent_addr.conversation_id}_{role_name}_{instructions[:10]}",
            agent_id=new_config.name,
        )

        sub_agent = BaseAgent(
            config=new_config,
            memory=self.memory_service,
            memory_address=child_addr,
            client_factory=self.client_factory
        )

        response = await sub_agent.arun(user_input=instructions)
        
        # Consume async generator if needed
        if hasattr(response, "__aiter__"):
            parts = []
            async for chunk in response:
                parts.append(chunk)
            response = "".join(parts)
        elif hasattr(response, "__iter__") and not isinstance(response, str):
            response = "".join(list(response))

        return {
            "subagent": role_name,
            "status": "finished",
            "response": response
        }

