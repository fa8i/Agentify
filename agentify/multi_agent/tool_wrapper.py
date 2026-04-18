from typing import Any, Dict, Optional, Protocol, Union, AsyncGenerator, Generator
import hashlib
import asyncio
import logging
import uuid
import time
from agentify.core.agent import BaseAgent
from agentify.core.tool import Tool
from agentify.memory.interfaces import MemoryAddress


logger = logging.getLogger(__name__)


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

    @staticmethod
    def _is_tool_call_consistency_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "tool_calls" in msg
            and "tool messages" in msg
            and "tool_call_id" in msg
        )

    def _build_child_addr(self) -> MemoryAddress:
        return MemoryAddress(
            user_id=self.parent_addr.user_id,
            conversation_id=self.parent_addr.conversation_id,
            agent_id=self.agent.config.name,
        )

    def _build_recovery_addr(self, instructions: str) -> MemoryAddress:
        digest = hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:10]
        recovery_suffix = uuid.uuid4().hex[:8]
        return MemoryAddress(
            user_id=self.parent_addr.user_id,
            conversation_id=(
                f"{self.parent_addr.conversation_id}"
                f"__rcv__{self.agent.config.name}__{digest}__{recovery_suffix}"
            ),
            agent_id=self.agent.config.name,
        )

    def _should_use_isolated_recovery(self, exc: Exception) -> bool:
        cfg = self.agent.config
        return (
            cfg.delegation_recovery_enabled
            and cfg.delegation_recovery_mode == "retry_isolated"
            and self._is_tool_call_consistency_error(exc)
        )

    def _run_with_recovery(self, instructions: str) -> Union[str, Dict[str, Any]]:
        base_addr = self._build_child_addr()

        try:
            return self.agent.run(user_input=instructions, addr=base_addr)
        except Exception as exc:
            if not self._should_use_isolated_recovery(exc):
                logger.error(
                    "Delegated sync execution failed for '%s' without recovery: %s",
                    self.agent.config.name,
                    exc,
                    exc_info=True,
                )
                return {
                    "error": "Delegated task failed. Please try again."
                }

            logger.debug(
                "Delegated sync execution consistency error for '%s'. "
                "Applying isolated recovery retry.",
                self.agent.config.name,
            )

            retries = max(1, self.agent.config.delegation_max_retries)
            for attempt in range(retries):
                try:
                    delay = (self.agent.config.delegation_retry_backoff_ms / 1000.0) * attempt
                    if delay > 0:
                        time.sleep(delay)
                    recovery_addr = self._build_recovery_addr(instructions)
                    return self.agent.run(user_input=instructions, addr=recovery_addr)
                except Exception:
                    if attempt == retries - 1:
                        logger.error(
                            "Delegated sync recovery failed for '%s' after %s retries.",
                            self.agent.config.name,
                            retries,
                            exc_info=True,
                        )
                        break

            return {
                "error": "Delegated task failed after automatic recovery."
            }

    async def _arun_with_recovery(self, instructions: str) -> Union[str, Dict[str, Any], AsyncGenerator[str, None]]:
        base_addr = self._build_child_addr()

        try:
            return await self.agent.arun(user_input=instructions, addr=base_addr)
        except Exception as exc:
            if not self._should_use_isolated_recovery(exc):
                logger.error(
                    "Delegated async execution failed for '%s' without recovery: %s",
                    self.agent.config.name,
                    exc,
                    exc_info=True,
                )
                return {
                    "error": "Delegated task failed. Please try again."
                }

            logger.debug(
                "Delegated async execution consistency error for '%s'. "
                "Applying isolated recovery retry.",
                self.agent.config.name,
            )

            retries = max(1, self.agent.config.delegation_max_retries)
            for attempt in range(retries):
                try:
                    delay = (self.agent.config.delegation_retry_backoff_ms / 1000.0) * attempt
                    if delay > 0:
                        await asyncio.sleep(delay)
                    recovery_addr = self._build_recovery_addr(instructions)
                    return await self.agent.arun(user_input=instructions, addr=recovery_addr)
                except Exception:
                    if attempt == retries - 1:
                        logger.error(
                            "Delegated async recovery failed for '%s' after %s retries.",
                            self.agent.config.name,
                            retries,
                            exc_info=True,
                        )
                        break

            return {
                "error": "Delegated task failed after automatic recovery."
            }

    def _run_agent(self, instructions: str) -> Dict[str, Any]:
        """Runs the wrapped agent synchronously."""
        response = self._run_with_recovery(instructions)

        if isinstance(response, dict):
            return response

        if hasattr(response, "__iter__") and not isinstance(response, str):
            response = "".join(list(response))

        return {"response": response}

    async def _arun_agent(self, instructions: str) -> Dict[str, Any]:
        """Runs the wrapped agent using arun()."""
        response = await self._arun_with_recovery(instructions)

        if isinstance(response, dict):
            return response

        # Consume async generator if needed
        if hasattr(response, "__aiter__"):
            parts = []
            async for chunk in response:
                parts.append(chunk)
            response = "".join(parts)
        elif hasattr(response, "__iter__") and not isinstance(response, str):
            response = "".join(list(response))

        return {"response": response}


class Flow(Protocol):
    """Protocol for any multi-agent flow (Team, Pipeline, etc)."""

    def run(
        self,
        user_input: str,
        session_id: str = "default_session",
        user_id: str = "default_user",
    ) -> Union[str, Generator[str, None, None]]: ...

    async def arun(
        self,
        user_input: str,
        session_id: str = "default_session",
        user_id: str = "default_user",
    ) -> Union[str, AsyncGenerator[str, None]]: ...


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
        response = self.flow.run(
            user_input=instructions,
            session_id=self.parent_addr.conversation_id,
            user_id=self.parent_addr.user_id,
        )

        if hasattr(response, "__iter__") and not isinstance(response, str):
            response = "".join(list(response))

        return {"response": response}

    async def _arun_flow(self, instructions: str) -> Dict[str, Any]:
        """Runs the wrapped flow using arun()."""

        response = await self.flow.arun(
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
        import copy

        new_config = copy.deepcopy(self.base_config)
        new_config.name = f"{self.base_config.name}.{role_name}"
        if system_prompt:
            new_config.system_prompt = system_prompt

        instr_hash = hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:16]
        child_addr = MemoryAddress(
            user_id=self.parent_addr.user_id,
            conversation_id=f"{self.parent_addr.conversation_id}_{role_name}_{instr_hash}",
            agent_id=new_config.name,
        )

        sub_agent = BaseAgent(
            config=new_config,
            memory=self.memory_service,
            memory_address=child_addr,
            client_factory=self.client_factory,
        )

        response = sub_agent.run(user_input=instructions)

        if hasattr(response, "__iter__") and not isinstance(response, str):
            response = "".join(list(response))

        return {
            "subagent": role_name,
            "status": "finished",
            "response": response,
        }

    async def _aspawn_and_run(
        self, 
        role_name: str, 
        instructions: str, 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Async version: creates and runs a new agent instance asynchronously."""
        from agentify.core.agent import BaseAgent
        import copy

        new_config = copy.deepcopy(self.base_config)
        new_config.name = f"{self.base_config.name}.{role_name}"
        if system_prompt:
            new_config.system_prompt = system_prompt
        
        instr_hash = hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:16]
        child_addr = MemoryAddress(
            user_id=self.parent_addr.user_id,
            conversation_id=f"{self.parent_addr.conversation_id}_{role_name}_{instr_hash}",
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
