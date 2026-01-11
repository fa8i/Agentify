from typing import List, Optional, Union, Generator, AsyncGenerator, Any, Dict
import copy
from agentify.core.runnable import Runnable
from agentify.core.agent import BaseAgent
from agentify.memory.interfaces import MemoryAddress
from agentify.multi_agent.tool_wrapper import AgentTool


class Team(Runnable):
    """Orchestrates a group of agents.

    The Team class:
    1. Manages a list of agents.
    2. Designates a 'supervisor' (entry point).
    3. Automatically registers other agents as tools for the supervisor.
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        supervisor: Optional[BaseAgent] = None,
    ):
        if not agents:
            raise ValueError("A Team must have at least one agent.")

        self.agents = agents
        # If no supervisor specified, the first agent is the entry point
        self.supervisor = supervisor or agents[0]

        # Identify 'worker' agents (everyone except the supervisor)
        self.workers = [a for a in agents if a != self.supervisor]

    def run(
        self,
        user_input: str,
        session_id: str = "default_session",
        user_id: str = "default_user",
        **kwargs: Any,
    ) -> Union[str, Generator[str, None, None]]:
        """Run the team workflow.

        1. Sets up the memory address for the supervisor.
        2. Registers workers as tools for the supervisor (scoped to this session).
        3. Runs the supervisor.
        """

        # 1. Setup Supervisor Address
        supervisor_addr = MemoryAddress(
            user_id=user_id,
            conversation_id=session_id,
            agent_id=self.supervisor.config.name,
        )

        # Clone supervisor to avoid tool pollution across runs
        # Use shallow copy + manual tools dict copy to avoid deepcopying net clients
        supervisor = copy.copy(self.supervisor)
        supervisor._tools = self.supervisor._tools.copy()

        # 2. Register Workers as Tools (Dynamic Registration)
        # Wrap each worker in an AgentTool, bound to the supervisor's address context
        worker_tools = []
        for worker in self.workers:
            tool_wrapper = AgentTool(agent=worker, parent_addr=supervisor_addr)
            worker_tools.append(tool_wrapper)

            # Register with CLONED supervisor
            supervisor.register_tool(tool_wrapper)

        # 3. Run Supervisor
        return supervisor.run(user_input=user_input, addr=supervisor_addr)

    async def arun(
        self,
        user_input: str,
        session_id: str = "default_session",
        user_id: str = "default_user",
        **kwargs: Any,
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Async version of run()."""
        # 1. Setup Supervisor Address
        supervisor_addr = MemoryAddress(
            user_id=user_id,
            conversation_id=session_id,
            agent_id=self.supervisor.config.name,
        )

        supervisor = copy.copy(self.supervisor)
        supervisor._tools = self.supervisor._tools.copy()

        # 2. Register Workers as Tools
        for worker in self.workers:
            tool_wrapper = AgentTool(agent=worker, parent_addr=supervisor_addr)
            supervisor.register_tool(tool_wrapper)

        # 3. Run Supervisor asynchronously
        return await supervisor.arun(user_input=user_input, addr=supervisor_addr)

