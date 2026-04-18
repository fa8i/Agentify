import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from agentify.multi_agent import Team, Pipeline, HierarchicalTeam
from agentify.multi_agent.tool_wrapper import AgentTool
from agentify.core import BaseAgent, AgentConfig
from agentify.memory import MemoryService, MemoryAddress
from agentify.memory.stores.in_memory_store import InMemoryStore

class AsyncMockAgent(BaseAgent):
    def __init__(self, name, response="Mock Response"):
        store = InMemoryStore()
        memory = MemoryService(store=store, log_enabled=False)
        config = AgentConfig(
            name=name,
            system_prompt="Mock Prompt",
            provider="openai",
            model_name="gpt-4o-mini"
        )
        mock_factory = MagicMock()
        mock_factory.create_client.return_value = MagicMock()
        mock_factory.create_async_client.return_value = MagicMock()
        super().__init__(config=config, memory=memory, client_factory=mock_factory)
        self._preset_response = response
        
        # Mock the internal methods to avoid actual API calls
        self._aget_llm_response = AsyncMock(return_value=MagicMock(content=response))
        self.arun = AsyncMock(return_value=response)

@pytest.fixture
def store():
    return InMemoryStore()

@pytest.fixture
def memory_service(store):
    return MemoryService(store=store)

@pytest.mark.asyncio
async def test_team_arun_parallelism():
    """Test that Team.arun delegates to supervisor and can handle logic."""
    # Setup supervisor
    supervisor = AsyncMockAgent("Supervisor", response="Team Result")
    
    # Setup workers
    worker1 = AsyncMockAgent("Worker1")
    worker2 = AsyncMockAgent("Worker2")
    
    team = Team(agents=[supervisor, worker1, worker2], supervisor=supervisor)
    
    result = await team.arun(user_input="Task", session_id="test_sess", user_id="test_user")
    
    assert result == "Team Result"
    assert supervisor.arun.called

@pytest.mark.asyncio
async def test_pipeline_arun_sequential():
    """Test that Pipeline.arun executes steps sequentially."""
    step1 = AsyncMockAgent("Step1", response="Output1")
    step2 = AsyncMockAgent("Step2", response="Output2")
    
    pipeline = Pipeline(steps=[step1, step2])
    
    result = await pipeline.arun(user_input="Input", session_id="test_sess")
    
    assert result == "Output2"
    # Verify strict order
    assert step1.arun.called
    assert step2.arun.called
    
    # Step 2 should have received Step 1's output
    call_args = step2.arun.call_args[1]
    assert call_args["user_input"] == "Output1"

@pytest.mark.asyncio
async def test_hierarchical_arun_delegation():
    """Test that HierarchicalTeam delegates to root."""
    root = AsyncMockAgent("Root", response="Hierarchy Result")
    child = AsyncMockAgent("Child")
    
    hierarchy = {root: [child]}
    h_team = HierarchicalTeam(root=root, hierarchy=hierarchy)
    
    result = await h_team.arun(user_input="Task", session_id="test_sess")
    
    assert result == "Hierarchy Result"
    assert root.arun.called


@pytest.mark.asyncio
async def test_agenttool_async_recovery_on_consistency_error():
    worker = AsyncMockAgent("Worker", response="Recovered")
    consistency_error = Exception(
        "An assistant message with 'tool_calls' must be followed by tool messages "
        "responding to each 'tool_call_id'."
    )
    worker.arun = AsyncMock(side_effect=[consistency_error, "Recovered"])

    parent_addr = MemoryAddress(user_id="u1", conversation_id="s1", agent_id="Supervisor")
    tool = AgentTool(agent=worker, parent_addr=parent_addr)

    result = await tool.async_func(instructions="do work")

    assert result["response"] == "Recovered"
    assert worker.arun.await_count == 2
    second_call_addr = worker.arun.await_args_list[1].kwargs["addr"]
    assert "__rcv__" in second_call_addr.conversation_id


@pytest.mark.asyncio
async def test_agenttool_async_recovery_can_be_disabled():
    worker = AsyncMockAgent("Worker", response="Recovered")
    worker.config.delegation_recovery_enabled = False
    consistency_error = Exception(
        "An assistant message with 'tool_calls' must be followed by tool messages "
        "responding to each 'tool_call_id'."
    )
    worker.arun = AsyncMock(side_effect=consistency_error)

    parent_addr = MemoryAddress(user_id="u1", conversation_id="s1", agent_id="Supervisor")
    tool = AgentTool(agent=worker, parent_addr=parent_addr)

    result = await tool.async_func(instructions="do work")

    assert "error" in result
    assert worker.arun.await_count == 1
