import asyncio
from unittest.mock import MagicMock

from agentify.core.agent import BaseAgent
from agentify.core.config import AgentConfig
from agentify.core.tool import Tool
from agentify.memory.interfaces import MemoryAddress
from agentify.memory.service import MemoryService
from agentify.memory.stores.in_memory_store import InMemoryStore


def _build_agent(*, tool_pre_hooks=None, tool_post_hooks=None, tool=None):
    mock_factory = MagicMock()
    mock_factory.create_client.return_value = MagicMock()

    return BaseAgent(
        config=AgentConfig(
            name="HookAgent",
            system_prompt="Test agent.",
            provider="openai",
            model_name="gpt-4o-mini",
        ),
        memory=MemoryService(store=InMemoryStore(), log_enabled=False),
        memory_address=MemoryAddress(user_id="u1", conversation_id="c1", agent_id="HookAgent"),
        client_factory=mock_factory,
        tools=[tool] if tool else [],
        tool_pre_hooks=tool_pre_hooks,
        tool_post_hooks=tool_post_hooks,
    )


def test_tool_hooks_run_before_and_after_tool_call():
    events = []

    def pre_hook(tool_name, arguments):
        events.append(("pre", tool_name, arguments["value"]))

    def post_hook(tool_name, result):
        events.append(("post", tool_name, result))

    tool = Tool(
        schema={
            "name": "double",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        },
        func=lambda value: value * 2,
    )
    agent = _build_agent(
        tool=tool,
        tool_pre_hooks=[pre_hook],
        tool_post_hooks=[post_hook],
    )

    result = asyncio.run(agent._aexecute_tool("double", {"value": 21}))

    assert result == "42"
    assert events == [("pre", "double", 21), ("post", "double", "42")]


def test_tool_post_hook_receives_tool_error():
    events = []

    def post_hook(tool_name, error):
        events.append((tool_name, str(error)))

    async def failing_tool():
        raise RuntimeError("boom")

    tool = Tool(schema={"name": "fail", "parameters": {"type": "object"}}, func=failing_tool)
    agent = _build_agent(tool=tool, tool_post_hooks=[post_hook])

    result = asyncio.run(agent._aexecute_tool("fail", {}))

    assert "Unexpected error executing tool 'fail': boom" in result
    assert events == [("fail", "boom")]
