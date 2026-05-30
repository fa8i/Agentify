import asyncio
from types import SimpleNamespace

from agentify.core.tool import Tool
from agentify.memory.interfaces import MemoryAddress
from agentify.multi_agent.tool_wrapper import SpawnAgentTool


def test_spawn_agent_tool_passes_inherited_tools(monkeypatch):
    captured = {}

    class DummyBaseAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, user_input):
            return f"ran: {user_input}"

    monkeypatch.setattr("agentify.core.agent.BaseAgent", DummyBaseAgent)

    inherited_tools = [Tool({"name": "example", "parameters": {}}, lambda: "ok")]
    pre_hooks = [lambda: None]
    post_hooks = [lambda: None]
    tool_pre_hooks = [lambda: None]
    tool_post_hooks = [lambda: None]
    spawn_tool = SpawnAgentTool(
        base_config=SimpleNamespace(name="Parent"),
        memory_service=object(),
        parent_addr=MemoryAddress(user_id="u1", conversation_id="c1", agent_id="Parent"),
        tools=inherited_tools,
        pre_hooks=pre_hooks,
        post_hooks=post_hooks,
        tool_pre_hooks=tool_pre_hooks,
        tool_post_hooks=tool_post_hooks,
    )

    result = spawn_tool._spawn_and_run(role_name="Child", instructions="do work")

    assert result["response"] == "ran: do work"
    assert captured["tools"] == inherited_tools
    assert captured["tools"] is not inherited_tools
    assert captured["pre_hooks"] == pre_hooks
    assert captured["post_hooks"] == post_hooks
    assert captured["tool_pre_hooks"] == tool_pre_hooks
    assert captured["tool_post_hooks"] == tool_post_hooks


def test_spawn_agent_tool_passes_inherited_tools_async(monkeypatch):
    captured = {}

    class DummyBaseAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def arun(self, user_input):
            return f"ran: {user_input}"

    monkeypatch.setattr("agentify.core.agent.BaseAgent", DummyBaseAgent)

    inherited_tools = [Tool({"name": "example", "parameters": {}}, lambda: "ok")]
    pre_hooks = [lambda: None]
    post_hooks = [lambda: None]
    tool_pre_hooks = [lambda: None]
    tool_post_hooks = [lambda: None]
    spawn_tool = SpawnAgentTool(
        base_config=SimpleNamespace(name="Parent"),
        memory_service=object(),
        parent_addr=MemoryAddress(user_id="u1", conversation_id="c1", agent_id="Parent"),
        tools=inherited_tools,
        pre_hooks=pre_hooks,
        post_hooks=post_hooks,
        tool_pre_hooks=tool_pre_hooks,
        tool_post_hooks=tool_post_hooks,
    )

    result = asyncio.run(
        spawn_tool._aspawn_and_run(role_name="Child", instructions="do work")
    )

    assert result["response"] == "ran: do work"
    assert captured["tools"] == inherited_tools
    assert captured["tools"] is not inherited_tools
    assert captured["pre_hooks"] == pre_hooks
    assert captured["post_hooks"] == post_hooks
    assert captured["tool_pre_hooks"] == tool_pre_hooks
    assert captured["tool_post_hooks"] == tool_post_hooks
