import asyncio

import pytest

from agentify.core.tool import Tool
from agentify.mcp.runtime_bridge import RuntimeMCPBridge
from agentify.mcp.runtime_server import _send_request


def test_runtime_mcp_bridge_lists_and_calls_live_tools():
    tool = Tool(
        schema={
            "name": "echo_tool",
            "description": "Echo text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
        func=lambda text: f"echo: {text}",
    )

    async def run_test():
        bridge = RuntimeMCPBridge()
        bridge.update_tools([tool])
        await bridge.start()
        try:
            config = bridge.codex_config()
            server = config["mcp_servers"]["agentify-runtime-tools"]
            port = int(server["args"][server["args"].index("--port") + 1])
            token = server["args"][server["args"].index("--token") + 1]

            listed = await _send_request(
                "127.0.0.1",
                port,
                token,
                {"action": "list_tools"},
            )
            called = await _send_request(
                "127.0.0.1",
                port,
                token,
                {
                    "action": "call_tool",
                    "name": "echo_tool",
                    "arguments": {"text": "hola"},
                },
            )
        finally:
            await bridge.close()
        return listed, called

    listed, called = asyncio.run(run_test())

    assert listed["tools"][0]["name"] == "echo_tool"
    assert listed["tools"][0]["inputSchema"]["required"] == ["text"]
    assert called["content"][0]["text"] == "echo: hola"
    assert called["isError"] is False


def test_runtime_mcp_bridge_rejects_bad_token():
    async def run_test():
        bridge = RuntimeMCPBridge()
        await bridge.start()
        try:
            config = bridge.codex_config()
            server = config["mcp_servers"]["agentify-runtime-tools"]
            port = int(server["args"][server["args"].index("--port") + 1])

            with pytest.raises(RuntimeError, match="Unauthorized"):
                await _send_request(
                    "127.0.0.1",
                    port,
                    "bad-token",
                    {"action": "list_tools"},
                )
        finally:
            await bridge.close()

    asyncio.run(run_test())


def test_runtime_mcp_bridge_returns_tool_timeout_error():
    async def slow_tool():
        await asyncio.sleep(1)
        return "too late"

    tool = Tool(
        schema={"name": "slow_tool", "parameters": {"type": "object"}},
        func=slow_tool,
    )

    async def run_test():
        bridge = RuntimeMCPBridge(tool_timeout=0.01)
        bridge.update_tools([tool])
        await bridge.start()
        try:
            config = bridge.codex_config()
            server = config["mcp_servers"]["agentify-runtime-tools"]
            port = int(server["args"][server["args"].index("--port") + 1])
            token = server["args"][server["args"].index("--token") + 1]

            return await _send_request(
                "127.0.0.1",
                port,
                token,
                {"action": "call_tool", "name": "slow_tool", "arguments": {}},
            )
        finally:
            await bridge.close()

    response = asyncio.run(run_test())

    assert response["isError"] is True
    assert "timed out" in response["content"][0]["text"]


def test_runtime_mcp_bridge_call_timeout_outlasts_tool_timeout():
    bridge = RuntimeMCPBridge(tool_timeout=300)
    assert bridge.call_timeout() == 330.0

    bridge_no_timeout = RuntimeMCPBridge()
    assert bridge_no_timeout.call_timeout() == 600.0

    async def run_test():
        await bridge.start()
        try:
            config = bridge.codex_config()
            server = config["mcp_servers"]["agentify-runtime-tools"]
            args = server["args"]
            return args[args.index("--call-timeout") + 1]
        finally:
            await bridge.close()

    assert asyncio.run(run_test()) == "330.0"


def test_runtime_mcp_bridge_tracks_connections():
    tool = Tool(
        schema={"name": "echo_tool", "parameters": {"type": "object"}},
        func=lambda: "ok",
    )

    async def run_test():
        bridge = RuntimeMCPBridge()
        bridge.update_tools([tool])
        await bridge.start()
        try:
            assert bridge.connection_count == 0
            config = bridge.codex_config()
            server = config["mcp_servers"]["agentify-runtime-tools"]
            port = int(server["args"][server["args"].index("--port") + 1])
            token = server["args"][server["args"].index("--token") + 1]
            await _send_request("127.0.0.1", port, token, {"action": "list_tools"})
            return bridge.connection_count
        finally:
            await bridge.close()

    assert asyncio.run(run_test()) == 1


def test_runtime_mcp_bridge_isolates_sessions_and_executors():
    tool_a = Tool(
        schema={"name": "tool_a", "parameters": {"type": "object"}},
        func=lambda: "a",
    )
    tool_b = Tool(
        schema={"name": "tool_b", "parameters": {"type": "object"}},
        func=lambda: "b",
    )

    async def executor_a(name, arguments):
        await asyncio.sleep(0.05)  # overlap with executor_b
        return "from-session-a"

    async def executor_b(name, arguments):
        return "from-session-b"

    async def run_test():
        bridge = RuntimeMCPBridge()
        bridge.register_session("sess-a", tools=[tool_a], tool_executor=executor_a)
        bridge.register_session("sess-b", tools=[tool_b], tool_executor=executor_b)
        await bridge.start()
        try:
            config_a = bridge.codex_config("sess-a")
            server = config_a["mcp_servers"]["agentify-runtime-tools"]
            args = server["args"]
            assert args[args.index("--session") + 1] == "sess-a"
            assert server["enabled_tools"] == ["tool_a"]
            port = int(args[args.index("--port") + 1])
            token = args[args.index("--token") + 1]

            result_a, result_b = await asyncio.gather(
                _send_request(
                    "127.0.0.1",
                    port,
                    token,
                    {"action": "call_tool", "name": "tool_a", "arguments": {}, "session": "sess-a"},
                ),
                _send_request(
                    "127.0.0.1",
                    port,
                    token,
                    {"action": "call_tool", "name": "tool_b", "arguments": {}, "session": "sess-b"},
                ),
            )
            listed_b = await _send_request(
                "127.0.0.1",
                port,
                token,
                {"action": "list_tools", "session": "sess-b"},
            )
            counts = (
                bridge.session_connection_count("sess-a"),
                bridge.session_connection_count("sess-b"),
            )
        finally:
            await bridge.close()
        return result_a, result_b, listed_b, counts

    result_a, result_b, listed_b, counts = asyncio.run(run_test())

    assert result_a["content"][0]["text"] == "from-session-a"
    assert result_b["content"][0]["text"] == "from-session-b"
    assert [t["name"] for t in listed_b["tools"]] == ["tool_b"]
    assert counts == (1, 2)


def test_runtime_mcp_bridge_rejects_unknown_session():
    async def run_test():
        bridge = RuntimeMCPBridge()
        bridge.register_session("known", tools=[])
        await bridge.start()
        try:
            config = bridge.codex_config("known")
            server = config["mcp_servers"]["agentify-runtime-tools"]
            port = int(server["args"][server["args"].index("--port") + 1])
            token = server["args"][server["args"].index("--token") + 1]

            with pytest.raises(RuntimeError, match="Unknown runtime MCP bridge session"):
                await _send_request(
                    "127.0.0.1",
                    port,
                    token,
                    {"action": "list_tools", "session": "missing"},
                )
        finally:
            await bridge.close()

    asyncio.run(run_test())


def test_runtime_mcp_bridge_per_session_call_timeout():
    bridge = RuntimeMCPBridge()
    bridge.register_session("fast", tools=[], tool_timeout=10)
    bridge.register_session("slow", tools=[], tool_timeout=120)

    assert bridge.call_timeout("fast") == 40.0
    assert bridge.call_timeout("slow") == 150.0


def test_runtime_mcp_bridge_uses_custom_tool_executor():
    calls = []
    tool = Tool(
        schema={
            "name": "echo_tool",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
            },
        },
        func=lambda text: f"direct: {text}",
    )

    async def executor(name, arguments):
        calls.append((name, arguments))
        return f"executor: {arguments['text']}"

    async def run_test():
        bridge = RuntimeMCPBridge(tool_executor=executor)
        bridge.update_tools([tool])
        await bridge.start()
        try:
            config = bridge.codex_config()
            server = config["mcp_servers"]["agentify-runtime-tools"]
            port = int(server["args"][server["args"].index("--port") + 1])
            token = server["args"][server["args"].index("--token") + 1]

            return await _send_request(
                "127.0.0.1",
                port,
                token,
                {
                    "action": "call_tool",
                    "name": "echo_tool",
                    "arguments": {"text": "hola"},
                },
            )
        finally:
            await bridge.close()

    response = asyncio.run(run_test())

    assert calls == [("echo_tool", {"text": "hola"})]
    assert response["content"][0]["text"] == "executor: hola"
