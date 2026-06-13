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
