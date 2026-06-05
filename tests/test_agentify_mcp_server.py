import asyncio
import json
import os
import sys

from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsRequest

from agentify.cli import main as agentify_cli_main
from agentify.core.tool import Tool
from agentify.mcp.client import MCPConnection
from agentify.mcp.server import (
    AgentifyMCPServer,
    agentify_tool_to_mcp_tool,
    build_mcp_stdio_server,
    generate_codex_mcp_config,
    load_tool_registry,
    main as mcp_server_main,
)


def test_agentify_tool_converts_to_mcp_definition():
    tool = Tool(
        schema={
            "name": "search_docs",
            "description": "Search project docs.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        func=lambda query: f"found {query}",
    )

    mcp_tool = agentify_tool_to_mcp_tool(tool)

    assert mcp_tool.name == "search_docs"
    assert mcp_tool.description == "Search project docs."
    assert mcp_tool.inputSchema == tool.schema["parameters"]


def test_mcp_server_handler_calls_agentify_tool():
    calls = []

    def search_docs(query: str) -> str:
        calls.append(query)
        return f"result for {query}"

    tool = Tool(
        schema={
            "name": "search_docs",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        func=search_docs,
    )
    server = AgentifyMCPServer([tool])

    result = asyncio.run(server.call_tool("search_docs", {"query": "codex"}))

    assert calls == ["codex"]
    assert result.isError is False
    assert result.content[0].text == "result for codex"


def test_mcp_server_handler_supports_async_agentify_tool():
    async def double(value: int) -> dict:
        return {"value": value * 2}

    tool = Tool(
        schema={
            "name": "double",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        },
        func=double,
    )
    server = AgentifyMCPServer([tool])

    result = asyncio.run(server.call_tool("double", {"value": 21}))

    assert result.isError is False
    assert json.loads(result.content[0].text) == {"value": 42}


def test_mcp_server_returns_controlled_tool_errors():
    def fail() -> str:
        raise RuntimeError("boom")

    tool = Tool(
        schema={"name": "fail", "parameters": {"type": "object"}},
        func=fail,
    )
    server = AgentifyMCPServer([tool])

    result = asyncio.run(server.call_tool("fail", {}))

    assert result.isError is True
    assert json.loads(result.content[0].text) == {"error": "boom"}


def test_mcp_server_validates_inputs():
    tool = Tool(
        schema={
            "name": "double",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        },
        func=lambda value: str(value * 2),
    )
    server = AgentifyMCPServer([tool])

    result = asyncio.run(server.call_tool("double", {"value": "bad"}))

    assert result.isError is True
    assert "schema validation" in json.loads(result.content[0].text)["error"]


def test_mcp_server_allowlist_limits_exposed_tools():
    allowed = Tool({"name": "allowed", "parameters": {"type": "object"}}, lambda: "ok")
    blocked = Tool({"name": "blocked", "parameters": {"type": "object"}}, lambda: "no")

    server = AgentifyMCPServer([allowed, blocked], allowlist=["allowed"])

    assert [tool.name for tool in server.list_tools()] == ["allowed"]
    result = asyncio.run(server.call_tool("blocked", {}))
    assert result.isError is True


def test_registry_loads_importable_tools(tmp_path, monkeypatch):
    registry_file = tmp_path / "registry_tools.py"
    registry_file.write_text(
        "from agentify.core.tool import Tool\n"
        "def build_agentify_tools():\n"
        "    return [Tool({'name': 'echo', 'parameters': {'type': 'object'}}, lambda text: text)]\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    tools = load_tool_registry("registry_tools:build_agentify_tools")

    assert [tool.name for tool in tools] == ["echo"]


def test_codex_e2e_fixture_registry_is_importable():
    tools = load_tool_registry("tests.fixtures.codex_mcp_registry:build_agentify_tools")

    assert [tool.name for tool in tools] == ["echo_tool"]


def test_registry_rejects_missing_import():
    try:
        load_tool_registry("missing_module:build_agentify_tools")
    except ValueError as exc:
        assert "Could not import registry module" in str(exc)
    else:
        raise AssertionError("Expected registry load to fail")


def test_registry_rejects_non_tools(tmp_path, monkeypatch):
    registry_file = tmp_path / "bad_registry.py"
    registry_file.write_text("def build_agentify_tools():\n    return ['not-a-tool']\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    try:
        load_tool_registry("bad_registry:build_agentify_tools")
    except ValueError as exc:
        assert "only Agentify Tool objects" in str(exc)
    else:
        raise AssertionError("Expected registry load to fail")


def test_entrypoint_fails_with_clear_error_for_missing_registry(capsys):
    exit_code = mcp_server_main(["--registry", "missing_module:build_agentify_tools"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "agentify-mcp-server:" in captured.err
    assert "Could not import registry module" in captured.err


def test_entrypoint_fails_if_registry_does_not_return_tools(tmp_path, monkeypatch, capsys):
    registry_file = tmp_path / "non_sequence_registry.py"
    registry_file.write_text("def build_agentify_tools():\n    return {'bad': 'value'}\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    exit_code = mcp_server_main(["--registry", "non_sequence_registry:build_agentify_tools"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "must return a sequence" in captured.err


def test_generate_codex_mcp_config():
    config = generate_codex_mcp_config(
        name="agentify-my-agent",
        registry="my_project.tools:build_agentify_tools",
        allow=["search_docs", "read_file"],
    )

    assert "[mcp_servers.agentify-my-agent]" in config
    assert 'command = "python"' in config
    assert '"-m",' in config
    assert '"agentify.mcp.server",' in config
    assert '"--registry",' in config
    assert '"my_project.tools:build_agentify_tools",' in config
    assert '"--allow",' in config
    assert '"search_docs,read_file",' in config
    assert 'enabled_tools = ["search_docs", "read_file"]' in config


def test_generate_codex_mcp_config_with_absolute_python():
    config = generate_codex_mcp_config(
        name="agentify-e2e",
        registry="tests.fixtures.codex_mcp_registry:build_agentify_tools",
        allow=["echo_tool"],
        command=sys.executable,
    )

    assert f"command = {json.dumps(sys.executable)}" in config
    assert 'enabled_tools = ["echo_tool"]' in config


def test_agentify_cli_generates_codex_mcp_config(capsys):
    exit_code = agentify_cli_main(
        [
            "codex",
            "mcp",
            "config",
            "--name",
            "agentify-my-agent",
            "--registry",
            "my_project.tools:build_agentify_tools",
            "--allow",
            "search_docs,read_file",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "[mcp_servers.agentify-my-agent]" in captured.out
    assert 'enabled_tools = ["search_docs", "read_file"]' in captured.out


def test_lowlevel_mcp_server_exposes_list_tools_and_call_tool():
    tool = Tool(
        schema={"name": "echo", "parameters": {"type": "object"}},
        func=lambda text: text,
    )
    lowlevel_server = build_mcp_stdio_server(AgentifyMCPServer([tool]))

    async def run_test():
        list_handler = lowlevel_server.request_handlers[ListToolsRequest]
        list_result = await list_handler(ListToolsRequest())

        call_handler = lowlevel_server.request_handlers[CallToolRequest]
        call_result = await call_handler(
            CallToolRequest(
                params=CallToolRequestParams(name="echo", arguments={"text": "hello"})
            )
        )
        return list_result.root, call_result.root

    list_result, call_result = asyncio.run(run_test())

    assert [tool.name for tool in list_result.tools] == ["echo"]
    assert call_result.isError is False
    assert call_result.content[0].text == "hello"


def test_stdio_entrypoint_exposes_tools_over_real_mcp_transport(tmp_path):
    registry_file = tmp_path / "stdio_registry.py"
    registry_file.write_text(
        "from agentify.core.tool import Tool\n"
        "def build_agentify_tools():\n"
        "    return [Tool({'name': 'echo_tool', 'parameters': {'type': 'object'}}, "
        "lambda text: f'echo: {text}')]\n"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), os.getcwd(), env.get("PYTHONPATH", "")]
    )

    async def run_test():
        async with MCPConnection.stdio(
            command=sys.executable,
            args=[
                "-m",
                "agentify.mcp.server",
                "--registry",
                "stdio_registry:build_agentify_tools",
                "--allow",
                "echo_tool",
            ],
            env=env,
        ) as connection:
            tools = await connection.get_tools()
            result = await tools[0].func(text="hello")
            return tools, result

    tools, result = asyncio.run(run_test())

    assert [tool.name for tool in tools] == ["echo_tool"]
    assert result == "echo: hello"


def test_debug_log_records_server_activity(tmp_path):
    debug_log = tmp_path / "agentify-mcp.log"
    tool = Tool(
        schema={"name": "echo", "parameters": {"type": "object"}},
        func=lambda text: text,
    )
    server = AgentifyMCPServer([tool], debug_log=debug_log)

    server.list_tools()
    asyncio.run(server.call_tool("echo", {"text": "hello"}))

    content = debug_log.read_text(encoding="utf-8")
    assert "tools loaded: echo" in content
    assert "list_tools called" in content
    assert "call_tool called: echo" in content
    assert "call_tool finished: echo" in content
