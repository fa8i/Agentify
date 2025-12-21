"""Unit tests for MCP integration."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentify.mcp.adapter import convert_mcp_tools_to_agentify
from agentify.mcp.client import MCPConnection, _Transport


class MockMCPTool:
    """Mock MCP tool definition."""
    def __init__(self, name: str, description: str, schema: dict):
        self.name = name
        self.description = description
        self.inputSchema = schema


class MockCallToolResult:
    """Mock MCP tool execution result."""
    def __init__(self, text_output: str):
        item = MagicMock()
        item.type = "text"
        item.text = text_output
        self.content = [item]


# --- Adapter Tests ---

@pytest.mark.asyncio
async def test_schema_conversion():
    """MCP tool schemas are correctly mapped to Agentify format."""
    mock_session = AsyncMock()
    mock_tools = [
        MockMCPTool(
            name="add",
            description="Adds two numbers",
            schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"]
            }
        )
    ]

    agentify_tools = await convert_mcp_tools_to_agentify(mock_session, mock_tools)

    assert len(agentify_tools) == 1
    tool = agentify_tools[0]
    assert tool.name == "add"
    assert tool.schema["description"] == "Adds two numbers"
    assert tool.schema["parameters"] == mock_tools[0].inputSchema
    assert callable(tool.func)


@pytest.mark.asyncio
async def test_tool_wrapper_calls_session():
    """Wrapper function correctly invokes the MCP session."""
    mock_session = AsyncMock()
    mock_session.call_tool.return_value = MockCallToolResult("result_value")
    mock_tools = [MockMCPTool("my_tool", "desc", {})]

    agentify_tools = await convert_mcp_tools_to_agentify(mock_session, mock_tools)
    result = await agentify_tools[0].func(param="test")

    mock_session.call_tool.assert_called_once_with("my_tool", arguments={"param": "test"})
    assert result == "result_value"


@pytest.mark.asyncio
async def test_multiple_tools_conversion():
    """Multiple MCP tools are all converted correctly."""
    mock_session = AsyncMock()
    mock_tools = [
        MockMCPTool("tool_a", "Description A", {}),
        MockMCPTool("tool_b", "Description B", {}),
        MockMCPTool("tool_c", "Description C", {}),
    ]

    agentify_tools = await convert_mcp_tools_to_agentify(mock_session, mock_tools)

    assert len(agentify_tools) == 3
    names = {t.name for t in agentify_tools}
    assert names == {"tool_a", "tool_b", "tool_c"}


@pytest.mark.asyncio
async def test_empty_description_handling():
    """Tools with None description default to empty string."""
    mock_session = AsyncMock()
    mock_tools = [MockMCPTool("no_desc_tool", None, {})]

    agentify_tools = await convert_mcp_tools_to_agentify(mock_session, mock_tools)

    assert agentify_tools[0].schema["description"] == ""


# --- Factory Pattern Tests ---

def test_stdio_factory_creates_correct_transport():
    """MCPConnection.stdio() sets transport to STDIO."""
    conn = MCPConnection.stdio(command="python", args=["server.py"])
    assert conn._transport == _Transport.STDIO
    assert conn._stdio_params is not None
    assert conn._stdio_params.command == "python"


def test_sse_factory_creates_correct_transport():
    """MCPConnection.sse() sets transport to SSE."""
    conn = MCPConnection.sse(url="http://localhost:8080/sse", headers={"Auth": "Bearer token"})
    assert conn._transport == _Transport.SSE
    assert conn._sse_url == "http://localhost:8080/sse"
    assert conn._sse_headers == {"Auth": "Bearer token"}


def test_sse_factory_default_timeouts():
    """MCPConnection.sse() uses default timeouts."""
    conn = MCPConnection.sse(url="http://example.com")
    assert conn._sse_timeout == 5.0
    assert conn._sse_read_timeout == 300.0


def test_sse_factory_custom_timeouts():
    """MCPConnection.sse() accepts custom timeouts."""
    conn = MCPConnection.sse(url="http://example.com", timeout=10.0, sse_read_timeout=60.0)
    assert conn._sse_timeout == 10.0
    assert conn._sse_read_timeout == 60.0
