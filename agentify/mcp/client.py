"""Async Context Manager for MCP Client connections."""
import os
from enum import Enum, auto
from typing import Any, Dict, List, Optional
from contextlib import AbstractAsyncContextManager, AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

from agentify.mcp.adapter import convert_mcp_tools_to_agentify
from agentify.core.tool import Tool


class _Transport(Enum):
    """Internal enum for transport type."""
    STDIO = auto()
    SSE = auto()


class MCPConnection(AbstractAsyncContextManager):
    """Manages MCP server connections via StdIO or SSE transport.

    Use the factory methods to create connections:
        - `MCPConnection.stdio(...)` for local process servers
        - `MCPConnection.sse(...)` for remote HTTP/SSE servers

    Example (StdIO):
        async with MCPConnection.stdio(command="uvx", args=["mcp-server-fetch"]) as mcp:
            tools = await mcp.get_tools()

    Example (SSE):
        async with MCPConnection.sse(url="http://localhost:8080/sse") as mcp:
            tools = await mcp.get_tools()
    """

    def __init__(self) -> None:
        """Private constructor. Use factory methods instead."""
        self._transport: Optional[_Transport] = None
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        # StdIO params
        self._stdio_params: Optional[StdioServerParameters] = None
        # SSE params
        self._sse_url: Optional[str] = None
        self._sse_headers: Optional[Dict[str, Any]] = None
        self._sse_timeout: float = 5.0
        self._sse_read_timeout: float = 300.0

    @classmethod
    def stdio(
        cls,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None
    ) -> "MCPConnection":
        """Create a connection to a local MCP server via StdIO.

        Args:
            command: The command to run (e.g., "python", "uvx").
            args: Arguments to pass to the command.
            env: Optional environment variables for the process.
        """
        instance = cls()
        instance._transport = _Transport.STDIO
        instance._stdio_params = StdioServerParameters(
            command=command,
            args=args,
            env=env or os.environ.copy()
        )
        return instance

    @classmethod
    def sse(
        cls,
        url: str,
        headers: Optional[Dict[str, Any]] = None,
        timeout: float = 5.0,
        sse_read_timeout: float = 300.0,
    ) -> "MCPConnection":
        """Create a connection to a remote MCP server via SSE/HTTP.

        Args:
            url: The SSE endpoint URL.
            headers: Optional HTTP headers (e.g., for authentication).
            timeout: HTTP timeout for regular operations.
            sse_read_timeout: Timeout for SSE read operations.
        """
        instance = cls()
        instance._transport = _Transport.SSE
        instance._sse_url = url
        instance._sse_headers = headers
        instance._sse_timeout = timeout
        instance._sse_read_timeout = sse_read_timeout
        return instance

    async def __aenter__(self) -> "MCPConnection":
        self._exit_stack = AsyncExitStack()
        try:
            if self._transport == _Transport.STDIO:
                read, write = await self._exit_stack.enter_async_context(
                    stdio_client(self._stdio_params)
                )
            elif self._transport == _Transport.SSE:
                read, write = await self._exit_stack.enter_async_context(
                    sse_client(
                        url=self._sse_url,
                        headers=self._sse_headers,
                        timeout=self._sse_timeout,
                        sse_read_timeout=self._sse_read_timeout,
                    )
                )
            else:
                raise ValueError("Transport not configured. Use MCPConnection.stdio() or MCPConnection.sse().")

            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await self._session.initialize()
            return self
        except Exception:
            await self.aclose()
            raise

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Gracefully closes the MCP connection."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None

    async def get_tools(self) -> List[Tool]:
        """Fetches tools from the MCP server and converts them to Agentify format."""
        if not self._session:
            raise RuntimeError("MCPConnection is not active. Use 'async with ...'")

        result = await self._session.list_tools()
        return await convert_mcp_tools_to_agentify(self._session, result.tools)
