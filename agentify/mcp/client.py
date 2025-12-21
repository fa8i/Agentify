"""Async Context Manager for MCP Client connections."""
import os
from typing import Dict, List, Optional
from contextlib import AbstractAsyncContextManager, AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agentify.mcp.adapter import convert_mcp_tools_to_agentify
from agentify.core.tool import Tool


class MCPConnection(AbstractAsyncContextManager):
    """Manages MCP server connections using StdIO transport.

    Usage:
        async with MCPConnection(command="uvx", args=["mcp-server-fetch"]) as mcp:
            tools = await mcp.get_tools()
    """

    def __init__(
        self,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None
    ) -> None:
        self.params = StdioServerParameters(
            command=command,
            args=args,
            env=env or os.environ.copy()
        )
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None

    async def __aenter__(self) -> "MCPConnection":
        self._exit_stack = AsyncExitStack()
        try:
            read, write = await self._exit_stack.enter_async_context(stdio_client(self.params))
            self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
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
