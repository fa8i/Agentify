"""Optional integration tests against the real Codex CLI + ChatGPT OAuth.

Skipped by default so CI never touches the network or a local login.
To run them:

    codex login                       # once, interactive ChatGPT OAuth
    AGENTIFY_CODEX_E2E=1 .venv/bin/python -m pytest tests/test_codex_real_e2e.py -v

Optional env vars:
    AGENTIFY_CODEX_E2E_MODEL   model name (default: gpt-5.4). The account must
    support it; list models with `codex` or the SDK `Codex.models()`.
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("AGENTIFY_CODEX_E2E"),
    reason="Real Codex E2E disabled. Set AGENTIFY_CODEX_E2E=1 (requires `codex login`).",
)

MODEL = os.environ.get("AGENTIFY_CODEX_E2E_MODEL", "gpt-5.4")


def _build_agent(memory_mode: str, tools=None):
    from agentify.core.agent import BaseAgent
    from agentify.core.config import AgentConfig
    from agentify.memory.interfaces import MemoryAddress
    from agentify.memory.service import MemoryService
    from agentify.memory.stores.in_memory_store import InMemoryStore

    addr = MemoryAddress(conversation_id=f"codex-e2e-{memory_mode}")
    agent = BaseAgent(
        config=AgentConfig(
            name="CodexE2E",
            system_prompt="Answer concisely in one short sentence.",
            provider="codex",
            model_name=MODEL,
            verbose=False,
            client_config_override={"memory_mode": memory_mode},
        ),
        memory=MemoryService(store=InMemoryStore(), log_enabled=False),
        memory_address=addr,
        tools=tools or [],
    )
    return agent, addr


@pytest.mark.parametrize("memory_mode", ["agentify", "codex_thread"])
def test_multi_turn_memory(memory_mode):
    async def run():
        agent, addr = _build_agent(memory_mode)
        try:
            await agent.arun("Remember this codeword: PAPAYA42.", addr=addr)
            answer = await agent.arun("What is the codeword I gave you?", addr=addr)
            assert "PAPAYA42" in answer
        finally:
            await agent.aclose()

    asyncio.run(run())


def test_agentify_tool_via_mcp():
    from agentify.core.tool import Tool

    tool = Tool(
        schema={
            "name": "echo_tool",
            "description": "Echo back the given text with a marker prefix.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
        func=lambda text: f"ECHO_FROM_AGENTIFY: {text}",
    )

    async def run():
        agent, addr = _build_agent("codex_thread", tools=[tool])
        try:
            answer = await agent.arun(
                'Call the echo_tool with text "hola-agentify" and return its exact '
                "output with no extra text.",
                addr=addr,
            )
            assert "ECHO_FROM_AGENTIFY: hola-agentify" in answer
        finally:
            await agent.aclose()

    asyncio.run(run())
