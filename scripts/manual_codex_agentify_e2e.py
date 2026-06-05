"""Manual E2E validation for BaseAgent(provider='codex') + Agentify MCP tools."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agentify.core.agent import BaseAgent
from agentify.core.config import AgentConfig
from agentify.memory.interfaces import MemoryAddress
from agentify.memory.service import MemoryService
from agentify.memory.stores.in_memory_store import InMemoryStore
from manual_codex_mcp_e2e import (
    EXPECTED,
    _codex_config_path,
    _install_temp_config,
    _restore_config,
)


async def _run_agent(model: str) -> str:
    memory = MemoryService(store=InMemoryStore(), log_enabled=False)
    addr = MemoryAddress(
        user_id="agentify-e2e",
        conversation_id="codex-agentify-mcp-e2e",
        agent_id="codex-agent",
    )
    agent = BaseAgent(
        config=AgentConfig(
            name="CodexAgentifyE2E",
            system_prompt=(
                "You are testing Agentify MCP tools through Codex. "
                "When asked to use echo_tool, call the MCP tool and return its exact result."
            ),
            provider="codex",
            model_name=model,
            verbose=False,
        ),
        memory=memory,
        memory_address=addr,
    )

    result = await agent.arun(
        'Usa la herramienta MCP echo_tool con el texto "hola-agentify". '
        "Devuelve exactamente el resultado de la herramienta y sin texto adicional.",
        addr=addr,
    )
    if not isinstance(result, str):
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        return "".join(chunks)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BaseAgent(provider='codex') MCP E2E.")
    parser.add_argument("--model", default="gpt-5.3-codex")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--debug-log", default="/tmp/agentify-codex-agentify-e2e.log")
    parser.add_argument("--keep-config", action="store_true")
    args = parser.parse_args()

    config_path = _codex_config_path()
    backup_path = _install_temp_config(config_path, args.python, Path(args.debug_log))
    debug_log = Path(args.debug_log)
    debug_start = debug_log.stat().st_size if debug_log.exists() else 0
    print(f"Temporary Codex MCP config installed at {config_path}")
    print(f"MCP debug log: {debug_log}")

    try:
        content = asyncio.run(_run_agent(args.model))
        print("BaseAgent response:")
        print(content)
        new_debug = debug_log.read_text(encoding="utf-8")[debug_start:] if debug_log.exists() else ""
        print("MCP debug activity:")
        print(new_debug.strip() or "<none>")

        if EXPECTED not in content:
            print(f"Expected marker not found: {EXPECTED}", file=sys.stderr)
            return 1
        if "call_tool called: echo_tool" not in new_debug:
            print("Expected MCP call_tool debug line not found.", file=sys.stderr)
            return 2
        print("E2E OK: BaseAgent(provider='codex') invoked echo_tool via Agentify MCP.")
        return 0
    finally:
        if args.keep_config:
            print("Leaving temporary Codex config in place because --keep-config was used.")
        else:
            _restore_config(config_path, backup_path)
            print("Codex config restored.")


if __name__ == "__main__":
    raise SystemExit(main())
