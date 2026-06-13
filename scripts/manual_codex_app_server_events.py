"""Manual Codex app-server event diagnostic for Agentify MCP tools."""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

from agentify.llm.codex_events import CodexEventBackend
from manual_codex_mcp_e2e import _install_temp_config, _restore_config, _codex_config_path


EXPECTED = "ECHO_FROM_AGENTIFY: hola-agentify"


async def _run(args: argparse.Namespace) -> int:
    backend = CodexEventBackend(timeout=300)
    result = await backend.run_turn_events(
        session_id=f"agentify-codex-events-{uuid.uuid4().hex}",
        model=args.model,
        prompt=(
            'Usa la herramienta MCP echo_tool con input "hola-agentify". '
            "Después responde exactamente con el resultado de la herramienta."
        ),
        events_log=args.events_log,
        timeout=args.timeout,
    )

    print(f"events: {result.event_count}")
    print(f"turn_completed: {result.turn_completed}")
    print(f"mcp_tool_calls: {result.mcp_tool_calls}")
    print(f"mcp_tool_results: {result.mcp_tool_results}")
    print("final_text:")
    print(result.final_text)

    if EXPECTED in result.final_text:
        print("STATE A: event stream exposes usable tool output.")
        return 0
    if result.mcp_tool_calls or result.mcp_tool_results:
        print("STATE A-diagnostic: MCP output is visible in events, but final text is not.")
        return 4
    print("STATE B: MCP may run, but usable output is not visible in event stream.", file=sys.stderr)
    return 3


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture raw Codex turn events for MCP diagnostics.")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--events-log", default="/tmp/agentify-codex-events.jsonl")
    parser.add_argument("--debug-log", default="/tmp/agentify-codex-mcp-events.log")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--keep-config", action="store_true")
    args = parser.parse_args()

    config_path = _codex_config_path()
    backup_path = _install_temp_config(config_path, args.python, Path(args.debug_log))
    print(f"Temporary Codex MCP config installed at {config_path}")
    print(f"Raw event log: {args.events_log}")
    print(f"MCP debug log: {args.debug_log}")

    try:
        return asyncio.run(_run(args))
    finally:
        if args.keep_config:
            print("Leaving temporary Codex config in place because --keep-config was used.")
        else:
            _restore_config(config_path, backup_path)
            print("Codex config restored.")


if __name__ == "__main__":
    raise SystemExit(main())
