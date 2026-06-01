"""Manual E2E validation for provider='codex' using Agentify tools via MCP.

This script temporarily appends an Agentify MCP server block to
``~/.codex/config.toml``, runs Codex through ``CodexThreadBackend``, and restores
the original config before exiting unless ``--keep-config`` is passed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import uuid
from pathlib import Path

from agentify.llm.codex_backend import CodexThreadBackend


EXPECTED = "ECHO_FROM_AGENTIFY: hola-agentify"
REGISTRY = "tests.fixtures.codex_mcp_registry:build_agentify_tools"
MARKER_START = "# >>> agentify codex mcp e2e"
MARKER_END = "# <<< agentify codex mcp e2e"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _codex_config_path() -> Path:
    return Path.home() / ".codex" / "config.toml"


def _strip_existing_block(content: str) -> str:
    while MARKER_START in content and MARKER_END in content:
        start = content.index(MARKER_START)
        end = content.index(MARKER_END, start) + len(MARKER_END)
        content = content[:start] + content[end:]
    return content.rstrip() + "\n" if content.strip() else ""


def _build_config_block(python: str, debug_log: Path) -> str:
    return (
        "[mcp_servers.agentify-e2e]\n"
        f"command = {json.dumps(python)}\n"
        f"cwd = {json.dumps(_repo_root().as_posix())}\n"
        "args = [\n"
        '  "-m",\n'
        '  "agentify.mcp.server",\n'
        '  "--registry",\n'
        f"  {json.dumps(REGISTRY)},\n"
        '  "--allow",\n'
        '  "echo_tool",\n'
        '  "--debug-log",\n'
        f"  {json.dumps(debug_log.as_posix())},\n"
        "]\n"
        'enabled_tools = ["echo_tool"]'
    )


def _install_temp_config(config_path: Path, python: str, debug_log: Path) -> Path | None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if config_path.exists():
        backup_path = config_path.with_suffix(config_path.suffix + f".agentify-e2e-{uuid.uuid4().hex}.bak")
        shutil.copy2(config_path, backup_path)
        content = config_path.read_text(encoding="utf-8")
    else:
        content = ""

    content = _strip_existing_block(content)
    block = _build_config_block(python, debug_log)
    config_path.write_text(
        content + f"\n{MARKER_START}\n{block}\n{MARKER_END}\n",
        encoding="utf-8",
    )
    return backup_path


def _restore_config(config_path: Path, backup_path: Path | None) -> None:
    if backup_path is not None and backup_path.exists():
        shutil.move(str(backup_path), str(config_path))
        return
    if config_path.exists():
        content = _strip_existing_block(config_path.read_text(encoding="utf-8"))
        if content.strip():
            config_path.write_text(content, encoding="utf-8")
        else:
            config_path.unlink()


async def _run_codex(model: str) -> str:
    backend = CodexThreadBackend(config={}, timeout=300)
    response = await backend.run_native(
        session_id=f"agentify-codex-mcp-e2e-{uuid.uuid4().hex}",
        model=model,
        prompt=(
            'Usa la herramienta MCP echo_tool con el texto "hola-agentify". '
            "Cuando recibas el resultado de la herramienta, responde en tu mensaje final "
            "exactamente con ese resultado y sin texto adicional."
        ),
    )
    return response.choices[0].message.content


def main() -> int:
    parser = argparse.ArgumentParser(description="Run manual Codex + Agentify MCP E2E validation.")
    parser.add_argument("--model", default="gpt-5.5", help="Codex model to request")
    parser.add_argument("--python", default=sys.executable, help="Absolute Python for Codex MCP config")
    parser.add_argument(
        "--debug-log",
        default="/tmp/agentify-codex-mcp-e2e.log",
        help="MCP server debug log path",
    )
    parser.add_argument("--keep-config", action="store_true", help="Do not restore ~/.codex/config.toml")
    args = parser.parse_args()

    config_path = _codex_config_path()
    debug_log = Path(args.debug_log)
    backup_path = _install_temp_config(config_path, args.python, debug_log)
    debug_start = debug_log.stat().st_size if debug_log.exists() else 0
    print(f"Temporary Codex MCP config installed at {config_path}")
    print(f"MCP debug log: {debug_log}")

    try:
        content = asyncio.run(_run_codex(args.model))
        print("Codex response:")
        print(content)
        new_debug = ""
        if debug_log.exists():
            new_debug = debug_log.read_text(encoding="utf-8")[debug_start:]
        if EXPECTED not in content:
            if "call_tool called: echo_tool" in new_debug and "call_tool finished: echo_tool" in new_debug:
                print(
                    "MCP tool was invoked, but openai-codex SDK returned no final response "
                    "containing the tool result.",
                    file=sys.stderr,
                )
                return 3
            print(f"Expected marker not found: {EXPECTED}", file=sys.stderr)
            return 1
        print("E2E OK: Codex invoked echo_tool via Agentify MCP.")
        return 0
    finally:
        if args.keep_config:
            print("Leaving temporary Codex config in place because --keep-config was used.")
        else:
            _restore_config(config_path, backup_path)
            print("Codex config restored.")


if __name__ == "__main__":
    raise SystemExit(main())
