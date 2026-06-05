"""Command line utilities for Agentify."""

import argparse
from collections.abc import Sequence

from agentify.mcp.server import generate_codex_mcp_config, parse_allowlist


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentify")
    subparsers = parser.add_subparsers(dest="command")

    codex_parser = subparsers.add_parser("codex")
    codex_subparsers = codex_parser.add_subparsers(dest="codex_command")

    mcp_parser = codex_subparsers.add_parser("mcp")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command")

    config_parser = mcp_subparsers.add_parser("config")
    config_parser.add_argument("--name", required=True, help="Codex MCP server name")
    config_parser.add_argument("--registry", required=True, help="Import path: module.path:function_name")
    config_parser.add_argument("--allow", help="Comma-separated tool allowlist")
    config_parser.add_argument(
        "--command",
        dest="python_command",
        default="python",
        help="Python command Codex should run",
    )

    args = parser.parse_args(argv)

    if args.command == "codex" and args.codex_command == "mcp" and args.mcp_command == "config":
        print(
            generate_codex_mcp_config(
                name=args.name,
                registry=args.registry,
                allow=parse_allowlist(args.allow),
                command=args.python_command,
            ),
            end="",
        )
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
