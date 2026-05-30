import shlex
import subprocess
from typing import Any, Dict, Optional

from agentify.core.tool import Tool

DEFAULT_TIMEOUT_SECONDS = 15
MAX_OUTPUT_CHARS = 8000


class ShellSafeTool(Tool):
    """Execute local shell commands with guardrails."""

    def __init__(self, *, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS):
        self.timeout_seconds = max(1, timeout_seconds)
        schema = {
            "name": "run_safe_command",
            "description": (
                "Execute a shell command in the local environment using argv parsing (no shell=True). "
                "Use this when the user asks to run terminal commands. "
                "Return stdout/stderr/return_code so the assistant can explain the real result. "
                "Avoid global pip/npm installs unless the user explicitly requests them; "
                "prefer dedicated integration install tools when available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "Exact command string to execute. The runtime prompts the user before execution. "
                            "Examples: pwd, cp -r src dst, rsync -av src/ dst/."
                        ),
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Optional timeout override (max 1800).",
                    },
                },
                "required": ["command"],
            },
        }
        super().__init__(schema=schema, func=self._run_safe_command)

    @staticmethod
    def _truncate(text: str) -> Dict[str, Any]:
        if len(text) <= MAX_OUTPUT_CHARS:
            return {"text": text, "truncated": False}
        return {"text": text[:MAX_OUTPUT_CHARS], "truncated": True}

    def _run_safe_command(self, command: str, timeout_seconds: Optional[int] = None) -> Dict[str, Any]:
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            return {"ok": False, "error": f"Invalid command syntax: {exc}"}

        if not tokens:
            return {"ok": False, "error": "Empty command."}

        timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        timeout = min(max(1, int(timeout)), 1800)

        try:
            completed = subprocess.run(
                tokens,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"Command timed out after {timeout} seconds."}
        except Exception as exc:
            return {"ok": False, "error": f"Command execution error: {exc}"}

        stdout = self._truncate(completed.stdout or "")
        stderr = self._truncate(completed.stderr or "")

        return {
            "ok": completed.returncode == 0,
            "return_code": completed.returncode,
            "command": command,
            "stdout": stdout["text"],
            "stderr": stderr["text"],
            "stdout_truncated": stdout["truncated"],
            "stderr_truncated": stderr["truncated"],
        }
