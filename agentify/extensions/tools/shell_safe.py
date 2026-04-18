import shlex
import subprocess
from typing import Any, Dict, List, Optional

from agentify.core.tool import Tool

DEFAULT_TIMEOUT_SECONDS = 15
MAX_OUTPUT_CHARS = 8000


class ShellSafeTool(Tool):
    """Execute a restricted set of local shell commands safely."""

    def __init__(self, *, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS):
        self.timeout_seconds = max(1, timeout_seconds)
        schema = {
            "name": "run_safe_command",
            "description": "Run an allowlisted shell command safely.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "Command to run. Allowed: pwd, ls [path], whoami, date, "
                            "uname -a, git status"
                        ),
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Optional timeout override (max 60).",
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

    @staticmethod
    def _is_allowed(tokens: List[str]) -> bool:
        if not tokens:
            return False

        cmd = tokens[0]
        if cmd in {"pwd", "whoami", "date"}:
            return len(tokens) == 1

        if cmd == "uname":
            return tokens == ["uname", "-a"]

        if cmd == "ls":
            # allow: ls OR ls <path>
            return len(tokens) in {1, 2}

        if cmd == "git":
            return tokens == ["git", "status"]

        return False

    def _run_safe_command(self, command: str, timeout_seconds: Optional[int] = None) -> Dict[str, Any]:
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            return {"ok": False, "error": f"Invalid command syntax: {exc}"}

        if not self._is_allowed(tokens):
            return {
                "ok": False,
                "error": "Command not allowed by safety policy.",
                "allowed_examples": [
                    "pwd",
                    "ls",
                    "ls .",
                    "whoami",
                    "date",
                    "uname -a",
                    "git status",
                ],
            }

        timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        timeout = min(max(1, int(timeout)), 60)

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
