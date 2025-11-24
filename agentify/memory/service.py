from __future__ import annotations
import logging
import os
from typing import Any, Dict, List, Optional
from .interfaces import ConversationStore, MemoryAddress, Message
from .policies import MemoryPolicy

# Flag to enable/disable memory logging (can be controlled via env var)
ENABLE_MEMORY_LOGS = os.getenv("AGENTIFY_MEMORY_LOGS", "true").lower()


# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BLUE = "\033[94m"  # system
    GREEN = "\033[92m"  # user
    YELLOW = "\033[93m"  # assistant
    CYAN = "\033[96m"  # tool
    MAGENTA = "\033[95m"  # tool calls
    GRAY = "\033[90m"  # metadata


# Configure logger with handler for terminal output
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

_ALLOWED_FIELDS = {"role", "content", "name", "tool_call_id", "metadata", "id", "ts"}


class MemoryService:
    """Facade consumed by the agent. It does NOT create addresses.
    The API requires a MemoryAddress provided by the application/API layer.
    """

    def __init__(
        self, store: ConversationStore, policy: Optional[MemoryPolicy] = None
    ) -> None:
        self.store = store
        self.policy = policy or MemoryPolicy(store)

    def _normalize_message(self, message: Dict[str, Any]) -> Message:
        """Accept OpenAI-shaped dicts; move unknown keys (e.g., 'tool_calls') into metadata.
        This keeps the Message dataclass stable without adding many optional fields.
        """
        incoming = dict(message)  # shallow copy
        base: Dict[str, Any] = {}
        meta: Dict[str, Any] = dict(incoming.get("metadata") or {})

        for k, v in list(incoming.items()):
            if k in _ALLOWED_FIELDS and k != "metadata":
                base[k] = v
            elif k == "metadata":
                pass
            else:
                meta[k] = v

        base["metadata"] = meta
        return Message(**base)

    def append_history(self, addr: MemoryAddress, message: Dict[str, Any]) -> None:
        """Append a dict message (OpenAI-ish) to the given address, normalizing extras."""
        msg = self._normalize_message(message)
        self.policy.on_append(addr, msg)

        # Log message with color coding by role (only if enabled)
        if ENABLE_MEMORY_LOGS:
            role_colors = {
                "system": Colors.BLUE,
                "user": Colors.GREEN,
                "assistant": Colors.YELLOW,
                "tool": Colors.CYAN,
            }

            color = role_colors.get(msg.role, Colors.RESET)
            content_preview = (
                (msg.content[:100] + "...")
                if msg.content and len(msg.content) > 100
                else msg.content
            )

            tool_info = ""
            if msg.metadata and "tool_calls" in msg.metadata:
                tool_names = [
                    tc.get("function", {}).get("name", "unknown")
                    for tc in msg.metadata["tool_calls"]
                ]
                tool_info = (
                    f"{Colors.MAGENTA} | tools: {', '.join(tool_names)}{Colors.RESET}"
                )

            logger.info(
                f"{color}[{msg.role}]{Colors.RESET} {content_preview}{tool_info}"
            )

    def reset_history(
        self, addr: MemoryAddress, system_message: Dict[str, Any]
    ) -> None:
        """Replace history with a single system message for the given address."""
        msg = Message(**system_message)
        self.store.replace_messages(addr, [msg])
        if self.policy.ttl:
            self.store.set_ttl(addr, self.policy.ttl)

    def get_history(self, addr: MemoryAddress) -> List[Dict[str, Any]]:
        """Read all messages for the given address as OpenAI-formatted dicts."""
        return [m.to_openai() for m in self.store.read_messages(addr)]

    def delete_history(self, addr: MemoryAddress) -> None:
        """Remove all messages for the given address."""
        self.store.delete_conversation(addr)
