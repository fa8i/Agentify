from __future__ import annotations
import logging
import os
from typing import Any, Dict, List, Optional
from .interfaces import ConversationStore, MemoryAddress, Message
from .policies import MemoryPolicy
from agentify.utils.style import Colors


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
        self,
        store: ConversationStore,
        policy: Optional[MemoryPolicy] = None,
        log_enabled: bool = True,
        max_log_length: Optional[int] = None,
    ) -> None:
        self.store = store
        self.policy = policy or MemoryPolicy(store)
        self.log_enabled = log_enabled
        self.max_log_length = max_log_length # max length preview of the log

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
        if self.log_enabled:
            role_colors = {
                "system": Colors.BLUE,
                "user": Colors.GREEN,
                "assistant": Colors.YELLOW,
                "tool": Colors.CYAN,
            }

            color = role_colors.get(msg.role, Colors.RESET)
            
            # Extract agent_id if available to show who is speaking
            agent_id = addr.agent_id if addr and addr.agent_id else "unknown"
            agent_tag = f"[{agent_id}]" if agent_id else ""
            
            # Log reasoning if present in metadata
            if msg.metadata and "reasoning_content" in msg.metadata:
                reasoning = msg.metadata["reasoning_content"]
                if self.max_log_length is not None and len(reasoning) > self.max_log_length:
                    reasoning_preview = reasoning[: self.max_log_length] + "..."
                else:
                    reasoning_preview = reasoning
                
                logger.info(
                    f"{Colors.GRAY}{agent_tag}{Colors.RESET}{Colors.GRAY}[Reasoning]{Colors.RESET} {Colors.GRAY}{reasoning_preview}{Colors.RESET}"
                )

            if self.max_log_length is None:
                content_preview = msg.content
            else:
                content_preview = (
                    (msg.content[: self.max_log_length] + "...")
                    if msg.content and len(msg.content) > self.max_log_length
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
                f"{Colors.GRAY}{agent_tag}{Colors.RESET}{color}[{msg.role}]{Colors.RESET} {content_preview}{tool_info}"
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

    def list_conversations(self, limit: int = 100, offset: int = 0) -> List[MemoryAddress]:
        """List active conversations from the underlying store."""
        return self.store.list_conversations(limit=limit, offset=offset)
