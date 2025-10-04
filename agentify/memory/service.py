from __future__ import annotations
from typing import Any, Dict, List, Optional
from .interfaces import ConversationStore, MemoryAddress, Message
from .policies import MemoryPolicy

_ALLOWED_FIELDS = {"role", "content", "name", "tool_call_id", "metadata", "id", "ts"}


class MemoryService:
    """
    Facade consumed by the agent. It does NOT create addresses.
    The API requires a MemoryAddress provided by the application/API layer.
    """

    def __init__(
        self, store: ConversationStore, policy: Optional[MemoryPolicy] = None
    ) -> None:
        self.store = store
        self.policy = policy or MemoryPolicy(store)

    def _normalize_message(self, message: Dict[str, Any]) -> Message:
        """
        Accept OpenAI-shaped dicts; move unknown keys (e.g., 'tool_calls') into metadata.
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
        print(msg)

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
