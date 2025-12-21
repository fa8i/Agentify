from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple
from datetime import datetime, timezone
import uuid


@dataclass(frozen=True, slots=True)
class MemoryAddress:
    """
    Composite key that identifies *where* messages live.
    You control these fields from your API boundary.
    Add/remove fields to match your domain (e.g., team_id, channel_id).
    """
    api_version: Optional[str] = None
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    agent_id: Optional[str] = None
    extras: Tuple[Tuple[str, str], ...] = ()  # extras lets pass stable routing dimensions if needed.

    def as_tuple(self) -> Tuple:
        """Stable, hashable representation for keys and indexing."""
        return (
            self.api_version,
            self.tenant_id,
            self.user_id,
            self.conversation_id,
            self.agent_id,
            self.extras,
        )

    def key_str(self, prefix: str = "mem") -> str:
        """Human-readable key for key-value backends (e.g., Redis)."""
        parts = [
            ("v", self.api_version),
            ("t", self.tenant_id),
            ("u", self.user_id),
            ("c", self.conversation_id),
            ("a", self.agent_id),
        ] + list(self.extras)
        joined = ":".join(f"{k}={v}" for k, v in parts if v)
        return f"{prefix}:{joined}" if joined else prefix


@dataclass(slots=True)
class Message:
    """
    Canonical chat message stored in the memory layer.
    - content can be str or a list of parts (vision/multimodal).
    - metadata is free-form and backend-agnostic.
    """
    role: str                                  # "system" | "user" | "assistant" | "tool"
    content: Optional[Any] = None              # str | list[dict]
    name: Optional[str] = None                 # tool name when role="tool"
    tool_call_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: int = field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp()))

    def to_openai(self) -> Dict[str, Any]:
        """Convert to OpenAI Chat Completions message format."""
        d: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.name is not None:
            d["name"] = self.name
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        d.update(self.metadata)  # carry extra fields (e.g., tool_calls)
        return d

    def to_dict(self) -> Dict[str, Any]:
        """Backend-agnostic serialization (works with slots dataclasses)."""
        return {
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "metadata": self.metadata,
            "id": self.id,
            "ts": self.ts,
        }

class ConversationStore(Protocol):
    """
    Minimal CRUD for conversation histories.
    The store does NOT generate addresses; it only consumes them.
    """

    def append_message(self, addr: MemoryAddress, msg: Message) -> None: ...
    def read_messages(self, addr: MemoryAddress, start: int = 0, end: int = -1) -> List[Message]: ...
    def replace_messages(self, addr: MemoryAddress, messages: List[Message]) -> None: ...
    def delete_conversation(self, addr: MemoryAddress) -> None: ...
    def set_ttl(self, addr: MemoryAddress, seconds: int) -> None: ...
    def list_conversations(self, limit: int = 100, offset: int = 0) -> List[MemoryAddress]: ...


class TokenCounter(Protocol):
    """Callable that returns token count for OpenAI-formatted messages."""
    def __call__(self, openai_messages: List[Dict[str, Any]]) -> int: ...


class KeyValueStore(Protocol):
    """
    Interface for long-term semantic memory or global state.
    """
    def get(self, key: str) -> Optional[Any]: ...
    def set(self, key: str, value: Any) -> None: ...
    def delete(self, key: str) -> None: ...
    def search(self, query: str, limit: int = 5) -> List[Any]: ...
