from __future__ import annotations
from typing import Dict, List
from collections import defaultdict
from ..interfaces import ConversationStore, MemoryAddress, Message


class InMemoryStore(ConversationStore):
    """Dict-backed store."""

    def __init__(self) -> None:
        self._history: Dict[MemoryAddress, List[Message]] = defaultdict(list)
        self._ttl: Dict[MemoryAddress, int] = {}

    def append_message(self, addr: MemoryAddress, msg: Message) -> None:
        self._history[addr].append(msg)

    def read_messages(self, addr: MemoryAddress, start: int = 0, end: int = -1) -> List[Message]:
        msgs = self._history.get(addr, [])
        return msgs[slice(start, None if end == -1 else end + 1)]

    def replace_messages(self, addr: MemoryAddress, messages: List[Message]) -> None:
        self._history[addr] = list(messages)

    def delete_conversation(self, addr: MemoryAddress) -> None:
        self._history.pop(addr, None)
        self._ttl.pop(addr, None)

    def set_ttl(self, addr: MemoryAddress, seconds: int) -> None:
        self._ttl[addr] = seconds  # not enforced in-memory

    def list_conversations(self, limit: int = 100, offset: int = 0) -> List[MemoryAddress]:
        keys = list(self._history.keys())
        # Sort for stability
        keys.sort(key=lambda k: k.key_str())
        return keys[offset : offset + limit]
