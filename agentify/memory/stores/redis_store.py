from __future__ import annotations
import json
from typing import List

try:
    import redis
except ImportError:
    redis = None
from ..interfaces import ConversationStore, MemoryAddress, Message


class RedisStore(ConversationStore):
    """
    Redis-backed store. The key is generated from MemoryAddress.key_str().
    """

    def __init__(
        self, url: str = "redis://localhost:6379/0", key_prefix: str = "mem"
    ) -> None:
        if redis is None:
            raise ImportError("Redis is not installed. Please install agentify[redis].")
        self.r = redis.from_url(url, decode_responses=True)
        self.prefix = key_prefix

    def _hkey(self, addr: MemoryAddress) -> str:
        return f"{addr.key_str(prefix=self.prefix)}:history"

    def append_message(self, addr: MemoryAddress, msg: Message) -> None:
        self.r.rpush(self._hkey(addr), json.dumps(msg.to_dict(), ensure_ascii=False))

    def read_messages(
        self, addr: MemoryAddress, start: int = 0, end: int = -1
    ) -> List[Message]:
        raw = self.r.lrange(self._hkey(addr), start, end)
        return [Message(**json.loads(x)) for x in raw]

    def replace_messages(self, addr: MemoryAddress, messages: List[Message]) -> None:
        key = self._hkey(addr)
        pipe = self.r.pipeline(transaction=True)
        pipe.delete(key)
        if messages:
            pipe.rpush(
                key, *[json.dumps(m.to_dict(), ensure_ascii=False) for m in messages]
            )
        pipe.execute()

    def delete_conversation(self, addr: MemoryAddress) -> None:
        self.r.delete(self._hkey(addr))

    def set_ttl(self, addr: MemoryAddress, seconds: int) -> None:
        self.r.expire(self._hkey(addr), seconds)

    def list_conversations(self, limit: int = 100, offset: int = 0) -> List[MemoryAddress]:
        """Scan keys to list active dialogues."""
        keys = []
        pattern = f"{self.prefix}:*:history"
        
        for key in self.r.scan_iter(match=pattern, count=100):
            keys.append(key)
        
        keys.sort()
        
        # Paginate
        slice_keys = keys[offset : offset + limit]
        
        results = []
        for k in slice_keys:
            if not k.endswith(":history"):
                continue
            core = k[:-8]
            
            if core.startswith(f"{self.prefix}:"):
                core = core[len(self.prefix)+1:]
            
            parts = core.split(":")
            kwargs = {}
            extras = []
            
            for part in parts:
                if "=" not in part:
                    continue
                key, val = part.split("=", 1)
                if key == "v": kwargs["api_version"] = val
                elif key == "t": kwargs["tenant_id"] = val
                elif key == "u": kwargs["user_id"] = val
                elif key == "c": kwargs["conversation_id"] = val
                elif key == "a": kwargs["agent_id"] = val
                else:
                    extras.append((key, val))
            
            if extras:
                kwargs["extras"] = tuple(extras)
                
            results.append(MemoryAddress(**kwargs))
            
        return results
