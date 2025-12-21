from __future__ import annotations
import sqlite3
import json
import logging
from typing import List, Tuple, Any
from ..interfaces import ConversationStore, MemoryAddress, Message

logger = logging.getLogger(__name__)


class SQLiteStore(ConversationStore):
    """
    SQLite-backed store.
    Zero-dependency, single-file persistent storage.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Create a new connection for thread safety if needed."""
        # SQLite connections are not thread-safe by default.
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        """Initialize table schema."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_version TEXT,
                    tenant_id TEXT,
                    user_id TEXT,
                    conversation_id TEXT,
                    agent_id TEXT,
                    address_key TEXT,
                    msg_id TEXT,
                    ts REAL,
                    payload TEXT
                )
            """)
            # Indices for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_addr ON messages (conversation_id, agent_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ts ON messages (ts)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_key ON messages (address_key)")
            conn.commit()
        finally:
            conn.close()

    def _addr_key(self, addr: MemoryAddress) -> str:
        return addr.key_str()

    def append_message(self, addr: MemoryAddress, msg: Message) -> None:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO messages (
                    api_version, tenant_id, user_id, conversation_id, agent_id, address_key,
                    msg_id, ts, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    addr.api_version,
                    addr.tenant_id,
                    addr.user_id,
                    addr.conversation_id,
                    addr.agent_id,
                    self._addr_key(addr),
                    msg.id,
                    msg.ts,
                    json.dumps(msg.to_dict(), ensure_ascii=False),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def read_messages(
        self, addr: MemoryAddress, start: int = 0, end: int = -1
    ) -> List[Message]:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            query = "SELECT payload FROM messages WHERE address_key = ? ORDER BY ts ASC"
            params: List[Any] = [self._addr_key(addr)]
            
            if end != -1:
                limit = end - start + 1
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, start])
            else:
                query += " LIMIT -1 OFFSET ?"
                params.append(start)
                
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            
            return [Message(**json.loads(row[0])) for row in rows]
        finally:
            conn.close()

    def replace_messages(self, addr: MemoryAddress, messages: List[Message]) -> None:
        conn = self._get_conn()
        try:
            # Transactional replace
            with conn: # auto commit/rollback
                cursor = conn.cursor()
                # 1. Delete existing
                cursor.execute("DELETE FROM messages WHERE address_key = ?", (self._addr_key(addr),))
                
                # 2. Insert new
                if messages:
                    data = []
                    for msg in messages:
                        data.append((
                            addr.api_version,
                            addr.tenant_id,
                            addr.user_id,
                            addr.conversation_id,
                            addr.agent_id,
                            self._addr_key(addr),
                            msg.id,
                            msg.ts,
                            json.dumps(msg.to_dict(), ensure_ascii=False),
                        ))
                    cursor.executemany(
                        """
                        INSERT INTO messages (
                            api_version, tenant_id, user_id, conversation_id, agent_id, address_key,
                            msg_id, ts, payload
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        data
                    )
        finally:
            conn.close()

    def delete_conversation(self, addr: MemoryAddress) -> None:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE address_key = ?", (self._addr_key(addr),))
            conn.commit()
        finally:
            conn.close()

    def set_ttl(self, addr: MemoryAddress, seconds: int) -> None:
        logger.warning("set_ttl is not supported in SQLiteStore (requires external cleanup job).")

    def list_conversations(self, limit: int = 100, offset: int = 0) -> List[MemoryAddress]:
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            # Select DISTINCT address_key to find unique conversations
            cursor.execute(
                "SELECT DISTINCT address_key FROM messages ORDER BY address_key LIMIT ? OFFSET ?",
                (limit, offset)
            )
            rows = cursor.fetchall()
            
            result = []
            for row in rows:
                key_str = row[0]
                
                core = key_str
                if core.startswith("mem:"):
                    core = core[4:]
                
                parts = core.split(":")
                kwargs = {}
                extras = []
                for part in parts:
                    if "=" not in part:
                        continue
                    k, v = part.split("=", 1)
                    if k == "v": kwargs["api_version"] = v
                    elif k == "t": kwargs["tenant_id"] = v
                    elif k == "u": kwargs["user_id"] = v
                    elif k == "c": kwargs["conversation_id"] = v
                    elif k == "a": kwargs["agent_id"] = v
                    else:
                        extras.append((k, v))
                
                if extras:
                    kwargs["extras"] = tuple(extras)
                    
                result.append(MemoryAddress(**kwargs))
            
            return result
        finally:
            conn.close()
