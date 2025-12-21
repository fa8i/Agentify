from __future__ import annotations
import json
import logging
from typing import List, Optional, Any, Dict
from datetime import datetime

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
except ImportError:
    Elasticsearch = None  # type: ignore
    bulk = None  # type: ignore

from ..interfaces import ConversationStore, MemoryAddress, Message

logger = logging.getLogger(__name__)


class ElasticsearchStore(ConversationStore):
    """
    Elasticsearch-backed store.
    Stores messages as individual documents to enable advanced search.
    """

    def __init__(
        self,
        url: str = "http://localhost:9200",
        api_key: Optional[str] = None,
        index_name: str = "agentify-memory",
        verify_certs: bool = True,
    ) -> None:
        if Elasticsearch is None:
            raise ImportError(
                "Elasticsearch is not installed. Please install agentify[elastic] or 'pip install elasticsearch'."
            )

        # Connection setup
        options = {}
        if api_key:
            options["api_key"] = api_key
        
        self.client = Elasticsearch(
            url,
            verify_certs=verify_certs,
            **options
        )
        self.index_name = index_name
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Create index with optimal mappings if it doesn't exist."""
        if self.client.indices.exists(index=self.index_name):
            return

        mapping = {
            "mappings": {
                "properties": {
                    # Address fields (Exact match)
                    "api_version": {"type": "keyword"},
                    "tenant_id": {"type": "keyword"},
                    "user_id": {"type": "keyword"},
                    "conversation_id": {"type": "keyword"},
                    "agent_id": {"type": "keyword"},
                    
                    # Message fields
                    "role": {"type": "keyword"},
                    "content": {"type": "text"},  # Full-text searchable
                    "name": {"type": "keyword"},
                    "id": {"type": "keyword"},
                    "ts": {"type": "date", "format": "epoch_second"},
                    
                    # Structured Objects 
                    "metadata": {"type": "object", "enabled": True}, 
                    "address_key": {"type": "keyword"}
                }
            }
        }
        try:
            self.client.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created Elasticsearch index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to create index {self.index_name}: {e}")
            raise

    def _addr_to_doc(self, addr: MemoryAddress, msg: Message) -> Dict[str, Any]:
        """Flatten address + message into a single document."""
        base = msg.to_dict()
        # Add filtering fields
        base.update({
            "api_version": addr.api_version,
            "tenant_id": addr.tenant_id,
            "user_id": addr.user_id,
            "conversation_id": addr.conversation_id,
            "agent_id": addr.agent_id,
            "address_key": addr.key_str(),
        })

        return base

    def _build_filter_query(self, addr: MemoryAddress) -> List[Dict[str, Any]]:
        """Construct bool/filter clauses for the exact address."""
        must = []
        if addr.api_version:
            must.append({"term": {"api_version": addr.api_version}})
        if addr.tenant_id:
            must.append({"term": {"tenant_id": addr.tenant_id}})
        if addr.user_id:
            must.append({"term": {"user_id": addr.user_id}})
        if addr.conversation_id:
            must.append({"term": {"conversation_id": addr.conversation_id}})
        if addr.agent_id:
            must.append({"term": {"agent_id": addr.agent_id}})
        return must

    def append_message(self, addr: MemoryAddress, msg: Message) -> None:
        doc = self._addr_to_doc(addr, msg)
        self.client.index(index=self.index_name, id=msg.id, document=doc, refresh=True) 
        # refresh='true' makes it visible immediately (good for chat consistency, possibly slower for bulk)

    def read_messages(self, addr: MemoryAddress, start: int = 0, end: int = -1) -> List[Message]:
        must = self._build_filter_query(addr)
        
        # We need to fetch enough messages to support the slice.
        # Since 'end' can be -1 (all), we might need a large size or scroll.
        # For chat history, usually 100-1000 is enough.
        size = 1000 if end == -1 else (end + 20)

        query = {
            "query": {"bool": {"filter": must}},
            "sort": [{"ts": {"order": "asc"}}],
            "size": size,
            "from": start
        }

        resp = self.client.search(index=self.index_name, body=query)
        hits = resp["hits"]["hits"]
        
        msgs = []
        for h in hits:
            src = h["_source"]
            # Extract only message fields to reconstruct
            m_kwargs = {
                "role": src.get("role"),
                "content": src.get("content"),
                "name": src.get("name"),
                "tool_call_id": src.get("tool_call_id"),
                "metadata": src.get("metadata", {}),
                "id": src.get("id"),
                "ts": int(src.get("ts", 0))
            }
            msgs.append(Message(**m_kwargs))
            
        # Slice locally if specific end requested
        if end != -1:
            limit = end - start + 1
            msgs = msgs[:limit]
            
        return msgs

    def replace_messages(self, addr: MemoryAddress, messages: List[Message]) -> None:
        # 1. Delete existing for this address
        self.delete_conversation(addr)
        
        if not messages:
            return

        # 2. Bulk insert new
        actions = []
        for msg in messages:
            doc = self._addr_to_doc(addr, msg)
            actions.append({
                "_index": self.index_name,
                "_id": msg.id,
                "_source": doc
            })
        
        if actions:
            bulk(self.client, actions, refresh=True)

    def delete_conversation(self, addr: MemoryAddress) -> None:
        must = self._build_filter_query(addr)
        query = {"query": {"bool": {"filter": must}}}
        self.client.delete_by_query(index=self.index_name, body=query, refresh=True)

    def set_ttl(self, addr: MemoryAddress, seconds: int) -> None:
        # Implementing TTL in ES usually requires Index Lifecycle Management (ILM) 
        # or a separate cleanup job, as TTL is per-index or per-document requires setup.
        # For simplicity, we log strictly: This is not natively supported per-conversation efficiently.
        logger.warning("set_ttl is not fully supported in simple ElasticsearchStore yet.")

    def list_conversations(self, limit: int = 100, offset: int = 0) -> List[MemoryAddress]:
        """
        List unique conversation addresses using aggregations.
        Uses a terms aggregation on 'address_key'.
        """
        size = offset + limit
        query = {
            "size": 0,
            "aggs": {
                "unique_conversations": {
                    "terms": {
                        "field": "address_key",
                        "size": size
                    }
                }
            }
        }
        
        resp = self.client.search(index=self.index_name, body=query)
        buckets = resp["aggregations"]["unique_conversations"]["buckets"]
        
        # Apply offset/limit locally
        buckets_slice = buckets[offset : offset + limit]
        
        results = []
        for b in buckets_slice:
            k = b["key"]
            core = k
            prefix_check = "mem:" 
            if core.startswith(prefix_check):
                core = core[len(prefix_check):]
            
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
