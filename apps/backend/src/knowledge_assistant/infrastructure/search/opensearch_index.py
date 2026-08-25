"""
OpenSearch / Elasticsearch BM25 Keyword Search Adapter.

Implements the `KeywordSearchIndex` domain port using OpenSearch (or Elasticsearch).
Provides distributed, persistent, multi-tenant BM25 lexical search with user and
collection isolation across backend API servers and Celery worker replicas.
"""
from __future__ import annotations

import logging
from typing import Any

from opensearchpy import AsyncOpenSearch, helpers

from knowledge_assistant.domain.entities import Chunk
from knowledge_assistant.domain.repositories import KeywordSearchIndex

logger = logging.getLogger(__name__)


class OpenSearchKeywordIndex(KeywordSearchIndex):
    """
    OpenSearch-backed BM25 keyword search index.
    Compatible with both OpenSearch and Elasticsearch clusters.
    """

    def __init__(
        self,
        hosts: str | list[str] = "http://localhost:9200",
        index_name: str = "knowledge_chunks",
        timeout: float = 30.0,
        max_retries: int = 3,
        **client_kwargs: Any,
    ) -> None:
        self.hosts = [hosts] if isinstance(hosts, str) else hosts
        self.index_name = index_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.client_kwargs = client_kwargs
        self._client: AsyncOpenSearch | None = None
        self._index_initialized = False

    def _get_client(self) -> AsyncOpenSearch:
        if self._client is None:
            use_ssl = any(str(h).startswith("https://") for h in self.hosts)
            kwargs: dict[str, Any] = {
                "hosts": self.hosts,
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "retry_on_timeout": True,
            }
            if use_ssl:
                kwargs.update({"use_ssl": True, "verify_certs": True})
            else:
                kwargs.update({"use_ssl": False, "verify_certs": False, "ssl_show_warn": False})
            kwargs.update(self.client_kwargs)
            self._client = AsyncOpenSearch(**kwargs)
        return self._client


    async def _ensure_index(self) -> None:
        if self._index_initialized:
            return
        client = self._get_client()
        try:
            exists = await client.indices.exists(index=self.index_name)
            if not exists:
                mapping = {
                    "mappings": {
                        "properties": {
                            "chunk_id": {"type": "keyword"},
                            "document_id": {"type": "keyword"},
                            "user_id": {"type": "keyword"},
                            "collection_id": {"type": "keyword"},
                            "level": {"type": "keyword"},
                            "text": {
                                "type": "text",
                                "analyzer": "standard",
                            },
                        }
                    },
                }
                try:
                    await client.indices.create(index=self.index_name, body=mapping)
                except Exception as exc:
                    if "already exists" not in str(exc).lower():
                        raise exc
                logger.info("Created OpenSearch index '%s'", self.index_name)
            self._index_initialized = True
        except Exception as exc:
            logger.warning("Could not verify/create OpenSearch index '%s': %s", self.index_name, exc)

    async def index(self, chunks: list[Chunk]) -> None:
        """Bulk index child chunks into OpenSearch."""
        child_chunks = [c for c in chunks if getattr(c.level, "value", c.level) == "child"]
        if not child_chunks:
            return

        await self._ensure_index()
        client = self._get_client()

        actions = [
            {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": chunk.id,
                "_source": {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "user_id": chunk.user_id,
                    "collection_id": chunk.collection_id,
                    "level": getattr(chunk.level, "value", str(chunk.level)),
                    "text": chunk.text,
                },
            }
            for chunk in child_chunks
        ]

        try:
            await helpers.async_bulk(client, actions, refresh=True)
        except Exception as exc:
            logger.exception("Failed to bulk index chunks into OpenSearch: %s", exc)
            raise

    async def search(
        self, query: str, user_id: str, collection_id: str | None, top_k: int
    ) -> list[tuple[str, float]]:
        """
        Execute BM25 match query isolated to the user and optional collection.
        Returns list of (chunk_id, bm25_score) tuples.
        """
        await self._ensure_index()
        client = self._get_client()

        filter_clauses: list[dict[str, Any]] = [
            {"term": {"user_id": user_id}},
            {"term": {"level": "child"}},
        ]
        if collection_id:
            filter_clauses.append({"term": {"collection_id": collection_id}})

        body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "text": {
                                    "query": query,
                                    "operator": "or",
                                }
                            }
                        }
                    ],
                    "filter": filter_clauses,
                }
            },
            "size": top_k,
            "_source": ["chunk_id"],
        }

        try:
            response = await client.search(index=self.index_name, body=body)
            hits = response.get("hits", {}).get("hits", [])
            results: list[tuple[str, float]] = []
            for hit in hits:
                chunk_id = hit.get("_source", {}).get("chunk_id") or hit.get("_id")
                score = float(hit.get("_score") or 0.0)
                if chunk_id and score > 0:
                    results.append((chunk_id, score))
            return results
        except Exception as exc:
            logger.warning("OpenSearch search failed: %s", exc)
            return []

    async def delete_by_document(self, document_id: str) -> None:
        """Delete all indexed chunks for a given document."""
        await self._ensure_index()
        client = self._get_client()
        body = {
            "query": {
                "term": {"document_id": document_id}
            }
        }
        try:
            await client.delete_by_query(index=self.index_name, body=body, refresh=True)
        except Exception as exc:
            logger.warning("Failed to delete chunks for document %s in OpenSearch: %s", document_id, exc)

    async def close(self) -> None:
        """Close client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
