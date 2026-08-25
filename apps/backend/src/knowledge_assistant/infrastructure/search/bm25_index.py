"""
BM25 keyword search.

Implementation note: rank-bm25 is an in-memory index. For a single-user
desktop-scale assistant this is fine and avoids standing up Elasticsearch.
The index is namespaced per (user_id, collection_id) and rebuilt
incrementally as documents are ingested/deleted. For larger deployments,
swap this class for an OpenSearch/Elasticsearch-backed implementation of
the same `KeywordSearchIndex` port -- nothing else in the codebase changes.
"""
from __future__ import annotations

import asyncio
import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from knowledge_assistant.domain.entities import Chunk
from knowledge_assistant.domain.repositories import KeywordSearchIndex

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class _Bucket:
    __slots__ = ("chunk_ids", "corpus", "bm25")

    def __init__(self) -> None:
        self.chunk_ids: list[str] = []
        self.corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None

    def rebuild(self) -> None:
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None


class InMemoryBM25Index(KeywordSearchIndex):
    def __init__(self) -> None:
        self._buckets: dict[tuple[str, str | None], _Bucket] = defaultdict(_Bucket)
        self._chunk_to_bucket: dict[str, tuple[str, str | None]] = {}
        self._chunk_to_doc: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def index(self, chunks: list[Chunk]) -> None:
        async with self._lock:
            for chunk in chunks:
                # only child (leaf) chunks are searched directly
                if chunk.level.value != "child":
                    continue
                key = (chunk.user_id, chunk.collection_id)
                bucket = self._buckets[key]
                bucket.chunk_ids.append(chunk.id)
                bucket.corpus.append(tokenize(chunk.text))
                self._chunk_to_bucket[chunk.id] = key
                self._chunk_to_doc[chunk.id] = chunk.document_id
            touched = {(c.user_id, c.collection_id) for c in chunks if c.level.value == "child"}
            for key in touched:
                self._buckets[key].rebuild()

    async def search(
        self, query: str, user_id: str, collection_id: str | None, top_k: int
    ) -> list[tuple[str, float]]:
        keys = [(user_id, collection_id)] if collection_id else [
            k for k in self._buckets if k[0] == user_id
        ]
        query_tokens = tokenize(query)
        results: list[tuple[str, float]] = []
        for key in keys:
            bucket = self._buckets.get(key)
            if not bucket or not bucket.bm25:
                continue
            scores = bucket.bm25.get_scores(query_tokens)
            for chunk_id, score in zip(bucket.chunk_ids, scores):
                if score > 0:
                    results.append((chunk_id, float(score)))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def delete_by_document(self, document_id: str) -> None:
        async with self._lock:
            dead_ids = {cid for cid, doc_id in self._chunk_to_doc.items() if doc_id == document_id}
            if not dead_ids:
                return
            touched_keys = {self._chunk_to_bucket[cid] for cid in dead_ids if cid in self._chunk_to_bucket}
            for key in touched_keys:
                bucket = self._buckets[key]
                keep = [
                    (cid, toks)
                    for cid, toks in zip(bucket.chunk_ids, bucket.corpus)
                    if cid not in dead_ids
                ]
                bucket.chunk_ids = [c for c, _ in keep]
                bucket.corpus = [t for _, t in keep]
                bucket.rebuild()
            for cid in dead_ids:
                self._chunk_to_bucket.pop(cid, None)
                self._chunk_to_doc.pop(cid, None)
