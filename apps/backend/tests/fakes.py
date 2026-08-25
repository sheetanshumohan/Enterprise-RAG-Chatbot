"""In-memory fakes for the domain repository ports, used by application-layer
tests so they don't need a real Postgres/Qdrant instance. Each fake fully
implements its ABC, matching production semantics closely enough (including
per-user isolation) to catch real bugs."""
from __future__ import annotations

from knowledge_assistant.domain.entities import Chunk, Document
from knowledge_assistant.domain.repositories import ChunkRepository, DocumentRepository, KeywordSearchIndex, VectorStore
from knowledge_assistant.infrastructure.embeddings.client import EmbeddingClient


class FakeDocumentRepository(DocumentRepository):
    def __init__(self):
        self._store: dict[str, Document] = {}

    async def create(self, document: Document) -> Document:
        self._store[document.id] = document
        return document

    async def update(self, document: Document) -> Document:
        self._store[document.id] = document
        return document

    async def get(self, document_id: str, user_id: str) -> Document | None:
        doc = self._store.get(document_id)
        return doc if doc and doc.user_id == user_id else None

    async def find_by_hash(self, user_id: str, content_hash: str) -> Document | None:
        for doc in self._store.values():
            if doc.user_id == user_id and doc.content_hash == content_hash:
                return doc
        return None

    async def list_for_user(self, user_id: str, collection_id: str | None = None) -> list[Document]:
        return [
            d for d in self._store.values()
            if d.user_id == user_id and (collection_id is None or d.collection_id == collection_id)
        ]

    async def delete(self, document_id: str, user_id: str) -> None:
        doc = self._store.get(document_id)
        if doc and doc.user_id == user_id:
            del self._store[document_id]


class FakeChunkRepository(ChunkRepository):
    def __init__(self):
        self._store: dict[str, Chunk] = {}

    async def bulk_create(self, chunks: list[Chunk]) -> list[Chunk]:
        for c in chunks:
            self._store[c.id] = c
        return chunks

    async def get_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        return [self._store[cid] for cid in chunk_ids if cid in self._store]

    async def get_children(self, parent_id: str) -> list[Chunk]:
        return [c for c in self._store.values() if c.parent_id == parent_id]

    async def get_parent(self, parent_id: str) -> Chunk | None:
        return self._store.get(parent_id)

    async def list_by_document(self, document_id: str) -> list[Chunk]:
        return [c for c in self._store.values() if c.document_id == document_id]

    async def delete_by_document(self, document_id: str) -> None:
        dead = [cid for cid, c in self._store.items() if c.document_id == document_id]
        for cid in dead:
            del self._store[cid]


class FakeVectorStore(VectorStore):
    def __init__(self):
        self.upserted: list[Chunk] = []
        self.deleted_documents: list[str] = []

    async def upsert(self, chunks: list[Chunk]) -> None:
        self.upserted.extend(chunks)

    async def search(self, query_embedding, user_id, collection_id, top_k, metadata_filters=None):
        return [(c.id, 0.9) for c in self.upserted if c.user_id == user_id][:top_k]

    async def delete_by_document(self, document_id: str) -> None:
        self.deleted_documents.append(document_id)
        self.upserted = [c for c in self.upserted if c.document_id != document_id]


class FakeKeywordIndex(KeywordSearchIndex):
    def __init__(self):
        self.indexed: list[Chunk] = []
        self.deleted_documents: list[str] = []

    async def index(self, chunks: list[Chunk]) -> None:
        self.indexed.extend(c for c in chunks if c.level.value == "child")

    async def search(self, query, user_id, collection_id, top_k):
        return [(c.id, 1.0) for c in self.indexed if c.user_id == user_id][:top_k]

    async def delete_by_document(self, document_id: str) -> None:
        self.deleted_documents.append(document_id)
        self.indexed = [c for c in self.indexed if c.document_id != document_id]


class FakeEmbeddingClient(EmbeddingClient):
    dim = 8

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # deterministic fake embedding: bag-of-length hash, good enough to exercise the pipeline
        return [[float((hash(t) >> i) % 7) for i in range(self.dim)] for t in texts]
