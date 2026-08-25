"""
Domain layer - Entities.

Clean Architecture rule: nothing in this file may import from
`application`, `infrastructure`, or `interfaces`. This layer has zero
framework dependencies (no FastAPI, no SQLAlchemy, no Qdrant client).
It is pure Python + stdlib, so it can be unit tested in isolation and
would survive a full swap of DB/vector-store/web-framework.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    TXT = "txt"


class ChunkLevel(str, Enum):
    PARENT = "parent"
    CHILD = "child"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class User:
    email: str
    hashed_password: str
    id: str = field(default_factory=new_id)
    full_name: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    is_active: bool = True


@dataclass
class Collection:
    user_id: str
    name: str
    id: str = field(default_factory=new_id)
    description: str = ""
    created_at: datetime = field(default_factory=utcnow)


@dataclass
class Document:
    user_id: str
    collection_id: str
    filename: str
    doc_type: DocumentType
    content_hash: str
    id: str = field(default_factory=new_id)
    status: DocumentStatus = DocumentStatus.PENDING
    version: int = 1
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    def mark_processing(self) -> None:
        self.status = DocumentStatus.PROCESSING
        self.updated_at = utcnow()

    def mark_indexed(self) -> None:
        self.status = DocumentStatus.INDEXED
        self.updated_at = utcnow()

    def mark_failed(self) -> None:
        self.status = DocumentStatus.FAILED
        self.updated_at = utcnow()

    def new_version(self, content_hash: str) -> "Document":
        """Return a new Document representing a re-upload/edit of this file."""
        return Document(
            user_id=self.user_id,
            collection_id=self.collection_id,
            filename=self.filename,
            doc_type=self.doc_type,
            content_hash=content_hash,
            version=self.version + 1,
            tags=list(self.tags),
            metadata=dict(self.metadata),
        )


@dataclass
class Chunk:
    document_id: str
    user_id: str
    collection_id: str
    text: str
    level: ChunkLevel
    id: str = field(default_factory=new_id)
    parent_id: str | None = None  # set on CHILD chunks, points at PARENT chunk id
    position: int = 0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class RetrievedChunk:
    """A chunk plus the retrieval metadata explaining why it was returned."""
    chunk: Chunk
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float | None = None
    source_query: str = ""


@dataclass
class ChatSession:
    user_id: str
    id: str = field(default_factory=new_id)
    collection_id: str | None = None  # None => search across all collections
    title: str = "New conversation"
    created_at: datetime = field(default_factory=utcnow)


@dataclass
class Citation:
    chunk_id: str
    document_id: str
    document_filename: str
    snippet: str
    score: float


@dataclass
class Message:
    session_id: str
    role: MessageRole
    content: str
    id: str = field(default_factory=new_id)
    citations: list[Citation] = field(default_factory=list)
    confidence: float | None = None
    reasoning_summary: str | None = None
    suggested_followups: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)


@dataclass
class RetrievalLog:
    query: str
    user_id: str
    id: str = field(default_factory=new_id)
    rewritten_queries: list[str] = field(default_factory=list)
    retriever_used: str = "hybrid"
    iterations: int = 1
    chunks_retrieved: int = 0
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    embedding_latency_ms: float = 0.0
    created_at: datetime = field(default_factory=utcnow)


@dataclass
class Feedback:
    message_id: str
    user_id: str
    rating: int  # -1, 0, +1
    id: str = field(default_factory=new_id)
    comment: str = ""
    created_at: datetime = field(default_factory=utcnow)


@dataclass
class EvaluationResult:
    query: str
    id: str = field(default_factory=new_id)
    precision: float = 0.0
    recall: float = 0.0
    groundedness: float = 0.0
    context_precision: float = 0.0
    answer_relevance: float = 0.0
    hallucination_rate: float = 0.0
    latency_ms: float = 0.0
    created_at: datetime = field(default_factory=utcnow)
