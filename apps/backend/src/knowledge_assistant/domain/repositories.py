"""
Domain layer - Repository interfaces ("ports" in hexagonal-architecture terms).

These are ABCs only. The `application` layer depends on these abstractions,
never on concrete infrastructure classes. `infrastructure/db/*` provides the
real SQLAlchemy/Qdrant/Redis implementations and is wired in at the
composition root (interfaces/api/main.py) via dependency injection.

This is what makes the system swappable: replace Postgres with anything
else, or Qdrant with pgvector/Pinecone, without touching domain or
application code, as long as the new class implements these interfaces.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from knowledge_assistant.domain.entities import (
    ChatSession,
    Chunk,
    Collection,
    Document,
    EvaluationResult,
    Feedback,
    Message,
    RetrievalLog,
    User,
)


class UserRepository(ABC):
    @abstractmethod
    async def create(self, user: User) -> User: ...

    @abstractmethod
    async def get_by_id(self, user_id: str) -> User | None: ...

    @abstractmethod
    async def get_by_email(self, email: str) -> User | None: ...


class CollectionRepository(ABC):
    @abstractmethod
    async def create(self, collection: Collection) -> Collection: ...

    @abstractmethod
    async def list_for_user(self, user_id: str) -> list[Collection]: ...

    @abstractmethod
    async def get(self, collection_id: str, user_id: str) -> Collection | None: ...

    @abstractmethod
    async def delete(self, collection_id: str, user_id: str) -> None: ...


class DocumentRepository(ABC):
    @abstractmethod
    async def create(self, document: Document) -> Document: ...

    @abstractmethod
    async def update(self, document: Document) -> Document: ...

    @abstractmethod
    async def get(self, document_id: str, user_id: str) -> Document | None: ...

    @abstractmethod
    async def find_by_hash(self, user_id: str, content_hash: str) -> Document | None:
        """Used for duplicate detection before re-ingesting a file."""
        ...

    @abstractmethod
    async def list_for_user(
        self, user_id: str, collection_id: str | None = None
    ) -> list[Document]: ...

    @abstractmethod
    async def delete(self, document_id: str, user_id: str) -> None: ...


class ChunkRepository(ABC):
    """Metadata store for chunks (Postgres). Vectors themselves live in VectorStore."""

    @abstractmethod
    async def bulk_create(self, chunks: list[Chunk]) -> list[Chunk]: ...

    @abstractmethod
    async def get_by_ids(self, chunk_ids: list[str]) -> list[Chunk]: ...

    @abstractmethod
    async def get_children(self, parent_id: str) -> list[Chunk]: ...

    @abstractmethod
    async def get_parent(self, parent_id: str) -> Chunk | None: ...

    @abstractmethod
    async def list_by_document(self, document_id: str) -> list[Chunk]: ...

    @abstractmethod
    async def delete_by_document(self, document_id: str) -> None: ...


class VectorStore(ABC):
    """Port for the vector database (Qdrant in infrastructure/vector_store)."""

    @abstractmethod
    async def upsert(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        user_id: str,
        collection_id: str | None,
        top_k: int,
        metadata_filters: dict | None = None,
    ) -> list[tuple[str, float]]:
        """Returns list of (chunk_id, similarity_score)."""
        ...

    @abstractmethod
    async def delete_by_document(self, document_id: str) -> None: ...


class KeywordSearchIndex(ABC):
    """Port for BM25 keyword search."""

    @abstractmethod
    async def index(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    async def search(
        self, query: str, user_id: str, collection_id: str | None, top_k: int
    ) -> list[tuple[str, float]]:
        """Returns list of (chunk_id, bm25_score)."""
        ...

    @abstractmethod
    async def delete_by_document(self, document_id: str) -> None: ...


class ChatRepository(ABC):
    @abstractmethod
    async def create_session(self, session: ChatSession) -> ChatSession: ...

    @abstractmethod
    async def get_session(self, session_id: str, user_id: str) -> ChatSession | None: ...

    @abstractmethod
    async def list_sessions(self, user_id: str) -> list[ChatSession]: ...

    @abstractmethod
    async def update_title(self, session_id: str, title: str) -> None: ...

    @abstractmethod
    async def delete_session(self, session_id: str, user_id: str) -> None: ...

    @abstractmethod
    async def add_message(self, message: Message) -> Message: ...

    @abstractmethod
    async def get_history(self, session_id: str, limit: int = 20) -> list[Message]: ...


class RetrievalLogRepository(ABC):
    @abstractmethod
    async def log(self, entry: RetrievalLog) -> RetrievalLog: ...


class FeedbackRepository(ABC):
    @abstractmethod
    async def create(self, feedback: Feedback) -> Feedback: ...


class EvaluationRepository(ABC):
    @abstractmethod
    async def save(self, result: EvaluationResult) -> EvaluationResult: ...

    @abstractmethod
    async def list_recent(self, limit: int = 50) -> list[EvaluationResult]: ...
