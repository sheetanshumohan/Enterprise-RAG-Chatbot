"""
SQLAlchemy implementations of the domain repository ports.

Each class here implements one ABC from `domain.repositories` and does
nothing except translate between domain dataclasses and ORM rows. No
business logic lives here -- that's a Clean Architecture rule: this layer
is allowed to know about SQLAlchemy, but the domain/application layers are
never allowed to know this file exists.
"""
from __future__ import annotations

from dataclasses import asdict

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_assistant.domain.entities import (
    ChatSession,
    Chunk,
    ChunkLevel,
    Collection,
    Document,
    DocumentStatus,
    DocumentType,
    EvaluationResult,
    Feedback,
    Message,
    MessageRole,
    RetrievalLog,
    User,
)
from knowledge_assistant.domain.repositories import (
    ChatRepository,
    ChunkRepository,
    CollectionRepository,
    DocumentRepository,
    EvaluationRepository,
    FeedbackRepository,
    RetrievalLogRepository,
    UserRepository,
)
from knowledge_assistant.infrastructure.db.models import (
    ChatSessionModel,
    ChunkModel,
    CollectionModel,
    DocumentModel,
    EvaluationResultModel,
    FeedbackModel,
    MessageModel,
    RetrievalLogModel,
    UserModel,
)


class SqlUserRepository(UserRepository):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, user: User) -> User:
        model = UserModel(**asdict(user))
        self._session.add(model)
        await self._session.commit()
        return user

    async def get_by_id(self, user_id: str) -> User | None:
        model = await self._session.get(UserModel, user_id)
        return _to_user(model) if model else None

    async def get_by_email(self, email: str) -> User | None:
        result = await self._session.execute(select(UserModel).where(UserModel.email == email))
        model = result.scalar_one_or_none()
        return _to_user(model) if model else None


class SqlCollectionRepository(CollectionRepository):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, collection: Collection) -> Collection:
        self._session.add(CollectionModel(**asdict(collection)))
        await self._session.commit()
        return collection

    async def list_for_user(self, user_id: str) -> list[Collection]:
        result = await self._session.execute(
            select(CollectionModel).where(CollectionModel.user_id == user_id)
        )
        return [_to_collection(m) for m in result.scalars().all()]

    async def get(self, collection_id: str, user_id: str) -> Collection | None:
        model = await self._session.get(CollectionModel, collection_id)
        if not model or model.user_id != user_id:
            return None
        return _to_collection(model)

    async def delete(self, collection_id: str, user_id: str) -> None:
        model = await self._session.get(CollectionModel, collection_id)
        if model and model.user_id == user_id:
            await self._session.delete(model)
            await self._session.commit()


class SqlDocumentRepository(DocumentRepository):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, document: Document) -> Document:
        self._session.add(_document_to_model(document))
        await self._session.commit()
        return document

    async def update(self, document: Document) -> Document:
        model = await self._session.get(DocumentModel, document.id)
        if not model:
            raise ValueError("Document not found")
        for field_, value in _document_to_model(document).__dict__.items():
            if not field_.startswith("_"):
                setattr(model, field_, value)
        await self._session.commit()
        return document

    async def get(self, document_id: str, user_id: str) -> Document | None:
        model = await self._session.get(DocumentModel, document_id)
        if not model or model.user_id != user_id:
            return None
        return _to_document(model)

    async def find_by_hash(self, user_id: str, content_hash: str) -> Document | None:
        result = await self._session.execute(
            select(DocumentModel).where(
                DocumentModel.user_id == user_id, DocumentModel.content_hash == content_hash
            )
        )
        model = result.scalar_one_or_none()
        return _to_document(model) if model else None

    async def list_for_user(self, user_id: str, collection_id: str | None = None) -> list[Document]:
        query = select(DocumentModel).where(DocumentModel.user_id == user_id)
        if collection_id:
            query = query.where(DocumentModel.collection_id == collection_id)
        result = await self._session.execute(query)
        return [_to_document(m) for m in result.scalars().all()]

    async def delete(self, document_id: str, user_id: str) -> None:
        model = await self._session.get(DocumentModel, document_id)
        if model and model.user_id == user_id:
            await self._session.delete(model)
            await self._session.commit()


class SqlChunkRepository(ChunkRepository):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def bulk_create(self, chunks: list[Chunk]) -> list[Chunk]:
        self._session.add_all(_chunk_to_model(c) for c in chunks)
        await self._session.commit()
        return chunks

    async def get_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        result = await self._session.execute(select(ChunkModel).where(ChunkModel.id.in_(chunk_ids)))
        return [_to_chunk(m) for m in result.scalars().all()]

    async def get_children(self, parent_id: str) -> list[Chunk]:
        result = await self._session.execute(select(ChunkModel).where(ChunkModel.parent_id == parent_id))
        return [_to_chunk(m) for m in result.scalars().all()]

    async def get_parent(self, parent_id: str) -> Chunk | None:
        model = await self._session.get(ChunkModel, parent_id)
        return _to_chunk(model) if model else None

    async def list_by_document(self, document_id: str) -> list[Chunk]:
        result = await self._session.execute(select(ChunkModel).where(ChunkModel.document_id == document_id))
        return [_to_chunk(m) for m in result.scalars().all()]

    async def delete_by_document(self, document_id: str) -> None:
        result = await self._session.execute(select(ChunkModel).where(ChunkModel.document_id == document_id))
        for model in result.scalars().all():
            await self._session.delete(model)
        await self._session.commit()


class SqlChatRepository(ChatRepository):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create_session(self, session_: ChatSession) -> ChatSession:
        self._session.add(ChatSessionModel(**asdict(session_)))
        await self._session.commit()
        return session_

    async def get_session(self, session_id: str, user_id: str) -> ChatSession | None:
        model = await self._session.get(ChatSessionModel, session_id)
        if not model or model.user_id != user_id:
            return None
        return _to_chat_session(model)

    async def list_sessions(self, user_id: str) -> list[ChatSession]:
        result = await self._session.execute(
            select(ChatSessionModel)
            .where(ChatSessionModel.user_id == user_id)
            .order_by(ChatSessionModel.created_at.desc())
        )
        return [_to_chat_session(m) for m in result.scalars().all()]

    async def update_title(self, session_id: str, title: str) -> None:
        model = await self._session.get(ChatSessionModel, session_id)
        if model:
            model.title = title
            await self._session.commit()

    async def delete_session(self, session_id: str, user_id: str) -> None:
        model = await self._session.get(ChatSessionModel, session_id)
        if model and model.user_id == user_id:
            # 1. Fetch all message IDs in this session
            msg_res = await self._session.execute(
                select(MessageModel.id).where(MessageModel.session_id == session_id)
            )
            msg_ids = [row[0] for row in msg_res.all()]

            if msg_ids:
                # 2. Delete feedback referencing these messages
                await self._session.execute(
                    delete(FeedbackModel).where(FeedbackModel.message_id.in_(msg_ids))
                )
                # 3. Delete messages
                await self._session.execute(
                    delete(MessageModel).where(MessageModel.session_id == session_id)
                )

            # 4. Delete the chat session itself
            await self._session.execute(
                delete(ChatSessionModel).where(ChatSessionModel.id == session_id)
            )
            await self._session.commit()

    async def add_message(self, message: Message) -> Message:
        data = asdict(message)
        data["role"] = message.role.value
        data["citations"] = [asdict(c) for c in message.citations]
        self._session.add(MessageModel(**data))
        await self._session.commit()
        return message

    async def get_history(self, session_id: str, limit: int = 20) -> list[Message]:
        result = await self._session.execute(
            select(MessageModel)
            .where(MessageModel.session_id == session_id)
            # created_at alone is the primary ordering, but two messages could
            # in principle share an identical timestamp (coarse clock
            # resolution on some platforms); id as a tiebreaker guarantees a
            # stable, deterministic order on every fetch rather than one that
            # could vary between page loads.
            .order_by(MessageModel.created_at, MessageModel.id)
            .limit(limit)
        )
        return [_to_message(m) for m in result.scalars().all()]


class SqlRetrievalLogRepository(RetrievalLogRepository):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def log(self, entry: RetrievalLog) -> RetrievalLog:
        self._session.add(RetrievalLogModel(**asdict(entry)))
        await self._session.commit()
        return entry


class SqlFeedbackRepository(FeedbackRepository):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, feedback: Feedback) -> Feedback:
        self._session.add(FeedbackModel(**asdict(feedback)))
        await self._session.commit()
        return feedback


class SqlEvaluationRepository(EvaluationRepository):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def save(self, result: EvaluationResult) -> EvaluationResult:
        self._session.add(EvaluationResultModel(**asdict(result)))
        await self._session.commit()
        return result

    async def list_recent(self, limit: int = 50) -> list[EvaluationResult]:
        result = await self._session.execute(
            select(EvaluationResultModel).order_by(EvaluationResultModel.created_at.desc()).limit(limit)
        )
        return [
            EvaluationResult(
                id=m.id, query=m.query, precision=m.precision, recall=m.recall,
                groundedness=m.groundedness, context_precision=m.context_precision,
                answer_relevance=m.answer_relevance, hallucination_rate=m.hallucination_rate,
                latency_ms=m.latency_ms, created_at=m.created_at,
            )
            for m in result.scalars().all()
        ]


# --- mapping helpers -------------------------------------------------------

def _to_user(m: UserModel) -> User:
    return User(id=m.id, email=m.email, hashed_password=m.hashed_password, full_name=m.full_name,
                created_at=m.created_at, is_active=m.is_active)


def _to_collection(m: CollectionModel) -> Collection:
    return Collection(id=m.id, user_id=m.user_id, name=m.name, description=m.description,
                       created_at=m.created_at)


def _document_to_model(d: Document) -> DocumentModel:
    return DocumentModel(
        id=d.id, user_id=d.user_id, collection_id=d.collection_id, filename=d.filename,
        doc_type=d.doc_type.value, content_hash=d.content_hash, status=d.status.value,
        version=d.version, tags=d.tags, doc_metadata=d.metadata,
        created_at=d.created_at, updated_at=d.updated_at,
    )


def _to_document(m: DocumentModel) -> Document:
    return Document(
        id=m.id, user_id=m.user_id, collection_id=m.collection_id, filename=m.filename,
        doc_type=DocumentType(m.doc_type), content_hash=m.content_hash,
        status=DocumentStatus(m.status), version=m.version, tags=m.tags, metadata=m.doc_metadata,
        created_at=m.created_at, updated_at=m.updated_at,
    )


def _chunk_to_model(c: Chunk) -> ChunkModel:
    return ChunkModel(
        id=c.id, document_id=c.document_id, user_id=c.user_id, collection_id=c.collection_id,
        text=c.text, level=c.level.value, parent_id=c.parent_id, position=c.position,
        token_count=c.token_count, chunk_metadata=c.metadata,
    )


def _to_chunk(m: ChunkModel) -> Chunk:
    return Chunk(
        id=m.id, document_id=m.document_id, user_id=m.user_id, collection_id=m.collection_id,
        text=m.text, level=ChunkLevel(m.level), parent_id=m.parent_id, position=m.position,
        token_count=m.token_count, metadata=m.chunk_metadata,
    )


def _to_chat_session(m: ChatSessionModel) -> ChatSession:
    return ChatSession(id=m.id, user_id=m.user_id, collection_id=m.collection_id, title=m.title,
                        created_at=m.created_at)


def _to_message(m: MessageModel) -> Message:
    from knowledge_assistant.domain.entities import Citation

    return Message(
        id=m.id, session_id=m.session_id, role=MessageRole(m.role), content=m.content,
        citations=[Citation(**c) for c in m.citations], confidence=m.confidence,
        reasoning_summary=m.reasoning_summary, suggested_followups=m.suggested_followups,
        created_at=m.created_at,
    )
