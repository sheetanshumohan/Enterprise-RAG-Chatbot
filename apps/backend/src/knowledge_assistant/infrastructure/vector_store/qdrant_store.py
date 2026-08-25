"""Qdrant-backed implementation of the VectorStore port."""
from __future__ import annotations

from qdrant_client import AsyncQdrantClient, models

from knowledge_assistant.domain.entities import Chunk
from knowledge_assistant.domain.repositories import VectorStore

class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        url: str,
        embedding_dim: int,
        api_key: str | None = None,
        collection_name: str = "knowledge_assistant_chunks",
        timeout: float = 30.0,
    ) -> None:
        self._client = AsyncQdrantClient(url=url, api_key=api_key, timeout=timeout)
        self._dim = embedding_dim
        self._collection_name = collection_name
        self._ensured = False

    async def _ensure_collection(self) -> None:
        if self._ensured:
            return
        exists = await self._client.collection_exists(self._collection_name)
        if exists:
            try:
                info = await self._client.get_collection(self._collection_name)
                current_dim = getattr(info.config.params.vectors, "size", None)
                if current_dim and current_dim != self._dim:
                    await self._client.delete_collection(self._collection_name)
                    exists = False
            except Exception:
                pass

        if not exists:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=self._dim, distance=models.Distance.COSINE
                ),
            )
        for field_name in ("user_id", "collection_id", "document_id", "level"):
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # payload index already exists
        self._ensured = True

    async def upsert(self, chunks: list[Chunk]) -> None:
        await self._ensure_collection()
        embedded = [c for c in chunks if c.embedding is not None]
        if not embedded:
            return
        points = [
            models.PointStruct(
                id=c.id,
                vector=c.embedding,
                payload={
                    "user_id": c.user_id,
                    "collection_id": c.collection_id,
                    "document_id": c.document_id,
                    "level": c.level.value,
                    "parent_id": c.parent_id,
                    "tags": c.metadata.get("tags", []),
                    "doc_type": c.metadata.get("doc_type"),
                },
            )
            for c in embedded
        ]
        await self._client.upsert(collection_name=self._collection_name, points=points)

    async def search(
        self,
        query_embedding: list[float],
        user_id: str,
        collection_id: str | None,
        top_k: int,
        metadata_filters: dict | None = None,
    ) -> list[tuple[str, float]]:
        await self._ensure_collection()
        must = [
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
            models.FieldCondition(key="level", match=models.MatchValue(value="child")),
        ]
        if collection_id:
            must.append(
                models.FieldCondition(
                    key="collection_id", match=models.MatchValue(value=collection_id)
                )
            )
        if metadata_filters:
            for key, value in metadata_filters.items():
                must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))

        result = await self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            query_filter=models.Filter(must=must),
            limit=top_k,
        )
        return [(str(point.id), float(point.score)) for point in result.points]

    async def delete_by_document(self, document_id: str) -> None:
        await self._ensure_collection()
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id", match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            ),
        )

