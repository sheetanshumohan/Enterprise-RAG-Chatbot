"""
Application layer - Hybrid retrieval use case.

Pipeline for a single retrieval pass:
  1. Vector search (Qdrant)          -> ranked list A
  2. BM25 keyword search             -> ranked list B
  3. Reciprocal Rank Fusion (A, B)   -> fused ranking
  4. Fetch chunk metadata from ChunkRepository
  5. Expand each CHILD hit to its PARENT chunk (parent-document retrieval)
  6. De-duplicate parents, cross-encoder-style rerank
  7. Context compression: trim to the token budget, keep highest-value chunks

This class only depends on domain ports (VectorStore, KeywordSearchIndex,
ChunkRepository, EmbeddingClient, Reranker) -- it is unaware of Qdrant,
Postgres, or any concrete SDK.
"""
from __future__ import annotations

from dataclasses import dataclass

from knowledge_assistant.domain.entities import Chunk, RetrievedChunk
from knowledge_assistant.domain.repositories import ChunkRepository, KeywordSearchIndex, VectorStore
from knowledge_assistant.infrastructure.embeddings.client import EmbeddingClient
from knowledge_assistant.infrastructure.reranking.reranker import Reranker


@dataclass
class RetrievalParams:
    user_id: str
    collection_id: str | None = None
    document_ids: list[str] | None = None
    top_k_vector: int = 20
    top_k_bm25: int = 20
    rrf_k: int = 60
    top_k_fused: int = 10
    token_budget: int = 4000
    metadata_filters: dict | None = None


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        keyword_index: KeywordSearchIndex,
        chunk_repo: ChunkRepository,
        embedding_client: EmbeddingClient,
        reranker: Reranker,
    ) -> None:
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._chunk_repo = chunk_repo
        self._embedding_client = embedding_client
        self._reranker = reranker

    async def retrieve(self, query: str, params: RetrievalParams) -> list[RetrievedChunk]:
        query_embedding = await self._embedding_client.embed_one(query)

        vector_hits = await self._vector_store.search(
            query_embedding=query_embedding,
            user_id=params.user_id,
            collection_id=params.collection_id,
            top_k=params.top_k_vector,
            metadata_filters=params.metadata_filters,
        )
        bm25_hits = await self._keyword_index.search(
            query=query,
            user_id=params.user_id,
            collection_id=params.collection_id,
            top_k=params.top_k_bm25,
        )

        fused_scores = reciprocal_rank_fusion(vector_hits, bm25_hits, k=params.rrf_k)
        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[: params.top_k_fused]
        if not ranked_ids:
            return []

        child_chunks = {c.id: c for c in await self._chunk_repo.get_by_ids(ranked_ids)}

        # Parent-document expansion: replace each child hit with its parent's full text,
        # but keep the child's relevance score (the child is what matched the query).
        expanded: dict[str, RetrievedChunk] = {}
        vector_by_id = dict(vector_hits)
        bm25_by_id = dict(bm25_hits)
        for chunk_id in ranked_ids:
            child = child_chunks.get(chunk_id)
            if not child:
                continue
            parent = await self._chunk_repo.get_parent(child.parent_id) if child.parent_id else None
            target = parent or child
            existing = expanded.get(target.id)
            new_score = fused_scores[chunk_id]
            if existing is None or new_score > existing.fused_score:
                expanded[target.id] = RetrievedChunk(
                    chunk=target,
                    vector_score=vector_by_id.get(chunk_id, 0.0),
                    bm25_score=bm25_by_id.get(chunk_id, 0.0),
                    fused_score=new_score,
                    source_query=query,
                )

        candidates = list(expanded.values())
        if params.document_ids:
            target_set = set(params.document_ids)
            candidates = [c for c in candidates if c.chunk.document_id in target_set]

        if not candidates:
            return []

        reranked = await self._reranker.rerank(query, candidates)
        return compress_to_budget(reranked, params.token_budget)


def reciprocal_rank_fusion(
    *ranked_lists: list[tuple[str, float]], k: int = 60
) -> dict[str, float]:
    """Combine multiple ranked (id, score) lists into one fused score per id.

    RRF score for a document = sum over lists of 1 / (k + rank_in_that_list).
    This is rank-based (not raw-score-based), so it works even though
    vector cosine similarity and BM25 scores live on completely different
    scales -- the classic problem with naive score-averaging hybrid search.
    """
    fused: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, (doc_id, _score) in enumerate(
            sorted(ranked_list, key=lambda x: x[1], reverse=True), start=1
        ):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


def compress_to_budget(chunks: list[RetrievedChunk], token_budget: int) -> list[RetrievedChunk]:
    """Context compression: keep highest-ranked chunks until the token budget
    is exhausted, so we never blow the LLM's context window or dilute it with
    low-value chunks (which also hurts groundedness and raises cost)."""
    from knowledge_assistant.infrastructure.chunking.chunker import count_tokens

    kept: list[RetrievedChunk] = []
    used = 0
    ordered = sorted(chunks, key=lambda rc: rc.rerank_score or rc.fused_score, reverse=True)
    for rc in ordered:
        tokens = count_tokens(rc.chunk.text)
        if used + tokens > token_budget and kept:
            continue
        kept.append(rc)
        used += tokens
    return kept
