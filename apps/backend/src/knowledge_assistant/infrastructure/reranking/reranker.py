"""
Reranking stage.

`CrossEncoderReranker` uses a local sentence-transformers cross-encoder
(e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) which scores (query, chunk)
pairs jointly -- much more accurate than the bi-encoder similarity used in
first-pass vector search, at the cost of being too slow to run over the
whole corpus (hence: retrieve broad with hybrid search, rerank narrow
over ~10-20 candidates).

`LexicalOverlapReranker` is a dependency-free fallback (used automatically
if sentence-transformers / model weights aren't available in the runtime
environment, e.g. offline CI) so the pipeline still degrades gracefully
rather than crashing.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from knowledge_assistant.domain.entities import RetrievedChunk


class Reranker(ABC):
    @abstractmethod
    async def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    async def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return candidates
        import asyncio

        pairs = [(query, rc.chunk.text) for rc in candidates]
        scores = await asyncio.to_thread(self._model.predict, pairs)
        for rc, score in zip(candidates, scores):
            rc.rerank_score = float(score)
        return sorted(candidates, key=lambda rc: rc.rerank_score, reverse=True)


class LexicalOverlapReranker(Reranker):
    """Zero-dependency fallback reranker based on query-term coverage.

    Note: this scores against `rc.chunk.text`, which by the time reranking
    runs has already been expanded to the PARENT chunk (up to ~1500 tokens,
    see HybridRetriever's parent-document expansion). A naive Jaccard score
    (overlap / union of all tokens) is dominated by that expansion -- the
    union balloons with chunk length, drowning out genuine differences
    between candidates and making most scores collapse toward the same
    small number. Using *coverage* (overlap / query length) instead keeps
    the denominator fixed to the query, so it isn't diluted by chunk size.
    """

    async def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        for rc in candidates:
            chunk_tokens = set(re.findall(r"[a-z0-9]+", rc.chunk.text.lower()))
            overlap = len(query_tokens & chunk_tokens)
            coverage = overlap / len(query_tokens) if query_tokens else 0.0
            rc.rerank_score = 0.8 * coverage + 0.2 * rc.fused_score
        return sorted(candidates, key=lambda rc: rc.rerank_score, reverse=True)


def get_reranker(use_cross_encoder: bool, model_name: str | None = None) -> Reranker:
    if use_cross_encoder:
        try:
            return CrossEncoderReranker(model_name) if model_name else CrossEncoderReranker()
        except Exception:
            pass
    return LexicalOverlapReranker()

