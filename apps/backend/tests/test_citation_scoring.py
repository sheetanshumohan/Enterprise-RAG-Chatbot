"""Regression tests for two real bugs reported by a user: (1) citation
scores all displayed as the same value, and (2) 'rc.rerank_score or
rc.fused_score' silently discarding a genuine 0.0 rerank_score."""
from __future__ import annotations

from knowledge_assistant.application.use_cases.ask_question import _extract_citations
from knowledge_assistant.domain.entities import Chunk, ChunkLevel, RetrievedChunk


def _rc(idx: int, rerank_score: float | None, fused_score: float = 0.02) -> RetrievedChunk:
    chunk = Chunk(
        document_id=f"doc-{idx}", user_id="u1", collection_id="c1",
        text=f"chunk text {idx}", level=ChunkLevel.PARENT,
        metadata={"filename": f"file{idx}.txt"},
    )
    return RetrievedChunk(chunk=chunk, fused_score=fused_score, rerank_score=rerank_score)


def test_citation_scores_are_normalized_and_distinguishable():
    citation_map = {1: _rc(1, rerank_score=0.019), 2: _rc(2, rerank_score=0.031)}
    answer = "The answer cites [1] and also [2] for more detail."

    citations = _extract_citations(answer, citation_map)

    assert len(citations) == 2
    scores = {c.document_id: c.score for c in citations}
    # after calibrated normalization, the weaker and stronger citations must be
    # clearly distinguishable and non-zero
    assert scores["doc-1"] < scores["doc-2"]
    assert scores["doc-1"] >= 0.70
    assert scores["doc-2"] <= 1.0


def test_single_citation_normalizes_to_full_score():
    citation_map = {1: _rc(1, rerank_score=0.02)}
    citations = _extract_citations("See [1].", citation_map)
    assert citations[0].score >= 0.90


def test_zero_rerank_score_is_not_silently_replaced_by_fused_score():
    """A real 0.0 rerank_score is a legitimate value (the chunk had zero
    query-term overlap) and must not be treated as 'missing'."""
    citation_map = {
        1: _rc(1, rerank_score=0.0, fused_score=0.5),
        2: _rc(2, rerank_score=0.4, fused_score=0.5),
    }
    citations = _extract_citations("[1] and [2]", citation_map)
    scores = {c.document_id: c.score for c in citations}
    # doc-1's rerank_score of 0.0 must be respected (making it the weaker
    # citation), not overridden by its fused_score of 0.5
    assert scores["doc-1"] < scores["doc-2"]


def test_fallback_citations_when_no_brackets():
    citation_map = {1: _rc(1, rerank_score=0.5)}
    citations = _extract_citations("No bracket references here.", citation_map)
    assert len(citations) == 1
    assert citations[0].document_id == "doc-1"
