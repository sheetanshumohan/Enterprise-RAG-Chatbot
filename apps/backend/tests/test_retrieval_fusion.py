from knowledge_assistant.application.use_cases.retrieve_context import compress_to_budget, reciprocal_rank_fusion
from knowledge_assistant.domain.entities import Chunk, ChunkLevel, RetrievedChunk


def test_rrf_favors_items_ranked_high_in_both_lists():
    vector_hits = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    bm25_hits = [("b", 15.0), ("a", 10.0), ("d", 5.0)]

    fused = reciprocal_rank_fusion(vector_hits, bm25_hits, k=60)

    # "a" is #1 vector / #2 bm25, "b" is #2 vector / #1 bm25 -> both should
    # outrank "c" (only in one list) and "d" (only in one list, lower rank)
    assert fused["a"] > fused["c"]
    assert fused["b"] > fused["d"]
    assert fused["a"] > fused["d"]


def test_rrf_handles_disjoint_lists():
    fused = reciprocal_rank_fusion([("x", 1.0)], [("y", 1.0)])
    assert set(fused.keys()) == {"x", "y"}
    assert fused["x"] == fused["y"]  # both are rank 1 in their own list


def _rc(text: str, score: float) -> RetrievedChunk:
    chunk = Chunk(
        document_id="d1", user_id="u1", collection_id="c1",
        text=text, level=ChunkLevel.PARENT,
    )
    return RetrievedChunk(chunk=chunk, fused_score=score)


def test_compress_to_budget_keeps_at_least_one_chunk_even_if_oversized():
    huge = _rc("word " * 10_000, score=1.0)
    kept = compress_to_budget([huge], token_budget=10)
    assert len(kept) == 1  # never returns empty just because the top chunk is big


def test_compress_to_budget_drops_low_ranked_chunks_past_budget():
    chunks = [_rc("word " * 100, score=1.0 - i * 0.1) for i in range(20)]
    kept = compress_to_budget(chunks, token_budget=500)
    assert len(kept) < len(chunks)
    # highest-scored chunk must be kept
    assert kept[0].fused_score == 1.0
