import pytest

from knowledge_assistant.domain.entities import Chunk, ChunkLevel, RetrievedChunk
from knowledge_assistant.infrastructure.reranking.reranker import LexicalOverlapReranker


def _rc(text: str, fused_score: float = 0.1) -> RetrievedChunk:
    chunk = Chunk(document_id="d1", user_id="u1", collection_id="c1", text=text, level=ChunkLevel.PARENT)
    return RetrievedChunk(chunk=chunk, fused_score=fused_score)


@pytest.mark.asyncio
async def test_reranker_prefers_higher_lexical_overlap():
    query = "sick leave policy days"
    candidates = [
        _rc("The office kitchen is stocked with coffee and snacks."),
        _rc("Employees receive twelve sick leave days per policy year."),
    ]
    ranked = await LexicalOverlapReranker().rerank(query, candidates)
    assert ranked[0].chunk.text.startswith("Employees receive")
    assert ranked[0].rerank_score > ranked[1].rerank_score


@pytest.mark.asyncio
async def test_reranker_handles_empty_candidate_list():
    ranked = await LexicalOverlapReranker().rerank("anything", [])
    assert ranked == []


@pytest.mark.asyncio
async def test_reranker_score_not_diluted_by_large_parent_chunks():
    """Regression test: the old Jaccard (overlap/union) formula collapsed
    toward the same tiny score once chunks were expanded to full
    parent-sized text, because the union term balloons with chunk length.
    Coverage-based scoring (overlap / query length) must stay meaningfully
    different between a strong match and a weak one, even when both chunks
    are long."""
    query = "sick leave policy"
    filler = ("Unrelated filler sentence about something else entirely. " * 40)
    strong_match = f"{filler} Our sick leave policy grants generous time off. {filler}"
    weak_match = f"{filler} The parking policy requires a permit. {filler}"

    candidates = [_rc(weak_match, fused_score=0.02), _rc(strong_match, fused_score=0.02)]
    ranked = await LexicalOverlapReranker().rerank(query, candidates)

    assert ranked[0].chunk.text == strong_match
    # the two scores must be clearly distinguishable, not collapsed together
    assert ranked[0].rerank_score - ranked[1].rerank_score > 0.1
