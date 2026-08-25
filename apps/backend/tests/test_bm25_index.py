import pytest

from knowledge_assistant.domain.entities import Chunk, ChunkLevel
from knowledge_assistant.infrastructure.search.bm25_index import InMemoryBM25Index


def _child(user_id, collection_id, doc_id, text) -> Chunk:
    return Chunk(
        document_id=doc_id, user_id=user_id, collection_id=collection_id,
        text=text, level=ChunkLevel.CHILD, parent_id="parent-1",
    )


def _filler_chunks(user_id, collection_id, doc_id, n=4) -> list:
    topics = [
        "The quarterly budget review is scheduled for next Tuesday.",
        "Parking permits must be renewed annually at the front desk.",
        "The onboarding checklist includes IT setup and badge issuance.",
        "Team lunches are reimbursed up to twenty five dollars.",
    ]
    return [_child(user_id, collection_id, doc_id, t) for t in topics[:n]]


@pytest.mark.asyncio
async def test_search_finds_relevant_chunk():
    # Note: BM25 IDF is degenerate on a 2-doc corpus where each term appears
    # in exactly one doc (idf collapses to exactly 0 -- correct BM25 math,
    # not a bug). Filler chunks keep IDF non-degenerate, like a real corpus.
    index = InMemoryBM25Index()
    target = _child("u1", "c1", "d1", "Employees get 12 days of paid sick leave per year.")
    chunks = [target, *_filler_chunks("u1", "c1", "d1")]
    await index.index(chunks)

    results = await index.search("sick leave days", user_id="u1", collection_id="c1", top_k=5)
    assert results, "expected at least one hit"
    top_chunk_id = results[0][0]
    assert top_chunk_id == target.id


@pytest.mark.asyncio
async def test_search_is_isolated_per_user():
    index = InMemoryBM25Index()
    await index.index([
        _child("u1", "c1", "d1", "confidential salary information for u1"),
        *_filler_chunks("u1", "c1", "d1"),
    ])
    await index.index([
        _child("u2", "c1", "d2", "confidential salary information for u2"),
        *_filler_chunks("u2", "c1", "d2"),
    ])

    results = await index.search("salary information", user_id="u1", collection_id="c1", top_k=10)
    # only u1's chunk should ever be reachable via u1's search
    assert len(results) == 1


@pytest.mark.asyncio
async def test_delete_by_document_removes_chunks():
    index = InMemoryBM25Index()
    chunk = _child("u1", "c1", "d1", "unique searchable phrase xyzzy")
    await index.index([chunk, *_filler_chunks("u1", "c1", "d1")])
    assert await index.search("xyzzy", "u1", "c1", 5)

    await index.delete_by_document("d1")
    assert await index.search("xyzzy", "u1", "c1", 5) == []
