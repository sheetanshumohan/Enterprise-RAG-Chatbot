from unittest.mock import AsyncMock, patch
import pytest

from knowledge_assistant.domain.entities import Chunk, ChunkLevel
from knowledge_assistant.infrastructure.search.opensearch_index import OpenSearchKeywordIndex


def _child(user_id, collection_id, doc_id, text) -> Chunk:
    return Chunk(
        document_id=doc_id,
        user_id=user_id,
        collection_id=collection_id,
        text=text,
        level=ChunkLevel.CHILD,
        parent_id="parent-1",
    )


@pytest.mark.asyncio
async def test_ensure_index_creates_mapping_when_missing():
    index = OpenSearchKeywordIndex(hosts="http://localhost:9200", index_name="test_chunks")
    mock_client = AsyncMock()
    mock_client.indices.exists.return_value = False
    mock_client.indices.create.return_value = {"acknowledged": True}
    index._client = mock_client

    await index._ensure_index()

    mock_client.indices.exists.assert_awaited_once_with(index="test_chunks")
    mock_client.indices.create.assert_awaited_once()
    assert index._index_initialized is True


@pytest.mark.asyncio
async def test_index_bulk_indexes_child_chunks():
    index = OpenSearchKeywordIndex(hosts="http://localhost:9200", index_name="test_chunks")
    mock_client = AsyncMock()
    mock_client.indices.exists.return_value = True
    index._client = mock_client
    index._index_initialized = True

    chunks = [
        _child("u1", "c1", "d1", "Child chunk text 1"),
        _child("u1", "c1", "d1", "Child chunk text 2"),
        Chunk(
            document_id="d1",
            user_id="u1",
            collection_id="c1",
            text="Parent chunk text",
            level=ChunkLevel.PARENT,
        ),
    ]

    with patch("opensearchpy.helpers.async_bulk", new_callable=AsyncMock) as mock_bulk:
        mock_bulk.return_value = (2, [])
        await index.index(chunks)

        mock_bulk.assert_awaited_once()
        args, kwargs = mock_bulk.await_args
        actions = args[1]
        assert len(actions) == 2, "Only child chunks should be indexed"
        assert actions[0]["_source"]["user_id"] == "u1"
        assert actions[0]["_source"]["collection_id"] == "c1"


@pytest.mark.asyncio
async def test_search_executes_user_isolated_bm25_query():
    index = OpenSearchKeywordIndex(hosts="http://localhost:9200", index_name="test_chunks")
    mock_client = AsyncMock()
    mock_client.indices.exists.return_value = True
    index._client = mock_client
    index._index_initialized = True

    mock_client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "chunk-1",
                    "_score": 12.5,
                    "_source": {"chunk_id": "chunk-1"},
                },
                {
                    "_id": "chunk-2",
                    "_score": 8.2,
                    "_source": {"chunk_id": "chunk-2"},
                },
            ]
        }
    }

    results = await index.search("sick leave", user_id="u1", collection_id="c1", top_k=5)

    mock_client.search.assert_awaited_once()
    _, kwargs = mock_client.search.await_args
    body = kwargs["body"]
    assert kwargs["index"] == "test_chunks"
    assert body["size"] == 5
    # Verify filter clauses isolate by user_id and collection_id
    filter_terms = body["query"]["bool"]["filter"]
    assert {"term": {"user_id": "u1"}} in filter_terms
    assert {"term": {"collection_id": "c1"}} in filter_terms
    assert {"term": {"level": "child"}} in filter_terms

    assert len(results) == 2
    assert results[0] == ("chunk-1", 12.5)
    assert results[1] == ("chunk-2", 8.2)


@pytest.mark.asyncio
async def test_delete_by_document_calls_delete_by_query():
    index = OpenSearchKeywordIndex(hosts="http://localhost:9200", index_name="test_chunks")
    mock_client = AsyncMock()
    mock_client.indices.exists.return_value = True
    index._client = mock_client
    index._index_initialized = True

    await index.delete_by_document("doc-123")

    mock_client.delete_by_query.assert_awaited_once_with(
        index="test_chunks",
        body={"query": {"term": {"document_id": "doc-123"}}},
        refresh=True,
    )
