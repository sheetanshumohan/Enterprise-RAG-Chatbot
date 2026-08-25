import pytest

from knowledge_assistant.application.use_cases.ingest_document import DuplicateDocumentError, IngestDocumentUseCase
from knowledge_assistant.domain.entities import DocumentStatus
from tests.fakes import FakeChunkRepository, FakeDocumentRepository, FakeEmbeddingClient, FakeKeywordIndex, FakeVectorStore


def _make_use_case():
    document_repo = FakeDocumentRepository()
    chunk_repo = FakeChunkRepository()
    vector_store = FakeVectorStore()
    keyword_index = FakeKeywordIndex()
    embedding_client = FakeEmbeddingClient()
    use_case = IngestDocumentUseCase(document_repo, chunk_repo, vector_store, keyword_index, embedding_client)
    return use_case, document_repo, chunk_repo, vector_store, keyword_index


@pytest.mark.asyncio
async def test_ingest_txt_document_end_to_end():
    use_case, document_repo, chunk_repo, vector_store, keyword_index = _make_use_case()
    text = ("Company policy overview. " * 5 + "\n\n") * 20
    document = await use_case.execute(
        user_id="u1", collection_id="c1", filename="policy.txt", raw_bytes=text.encode()
    )

    assert document.status == DocumentStatus.INDEXED
    assert document.metadata["chunk_count"] > 0

    chunks = await chunk_repo.list_by_document(document.id)
    assert any(c.level.value == "parent" for c in chunks)
    assert any(c.level.value == "child" for c in chunks)
    assert vector_store.upserted, "child chunks should have been upserted to the vector store"
    assert keyword_index.indexed, "child chunks should have been indexed for BM25"


@pytest.mark.asyncio
async def test_duplicate_upload_is_rejected():
    use_case, *_ = _make_use_case()
    raw = b"identical content for dedup test"
    await use_case.execute(user_id="u1", collection_id="c1", filename="a.txt", raw_bytes=raw)

    with pytest.raises(DuplicateDocumentError):
        await use_case.execute(user_id="u1", collection_id="c1", filename="a-renamed.txt", raw_bytes=raw)


@pytest.mark.asyncio
async def test_same_content_different_user_is_not_a_duplicate():
    use_case, *_ = _make_use_case()
    raw = b"shared content uploaded by two different users"
    doc1 = await use_case.execute(user_id="u1", collection_id="c1", filename="a.txt", raw_bytes=raw)
    doc2 = await use_case.execute(user_id="u2", collection_id="c1", filename="a.txt", raw_bytes=raw)
    assert doc1.id != doc2.id


@pytest.mark.asyncio
async def test_empty_document_marks_failed_not_silently_indexed():
    use_case, document_repo, *_ = _make_use_case()
    with pytest.raises(ValueError):
        await use_case.execute(user_id="u1", collection_id="c1", filename="empty.txt", raw_bytes=b"   \n\n  ")

    docs = await document_repo.list_for_user("u1")
    assert docs[0].status == DocumentStatus.FAILED


@pytest.mark.asyncio
async def test_delete_document_cleans_up_all_stores():
    use_case, document_repo, chunk_repo, vector_store, keyword_index = _make_use_case()
    document = await use_case.execute(
        user_id="u1", collection_id="c1", filename="a.txt", raw_bytes=b"Some content. " * 50
    )

    await use_case.delete_document(document.id, "u1")

    assert await document_repo.get(document.id, "u1") is None
    assert await chunk_repo.list_by_document(document.id) == []
    assert document.id in vector_store.deleted_documents
    assert document.id in keyword_index.deleted_documents


@pytest.mark.asyncio
async def test_delete_document_respects_user_isolation():
    use_case, document_repo, *_ = _make_use_case()
    document = await use_case.execute(
        user_id="u1", collection_id="c1", filename="a.txt", raw_bytes=b"Some content. " * 50
    )
    # a different user attempting to delete u1's document should be a no-op
    await use_case.delete_document(document.id, "u2")
    assert await document_repo.get(document.id, "u1") is not None
