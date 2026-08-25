from knowledge_assistant.domain.entities import Document, DocumentType
from knowledge_assistant.infrastructure.chunking.chunker import build_parent_child_chunks, count_tokens


def _make_doc() -> Document:
    return Document(
        user_id="u1", collection_id="c1", filename="policy.txt",
        doc_type=DocumentType.TXT, content_hash="abc123",
    )


def test_empty_text_returns_no_chunks():
    assert build_parent_child_chunks(_make_doc(), "") == []


def test_produces_parent_and_child_chunks():
    text = "\n\n".join([f"Paragraph {i}. " + ("Sentence filler content here. " * 20) for i in range(10)])
    chunks = build_parent_child_chunks(_make_doc(), text, parent_max_tokens=200, child_max_tokens=50)

    parents = [c for c in chunks if c.level.value == "parent"]
    children = [c for c in chunks if c.level.value == "child"]

    assert len(parents) > 1, "long doc should split into multiple parent sections"
    assert len(children) > len(parents), "each parent should yield multiple children"
    for child in children:
        assert child.parent_id in {p.id for p in parents}
        assert count_tokens(child.text) <= 60  # small overlap tolerance


def test_single_short_paragraph_stays_as_one_chunk():
    text = "This is a short single paragraph document."
    chunks = build_parent_child_chunks(_make_doc(), text)
    parents = [c for c in chunks if c.level.value == "parent"]
    children = [c for c in chunks if c.level.value == "child"]
    assert len(parents) == 1
    assert len(children) == 1
    assert children[0].parent_id == parents[0].id
