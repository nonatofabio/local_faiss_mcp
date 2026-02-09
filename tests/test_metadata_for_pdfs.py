from local_faiss_mcp.metadata_extraction import (
    build_metadata_for_pdf,
    build_metadata_for_pdfs,
    DocumentMetadata,
)


def test_build_metadata_for_pdf(test_pdfs):
    """Test that metadata extraction works for a single PDF."""
    name, path = next(iter(test_pdfs.items()))
    result = build_metadata_for_pdf(path)

    assert isinstance(result, DocumentMetadata)
    assert result.pages > 0
    assert result.word_count >= 0
    assert result.id == name
    assert result.path == str(path)
    assert result.document_type == "pdf"


def test_build_metadata_for_pdfs(test_pdfs):
    """Test batch metadata extraction for multiple PDFs."""
    paths = list(test_pdfs.values())
    results = build_metadata_for_pdfs(paths)

    assert len(results) == len(paths)
    for meta in results:
        assert isinstance(meta, DocumentMetadata)
        assert meta.pages > 0


def test_metadata_to_dict_roundtrip(test_pdfs):
    """Test that metadata serializes and deserializes correctly."""
    path = next(iter(test_pdfs.values()))
    original = build_metadata_for_pdf(path)

    as_dict = original.to_dict()
    restored = DocumentMetadata.from_dict(as_dict)

    assert restored.id == original.id
    assert restored.pages == original.pages
    assert restored.word_count == original.word_count
    assert restored.title == original.title
    assert restored.author == original.author
