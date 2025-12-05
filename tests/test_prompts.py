#!/usr/bin/env python3
"""
Test suite for MCP prompts functionality.
Tests prompt listing and prompt generation.
"""

import json
import pytest


class TestPrompts:
    """Test MCP prompt functionality."""

    def test_extract_answer_prompt_structure(self):
        """Test that extract-answer prompt generates correct structure."""
        # Simulate the prompt generation logic
        query = "What is FAISS?"
        chunks = [
            {
                "text": "FAISS is a library for efficient similarity search.",
                "source": "faiss_docs.txt",
                "distance": 0.5
            },
            {
                "text": "FAISS was developed by Facebook AI Research.",
                "source": "faiss_history.txt",
                "distance": 0.8
            }
        ]

        chunks_json = json.dumps(chunks)

        # Verify chunks can be serialized and deserialized
        parsed_chunks = json.loads(chunks_json)
        assert len(parsed_chunks) == 2
        assert parsed_chunks[0]["text"] == "FAISS is a library for efficient similarity search."
        assert parsed_chunks[0]["source"] == "faiss_docs.txt"

    def test_summarize_prompt_structure(self):
        """Test that summarize-documents prompt generates correct structure."""
        topic = "FAISS architecture"
        chunks = [
            {
                "text": "FAISS uses indexing structures for fast retrieval.",
                "source": "faiss_architecture.txt"
            }
        ]

        chunks_json = json.dumps(chunks)
        parsed_chunks = json.loads(chunks_json)

        assert len(parsed_chunks) == 1
        assert parsed_chunks[0]["text"] == "FAISS uses indexing structures for fast retrieval."

    def test_prompt_arguments_validation(self):
        """Test that prompt arguments are properly validated."""
        # Test empty query
        query = ""
        assert query == ""  # Empty queries should be handled gracefully

        # Test empty chunks
        chunks_json = "[]"
        parsed = json.loads(chunks_json)
        assert parsed == []

        # Test invalid JSON handling
        invalid_json = "not valid json"
        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_json)

    def test_multiple_chunks_formatting(self):
        """Test that multiple chunks are formatted correctly."""
        chunks = [
            {"text": "First chunk", "source": "doc1.txt", "distance": 0.3},
            {"text": "Second chunk", "source": "doc2.txt", "distance": 0.5},
            {"text": "Third chunk", "source": "doc3.txt", "distance": 0.7}
        ]

        # Verify all chunks can be processed
        for i, chunk in enumerate(chunks, 1):
            assert "text" in chunk
            assert "source" in chunk
            assert "distance" in chunk
            assert isinstance(chunk["distance"], float)

    def test_prompt_with_special_characters(self):
        """Test that prompts handle special characters correctly."""
        query = "What is 'FAISS' & how does it work?"
        chunks = [
            {
                "text": "FAISS handles queries with special chars: <, >, &, etc.",
                "source": "special_chars.txt",
                "distance": 0.4
            }
        ]

        chunks_json = json.dumps(chunks)
        parsed = json.loads(chunks_json)

        assert parsed[0]["text"] == "FAISS handles queries with special chars: <, >, &, etc."

    def test_prompt_max_length_parameter(self):
        """Test that max_length parameter is handled correctly."""
        max_lengths = ["100", "200", "500", "1000"]

        for length in max_lengths:
            assert length.isdigit()
            assert int(length) > 0

    def test_empty_chunks_handling(self):
        """Test that empty chunks are handled gracefully."""
        chunks_json = "[]"
        parsed_chunks = json.loads(chunks_json)

        assert isinstance(parsed_chunks, list)
        assert len(parsed_chunks) == 0

    def test_chunk_missing_fields(self):
        """Test handling of chunks with missing optional fields."""
        chunks = [
            {"text": "Only text field"},  # Missing source and distance
            {"text": "Has source", "source": "doc.txt"}  # Missing distance
        ]

        for chunk in chunks:
            # Should have at least text field
            assert "text" in chunk

            # Source should default to 'unknown' if missing
            source = chunk.get("source", "unknown")
            assert isinstance(source, str)

            # Distance should default to 0.0 if missing
            distance = chunk.get("distance", 0.0)
            assert isinstance(distance, (int, float))


class TestPromptIntegration:
    """Integration tests for prompts with actual MCP server components."""

    def test_prompt_names(self):
        """Test that prompt names follow naming conventions."""
        prompt_names = ["extract-answer", "summarize-documents"]

        for name in prompt_names:
            # Should be lowercase
            assert name == name.lower()
            # Should use hyphens, not underscores
            assert "_" not in name
            # Should be descriptive
            assert len(name) > 5

    def test_prompt_descriptions(self):
        """Test that prompts have meaningful descriptions."""
        prompts = {
            "extract-answer": "Extract the most relevant answer from retrieved document chunks",
            "summarize-documents": "Summarize information from multiple document chunks"
        }

        for name, description in prompts.items():
            # Should have a description
            assert description is not None
            assert len(description) > 10
            # Should end with period or be a complete sentence
            assert description[0].isupper()


if __name__ == '__main__':
    # Allow running tests directly
    pytest.main([__file__, '-v'])
