#!/usr/bin/env python3
"""
Essential test suite for MCP prompts functionality.
Tests only critical user-facing behavior.
"""

import json
import pytest
from local_faiss_mcp.server import list_prompts, get_prompt


class TestPrompts:
    """Essential tests for MCP prompt functionality."""

    @pytest.mark.asyncio
    async def test_list_prompts_returns_two_prompts(self):
        """Test that users can discover both available prompts."""
        prompts = await list_prompts()

        assert len(prompts) == 2

        prompt_names = [p.name for p in prompts]
        assert "extract-answer" in prompt_names
        assert "summarize-documents" in prompt_names

    @pytest.mark.asyncio
    async def test_extract_answer_includes_query_and_chunks(self):
        """Test that extract-answer prompt generates text with query and chunks."""
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

        message = await get_prompt("extract-answer", {
            "query": "What is FAISS?",
            "chunks": json.dumps(chunks)
        })

        # Verify message structure
        assert message.role == "user"
        assert hasattr(message.content, 'text')

        # Verify content includes query and all chunks
        text = message.content.text
        assert "What is FAISS?" in text
        assert "FAISS is a library for efficient similarity search." in text
        assert "FAISS was developed by Facebook AI Research." in text
        assert "faiss_docs.txt" in text
        assert "faiss_history.txt" in text

    @pytest.mark.asyncio
    async def test_summarize_includes_topic_and_chunks(self):
        """Test that summarize-documents prompt generates text with topic and chunks."""
        chunks = [
            {
                "text": "FAISS uses indexing structures for fast retrieval.",
                "source": "faiss_architecture.txt"
            },
            {
                "text": "FAISS supports multiple distance metrics.",
                "source": "faiss_metrics.txt"
            }
        ]

        message = await get_prompt("summarize-documents", {
            "topic": "FAISS architecture",
            "chunks": json.dumps(chunks),
            "max_length": "150"
        })

        # Verify message structure
        assert message.role == "user"
        assert hasattr(message.content, 'text')

        # Verify content includes topic and all chunks
        text = message.content.text
        assert "FAISS architecture" in text
        assert "150" in text  # max_length
        assert "FAISS uses indexing structures for fast retrieval." in text
        assert "FAISS supports multiple distance metrics." in text
        assert "faiss_architecture.txt" in text
        assert "faiss_metrics.txt" in text

    @pytest.mark.asyncio
    async def test_unknown_prompt_raises_error(self):
        """Test that requesting an unknown prompt raises an error."""
        with pytest.raises(ValueError, match="Unknown prompt"):
            await get_prompt("nonexistent-prompt", {
                "query": "test"
            })

    @pytest.mark.asyncio
    async def test_invalid_json_chunks_handled(self):
        """Test that invalid JSON in chunks doesn't crash the server."""
        # Should handle gracefully by treating as empty chunks
        message = await get_prompt("extract-answer", {
            "query": "What is FAISS?",
            "chunks": "this is not valid json"
        })

        # Should still generate a valid message
        assert message.role == "user"
        assert hasattr(message.content, 'text')

        # Should include the query even if chunks are invalid
        text = message.content.text
        assert "What is FAISS?" in text


if __name__ == '__main__':
    # Allow running tests directly
    pytest.main([__file__, '-v'])
