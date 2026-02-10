#!/usr/bin/env python3
"""
Tests for the MCP-native memory protocol: recall/remember tools and memory-protocol prompt.
"""

import pytest

from local_faiss_mcp.server import (
    list_tools,
    call_tool,
    get_prompt,
    FAISSVectorStore,
)

# Module-level vector store for tool tests
import local_faiss_mcp.server as server_module


@pytest.fixture(autouse=True)
def setup_vector_store(tmp_path):
    """Set up a temporary vector store for each test."""
    index_path = tmp_path / "faiss.index"
    metadata_path = tmp_path / "metadata.json"

    server_module.vector_store = FAISSVectorStore(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
    )
    yield
    server_module.vector_store = None


class TestMemoryTools:
    """Tests for recall and remember tool aliases."""

    @pytest.mark.asyncio
    async def test_tools_list_includes_recall_and_remember(self):
        """Test that recall and remember appear in the tools list."""
        tools = await list_tools()
        tool_names = [t.name for t in tools]

        assert "recall" in tool_names
        assert "remember" in tool_names
        # Originals still present
        assert "ingest_document" in tool_names
        assert "query_rag_store" in tool_names

    @pytest.mark.asyncio
    async def test_remember_stores_and_recall_retrieves(self):
        """Test the full remember -> recall round trip."""
        # Store a memory
        result = await call_tool("remember", {
            "document": "We decided to use PostgreSQL for the user database because of JSONB support.",
            "source": "decision:use-postgres",
        })
        assert len(result) == 1
        assert "Successfully ingested" in result[0].text
        assert "decision:use-postgres" in result[0].text

        # Recall it
        result = await call_tool("recall", {
            "query": "which database did we choose",
            "top_k": 1,
        })
        assert len(result) == 1
        assert "PostgreSQL" in result[0].text

    @pytest.mark.asyncio
    async def test_remember_default_source_is_memory(self):
        """Test that remember defaults source to 'memory' not 'unknown'."""
        result = await call_tool("remember", {
            "document": "User prefers tabs over spaces.",
        })
        assert "memory" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_recall_empty_store(self):
        """Test that recall handles an empty store gracefully."""
        result = await call_tool("recall", {
            "query": "anything at all",
        })
        assert "No results found" in result[0].text or "empty" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_remember_is_compatible_with_query_rag_store(self):
        """Test that data stored via remember is retrievable via query_rag_store."""
        await call_tool("remember", {
            "document": "The auth timeout bug was caused by a missing token refresh.",
            "source": "bug-fix:auth-timeout",
        })

        result = await call_tool("query_rag_store", {
            "query": "auth timeout",
            "top_k": 1,
        })
        assert "token refresh" in result[0].text

    @pytest.mark.asyncio
    async def test_ingest_document_is_retrievable_via_recall(self):
        """Test that data stored via ingest_document is retrievable via recall."""
        await call_tool("ingest_document", {
            "document": "MCP servers communicate over stdio transport.",
            "source": "mcp-docs",
        })

        result = await call_tool("recall", {
            "query": "how do MCP servers communicate",
            "top_k": 1,
        })
        assert "stdio" in result[0].text


class TestMemoryProtocolPrompt:
    """Tests for the memory-protocol MCP prompt."""

    @pytest.mark.asyncio
    async def test_memory_protocol_prompt_returns_valid_message(self):
        """Test that the memory-protocol prompt returns a well-formed message."""
        message = await get_prompt("memory-protocol", {})

        assert message.role == "user"
        assert hasattr(message.content, "text")

    @pytest.mark.asyncio
    async def test_memory_protocol_contains_key_sections(self):
        """Test that the prompt includes the core protocol rules."""
        message = await get_prompt("memory-protocol", {})
        text = message.content.text

        assert "Memory First" in text
        assert "Auto-Archive" in text
        assert "recall" in text
        assert "remember" in text

    @pytest.mark.asyncio
    async def test_memory_protocol_contains_conflict_resolution(self):
        """Test that the prompt addresses how to handle conflicts."""
        message = await get_prompt("memory-protocol", {})
        text = message.content.text

        assert "Conflict" in text
        assert "codebase" in text.lower()

    @pytest.mark.asyncio
    async def test_memory_protocol_no_arguments_required(self):
        """Test that the prompt works with None arguments."""
        message = await get_prompt("memory-protocol", None)

        assert message.role == "user"
        assert len(message.content.text) > 0
