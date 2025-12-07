#!/usr/bin/env python3
"""
Essential test suite for re-ranking functionality.
Tests critical re-ranking behavior for users.
"""

import os
import tempfile
import pytest
from local_faiss_mcp.server import FAISSVectorStore


class TestReranking:
    """Essential tests for re-ranking functionality."""

    def test_reranker_initialization_with_model(self):
        """Test that re-ranker can be initialized with a model name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, 'test.index')
            metadata_path = os.path.join(tmpdir, 'metadata.json')

            # Initialize with re-ranker (using a small, fast model for testing)
            store = FAISSVectorStore(
                index_path=index_path,
                metadata_path=metadata_path,
                embedding_model_name='all-MiniLM-L6-v2',
                rerank_model_name='cross-encoder/ms-marco-TinyBERT-L-2-v2'
            )

            assert store.reranker is not None
            assert store.rerank_model_name == 'cross-encoder/ms-marco-TinyBERT-L-2-v2'

    def test_no_reranker_when_not_specified(self):
        """Test that re-ranker is None when not specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, 'test.index')
            metadata_path = os.path.join(tmpdir, 'metadata.json')

            store = FAISSVectorStore(
                index_path=index_path,
                metadata_path=metadata_path,
                embedding_model_name='all-MiniLM-L6-v2'
            )

            assert store.reranker is None
            assert store.rerank_model_name is None

    def test_query_with_reranking(self):
        """Test that queries with re-ranking include rerank scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, 'test.index')
            metadata_path = os.path.join(tmpdir, 'metadata.json')

            # Initialize with re-ranker
            store = FAISSVectorStore(
                index_path=index_path,
                metadata_path=metadata_path,
                embedding_model_name='all-MiniLM-L6-v2',
                rerank_model_name='cross-encoder/ms-marco-TinyBERT-L-2-v2'
            )

            # Ingest some test documents
            store.ingest("FAISS is a library for efficient similarity search.", "doc1.txt")
            store.ingest("Python is a programming language.", "doc2.txt")
            store.ingest("FAISS was developed by Facebook AI Research.", "doc3.txt")

            # Query with re-ranking
            results = store.query("What is FAISS?", top_k=2)

            assert len(results) > 0
            # Results should have rerank_score when re-ranker is enabled
            for result in results:
                assert 'rerank_score' in result
                assert isinstance(result['rerank_score'], float)
                # Rerank scores should be between 0 and 1 typically
                assert result['rerank_score'] is not None

    def test_query_without_reranking(self):
        """Test that queries without re-ranking don't include rerank scores."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, 'test.index')
            metadata_path = os.path.join(tmpdir, 'metadata.json')

            # Initialize without re-ranker
            store = FAISSVectorStore(
                index_path=index_path,
                metadata_path=metadata_path,
                embedding_model_name='all-MiniLM-L6-v2'
            )

            # Ingest some test documents
            store.ingest("FAISS is a library for efficient similarity search.", "doc1.txt")
            store.ingest("Python is a programming language.", "doc2.txt")

            # Query without re-ranking
            results = store.query("What is FAISS?", top_k=2)

            assert len(results) > 0
            # Results should NOT have rerank_score when re-ranker is disabled
            for result in results:
                assert 'rerank_score' not in result
                assert 'distance' in result  # But should have FAISS distance

    def test_reranking_improves_relevance(self):
        """Test that re-ranking can reorder results for better relevance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, 'test.index')
            metadata_path = os.path.join(tmpdir, 'metadata.json')

            # Initialize with re-ranker
            store_with_rerank = FAISSVectorStore(
                index_path=index_path,
                metadata_path=metadata_path,
                embedding_model_name='all-MiniLM-L6-v2',
                rerank_model_name='cross-encoder/ms-marco-TinyBERT-L-2-v2'
            )

            # Ingest documents
            store_with_rerank.ingest("FAISS is a library for efficient similarity search and clustering.", "doc1.txt")
            store_with_rerank.ingest("Unrelated content about cooking recipes.", "doc2.txt")
            store_with_rerank.ingest("FAISS provides fast nearest neighbor search.", "doc3.txt")

            # Query with re-ranking
            results = store_with_rerank.query("similarity search library", top_k=3)

            # The top result should be highly relevant
            assert len(results) > 0
            # Check that we got rerank scores
            assert 'rerank_score' in results[0]
            # Top result should be about FAISS (more relevant than cooking)
            assert "FAISS" in results[0]['text']


if __name__ == '__main__':
    # Allow running tests directly
    pytest.main([__file__, '-v'])
