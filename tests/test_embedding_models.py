#!/usr/bin/env python3
"""
Test suite for custom embedding model functionality.
Tests various embedding models and dimension compatibility.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path

# Import from the package
from local_faiss_mcp import FAISSVectorStore


class TestEmbeddingModels:
    """Test custom embedding model functionality."""

    def test_default_model(self):
        """Test that the default embedding model works correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, 'test.index')
            metadata_path = os.path.join(tmpdir, 'metadata.json')

            # Create store with default model
            store = FAISSVectorStore(
                index_path=index_path,
                metadata_path=metadata_path
            )

            # Verify default model
            assert store.embedding_model_name == 'all-MiniLM-L6-v2'
            assert store.dimension == 384

            # Test ingestion
            result = store.ingest('This is a test document about FAISS.', 'test.txt')
            assert result['success'] is True
            assert result['chunks_added'] >= 1
            assert result['total_documents'] >= 1

    def test_custom_model_384_dims(self):
        """Test using a custom model with 384 dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, 'test.index')
            metadata_path = os.path.join(tmpdir, 'metadata.json')

            # Create store with custom model (same dimensions as default)
            store = FAISSVectorStore(
                index_path=index_path,
                metadata_path=metadata_path,
                embedding_model_name='all-MiniLM-L12-v2'
            )

            # Verify custom model
            assert store.embedding_model_name == 'all-MiniLM-L12-v2'
            assert store.dimension == 384

            # Test ingestion and query
            result = store.ingest('Vector databases store embeddings efficiently.', 'test.txt')
            assert result['success'] is True

            results = store.query('vector databases', top_k=1)
            assert len(results) >= 1
            assert results[0]['distance'] >= 0.0  # Distance should be non-negative

    def test_custom_model_768_dims(self):
        """Test using a custom model with 768 dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, 'test.index')
            metadata_path = os.path.join(tmpdir, 'metadata.json')

            # Create store with larger model
            store = FAISSVectorStore(
                index_path=index_path,
                metadata_path=metadata_path,
                embedding_model_name='all-mpnet-base-v2'
            )

            # Verify model dimensions
            assert store.embedding_model_name == 'all-mpnet-base-v2'
            assert store.dimension == 768

            # Test basic functionality
            result = store.ingest('FAISS enables fast similarity search.', 'test.txt')
            assert result['success'] is True

    def test_dimension_mismatch_detection(self):
        """Test that dimension mismatches are detected and prevented."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, 'test.index')
            metadata_path = os.path.join(tmpdir, 'metadata.json')

            # Create index with first model (384 dims)
            store1 = FAISSVectorStore(
                index_path=index_path,
                metadata_path=metadata_path,
                embedding_model_name='all-MiniLM-L6-v2'
            )
            store1.ingest('Test document', 'test.txt')

            # Verify the index was created
            assert os.path.exists(index_path)
            assert store1.dimension == 384

            # Try to load with different model (768 dims) - should raise ValueError
            with pytest.raises(ValueError) as exc_info:
                store2 = FAISSVectorStore(
                    index_path=index_path,
                    metadata_path=metadata_path,
                    embedding_model_name='all-mpnet-base-v2'
                )

            # Verify error message is informative
            error_message = str(exc_info.value)
            assert 'dimension' in error_message.lower()
            assert '384' in error_message
            assert '768' in error_message

    def test_model_persistence_reload(self):
        """Test that we can reload an index with the same model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, 'test.index')
            metadata_path = os.path.join(tmpdir, 'metadata.json')

            # Create and populate store
            store1 = FAISSVectorStore(
                index_path=index_path,
                metadata_path=metadata_path,
                embedding_model_name='all-MiniLM-L12-v2'
            )

            test_text = 'FAISS is a library for similarity search.'
            result1 = store1.ingest(test_text, 'doc1.txt')
            assert result1['success'] is True
            original_count = result1['total_documents']

            # Reload with same model
            store2 = FAISSVectorStore(
                index_path=index_path,
                metadata_path=metadata_path,
                embedding_model_name='all-MiniLM-L12-v2'
            )

            # Verify data persisted
            assert store2.embedding_model_name == 'all-MiniLM-L12-v2'
            assert store2.dimension == 384
            assert len(store2.metadata['documents']) == original_count

            # Verify query works on reloaded store
            results = store2.query('similarity search', top_k=1)
            assert len(results) >= 1

    def test_metadata_stores_model_name(self):
        """Test that the model name is stored in metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, 'test.index')
            metadata_path = os.path.join(tmpdir, 'metadata.json')

            # Create store with custom model
            store = FAISSVectorStore(
                index_path=index_path,
                metadata_path=metadata_path,
                embedding_model_name='all-MiniLM-L12-v2'
            )

            # Verify metadata contains model info
            assert 'model' in store.metadata
            assert store.metadata['model'] == 'all-MiniLM-L12-v2'

    def test_same_dimension_different_models(self):
        """Test that we can switch between models with same dimensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, 'test.index')
            metadata_path = os.path.join(tmpdir, 'metadata.json')

            # Create index with first 384-dim model
            store1 = FAISSVectorStore(
                index_path=index_path,
                metadata_path=metadata_path,
                embedding_model_name='all-MiniLM-L6-v2'
            )
            store1.ingest('Test document', 'test.txt')

            # Load with different 384-dim model - should work (same dimensions)
            # Note: In practice this might give unexpected results, but technically allowed
            store2 = FAISSVectorStore(
                index_path=index_path,
                metadata_path=metadata_path,
                embedding_model_name='all-MiniLM-L12-v2'
            )

            # Both have 384 dimensions, so no error
            assert store2.dimension == 384


class TestEmbeddingModelCLI:
    """Test CLI argument parsing for embedding models."""

    def test_embed_flag_parsing(self):
        """Test that --embed flag is parsed correctly."""
        import argparse

        # Simulate CLI parsing
        parser = argparse.ArgumentParser()
        parser.add_argument('--index-dir', type=str, default='.')
        parser.add_argument('--embed', type=str, default='all-MiniLM-L6-v2')

        # Test default
        args = parser.parse_args([])
        assert args.embed == 'all-MiniLM-L6-v2'

        # Test custom model
        args = parser.parse_args(['--embed', 'all-mpnet-base-v2'])
        assert args.embed == 'all-mpnet-base-v2'

        # Test with index-dir
        args = parser.parse_args(['--index-dir', '/tmp/test', '--embed', 'custom-model'])
        assert args.index_dir == '/tmp/test'
        assert args.embed == 'custom-model'


if __name__ == '__main__':
    # Allow running tests directly
    pytest.main([__file__, '-v'])
