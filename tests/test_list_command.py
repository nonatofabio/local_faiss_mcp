#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the `local-faiss list` command.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from local_faiss_mcp.cli import cmd_list


class MockArgs:
    """Mock args object for testing."""
    def __init__(self, json_output=False):
        self.json = json_output


@pytest.fixture
def temp_index_dir():
    """Create a temporary directory for test index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config(temp_index_dir):
    """Mock get_faiss_config to return temp directory."""
    config = {
        'index_dir': str(temp_index_dir),
        'embed_model': 'all-MiniLM-L6-v2',
        'rerank_model': None
    }
    with patch('local_faiss_mcp.cli.get_faiss_config', return_value=config):
        yield config


class TestListEmptyIndex:
    """Tests for listing an empty index."""

    def test_list_no_metadata_file(self, mock_config, capsys):
        """Test list when no metadata file exists."""
        args = MockArgs(json_output=False)
        result = cmd_list(args)
        
        assert result == 0
        captured = capsys.readouterr()
        assert "No documents indexed yet" in captured.out

    def test_list_no_metadata_file_json(self, mock_config, capsys):
        """Test list with --json when no metadata file exists."""
        args = MockArgs(json_output=True)
        result = cmd_list(args)
        
        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output == {"documents": [], "total": 0}

    def test_list_empty_documents_array(self, mock_config, temp_index_dir, capsys):
        """Test list when metadata exists but documents array is empty."""
        metadata_path = temp_index_dir / "metadata.json"
        metadata_path.write_text(json.dumps({"documents": [], "model": "test"}))
        
        args = MockArgs(json_output=False)
        result = cmd_list(args)
        
        assert result == 0
        captured = capsys.readouterr()
        assert "No documents indexed yet" in captured.out


class TestListWithDocuments:
    """Tests for listing indexed documents."""

    def test_list_single_document(self, mock_config, temp_index_dir, capsys):
        """Test listing a single document."""
        metadata = {
            "documents": [
                {"id": 0, "source": "test.pdf", "text": "chunk 1"}
            ],
            "model": "test"
        }
        metadata_path = temp_index_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata))
        
        args = MockArgs(json_output=False)
        result = cmd_list(args)
        
        assert result == 0
        captured = capsys.readouterr()
        assert "Indexed Documents (1 total)" in captured.out
        assert "test.pdf" in captured.out
        assert "Chunks: 1" in captured.out

    def test_list_multiple_documents_grouped(self, mock_config, temp_index_dir, capsys):
        """Test listing multiple documents with chunk grouping."""
        metadata = {
            "documents": [
                {"id": 0, "source": "doc1.pdf", "text": "chunk 1"},
                {"id": 1, "source": "doc1.pdf", "text": "chunk 2"},
                {"id": 2, "source": "doc1.pdf", "text": "chunk 3"},
                {"id": 3, "source": "doc2.md", "text": "chunk 1"},
                {"id": 4, "source": "doc2.md", "text": "chunk 2"},
            ],
            "model": "test"
        }
        metadata_path = temp_index_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata))
        
        args = MockArgs(json_output=False)
        result = cmd_list(args)
        
        assert result == 0
        captured = capsys.readouterr()
        assert "Indexed Documents (2 total)" in captured.out
        assert "doc1.pdf" in captured.out
        assert "Chunks: 3" in captured.out
        assert "doc2.md" in captured.out
        assert "Chunks: 2" in captured.out


class TestListJsonOutput:
    """Tests for JSON output format."""

    def test_list_json_single_document(self, mock_config, temp_index_dir, capsys):
        """Test JSON output with single document."""
        metadata = {
            "documents": [
                {"id": 0, "source": "test.pdf", "text": "chunk 1"},
                {"id": 1, "source": "test.pdf", "text": "chunk 2"}
            ],
            "model": "test"
        }
        metadata_path = temp_index_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata))
        
        args = MockArgs(json_output=True)
        result = cmd_list(args)
        
        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        
        assert output["total"] == 1
        assert len(output["documents"]) == 1
        assert output["documents"][0]["source"] == "test.pdf"
        assert output["documents"][0]["chunks"] == 2

    def test_list_json_multiple_documents(self, mock_config, temp_index_dir, capsys):
        """Test JSON output with multiple documents."""
        metadata = {
            "documents": [
                {"id": 0, "source": "a_file.txt", "text": "chunk 1"},
                {"id": 1, "source": "b_file.md", "text": "chunk 1"},
                {"id": 2, "source": "b_file.md", "text": "chunk 2"},
                {"id": 3, "source": "c_file.pdf", "text": "chunk 1"},
            ],
            "model": "test"
        }
        metadata_path = temp_index_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata))
        
        args = MockArgs(json_output=True)
        result = cmd_list(args)
        
        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        
        assert output["total"] == 3
        # Verify alphabetical sorting
        sources = [doc["source"] for doc in output["documents"]]
        assert sources == ["a_file.txt", "b_file.md", "c_file.pdf"]
        # Verify chunk counts
        chunks = {doc["source"]: doc["chunks"] for doc in output["documents"]}
        assert chunks["a_file.txt"] == 1
        assert chunks["b_file.md"] == 2
        assert chunks["c_file.pdf"] == 1

    def test_list_json_with_indexed_at(self, mock_config, temp_index_dir, capsys):
        """Test JSON output includes indexed_at field."""
        metadata = {
            "documents": [
                {"id": 0, "source": "test.pdf", "text": "chunk 1", "indexed_at": "2024-12-08T14:23:00"},
                {"id": 1, "source": "test.pdf", "text": "chunk 2", "indexed_at": "2024-12-08T14:23:00"}
            ],
            "model": "test"
        }
        metadata_path = temp_index_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata))
        
        args = MockArgs(json_output=True)
        result = cmd_list(args)
        
        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        
        assert output["documents"][0]["indexed_at"] == "2024-12-08T14:23:00"


class TestListWithIndexedDate:
    """Tests for indexed date display."""

    def test_list_shows_indexed_date(self, mock_config, temp_index_dir, capsys):
        """Test that human-readable output shows indexed date."""
        metadata = {
            "documents": [
                {"id": 0, "source": "test.pdf", "text": "chunk 1", "indexed_at": "2024-12-08T14:23:00"}
            ],
            "model": "test"
        }
        metadata_path = temp_index_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata))
        
        args = MockArgs(json_output=False)
        result = cmd_list(args)
        
        assert result == 0
        captured = capsys.readouterr()
        assert "Indexed: 2024-12-08 14:23" in captured.out

    def test_list_handles_missing_indexed_at(self, mock_config, temp_index_dir, capsys):
        """Test that list works when indexed_at is missing (backward compatible)."""
        metadata = {
            "documents": [
                {"id": 0, "source": "test.pdf", "text": "chunk 1"}  # No indexed_at
            ],
            "model": "test"
        }
        metadata_path = temp_index_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata))

        args = MockArgs(json_output=False)
        result = cmd_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "test.pdf" in captured.out
        assert "Indexed:" not in captured.out  # Should not show "Indexed:" line

    def test_list_keeps_most_recent_indexed_at(self, mock_config, temp_index_dir, capsys):
        """Test that when a source has multiple chunks with different timestamps, the most recent is kept."""
        metadata = {
            "documents": [
                {"id": 0, "source": "test.pdf", "text": "chunk 1", "indexed_at": "2024-12-08T10:00:00"},
                {"id": 1, "source": "test.pdf", "text": "chunk 2", "indexed_at": "2024-12-08T15:30:00"},  # Most recent
                {"id": 2, "source": "test.pdf", "text": "chunk 3", "indexed_at": "2024-12-08T12:00:00"}
            ],
            "model": "test"
        }
        metadata_path = temp_index_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata))

        args = MockArgs(json_output=False)
        result = cmd_list(args)

        assert result == 0
        captured = capsys.readouterr()
        # Should show the most recent timestamp (15:30), not the first or middle one
        assert "Indexed: 2024-12-08 15:30" in captured.out
        assert "Chunks: 3" in captured.out

    def test_list_keeps_most_recent_indexed_at_json(self, mock_config, temp_index_dir, capsys):
        """Test that JSON output includes the most recent indexed_at."""
        metadata = {
            "documents": [
                {"id": 0, "source": "test.pdf", "text": "chunk 1", "indexed_at": "2024-12-08T10:00:00"},
                {"id": 1, "source": "test.pdf", "text": "chunk 2", "indexed_at": "2024-12-08T15:30:00"},
                {"id": 2, "source": "test.pdf", "text": "chunk 3", "indexed_at": "2024-12-08T12:00:00"}
            ],
            "model": "test"
        }
        metadata_path = temp_index_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata))

        args = MockArgs(json_output=True)
        result = cmd_list(args)

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output["documents"][0]["indexed_at"] == "2024-12-08T15:30:00"

    def test_list_invalid_date_format_fallback(self, mock_config, temp_index_dir, capsys):
        """Test that invalid date formats are handled gracefully."""
        metadata = {
            "documents": [
                {"id": 0, "source": "test.pdf", "text": "chunk 1", "indexed_at": "not-a-valid-date"}
            ],
            "model": "test"
        }
        metadata_path = temp_index_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata))

        args = MockArgs(json_output=False)
        result = cmd_list(args)

        assert result == 0
        captured = capsys.readouterr()
        # Should show raw value when parsing fails
        assert "Indexed: not-a-valid-date" in captured.out


class TestListEdgeCases:
    """Tests for edge cases and error handling."""

    def test_list_missing_source_fallback(self, mock_config, temp_index_dir, capsys):
        """Test that documents without 'source' field default to 'unknown'."""
        metadata = {
            "documents": [
                {"id": 0, "text": "chunk 1"},  # No source field
                {"id": 1, "text": "chunk 2"}
            ],
            "model": "test"
        }
        metadata_path = temp_index_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata))

        args = MockArgs(json_output=False)
        result = cmd_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "unknown" in captured.out
        assert "Chunks: 2" in captured.out

    def test_list_missing_source_fallback_json(self, mock_config, temp_index_dir, capsys):
        """Test that JSON output handles missing source field."""
        metadata = {
            "documents": [
                {"id": 0, "text": "chunk 1"},  # No source field
                {"id": 1, "text": "chunk 2"}
            ],
            "model": "test"
        }
        metadata_path = temp_index_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata))

        args = MockArgs(json_output=True)
        result = cmd_list(args)

        assert result == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert output["documents"][0]["source"] == "unknown"
        assert output["documents"][0]["chunks"] == 2

    def test_list_malformed_metadata_json(self, mock_config, temp_index_dir, capsys):
        """Test that malformed JSON is handled gracefully."""
        metadata_path = temp_index_dir / "metadata.json"
        # Write invalid JSON
        metadata_path.write_text("{invalid json content")

        args = MockArgs(json_output=False)
        result = cmd_list(args)

        # Should return error code
        assert result == 1
        captured = capsys.readouterr()
        assert "Failed to read metadata" in captured.err

