#!/usr/bin/env python3
"""
Essential tests for document parsing functionality.
"""

import os
import tempfile
import pytest
from pathlib import Path
from local_faiss_mcp.document_parser import (
    parse_text_file,
    parse_document,
    is_file_path
)


class TestDocumentParser:
    """Essential tests for document parsing."""

    def test_parse_text_file(self):
        """Test parsing plain text files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.\nWith multiple lines.")
            temp_path = f.name

        try:
            content = parse_text_file(Path(temp_path))
            assert "This is a test document" in content
            assert "multiple lines" in content
        finally:
            os.unlink(temp_path)

    def test_parse_markdown_file(self):
        """Test parsing markdown files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Heading\n\nThis is **bold** text.")
            temp_path = f.name

        try:
            content = parse_document(temp_path)
            assert "# Heading" in content
            assert "**bold**" in content
        finally:
            os.unlink(temp_path)

    def test_is_file_path_detection(self):
        """Test file path detection heuristic."""
        # Should detect as paths (files that exist)
        assert is_file_path(__file__) == True  # This test file exists

        # Should detect as paths (contain path separators and extensions)
        assert is_file_path("/path/to/file.txt") == True
        assert is_file_path("./relative/path.md") == True
        assert is_file_path("docs/readme.pdf") == True

        # Should NOT detect as paths (natural language, no separators)
        assert is_file_path("This is a long document with many words") == False
        assert is_file_path("FAISS is a library for similarity search") == False
        assert is_file_path("document.pdf") == False  # No separator, doesn't exist
        assert is_file_path("") == False

    def test_is_file_path_handles_very_long_text(self):
        """Test that is_file_path doesn't crash on very long text (regression test for 'file name too long' bug)."""
        # Simulate what happens when LLM passes a long document with path separators
        long_text = """
        This is a very long document that contains forward slashes / and might
        have URLs like https://example.com/very/long/path/to/something.
        """ * 1000  # Make it very long (>50,000 characters)

        # Should not crash with OSError, should return False
        result = is_file_path(long_text)
        assert result == False

        # Also test with backslashes (Windows-style paths)
        long_text_windows = "Some text with backslashes\\path\\to\\file " * 1000
        result = is_file_path(long_text_windows)
        assert result == False

    def test_is_file_path_length_limits(self):
        """Test that is_file_path respects reasonable path length limits."""
        # Paths over 2000 characters should be rejected immediately
        very_long_path = "a/" * 2000  # Creates a path > 2000 characters
        assert is_file_path(very_long_path) == False

        # Reasonable path under 500 chars should still work (if it has proper structure)
        reasonable_path = "/some/deeply/nested/" + "dir/" * 30 + "file.txt"
        # This path is ~160 characters, well under 500, should be detected
        assert len(reasonable_path) < 500
        assert is_file_path(reasonable_path) == True

    def test_parse_document_file_not_found(self):
        """Test that parsing non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            parse_document("/non/existent/file.txt")

    def test_parse_document_unsupported_format_without_pandoc(self):
        """Test parsing unsupported format without pandoc."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            f.write("content")
            temp_path = f.name

        try:
            # Should raise ValueError for unsupported format
            with pytest.raises(ValueError, match="Unsupported file format"):
                parse_document(temp_path)
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
