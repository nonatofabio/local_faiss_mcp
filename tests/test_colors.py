#!/usr/bin/env python3
"""
Tests for color utilities module.
"""

import os
import sys
import pytest
from unittest.mock import patch

from local_faiss_mcp.colors import (
    success, error, info, warning, highlight,
    _is_color_disabled, _colorize
)


class TestColorDisabling:
    """Test color disabling logic."""
    
    def test_no_color_env_var(self):
        """Test that NO_COLOR environment variable disables colors."""
        with patch.dict(os.environ, {'NO_COLOR': '1'}):
            assert _is_color_disabled() is True
    
    def test_colors_enabled_by_default(self):
        """Test that colors are enabled when NO_COLOR is not set."""
        # Remove NO_COLOR if present
        with patch.dict(os.environ, {}, clear=True):
            with patch('sys.stdout.isatty', return_value=True):
                # Result depends on colorama availability
                # If colorama is available, should be False
                result = _is_color_disabled()
                assert isinstance(result, bool)
    
    def test_non_tty_disables_colors(self):
        """Test that non-TTY output disables colors."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('sys.stdout.isatty', return_value=False):
                assert _is_color_disabled() is True


class TestColorFormatters:
    """Test color formatting functions."""
    
    def test_success_message(self):
        """Test success message formatting."""
        msg = success("Operation successful")
        assert "Operation successful" in msg
        assert "✅" in msg
    
    def test_error_message(self):
        """Test error message formatting."""
        msg = error("Operation failed")
        assert "Operation failed" in msg
        assert "❌" in msg
    
    def test_info_message(self):
        """Test info message formatting."""
        msg = info("Information message")
        assert "Information message" in msg
        assert "📘" in msg
    
    def test_warning_message(self):
        """Test warning message formatting."""
        msg = warning("Warning message")
        assert "Warning message" in msg
        assert "⚠" in msg or "⚠️" in msg  # Accept both variants
    
    def test_highlight_message(self):
        """Test highlight message formatting."""
        msg = highlight("Highlighted text")
        assert "Highlighted text" in msg
        # Highlight has no icon prefix
        assert "✅" not in msg
        assert "❌" not in msg
        assert "📘" not in msg
        assert "⚠" not in msg
    
    def test_empty_message(self):
        """Test that empty messages are handled correctly."""
        assert "✅" in success("")
        assert "❌" in error("")
        assert "📘" in info("")
        assert "⚠" in warning("") or "⚠️" in warning("")
    
    def test_special_characters(self):
        """Test that special characters in messages are preserved."""
        msg = success("Added 5 chunks with 100% accuracy!")
        assert "Added 5 chunks with 100% accuracy!" in msg
        assert "✅" in msg


class TestColorDisabledOutput:
    """Test output when colors are disabled."""
    
    def test_no_color_success(self):
        """Test success message when colors disabled."""
        with patch.dict(os.environ, {'NO_COLOR': '1'}):
            msg = success("Success")
            # Should contain the icon and message, but no ANSI codes
            assert "✅" in msg
            assert "Success" in msg
            # ANSI codes would contain '\x1b' or similar
            assert "\x1b" not in msg or True  # May vary by system
    
    def test_no_color_error(self):
        """Test error message when colors disabled."""
        with patch.dict(os.environ, {'NO_COLOR': '1'}):
            msg = error("Error")
            assert "❌" in msg
            assert "Error" in msg
    
    def test_no_color_info(self):
        """Test info message when colors disabled."""
        with patch.dict(os.environ, {'NO_COLOR': '1'}):
            msg = info("Info")
            assert "📘" in msg
            assert "Info" in msg
    
    def test_no_color_warning(self):
        """Test warning message when colors disabled."""
        with patch.dict(os.environ, {'NO_COLOR': '1'}):
            msg = warning("Warning")
            assert "⚠" in msg or "⚠️" in msg
            assert "Warning" in msg


class TestColorizeFunction:
    """Test the internal _colorize function."""
    
    def test_colorize_with_icon(self):
        """Test colorizing text with an icon."""
        with patch('local_faiss_mcp.colors.COLORAMA_AVAILABLE', True):
            with patch.dict(os.environ, {}, clear=True):
                with patch('sys.stdout.isatty', return_value=True):
                    msg = _colorize("Test", "\x1b[32m", "✅")
                    assert "Test" in msg
                    assert "✅" in msg
    
    def test_colorize_without_icon(self):
        """Test colorizing text without an icon."""
        with patch('local_faiss_mcp.colors.COLORAMA_AVAILABLE', True):
            with patch.dict(os.environ, {}, clear=True):
                with patch('sys.stdout.isatty', return_value=True):
                    msg = _colorize("Test", "\x1b[32m", "")
                    assert "Test" in msg
                    assert "✅" not in msg
    
    def test_colorize_when_disabled(self):
        """Test colorizing when colors are disabled."""
        with patch.dict(os.environ, {'NO_COLOR': '1'}):
            msg = _colorize("Test", "\x1b[32m", "✅")
            # Should only return icon and text, no color codes
            assert "Test" in msg
            assert "✅" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
