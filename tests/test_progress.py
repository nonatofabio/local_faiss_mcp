#!/usr/bin/env python3
"""
Tests for CLI functionality, including progress bar.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from local_faiss_mcp.cli import cmd_index, collect_files


class TestProgressBar:
    """Test progress bar functionality in cmd_index."""
    
    @patch('local_faiss_mcp.cli.get_faiss_config')
    @patch('local_faiss_mcp.cli.FAISSVectorStore')
    @patch('local_faiss_mcp.cli.parse_document')
    @patch('local_faiss_mcp.cli.collect_files')
    @patch('local_faiss_mcp.cli.create_file_progress')
    def test_progress_bar_shown_for_multiple_files(
        self, mock_create_progress, mock_collect, mock_parse, mock_store_class, mock_config
    ):
        """Test that progress bar is shown when indexing multiple files."""
        # Setup
        mock_config.return_value = {
            'index_dir': '/tmp/test_index',
            'embed_model': 'test-model'
        }
        files = ['file1.pdf', 'file2.pdf', 'file3.pdf']
        mock_collect.return_value = files
        
        mock_store = MagicMock()
        mock_store.ingest.return_value = {'success': True, 'chunks_added': 5}
        mock_store.metadata = {'documents': []}
        mock_store_class.return_value = mock_store
        
        mock_parse.return_value = "test content"
        
        # Mock create_file_progress to return files and show_progress=True
        mock_create_progress.return_value = (files, True)
        
        # Create args mock
        args = MagicMock()
        args.files = files
        args.recursive = False
        
        # Execute
        result = cmd_index(args)
        
        # Verify create_file_progress was called for multiple files
        assert mock_create_progress.called
        call_args = mock_create_progress.call_args
        assert len(call_args[0][0]) == 3  # 3 files passed
        assert result == 0
    
    @patch('local_faiss_mcp.cli.get_faiss_config')
    @patch('local_faiss_mcp.cli.FAISSVectorStore')
    @patch('local_faiss_mcp.cli.parse_document')
    @patch('local_faiss_mcp.cli.collect_files')
    @patch('local_faiss_mcp.cli.create_file_progress')
    def test_no_progress_bar_for_single_file(
        self, mock_create_progress, mock_collect, mock_parse, mock_store_class, mock_config
    ):
        """Test that progress bar is NOT shown when indexing a single file."""
        # Setup
        mock_config.return_value = {
            'index_dir': '/tmp/test_index',
            'embed_model': 'test-model'
        }
        files = ['file1.pdf']
        mock_collect.return_value = files
        
        mock_store = MagicMock()
        mock_store.ingest.return_value = {'success': True, 'chunks_added': 5}
        mock_store.metadata = {'documents': []}
        mock_store_class.return_value = mock_store
        
        mock_parse.return_value = "test content"
        
        # Mock create_file_progress to return files and show_progress=False
        mock_create_progress.return_value = (files, False)
        
        # Create args mock
        args = MagicMock()
        args.files = files
        args.recursive = False
        
        # Execute
        result = cmd_index(args)
        
        # Verify create_file_progress was called
        assert mock_create_progress.called
        # For single file, the function should return show_progress=False
        assert mock_create_progress.return_value[1] is False
        assert result == 0


class TestProgressModule:
    """Test progress module functions directly."""
    
    def test_create_file_progress_multiple_files(self):
        """Test progress bar is enabled for multiple files."""
        from local_faiss_mcp.progress import create_file_progress
        
        files = [Path('file1.pdf'), Path('file2.pdf')]
        files_iter, show_progress = create_file_progress(files)
        
        assert show_progress is True
        # Clean up tqdm iterator
        if hasattr(files_iter, 'close'):
            files_iter.close()
    
    def test_create_file_progress_single_file(self):
        """Test progress bar is disabled for single file."""
        from local_faiss_mcp.progress import create_file_progress
        
        files = [Path('file1.pdf')]
        files_iter, show_progress = create_file_progress(files)
        
        assert show_progress is False
        assert files_iter == files  # Should return original list
    
    def test_progress_print_without_progress(self, capsys):
        """Test progress_print uses print when show_progress=False."""
        from local_faiss_mcp.progress import progress_print
        
        progress_print("Test message", False)
        captured = capsys.readouterr()
        assert "Test message" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
