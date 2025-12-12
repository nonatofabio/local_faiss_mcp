#!/usr/bin/env python3
"""
Progress bar utilities for CLI operations.
"""

from pathlib import Path
from typing import Iterable, List, Optional
from tqdm import tqdm


def create_file_progress(
    files: List[Path],
    desc: str = "Processing"
) -> tuple[Iterable, bool]:
    """
    Create a progress bar wrapper for file operations.
    
    Only shows progress bar for 2+ files to avoid clutter on single operations.
    
    Args:
        files: List of file paths to iterate over
        desc: Description prefix for the progress bar
        
    Returns:
        Tuple of (iterator, show_progress_flag)
    """
    show_progress = len(files) > 1
    
    if show_progress:
        files_iter = tqdm(
            files,
            desc=desc,
            unit="file",
            bar_format='{desc} {bar}| {n_fmt}/{total_fmt} ({percentage:3.0f}%)',
            disable=False,
            leave=True
        )
    else:
        files_iter = files
    
    return files_iter, show_progress


def update_progress_description(
    files_iter: Iterable,
    file_path: Path,
    desc: str = "Indexing"
) -> None:
    """
    Update the progress bar description with current filename.
    
    Args:
        files_iter: The tqdm iterator
        file_path: Current file being processed
        desc: Description prefix
    """
    if hasattr(files_iter, 'set_description'):
        files_iter.set_description(f"{desc} {Path(file_path).name}")


def progress_print(message: str, show_progress: bool) -> None:
    """
    Print a message, using tqdm.write() if progress bar is active.
    
    This prevents interference between output and the progress bar.
    
    Args:
        message: Message to print
        show_progress: Whether progress bar is currently displayed
    """
    if show_progress:
        tqdm.write(message)
    else:
        print(message)
