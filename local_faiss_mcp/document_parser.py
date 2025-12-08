#!/usr/bin/env python3
"""
Document parser for various file formats.

Supports:
- Native: TXT, MD, PDF
- Via pandoc: DOCX, ODT, HTML, RTF, EPUB, and 40+ other formats
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional
from pypdf import PdfReader


def parse_text_file(file_path: Path) -> str:
    """Parse plain text files (TXT, MD, etc.)."""
    try:
        return file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # Try with latin-1 fallback
        return file_path.read_text(encoding='latin-1')


def parse_pdf(file_path: Path) -> str:
    """Parse PDF files using pypdf."""
    reader = PdfReader(file_path)
    text_parts = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)

    return '\n\n'.join(text_parts)


def parse_with_pandoc(file_path: Path) -> str:
    """Parse document using pandoc (if available)."""
    if not shutil.which('pandoc'):
        raise RuntimeError(
            f"Pandoc is required to parse {file_path.suffix} files. "
            f"Install with: brew install pandoc (macOS) or apt install pandoc (Linux)"
        )

    result = subprocess.run(
        ['pandoc', str(file_path), '-t', 'plain', '-o', '-'],
        capture_output=True,
        text=True,
        timeout=30
    )

    if result.returncode != 0:
        raise RuntimeError(f"Pandoc failed: {result.stderr}")

    return result.stdout


def parse_document(file_path: str | Path) -> str:
    """
    Parse document from file path.

    Supports:
    - TXT, MD: Native text parsing
    - PDF: pypdf extraction
    - DOCX, ODT, HTML, etc.: pandoc (if installed)

    Args:
        file_path: Path to document file

    Returns:
        Extracted text content

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is unsupported
        RuntimeError: If parsing fails
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Not a file: {file_path}")

    ext = file_path.suffix.lower()

    # Native parsers
    if ext in ['.txt', '.md', '.rst', '.log']:
        return parse_text_file(file_path)

    elif ext == '.pdf':
        return parse_pdf(file_path)

    # Pandoc fallback for other formats
    elif ext in ['.docx', '.odt', '.html', '.htm', '.rtf', '.epub', '.org', '.tex']:
        return parse_with_pandoc(file_path)

    # Unknown format - try pandoc if available
    else:
        if shutil.which('pandoc'):
            try:
                return parse_with_pandoc(file_path)
            except RuntimeError:
                raise ValueError(
                    f"Unsupported file format: {ext}. "
                    f"Supported formats: TXT, MD, PDF, DOCX (with pandoc)"
                )
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported formats: TXT, MD, PDF. "
                f"For DOCX/HTML/etc, install pandoc: brew install pandoc"
            )


def is_file_path(text: str) -> bool:
    """
    Check if a string looks like a file path.

    Heuristic: Contains path separators and doesn't look like natural language.
    """
    if not text:
        return False

    # Check for path-like characteristics
    path = Path(text)

    # If it exists as a file, it's definitely a path
    if path.exists() and path.is_file():
        return True

    # Check for path separators and reasonable length
    # (most documents are multi-sentence, paths are shorter)
    if ('/' in text or '\\' in text) and len(text) < 500:
        # Check if it has a file extension
        if path.suffix:
            return True

    return False
