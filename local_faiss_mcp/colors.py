#!/usr/bin/env python3
"""
Color utilities for CLI output using colorama.

Provides colored output helpers with cross-platform support and
NO_COLOR environment variable compliance.
"""

import os
import sys
from typing import Optional

try:
    from colorama import init, Fore, Style
    # Initialize colorama for Windows compatibility
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
    
    # Define pastel color codes using lighter ANSI colors
    PASTEL_GREEN = '\033[38;2;119;221;119m'
    PASTEL_RED = '\033[38;2;221;119;119m'
    PASTEL_BLUE = Fore.LIGHTCYAN_EX
    PASTEL_YELLOW = '\033[38;2;221;221;119m' 
except ImportError:
    COLORAMA_AVAILABLE = False
    PASTEL_GREEN = ""
    PASTEL_RED = ""
    PASTEL_BLUE = ""
    PASTEL_YELLOW = ""


def _is_color_disabled() -> bool:
    """
    Check if colored output should be disabled.
    
    Returns True if:
    - NO_COLOR environment variable is set
    - colorama is not available
    - stdout is not a TTY (e.g., piped output)
    """
    if not COLORAMA_AVAILABLE:
        return True
    
    if os.environ.get('NO_COLOR'):
        return True
    
    # Disable colors if output is piped
    if not sys.stdout.isatty():
        return True
    
    return False


def _colorize(text: str, color: str, icon: str = "") -> str:
    """
    Apply color to text if colors are enabled.
    
    Args:
        text: Text to colorize
        color: ANSI color code
        icon: Optional icon/emoji prefix
        
    Returns:
        Colored text or plain text if colors disabled
    """
    if _is_color_disabled():
        return f"{icon} {text}".strip() if icon else text
    
    prefix = f"{icon} " if icon else ""
    return f"{color}{prefix}{text}{Style.RESET_ALL}"


def success(message: str) -> str:
    """
    Format success message in pastel green with checkmark emoji.
    
    Args:
        message: Success message text
        
    Returns:
        Formatted success message
        
    Example:
        print(success("Added 5 chunks"))
        # Output: ✅ Added 5 chunks (in pastel green)
    """
    return _colorize(message, PASTEL_GREEN if COLORAMA_AVAILABLE else "", "✅")


def error(message: str) -> str:
    """
    Format error message in pastel red with X emoji.
    
    Args:
        message: Error message text
        
    Returns:
        Formatted error message
        
    Example:
        print(error("File not found"))
        # Output: ❌ File not found (in pastel red)
    """
    return _colorize(message, PASTEL_RED if COLORAMA_AVAILABLE else "", "❌")


def info(message: str) -> str:
    """
    Format info message in pastel blue with book emoji.
    
    Args:
        message: Info message text
        
    Returns:
        Formatted info message
        
    Example:
        print(info("Using MCP config: .mcp.json"))
        # Output: 📘 Using MCP config: .mcp.json (in pastel blue)
    """
    return _colorize(message, PASTEL_BLUE if COLORAMA_AVAILABLE else "", "📘")


def warning(message: str) -> str:
    """
    Format warning message in pastel yellow with warning emoji.
    
    Args:
        message: Warning message text
        
    Returns:
        Formatted warning message
        
    Example:
        print(warning("Large file may take time"))
        # Output: ⚠️ Large file may take time (in pastel yellow)
    """
    return _colorize(message, PASTEL_YELLOW if COLORAMA_AVAILABLE else "", "⚠️")


def highlight(message: str, color: Optional[str] = None) -> str:
    """
    Highlight text without icon prefix.
    
    Args:
        message: Text to highlight
        color: Optional specific color (default: cyan)
        
    Returns:
        Highlighted text
    """
    if not COLORAMA_AVAILABLE:
        return message
    
    color_code = color if color else Fore.CYAN
    return _colorize(message, color_code, "")
