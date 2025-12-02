"""
Entry point for running the MCP server as a module.

Usage:
    python -m local_faiss_mcp [--index-dir PATH]
"""

import asyncio
from .server import main

if __name__ == "__main__":
    asyncio.run(main())
