"""
Local FAISS MCP Server

A Model Context Protocol (MCP) server that provides local vector database
functionality using FAISS for Retrieval-Augmented Generation (RAG) applications.
"""

__version__ = "0.2.0"

from .server import FAISSVectorStore, app, main

__all__ = ["FAISSVectorStore", "app", "main", "__version__"]
