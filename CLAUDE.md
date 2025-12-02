# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MCP (Model Context Protocol) server implementation that uses FAISS (Facebook AI Similarity Search) as a local vector database for Retrieval-Augmented Generation (RAG) applications.

The goal is to create a self-contained, local RAG system where:
- FAISS handles vector storage and similarity search operations
- The MCP server exposes FAISS functionality as tools for AI agent interaction
- Documents can be ingested, chunked, embedded, and stored locally
- AI agents can query the vector store using natural language

## Architecture

The system has three main components:

1. **FAISS Vector Store**: Local vector database that stores and indexes document embeddings
   - Supports in-memory or disk-persisted indexes
   - Uses similarity metrics (L2 distance, dot product, cosine similarity)

2. **MCP Server**: Provides tool interface for agent interaction
   - Tool: `ingest_document` - handles document chunking, embedding generation, and storage in FAISS
   - Tool: `query_rag_store` - performs similarity searches to retrieve relevant document chunks

3. **Agent Integration**: Enables natural language interaction with the vector store
   - AI agents use MCP tools to interact with FAISS-backed storage
   - Retrieved chunks augment agent responses for RAG

## Key Design Principles

- **Local-first**: All storage and operations happen locally, no external vector DB services required
- **MCP Protocol**: Follows Model Context Protocol specifications for tool definitions and agent interaction
- **Embedding-based Search**: Uses vector embeddings for semantic similarity search rather than keyword matching

## Development Commands

### Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Running the Server
```bash
# Using the installed command (easiest)
local-faiss-mcp --index-dir /path/to/index/directory

# As a Python module
python -m local_faiss_mcp --index-dir /path/to/index/directory

# Direct execution (for development)
python local_faiss_mcp/server.py --index-dir /path/to/index/directory
```

**Command-line Arguments:**
- `--index-dir`: Directory to store FAISS index and metadata (default: current directory)

### Testing
```bash
# Run tests
pytest
```

### Building and Publishing
```bash
# Build package
python -m build

# Check package
twine check dist/*

# Publish to PyPI (see PUBLISHING.md for details)
twine upload dist/*
```

## Implementation Details

### File Structure
- `local_faiss_mcp/`: Main package directory
  - `server.py`: MCP server implementation with FAISSVectorStore class
  - `__init__.py`: Package initialization
  - `__main__.py`: Entry point for `python -m local_faiss_mcp`
- `test_standalone.py`: Standalone tests
- `pyproject.toml`: Project configuration and dependencies
- `requirements.txt`: Python dependencies
- `PUBLISHING.md`: Guide for publishing to PyPI

### Data Files (Created at Runtime)
- `faiss.index`: Persisted FAISS index (created in --index-dir)
- `metadata.json`: Document metadata storage (created in --index-dir)

### Configuring Index Directory

The server accepts an optional `--index-dir` argument to specify where to store the FAISS index and metadata files. This is particularly useful for:
- Project-specific vector stores (each project has its own index)
- Shared vector stores across multiple projects
- Organizing indexes by topic or domain

**Example MCP configurations:**

User-wide (`~/.claude/.mcp.json`):
```json
{
  "mcpServers": {
    "local-faiss-mcp": {
      "command": "local-faiss-mcp",
      "args": [
        "--index-dir",
        "/path/to/vector_indexes/general"
      ]
    }
  }
}
```

Project-specific (`./.mcp.json` in project root):
```json
{
  "mcpServers": {
    "project-faiss": {
      "command": "local-faiss-mcp",
      "args": [
        "--index-dir",
        "./.vector_store"
      ]
    }
  }
}
```

Alternative using Python module (if command not in PATH):
```json
{
  "mcpServers": {
    "local-faiss-mcp": {
      "command": "python",
      "args": [
        "-m", "local_faiss_mcp",
        "--index-dir",
        "./.vector_store"
      ]
    }
  }
}
```

### FAISSVectorStore Class (server.py)

The core class managing vector operations:

- **Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Index Type**: `IndexFlatL2` - exact L2 distance search
- **Chunking Strategy**: 500 words per chunk with 50-word overlap
- **Persistence**: Auto-saves index and metadata after ingestion
- **Directory Management**: Automatically creates index directory if it doesn't exist

Key methods:
- `__init__(index_path, metadata_path)`: Initializes store and creates directories
- `chunk_text()`: Splits documents into overlapping chunks
- `ingest()`: Embeds and stores document chunks in FAISS
- `query()`: Performs similarity search and retrieves top-k results
- `save()`: Persists index and metadata to disk

### MCP Tools

Both tools are defined in `server.py`:

1. **ingest_document**
   - Accepts `document` (text) and optional `source` (identifier)
   - Returns success status and chunk count
   - Automatically saves to disk in configured index directory

2. **query_rag_store**
   - Accepts `query` (text) and optional `top_k` (number)
   - Returns ranked results with distance scores
   - Handles empty index gracefully

### Server Initialization (server.py:main)

The server parses command-line arguments and initializes the vector store with the specified index directory:
- Resolves the index directory path (creates if needed)
- Constructs full paths for `faiss.index` and `metadata.json`
- Initializes FAISSVectorStore with these paths
- Starts MCP server via stdio transport
