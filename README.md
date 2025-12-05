# Local FAISS MCP Server

<!-- mcp-name: io.github.nonatofabio/local-faiss-mcp -->

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/nonatofabio/local_faiss_mcp/workflows/Tests/badge.svg)](https://github.com/nonatofabio/local_faiss_mcp/actions)
<!-- [![PyPI version](https://badge.fury.io/py/local-faiss-mcp.svg)](https://badge.fury.io/py/local-faiss-mcp) -->

A Model Context Protocol (MCP) server that provides local vector database functionality using FAISS for Retrieval-Augmented Generation (RAG) applications.

![demo](./static/demo.gif)

## Features

- **Local Vector Storage**: Uses FAISS for efficient similarity search without external dependencies
- **Document Ingestion**: Automatically chunks and embeds documents for storage
- **Semantic Search**: Query documents using natural language with sentence embeddings
- **Persistent Storage**: Indexes and metadata are saved to disk
- **MCP Compatible**: Works with any MCP-compatible AI agent or client

## Quickstart

```bash
pip install local-faiss-mcp
```

Then configure your MCP client (see [Configuration](#configuration-with-mcp-clients)) and try your first query in Claude:

```
Use the query_rag_store tool to search for: "How does FAISS perform similarity search?"
```

Claude will retrieve relevant document chunks from your vector store and use them to answer your question.

## Installation

### From PyPI (Recommended)

```bash
pip install local-faiss-mcp
```

### From Source

```bash
git clone https://github.com/nonatofabio/local_faiss_mcp.git
cd local_faiss_mcp
pip install -e .
```

## Usage

### Running the Server

After installation, you can run the server in three ways:

**1. Using the installed command (easiest):**
```bash
local-faiss-mcp --index-dir /path/to/index/directory
```

**2. As a Python module:**
```bash
python -m local_faiss_mcp --index-dir /path/to/index/directory
```

**3. For development/testing:**
```bash
python local_faiss_mcp/server.py --index-dir /path/to/index/directory
```

**Command-line Arguments:**
- `--index-dir`: Directory to store FAISS index and metadata files (default: current directory)
- `--embed`: Hugging Face embedding model name (default: `all-MiniLM-L6-v2`)

**Using a Custom Embedding Model:**
```bash
# Use a larger, more accurate model
local-faiss-mcp --index-dir ./.vector_store --embed all-mpnet-base-v2

# Use a multilingual model
local-faiss-mcp --index-dir ./.vector_store --embed paraphrase-multilingual-MiniLM-L12-v2

# Use any Hugging Face sentence-transformers model
local-faiss-mcp --index-dir ./.vector_store --embed sentence-transformers/model-name
```

The server will:
- Create the index directory if it doesn't exist
- Load existing FAISS index from `{index-dir}/faiss.index` (or create a new one)
- Load document metadata from `{index-dir}/metadata.json` (or create new)
- Listen for MCP tool calls via stdin/stdout

### Available Tools

The server provides two tools for document management:

#### 1. ingest_document

Ingest a document into the vector store.

**Parameters:**
- `document` (required): The text content to ingest
- `source` (optional): Identifier for the document source (default: "unknown")

**Example:**
```json
{
  "document": "FAISS is a library for efficient similarity search...",
  "source": "faiss_docs.txt"
}
```

#### 2. query_rag_store

Query the vector store for relevant document chunks.

**Parameters:**
- `query` (required): The search query text
- `top_k` (optional): Number of results to return (default: 3)

**Example:**
```json
{
  "query": "How does FAISS perform similarity search?",
  "top_k": 5
}
```

### Available Prompts

The server provides MCP prompts to help extract answers and summarize information from retrieved documents:

#### 1. extract-answer

Extract the most relevant answer from retrieved document chunks with proper citations.

**Arguments:**
- `query` (required): The original user query or question
- `chunks` (required): Retrieved document chunks as JSON array with fields: `text`, `source`, `distance`

**Use Case:** After querying the RAG store, use this prompt to get a well-formatted answer that cites sources and explains relevance.

**Example workflow in Claude:**
1. Use `query_rag_store` tool to retrieve relevant chunks
2. Use `extract-answer` prompt with the query and results
3. Get a comprehensive answer with citations

#### 2. summarize-documents

Create a focused summary from multiple document chunks.

**Arguments:**
- `topic` (required): The topic or theme to summarize
- `chunks` (required): Document chunks to summarize as JSON array
- `max_length` (optional): Maximum summary length in words (default: 200)

**Use Case:** Synthesize information from multiple retrieved documents into a concise summary.

**Example Usage:**

In Claude Code, after retrieving documents with `query_rag_store`, you can use the prompts like:

```
Use the extract-answer prompt with:
- query: "What is FAISS?"
- chunks: [the JSON results from query_rag_store]
```

The prompts will guide the LLM to provide structured, citation-backed answers based on your vector store data.

## Configuration with MCP Clients

### Claude Code

Add this server to your Claude Code MCP configuration (`.mcp.json`):

**User-wide configuration** (`~/.claude/.mcp.json`):
```json
{
  "mcpServers": {
    "local-faiss-mcp": {
      "command": "local-faiss-mcp"
    }
  }
}
```

**With custom index directory**:
```json
{
  "mcpServers": {
    "local-faiss-mcp": {
      "command": "local-faiss-mcp",
      "args": [
        "--index-dir",
        "/home/user/vector_indexes/my_project"
      ]
    }
  }
}
```

**With custom embedding model**:
```json
{
  "mcpServers": {
    "local-faiss-mcp": {
      "command": "local-faiss-mcp",
      "args": [
        "--index-dir",
        "./.vector_store",
        "--embed",
        "all-mpnet-base-v2"
      ]
    }
  }
}
```

**Project-specific configuration** (`./.mcp.json` in your project):
```json
{
  "mcpServers": {
    "local-faiss-mcp": {
      "command": "local-faiss-mcp",
      "args": [
        "--index-dir",
        "./.vector_store"
      ]
    }
  }
}
```

**Alternative: Using Python module** (if the command isn't in PATH):
```json
{
  "mcpServers": {
    "local-faiss-mcp": {
      "command": "python",
      "args": ["-m", "local_faiss_mcp", "--index-dir", "./.vector_store"]
    }
  }
}
```

### Claude Desktop

Add this server to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "local-faiss-mcp": {
      "command": "local-faiss-mcp",
      "args": ["--index-dir", "/path/to/index/directory"]
    }
  }
}
```

## Architecture

- **Embedding Model**: Configurable via `--embed` flag (default: `all-MiniLM-L6-v2` with 384 dimensions)
  - Supports any Hugging Face sentence-transformers model
  - Automatically detects embedding dimensions
  - Model choice persisted with the index
- **Index Type**: FAISS IndexFlatL2 for exact L2 distance search
- **Chunking**: Documents are split into ~500 word chunks with 50 word overlap
- **Storage**: Index saved as `faiss.index`, metadata saved as `metadata.json`

### Choosing an Embedding Model

Different models offer different trade-offs:

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | Default, balanced performance |
| `all-mpnet-base-v2` | 768 | Medium | Better | Higher quality embeddings |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Fast | Good | Multilingual support |
| `all-MiniLM-L12-v2` | 384 | Medium | Better | Better quality at same size |

**Important:** Once you create an index with a specific model, you must use the same model for subsequent runs. The server will detect dimension mismatches and warn you.

## Development

### Standalone Test

Test the FAISS vector store functionality without MCP infrastructure:

```bash
source venv/bin/activate
python test_standalone.py
```

This test:
- Initializes the vector store
- Ingests sample documents
- Performs semantic search queries
- Tests persistence and reload
- Cleans up test files

### Unit Tests

Run the complete test suite:
```bash
pytest tests/ -v
```

Run specific test files:
```bash
# Test embedding model functionality
pytest tests/test_embedding_models.py -v

# Run standalone integration test
python tests/test_standalone.py
```

The test suite includes:
- **test_embedding_models.py**: Comprehensive tests for custom embedding models, dimension detection, and compatibility
- **test_standalone.py**: End-to-end integration test without MCP infrastructure

## License

MIT
