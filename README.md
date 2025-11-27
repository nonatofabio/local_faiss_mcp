# Local FAISS MCP Server

A Model Context Protocol (MCP) server that provides local vector database functionality using FAISS for Retrieval-Augmented Generation (RAG) applications.

## Features

- **Local Vector Storage**: Uses FAISS for efficient similarity search without external dependencies
- **Document Ingestion**: Automatically chunks and embeds documents for storage
- **Semantic Search**: Query documents using natural language with sentence embeddings
- **Persistent Storage**: Indexes and metadata are saved to disk
- **MCP Compatible**: Works with any MCP-compatible AI agent or client

## Quickstart

```bash
git clone https://github.com/nonatofabio/local_faiss_mcp.git
cd local_faiss_mcp && pip install -r requirements.txt
python server.py --index-dir ./.vector_store
```

Then configure your MCP client (see [Configuration](#configuration-with-mcp-clients)) and try your first query in Claude:

```
Use the query_rag_store tool to search for: "How does FAISS perform similarity search?"
```

Claude will retrieve relevant document chunks from your vector store and use them to answer your question.

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Server

Run the MCP server using stdio transport:

```bash
# Use current directory for index storage (default)
python server.py

# Specify a custom directory for index storage
python server.py --index-dir /path/to/index/directory
```

**Command-line Arguments:**
- `--index-dir`: Directory to store FAISS index and metadata files (default: current directory)

The server will:
- Create the index directory if it doesn't exist
- Load existing FAISS index from `{index-dir}/faiss.index` (or create a new one)
- Load document metadata from `{index-dir}/metadata.json` (or create new)
- Listen for MCP tool calls via stdin/stdout

### Available Tools

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

## Configuration with MCP Clients

### Claude Code

Add this server to your Claude Code MCP configuration (`.mcp.json`):

**User-wide configuration** (`~/.claude/.mcp.json`):
```json
{
  "mcpServers": {
    "local-faiss-mcp": {
      "command": "python",
      "args": ["/home/user/localdev/local_faiss_mcp/server.py"]
    }
  }
}
```

**With custom index directory**:
```json
{
  "mcpServers": {
    "local-faiss-mcp": {
      "command": "python",
      "args": [
        "/home/user/localdev/local_faiss_mcp/server.py",
        "--index-dir",
        "/home/user/vector_indexes/my_project"
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
      "command": "python",
      "args": [
        "/home/user/localdev/local_faiss_mcp/server.py",
        "--index-dir",
        "./.vector_store"
      ]
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
      "command": "python",
      "args": ["/path/to/local_faiss_mcp/server.py", "--index-dir", "/path/to/index/directory"]
    }
  }
}
```

## Architecture

- **Embedding Model**: Uses `all-MiniLM-L6-v2` from sentence-transformers (384 dimensions)
- **Index Type**: FAISS IndexFlatL2 for exact L2 distance search
- **Chunking**: Documents are split into ~500 word chunks with 50 word overlap
- **Storage**: Index saved as `faiss.index`, metadata saved as `metadata.json`

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

Run unit tests:
```bash
pytest
```

## License

MIT
