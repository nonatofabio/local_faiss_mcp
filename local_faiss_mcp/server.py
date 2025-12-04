#!/usr/bin/env python3
"""
Local FAISS MCP Server for RAG

This MCP server provides tools for document ingestion and retrieval using FAISS
as a local vector database.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Any
import faiss
from sentence_transformers import SentenceTransformer
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server


class FAISSVectorStore:
    """Manages FAISS index and document storage."""

    def __init__(
        self,
        index_path: str = "faiss.index",
        metadata_path: str = "metadata.json",
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_model_name = embedding_model_name

        # Load the embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Get dimension from the model by encoding a test string
        test_embedding = self.embedding_model.encode(["test"], convert_to_numpy=True)
        self.dimension = test_embedding.shape[1]

        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(index_path)) or '.', exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(metadata_path)) or '.', exist_ok=True)

        # Load or create index
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            # Verify dimension matches
            if self.index.d != self.dimension:
                raise ValueError(
                    f"Existing index dimension ({self.index.d}) does not match "
                    f"embedding model dimension ({self.dimension}). "
                    f"Please use a different index directory or the same embedding model."
                )
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

        # Load or create metadata
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"documents": [], "model": embedding_model_name}

    def save(self):
        """Persist index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        words = text.split()

        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)

        return chunks

    def ingest(self, document: str, source: str = "unknown") -> dict[str, Any]:
        """Ingest a document into the vector store."""
        # Chunk the document
        chunks = self.chunk_text(document)

        if not chunks:
            return {"success": False, "error": "No chunks created from document"}

        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)

        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))

        # Store metadata
        start_idx = len(self.metadata["documents"])
        for i, chunk in enumerate(chunks):
            self.metadata["documents"].append({
                "id": start_idx + i,
                "source": source,
                "text": chunk
            })

        # Save to disk
        self.save()

        return {
            "success": True,
            "chunks_added": len(chunks),
            "total_documents": len(self.metadata["documents"])
        }

    def query(self, query_text: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Query the vector store for relevant documents."""
        if self.index.ntotal == 0:
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text], convert_to_numpy=True)

        # Search FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), min(top_k, self.index.ntotal))

        # Retrieve matching documents
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata["documents"]):
                doc = self.metadata["documents"][idx]
                results.append({
                    "text": doc["text"],
                    "source": doc["source"],
                    "distance": float(dist)
                })

        return results


# Initialize the MCP server
app = Server("local-faiss-mcp")
vector_store = None  # Will be initialized in main()


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="ingest_document",
            description="Ingest a document into the FAISS vector store. The document will be chunked, embedded, and stored for later retrieval.",
            inputSchema={
                "type": "object",
                "properties": {
                    "document": {
                        "type": "string",
                        "description": "The text content of the document to ingest"
                    },
                    "source": {
                        "type": "string",
                        "description": "Optional source identifier for the document (e.g., filename, URL)",
                        "default": "unknown"
                    }
                },
                "required": ["document"]
            }
        ),
        Tool(
            name="query_rag_store",
            description="Query the FAISS vector store to retrieve relevant document chunks based on semantic similarity.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query text"
                    },
                    "top_k": {
                        "type": "number",
                        "description": "Number of top results to return",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""

    if name == "ingest_document":
        document = arguments.get("document")
        source = arguments.get("source", "unknown")

        result = vector_store.ingest(document, source)

        if result["success"]:
            message = f"Successfully ingested document from '{source}'.\n"
            message += f"Created {result['chunks_added']} chunks.\n"
            message += f"Total documents in store: {result['total_documents']}"
        else:
            message = f"Failed to ingest document: {result.get('error', 'Unknown error')}"

        return [TextContent(type="text", text=message)]

    elif name == "query_rag_store":
        query = arguments.get("query")
        top_k = arguments.get("top_k", 3)

        results = vector_store.query(query, top_k)

        if not results:
            message = "No results found. The vector store may be empty."
        else:
            message = f"Found {len(results)} relevant chunks:\n\n"
            for i, result in enumerate(results, 1):
                message += f"{i}. Source: {result['source']}\n"
                message += f"   Distance: {result['distance']:.4f}\n"
                message += f"   Text: {result['text'][:200]}...\n\n"

        return [TextContent(type="text", text=message)]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    global vector_store

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Local FAISS MCP Server for RAG")
    parser.add_argument(
        "--index-dir",
        type=str,
        default=".",
        help="Directory to store FAISS index and metadata (default: current directory)"
    )
    parser.add_argument(
        "--embed",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Hugging Face embedding model name (default: all-MiniLM-L6-v2)"
    )
    args = parser.parse_args()

    # Initialize vector store with configured paths and embedding model
    index_dir = Path(args.index_dir).resolve()
    index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.json"

    print(f"Initializing with embedding model: {args.embed}", file=sys.stderr)

    vector_store = FAISSVectorStore(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        embedding_model_name=args.embed
    )

    print(f"Vector store initialized (dimension: {vector_store.dimension})", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


def cli_main():
    """Console script entry point."""
    import asyncio
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
