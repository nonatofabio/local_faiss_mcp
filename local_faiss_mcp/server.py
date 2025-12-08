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
from sentence_transformers import SentenceTransformer, CrossEncoder
from mcp.server import Server
from mcp.types import Tool, TextContent, Prompt, PromptMessage, PromptArgument
from mcp.server.stdio import stdio_server
from .document_parser import parse_document, is_file_path


class FAISSVectorStore:
    """Manages FAISS index and document storage."""

    def __init__(
        self,
        index_path: str = "faiss.index",
        metadata_path: str = "metadata.json",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        rerank_model_name: str | None = None
    ):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_model_name = embedding_model_name
        self.rerank_model_name = rerank_model_name

        # Load the embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Load the re-ranker model if specified
        self.reranker = None
        if rerank_model_name:
            self.reranker = CrossEncoder(rerank_model_name)

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

        # If re-ranker is enabled, retrieve more candidates for re-ranking
        if self.reranker:
            # Retrieve 10x more candidates for re-ranking
            candidate_k = min(top_k * 10, self.index.ntotal)
        else:
            candidate_k = min(top_k, self.index.ntotal)

        # Search FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), candidate_k)

        # Retrieve matching documents
        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.metadata["documents"]):
                doc = self.metadata["documents"][idx]
                candidates.append({
                    "text": doc["text"],
                    "source": doc["source"],
                    "distance": float(dist)
                })

        # Apply re-ranking if enabled
        if self.reranker and candidates:
            # Create (query, document) pairs for re-ranker
            pairs = [(query_text, candidate["text"]) for candidate in candidates]

            # Get re-ranker scores
            rerank_scores = self.reranker.predict(pairs)

            # Add rerank scores to candidates and sort by score (descending)
            for candidate, score in zip(candidates, rerank_scores):
                candidate["rerank_score"] = float(score)

            # Sort by rerank score (higher is better)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

            # Return top-k re-ranked results
            return candidates[:top_k]

        return candidates


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

        # Auto-detect if document is a file path
        if is_file_path(document):
            try:
                file_path = Path(document)
                # Parse the document file
                document_text = parse_document(file_path)
                # Use filename as source if not specified
                if source == "unknown":
                    source = file_path.name
                document = document_text
            except Exception as e:
                return [TextContent(type="text", text=f"Failed to parse document: {str(e)}")]

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
                if 'rerank_score' in result:
                    message += f"   Rerank Score: {result['rerank_score']:.4f}\n"
                message += f"   Text: {result['text'][:200]}...\n\n"

        return [TextContent(type="text", text=message)]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


@app.list_prompts()
async def list_prompts() -> list[Prompt]:
    """List available prompts for RAG workflows."""
    return [
        Prompt(
            name="extract-answer",
            description="Extract the most relevant answer from retrieved document chunks",
            arguments=[
                PromptArgument(
                    name="query",
                    description="The original user query or question",
                    required=True
                ),
                PromptArgument(
                    name="chunks",
                    description="Retrieved document chunks (JSON array with 'text', 'source', 'distance' fields)",
                    required=True
                )
            ]
        ),
        Prompt(
            name="summarize-documents",
            description="Summarize information from multiple document chunks",
            arguments=[
                PromptArgument(
                    name="topic",
                    description="The topic or theme to summarize",
                    required=True
                ),
                PromptArgument(
                    name="chunks",
                    description="Document chunks to summarize (JSON array)",
                    required=True
                ),
                PromptArgument(
                    name="max_length",
                    description="Maximum summary length in words (default: 200)",
                    required=False
                )
            ]
        )
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None) -> PromptMessage:
    """Generate a prompt with the given arguments."""

    if name == "extract-answer":
        query = arguments.get("query", "") if arguments else ""
        chunks_json = arguments.get("chunks", "[]") if arguments else "[]"

        try:
            chunks = json.loads(chunks_json)
        except json.JSONDecodeError:
            chunks = []

        # Build the prompt
        prompt_text = f"""You are a helpful assistant with access to a vector database. A user has asked the following question:

**Question:** {query}

I have retrieved the most relevant document chunks from the database. Please:
1. Carefully read through all the retrieved chunks below
2. Extract the most relevant information that answers the question
3. Provide a clear, concise answer based ONLY on the information in these chunks
4. Include direct quotes from the source documents when appropriate
5. If the chunks don't contain enough information to answer the question, say so clearly

Retrieved Document Chunks:
"""

        if not chunks:
            prompt_text += "\n(No chunks provided)\n"
        else:
            for i, chunk in enumerate(chunks, 1):
                text = chunk.get('text', '')
                source = chunk.get('source', 'unknown')
                distance = chunk.get('distance', 0.0)

                prompt_text += f"""
---
**Chunk {i}** (Source: {source}, Relevance Score: {distance:.4f})
{text}
"""

        prompt_text += """
---

Now, please provide your answer based on these chunks. Remember to cite sources when making specific claims."""

        return PromptMessage(
            role="user",
            content=TextContent(type="text", text=prompt_text)
        )

    elif name == "summarize-documents":
        topic = arguments.get("topic", "the documents") if arguments else "the documents"
        chunks_json = arguments.get("chunks", "[]") if arguments else "[]"
        max_length = arguments.get("max_length", "200") if arguments else "200"

        try:
            chunks = json.loads(chunks_json)
        except json.JSONDecodeError:
            chunks = []

        prompt_text = f"""Please create a comprehensive summary about "{topic}" based on the following document chunks retrieved from a vector database.

Requirements:
- Maximum length: {max_length} words
- Focus on the key points related to {topic}
- Maintain factual accuracy
- Cite sources when making specific claims

Document Chunks:
"""

        if not chunks:
            prompt_text += "\n(No chunks provided)\n"
        else:
            for i, chunk in enumerate(chunks, 1):
                text = chunk.get('text', '')
                source = chunk.get('source', 'unknown')

                prompt_text += f"""
---
**Source {i}:** {source}
{text}
"""

        prompt_text += """
---

Please provide your summary now."""

        return PromptMessage(
            role="user",
            content=TextContent(type="text", text=prompt_text)
        )

    else:
        raise ValueError(f"Unknown prompt: {name}")


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
    parser.add_argument(
        "--rerank",
        type=str,
        nargs="?",
        const="BAAI/bge-reranker-base",
        default=None,
        help="Enable re-ranking with specified model (default: BAAI/bge-reranker-base if flag provided without model)"
    )
    args = parser.parse_args()

    # Initialize vector store with configured paths and embedding model
    index_dir = Path(args.index_dir).resolve()
    index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.json"

    print(f"Initializing with embedding model: {args.embed}", file=sys.stderr)
    if args.rerank:
        print(f"Re-ranking enabled with model: {args.rerank}", file=sys.stderr)

    vector_store = FAISSVectorStore(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        embedding_model_name=args.embed,
        rerank_model_name=args.rerank
    )

    print(f"Vector store initialized (dimension: {vector_store.dimension})", file=sys.stderr)
    if vector_store.reranker:
        print(f"Re-ranker loaded successfully", file=sys.stderr)

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
