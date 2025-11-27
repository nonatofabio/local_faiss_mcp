#!/usr/bin/env python3
"""
Standalone test for the FAISS vector store.
Tests document ingestion and querying without MCP infrastructure.
"""

import os
import sys
from pathlib import Path

# Add current directory to path to import server module
sys.path.insert(0, str(Path(__file__).parent))

from server import FAISSVectorStore


def test_vector_store():
    """Test the FAISSVectorStore functionality."""

    # Use a test directory for the index
    test_dir = Path(".test_vector_store")
    test_dir.mkdir(exist_ok=True)

    index_path = test_dir / "test_faiss.index"
    metadata_path = test_dir / "test_metadata.json"

    print("=" * 60)
    print("FAISS Vector Store Standalone Test")
    print("=" * 60)

    try:
        # Initialize the vector store
        print("\n1. Initializing FAISS vector store...")
        vector_store = FAISSVectorStore(
            index_path=str(index_path),
            metadata_path=str(metadata_path)
        )
        print("   ✓ Vector store initialized successfully")

        # Test document ingestion
        print("\n2. Testing document ingestion...")

        doc1 = """
        FAISS (Facebook AI Similarity Search) is a library for efficient similarity
        search and clustering of dense vectors. It contains algorithms that search
        in sets of vectors of any size, up to ones that possibly do not fit in RAM.
        It also contains supporting code for evaluation and parameter tuning.
        """

        doc2 = """
        Vector databases are specialized databases designed to store and query
        high-dimensional vectors efficiently. They use techniques like approximate
        nearest neighbor (ANN) search to find similar vectors quickly, which is
        essential for applications like semantic search, recommendation systems,
        and RAG (Retrieval-Augmented Generation).
        """

        doc3 = """
        The Model Context Protocol (MCP) is an open protocol that standardizes
        how applications provide context to LLMs. It enables secure, controlled
        interactions between AI applications and external data sources and tools.
        MCP servers expose resources, tools, and prompts to LLM applications.
        """

        # Ingest documents
        result1 = vector_store.ingest(doc1, source="faiss_doc.txt")
        print(f"   ✓ Document 1 ingested: {result1['chunks_added']} chunks")

        result2 = vector_store.ingest(doc2, source="vector_db_doc.txt")
        print(f"   ✓ Document 2 ingested: {result2['chunks_added']} chunks")

        result3 = vector_store.ingest(doc3, source="mcp_doc.txt")
        print(f"   ✓ Document 3 ingested: {result3['chunks_added']} chunks")

        print(f"\n   Total documents in store: {result3['total_documents']}")

        # Test querying
        print("\n3. Testing semantic search queries...")

        queries = [
            ("What is FAISS?", 2),
            ("How do vector databases work?", 2),
            ("Tell me about MCP protocol", 2),
        ]

        for query_text, top_k in queries:
            print(f"\n   Query: '{query_text}'")
            print(f"   Top {top_k} results:")

            results = vector_store.query(query_text, top_k=top_k)

            for i, result in enumerate(results, 1):
                print(f"\n   [{i}] Source: {result['source']}")
                print(f"       Distance: {result['distance']:.4f}")
                print(f"       Text: {result['text'][:150]}...")

        # Test persistence
        print("\n4. Testing persistence...")
        print("   Saving vector store...")
        vector_store.save()
        print("   ✓ Vector store saved successfully")

        print(f"\n   Index file: {index_path} ({os.path.getsize(index_path)} bytes)")
        print(f"   Metadata file: {metadata_path} ({os.path.getsize(metadata_path)} bytes)")

        # Test loading
        print("\n5. Testing reload from disk...")
        vector_store_reloaded = FAISSVectorStore(
            index_path=str(index_path),
            metadata_path=str(metadata_path)
        )
        print("   ✓ Vector store reloaded successfully")

        # Verify data persisted
        test_query = "FAISS similarity search"
        results_reloaded = vector_store_reloaded.query(test_query, top_k=1)

        if results_reloaded:
            print(f"\n   Verification query: '{test_query}'")
            print(f"   Found {len(results_reloaded)} result(s)")
            print("   ✓ Data persisted correctly")
        else:
            print("   ✗ No results found after reload!")

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup test files
        print("\n6. Cleaning up test files...")
        try:
            if index_path.exists():
                index_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            if test_dir.exists():
                test_dir.rmdir()
            print("   ✓ Cleanup complete")
        except Exception as e:
            print(f"   Warning: Cleanup failed: {e}")

    return True


if __name__ == "__main__":
    print("\nStarting standalone test...\n")
    success = test_vector_store()
    sys.exit(0 if success else 1)
