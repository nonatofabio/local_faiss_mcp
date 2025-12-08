#!/usr/bin/env python3
"""
Command-line interface for local-faiss.

Commands:
- index: Index documents into FAISS vector store
- search: Search the vector store
"""

import sys
import os
import json
import argparse
from pathlib import Path
from glob import glob as glob_files
from typing import List, Optional, Dict, Any

from .server import FAISSVectorStore
from .document_parser import parse_document


def find_mcp_config() -> Optional[Path]:
    """
    Find MCP configuration file.

    Search order:
    1. ./.mcp.json (local/project-specific)
    2. ~/.*/mcp.json (user configs like ~/.claude/.mcp.json)
    3. ~/.mcp.json (fallback)

    Returns:
        Path to MCP config file, or None if not found
    """
    # 1. Local config
    local_config = Path('./.mcp.json')
    if local_config.exists():
        return local_config

    # 2. Search home directory for any .*/mcp.json
    home = Path.home()
    for config_path in home.glob('*/mcp.json'):
        # Check it's a hidden directory (starts with .)
        if config_path.parent.name.startswith('.'):
            return config_path

    # 3. Fallback to ~/.mcp.json
    fallback_config = home / '.mcp.json'
    if fallback_config.exists():
        return fallback_config

    return None


def read_mcp_config(config_path: Path) -> Dict[str, Any]:
    """Read and parse MCP configuration file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to read MCP config: {e}", file=sys.stderr)
        return {}


def get_faiss_config() -> Dict[str, Any]:
    """
    Get local-faiss-mcp configuration from MCP config.

    Returns:
        Dict with: index_dir, embed_model, rerank_model
    """
    config_path = find_mcp_config()

    if config_path:
        print(f"Using MCP config: {config_path}", file=sys.stderr)
        mcp_config = read_mcp_config(config_path)

        # Extract local-faiss-mcp server config
        servers = mcp_config.get('mcpServers', {})
        faiss_config = servers.get('local-faiss-mcp', {})

        if faiss_config:
            args = faiss_config.get('args', [])

            # Parse args to extract configuration
            config = {
                'index_dir': '.',
                'embed_model': 'all-MiniLM-L6-v2',
                'rerank_model': None
            }

            # Parse args list
            i = 0
            while i < len(args):
                if args[i] == '--index-dir' and i + 1 < len(args):
                    config['index_dir'] = args[i + 1]
                    i += 2
                elif args[i] == '--embed' and i + 1 < len(args):
                    config['embed_model'] = args[i + 1]
                    i += 2
                elif args[i] == '--rerank':
                    if i + 1 < len(args) and not args[i + 1].startswith('--'):
                        config['rerank_model'] = args[i + 1]
                        i += 2
                    else:
                        config['rerank_model'] = 'BAAI/bge-reranker-base'
                        i += 1
                else:
                    i += 1

            return config

    # No config found - create default local config
    return create_default_config()


def create_default_config() -> Dict[str, Any]:
    """
    Create default .mcp.json in current directory.

    Returns:
        Default configuration dict
    """
    config_path = Path('./.mcp.json')

    default_config = {
        'mcpServers': {
            'local-faiss-mcp': {
                'command': 'local-faiss-mcp',
                'args': [
                    '--index-dir',
                    './.vector_store'
                ]
            }
        }
    }

    # Only create if it doesn't exist
    if not config_path.exists():
        print(f"Creating default MCP config: {config_path}", file=sys.stderr)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)

    return {
        'index_dir': './.vector_store',
        'embed_model': 'all-MiniLM-L6-v2',
        'rerank_model': None
    }


def collect_files(patterns: List[str], recursive: bool = False) -> List[Path]:
    """
    Collect files from patterns and folders.

    Args:
        patterns: List of file paths, glob patterns, or folders
        recursive: Whether to search folders recursively

    Returns:
        List of file paths to index
    """
    files = []

    for pattern in patterns:
        path = Path(pattern)

        # If it's a directory
        if path.is_dir():
            if recursive:
                # Recursively find all files
                for ext in ['*.txt', '*.md', '*.pdf', '*.docx', '*.html', '*.rst', '*.log']:
                    files.extend(path.rglob(ext))
            else:
                # Only files in this directory
                for item in path.iterdir():
                    if item.is_file():
                        files.append(item)

        # If it's a glob pattern
        elif '*' in pattern or '?' in pattern:
            matched = glob_files(pattern, recursive=recursive)
            files.extend([Path(f) for f in matched if Path(f).is_file()])

        # If it's a single file
        elif path.is_file():
            files.append(path)

        else:
            print(f"Warning: Path not found: {pattern}", file=sys.stderr)

    # Remove duplicates and sort
    unique_files = sorted(set(files))
    return unique_files


def cmd_index(args):
    """Index documents into the vector store."""
    # Get configuration
    config = get_faiss_config()

    # Collect files to index
    files = collect_files(args.files, recursive=args.recursive)

    if not files:
        print("Error: No files found to index", file=sys.stderr)
        return 1

    print(f"\nIndexing {len(files)} file(s)...\n")

    # Initialize vector store
    index_dir = Path(config['index_dir']).resolve()
    index_dir.mkdir(parents=True, exist_ok=True)  # Create if doesn't exist
    index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.json"

    # Check if index already exists
    if index_path.exists():
        print(f"Adding to existing index at: {index_dir}")
    else:
        print(f"Creating new index at: {index_dir}")

    vector_store = FAISSVectorStore(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        embedding_model_name=config['embed_model']
    )

    # Index each file
    success_count = 0
    fail_count = 0

    for file_path in files:
        try:
            print(f"📄 Indexing: {file_path}")

            # Parse document
            document_text = parse_document(file_path)

            # Ingest into vector store (adds to existing index)
            result = vector_store.ingest(document_text, source=str(file_path))

            if result["success"]:
                print(f"   ✓ Added {result['chunks_added']} chunks")
                success_count += 1
            else:
                print(f"   ✗ Failed: {result.get('error', 'Unknown error')}")
                fail_count += 1

        except Exception as e:
            print(f"   ✗ Error: {str(e)}")
            fail_count += 1

    print(f"\n{'='*60}")
    print(f"Indexing complete!")
    print(f"  Success: {success_count} file(s)")
    print(f"  Failed: {fail_count} file(s)")
    print(f"  Total documents in store: {len(vector_store.metadata['documents'])}")
    print(f"  Index location: {index_dir}")
    print(f"{'='*60}")

    return 0 if fail_count == 0 else 1


def cmd_search(args):
    """Search the vector store."""
    # Get configuration
    config = get_faiss_config()

    # Initialize vector store
    index_dir = Path(config['index_dir']).resolve()
    index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.json"

    if not index_path.exists():
        print(f"Error: No index found at {index_dir}", file=sys.stderr)
        print("Run 'local-faiss index <files>' first to create an index", file=sys.stderr)
        return 1

    vector_store = FAISSVectorStore(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        embedding_model_name=config['embed_model'],
        rerank_model_name=config['rerank_model']
    )

    # Perform search
    results = vector_store.query(args.query, top_k=args.top_k)

    if not results:
        print("No results found.")
        return 0

    print(f"\nFound {len(results)} relevant chunk(s):\n")
    print("="*60)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Source: {result['source']}")
        print(f"   Distance: {result['distance']:.4f}")
        if 'rerank_score' in result:
            print(f"   Rerank Score: {result['rerank_score']:.4f}")
        print(f"\n   {result['text'][:300]}...")
        print("-"*60)

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Local FAISS vector database CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The CLI uses configuration from MCP config files in this order:
  1. ./.mcp.json (local/project-specific)
  2. ~/.claude/.mcp.json (Claude Code config)
  3. ~/.mcp.json (fallback)

If no config exists, creates ./.mcp.json with default settings.

Examples:
  # Index single file
  local-faiss index document.pdf

  # Index multiple files
  local-faiss index doc1.pdf doc2.txt doc3.md

  # Index all files in a folder
  local-faiss index documents/

  # Index recursively
  local-faiss index -r documents/

  # Index with glob pattern
  local-faiss index "docs/**/*.pdf"

  # Search the index
  local-faiss search "What is FAISS?"

  # Search with more results
  local-faiss search -k 5 "similarity search"
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Index command
    index_parser = subparsers.add_parser('index', help='Index documents into vector store')
    index_parser.add_argument(
        'files',
        nargs='+',
        help='Files, folders, or glob patterns to index'
    )
    index_parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Recursively search folders for documents'
    )

    # Search command
    search_parser = subparsers.add_parser('search', help='Search the vector store')
    search_parser.add_argument(
        'query',
        type=str,
        help='Natural language search query'
    )
    search_parser.add_argument(
        '-k', '--top-k',
        type=int,
        default=3,
        dest='top_k',
        help='Number of results to return (default: 3)'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == 'index':
        return cmd_index(args)
    elif args.command == 'search':
        return cmd_search(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
