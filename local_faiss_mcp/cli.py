#!/usr/bin/env python3
"""
Command-line interface for local-faiss.

Commands:
- index: Index documents into FAISS vector store
- search: Search the vector store
- list: List all indexed documents
"""

import sys
import os
import json
import argparse
import fnmatch
from pathlib import Path
from glob import glob as glob_files
from typing import List, Optional, Dict, Any
from collections import defaultdict
from .server import FAISSVectorStore
from .document_parser import parse_document
from .colors import success, error, info, warning
from .progress import create_file_progress, update_progress_description, progress_print


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
        print(warning(f"Failed to read MCP config: {e}"), file=sys.stderr)
        return {}


def get_faiss_config() -> Dict[str, Any]:
    """
    Get local-faiss-mcp configuration from MCP config.

    Returns:
        Dict with: index_dir, embed_model, rerank_model
    """
    config_path = find_mcp_config()

    if config_path:
        print(info(f"Using MCP config: {config_path}"), file=sys.stderr)
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
        print(info(f"Creating default MCP config: {config_path}"), file=sys.stderr)
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
            print(warning(f"Path not found: {pattern}"), file=sys.stderr)

    # Remove duplicates and sort
    unique_files = sorted(set(files))
    return unique_files


def load_source_counts(metadata_path: Path) -> Dict[str, int]:
    """Load indexed source names and chunk counts from metadata."""
    if not metadata_path.exists():
        return {}

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    source_chunks = defaultdict(int)
    for doc in metadata.get('documents', []):
        source_chunks[doc.get('source', 'unknown')] += 1
    return dict(source_chunks)


def source_matches_pattern(source: str, pattern: str) -> bool:
    """Return whether an indexed source matches a delete/update pattern."""
    normalized_source = source.replace("\\", "/")
    normalized_pattern = pattern.replace("\\", "/")
    source_name = Path(source).name
    pattern_name = Path(pattern).name

    return (
        source == pattern
        or normalized_source == normalized_pattern
        or source_name == pattern
        or source_name == pattern_name
        or fnmatch.fnmatch(normalized_source, normalized_pattern)
        or fnmatch.fnmatch(source_name, normalized_pattern)
        or fnmatch.fnmatch(source_name, pattern_name)
    )


def resolve_indexed_sources(patterns: List[str], metadata_path: Path) -> Dict[str, int]:
    """Resolve user patterns to indexed source names and chunk counts."""
    source_counts = load_source_counts(metadata_path)
    matched = {}

    for pattern in patterns:
        for source, count in source_counts.items():
            if source_matches_pattern(source, pattern):
                matched[source] = count

    return matched


def make_vector_store(config: Dict[str, Any]) -> FAISSVectorStore:
    """Create a vector store using CLI configuration."""
    index_dir = Path(config['index_dir']).resolve()
    index_dir.mkdir(parents=True, exist_ok=True)
    return FAISSVectorStore(
        index_path=str(index_dir / "faiss.index"),
        metadata_path=str(index_dir / "metadata.json"),
        embedding_model_name=config['embed_model'],
        rerank_model_name=config['rerank_model']
    )


def cmd_index(args):
    """Index documents into the vector store."""
    # Get configuration
    config = get_faiss_config()

    # Collect files to index
    files = collect_files(args.files, recursive=args.recursive)

    if not files:
        print(error("No files found to index"), file=sys.stderr)
        return 1

    # Print appropriate header based on file count
    if len(files) > 1:
        print(f"\nIndexing {len(files)} files...\n")
    else:
        print(f"\nIndexing {len(files)} file(s)...\n")

    # Initialize vector store
    index_dir = Path(config['index_dir']).resolve()
    index_dir.mkdir(parents=True, exist_ok=True)  # Create if doesn't exist
    index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.json"

    # Check if index already exists
    if index_path.exists():
        print(info(f"Adding to existing index at: {index_dir}"))
    else:
        print(info(f"Creating new index at: {index_dir}"))

    vector_store = FAISSVectorStore(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
        embedding_model_name=config['embed_model']
    )

    # Index each file
    success_count = 0
    fail_count = 0

    # Create progress bar wrapper for files
    files_iter, show_progress = create_file_progress(files, desc="Indexing")

    for file_path in files_iter:
        try:
            # Update progress bar with current filename
            update_progress_description(files_iter, file_path, "Indexing")
            progress_print(f"📄 Indexing: {file_path}", show_progress)

            # Parse document
            document_text = parse_document(file_path)

            # Ingest into vector store (adds to existing index)
            result = vector_store.ingest(document_text, source=str(file_path))

            if result["success"]:
                chunks_added = result["chunks_added"]
                progress_print(f"   {success(f'Added {chunks_added} chunks')}", show_progress)
                success_count += 1
            else:
                err_msg = result.get("error", "Unknown error")
                progress_print(f"   {error(f'Failed: {err_msg}')}", show_progress)
                fail_count += 1

        except Exception as e:
            err_str = str(e)
            progress_print(f"   {error(f'Error: {err_str}')}", show_progress)
            fail_count += 1

    print(f"\n{'='*60}")
    print("Indexing complete!")
    if success_count > 0:
        print(f"  {success(f'Success: {success_count} file(s)')}")   
    if fail_count > 0:
        print(f"  {error(f'Failed: {fail_count} file(s)')}")    
    total_docs = len(vector_store.metadata['documents'])
    print(info(f"Total documents in store: {total_docs}"))
    print(info(f"Index location: {index_dir}"))
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
        print(error(f"No index found at {index_dir}"), file=sys.stderr)
        print(info("Run 'local-faiss index <files>' first to create an index"), file=sys.stderr)
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
        print(info("No results found."))
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


def cmd_delete(args):
    """Delete indexed documents from the vector store."""
    config = get_faiss_config()
    index_dir = Path(config['index_dir']).resolve()
    index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.json"

    if not metadata_path.exists():
        print(info("No documents indexed yet."))
        return 0

    matched_sources = resolve_indexed_sources(args.sources, metadata_path)
    if not matched_sources:
        print(warning("No indexed sources matched the supplied pattern(s)."))
        return 0

    if args.dry_run:
        print("\nWould delete:")
        for source, chunks in sorted(matched_sources.items()):
            print(f"- {source} ({chunks} chunk(s))")
        print(info(f"Total chunks: {sum(matched_sources.values())}"))
        return 0

    if not index_path.exists():
        print(error(f"No index found at {index_dir}"), file=sys.stderr)
        return 1

    vector_store = make_vector_store(config)
    result = vector_store.remove_sources(set(matched_sources))

    removed_chunks = result["removed_chunks"]
    removed_source_count = len(result["removed_sources"])
    print(success(f"Removed {removed_chunks} chunk(s) from {removed_source_count} source(s)."))
    return 0


def cmd_update(args):
    """Update indexed documents by deleting and re-indexing them."""
    config = get_faiss_config()
    index_dir = Path(config['index_dir']).resolve()
    metadata_path = index_dir / "metadata.json"

    matched_sources = resolve_indexed_sources(args.files, metadata_path)
    if not matched_sources:
        print(warning("No existing indexed sources matched; indexing as new document(s)."))
    elif args.dry_run:
        print("\nWould update:")
        for source, chunks in sorted(matched_sources.items()):
            print(f"- {source} ({chunks} existing chunk(s))")
        return 0
    else:
        vector_store = make_vector_store(config)
        result = vector_store.remove_sources(set(matched_sources))
        print(info(f"Removed {result['removed_chunks']} existing chunk(s) before update."))

    return cmd_index(args)


def cmd_list(args):
    """List all indexed documents in the vector store."""
    # Get configuration
    config = get_faiss_config()

    # Build metadata path
    index_dir = Path(config['index_dir']).resolve()
    metadata_path = index_dir / "metadata.json"

    # Check if metadata exists
    if not metadata_path.exists():
        if args.json:
            print(json.dumps({"documents": [], "total": 0}))
        else:
            print(info("No documents indexed yet."))
            print(info("Run 'local-faiss index <files>' to index documents."))
        return 0

    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(error(f"Failed to read metadata: {e}"), file=sys.stderr)
        return 1

    documents = metadata.get('documents', [])

    if not documents:
        if args.json:
            print(json.dumps({"documents": [], "total": 0}))
        else:
            print(info("No documents indexed yet."))
            print(info("Run 'local-faiss index <files>' to index documents."))
        return 0

    # Group by source and count chunks, track indexed dates
    source_chunks = defaultdict(int)
    source_dates = {}
    for doc in documents:
        source = doc.get('source', 'unknown')
        source_chunks[source] += 1
        # Keep the most recent indexed_at date for each source
        indexed_at = doc.get('indexed_at')
        if indexed_at and (source not in source_dates or indexed_at > source_dates[source]):
            source_dates[source] = indexed_at

    # Sort sources alphabetically
    sorted_sources = sorted(source_chunks.keys())

    if args.json:
        # JSON output
        output = {
            "documents": [
                {
                    "source": source,
                    "chunks": source_chunks[source],
                    "indexed_at": source_dates.get(source)
                }
                for source in sorted_sources
            ],
            "total": len(sorted_sources)
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        print(f"\nIndexed Documents ({len(sorted_sources)} total):\n")
        print("="*60)

        for i, source in enumerate(sorted_sources, 1):
            chunk_count = source_chunks[source]
            indexed_at = source_dates.get(source)
            print(f"\n{i}. {source}")
            print(f"   Chunks: {chunk_count}")
            if indexed_at:
                # Format the ISO date more readably
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(indexed_at)
                    formatted_date = dt.strftime("%Y-%m-%d %H:%M")
                    print(f"   Indexed: {formatted_date}")
                except (ValueError, TypeError):
                    print(f"   Indexed: {indexed_at}")

        print("\n" + "="*60)
        print(info(f"Index location: {index_dir}"))

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

    # List command
    list_parser = subparsers.add_parser('list', help='List all indexed documents')
    list_parser.add_argument(
        '--json',
        action='store_true',
        help='Output in JSON format for machine-readable output'
    )

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete indexed documents from vector store')
    delete_parser.add_argument(
        'sources',
        nargs='+',
        help='Indexed source names or glob patterns to delete'
    )
    delete_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without modifying the index'
    )

    # Update command
    update_parser = subparsers.add_parser('update', help='Delete and re-index documents')
    update_parser.add_argument(
        'files',
        nargs='+',
        help='Files, folders, or glob patterns to update'
    )
    update_parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Recursively search folders for documents'
    )
    update_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be updated without modifying the index'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == 'index':
        return cmd_index(args)
    elif args.command == 'search':
        return cmd_search(args)
    elif args.command == 'list':
        return cmd_list(args)
    elif args.command == 'delete':
        return cmd_delete(args)
    elif args.command == 'update':
        return cmd_update(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())

