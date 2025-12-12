from pathlib import Path
from local_faiss_mcp.metadata_extraction import (
    build_metadata_for_pdfs,
    DocumentMetadata,
)

docs_dir = Path("pdf_docs") 
pdf_paths = list(docs_dir.glob("*.pdf"))

print(f"Found {len(pdf_paths)} PDF documents for metadata extraction.")

metas = build_metadata_for_pdfs(pdf_paths)

for m in metas:
    print("------------------------------")
    print(f"ID: {m.id}")
    print(f"Title: {m.title}")
    print(f"Author: {m.author}")
    print(f"Pages: {m.pages}")
    print(f"Word Count: {m.word_count}")
    print(f"Created At: {m.created_at}")
    print(f"Chunks: {m.chunks}")
