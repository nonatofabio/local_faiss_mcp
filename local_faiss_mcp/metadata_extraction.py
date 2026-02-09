from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Union, Iterable
from pypdf import PdfReader


@dataclass
class DocumentMetadata:
    # Core identifiers
    id: str
    path: str

    # PDF metadata
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    created_at: Optional[str] = None          # store as ISO string "YYYY-MM-DD"
    modified_at: Optional[str] = None         # same
    document_type: str = "pdf"

    # Stats
    pages: int = 0
    word_count: int = 0
    chunks: int = 0

    # Tags
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """For storing in JSON or similar formats."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentMetadata":
        """For loading from metadata.json."""
        return cls(**data)


def _parse_pdf_date(raw: Optional[Union[str, datetime]]) -> Optional[str]:
    """Convert a PDF date like 'D:20240115123045Z' to '2024-01-15'."""
    if not raw:
        return None

    s = str(raw)
    if s.startswith("D:"):
        s = s[2:]

    # Take just YYYYMMDD
    date_part = s[:8]
    try:
        dt = datetime.strptime(date_part, "%Y%m%d")
        return dt.date().isoformat()  # "YYYY-MM-DD"
    except ValueError:
        return None


def build_metadata_for_pdf(
    path: Union[str, Path],
    chunks: int = 0,
    tags: Optional[List[str]] = None,
) -> DocumentMetadata:
    path = Path(path)
    reader = PdfReader(path)
    meta = reader.metadata  

    # These may all be None, depends on the pdf
    title = meta.title
    author = meta.author
    subject = getattr(meta, "subject", None)
    creator = getattr(meta, "creator", None)
    producer = getattr(meta, "producer", None)

    created_at = _parse_pdf_date(getattr(meta, "creation_date", None))
    modified_at = _parse_pdf_date(getattr(meta, "modification_date", None))

    pages = len(reader.pages)

    # Quick word count
    text_chunks: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text_chunks.append(text)
    word_count = len(" ".join(text_chunks).split())

    tags = tags or []

    return DocumentMetadata(
        id=path.name,
        path=str(path),
        title=title,
        author=author,
        subject=subject,
        creator=creator,
        producer=producer,
        created_at=created_at,
        modified_at=modified_at,
        pages=pages,
        word_count=word_count,
        tags=tags,
        chunks=chunks,
    )

def build_metadata_for_pdfs(paths: Iterable[str | Path]) -> List[DocumentMetadata]:
    metas: List[DocumentMetadata] = []
    for p in paths:
        meta = build_metadata_for_pdf(p)  # from earlier
        metas.append(meta)
    return metas