import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from local_faiss_mcp.cli import cmd_delete, resolve_indexed_sources
from local_faiss_mcp.server import FAISSVectorStore


class FakeEmbeddingModel:
    def __init__(self, *_args, **_kwargs):
        self.dimension = 4

    def encode(self, texts, convert_to_numpy=True):
        rows = []
        for text in texts:
            value = (sum(ord(ch) for ch in text) % 97) / 97.0
            rows.append([value, value + 1.0, value + 2.0, value + 3.0])
        array = np.array(rows, dtype=np.float32)
        return array if convert_to_numpy else array.tolist()


class DeleteArgs:
    def __init__(self, sources, dry_run=False):
        self.sources = sources
        self.dry_run = dry_run


def make_store(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("local_faiss_mcp.server.SentenceTransformer", FakeEmbeddingModel)
    return FAISSVectorStore(
        index_path=str(tmp_path / "faiss.index"),
        metadata_path=str(tmp_path / "metadata.json"),
    )


def test_remove_sources_rebuilds_index_and_metadata(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    store.ingest("alpha beta gamma", source="alpha.txt")
    store.ingest("delta epsilon zeta", source="beta.txt")

    result = store.remove_sources({"alpha.txt"})

    assert result["success"] is True
    assert result["removed_chunks"] == 1
    assert result["removed_sources"] == {"alpha.txt": 1}
    assert store.index.ntotal == 1
    assert [doc["source"] for doc in store.metadata["documents"]] == ["beta.txt"]
    assert store.metadata["documents"][0]["id"] == 0

    reloaded = make_store(tmp_path, monkeypatch)
    assert reloaded.index.ntotal == 1
    assert [doc["source"] for doc in reloaded.metadata["documents"]] == ["beta.txt"]


def test_remove_sources_dry_run_does_not_modify_store(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)
    store.ingest("alpha beta gamma", source="alpha.txt")

    result = store.remove_sources({"alpha.txt"}, dry_run=True)

    assert result["removed_chunks"] == 1
    assert store.index.ntotal == 1
    assert [doc["source"] for doc in store.metadata["documents"]] == ["alpha.txt"]


def test_resolve_indexed_sources_matches_globs_and_basenames(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "documents": [
            {"id": 0, "source": "docs/a.md", "text": "a"},
            {"id": 1, "source": "docs/b.md", "text": "b"},
            {"id": 2, "source": "notes/c.txt", "text": "c"},
        ]
    }))

    assert resolve_indexed_sources(["*.md"], metadata_path) == {
        "docs/a.md": 1,
        "docs/b.md": 1,
    }
    assert resolve_indexed_sources(["c.txt"], metadata_path) == {"notes/c.txt": 1}


def test_delete_dry_run_reports_without_loading_vector_store(tmp_path, capsys):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "documents": [
            {"id": 0, "source": "docs/a.md", "text": "a"},
            {"id": 1, "source": "docs/a.md", "text": "a2"},
        ]
    }))

    config = {
        "index_dir": str(tmp_path),
        "embed_model": "all-MiniLM-L6-v2",
        "rerank_model": None,
    }

    with patch("local_faiss_mcp.cli.get_faiss_config", return_value=config), patch(
        "local_faiss_mcp.cli.FAISSVectorStore"
    ) as store_cls:
        result = cmd_delete(DeleteArgs(["*.md"], dry_run=True))

    assert result == 0
    store_cls.assert_not_called()
    captured = capsys.readouterr()
    assert "Would delete:" in captured.out
    assert "docs/a.md (2 chunk(s))" in captured.out
