"""
Shared pytest fixtures.

Key design decision: these tests never talk to a real Ollama server or a
real sentence-transformers model. Both are external services/heavy
downloads that shouldn't be a prerequisite for running `pytest`. Instead:

  - Ollama-backed classes (ChatOllama, OllamaEmbeddings) are replaced with
    lightweight fakes that implement just the interface our code calls.
  - The cross-encoder in src/reranker.py is monkeypatched at the
    `get_reranker` boundary so reranking logic is tested without loading
    a real model.

This mirrors how you'd test this kind of app in industry: the RAG
*orchestration logic* (chunking, retrieval wiring, citation formatting,
dedup-on-reindex) is what should be unit tested; the model weights
themselves are the vendor's problem, not ours.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_workdir(monkeypatch, tmp_path):
    """Point config's PERSIST_DIR / UPLOAD_DIR at a scratch dir for this test only."""
    import src.config as config

    persist_dir = tmp_path / "chroma_db"
    upload_dir = tmp_path / "temp_uploads"
    upload_dir.mkdir()

    monkeypatch.setattr(config, "PERSIST_DIR", str(persist_dir))
    monkeypatch.setattr(config, "UPLOAD_DIR", upload_dir)
    yield tmp_path


@pytest.fixture
def sample_pdf_path():
    return ROOT / "sample_docs" / "sample_return_policy.pdf"


@pytest.fixture
def sample_txt_path(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text(
        "Refunds are processed within 5-7 business days once the return "
        "is received and inspected.\n\nExchanges are free within 30 days "
        "of purchase, provided the item is unused and in original packaging."
    )
    return p
