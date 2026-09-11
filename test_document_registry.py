"""Tests for src/document_registry.py — the JSON sidecar manifest."""
import json

import pytest


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch, tmp_path):
    """Every test gets its own registry file, never the real project's."""
    import src.document_registry as registry

    monkeypatch.setattr(registry, "REGISTRY_PATH", tmp_path / "chroma_db" / "document_registry.json")
    yield registry


def test_register_and_list_document(_isolated_registry):
    _isolated_registry.register_document("policy.pdf", size_bytes=2048, num_pages=3)

    docs = _isolated_registry.list_documents()
    assert "policy.pdf" in docs
    assert docs["policy.pdf"]["size_bytes"] == 2048
    assert docs["policy.pdf"]["num_pages"] == 3
    assert docs["policy.pdf"]["file_type"] == "PDF"
    assert "uploaded_at" in docs["policy.pdf"]


def test_registering_same_filename_twice_overwrites_not_duplicates(_isolated_registry):
    _isolated_registry.register_document("notes.txt", size_bytes=100, num_pages=0)
    _isolated_registry.register_document("notes.txt", size_bytes=999, num_pages=0)

    docs = _isolated_registry.list_documents()
    assert len(docs) == 1
    assert docs["notes.txt"]["size_bytes"] == 999


def test_remove_document(_isolated_registry):
    _isolated_registry.register_document("a.pdf", 100, 1)
    _isolated_registry.register_document("b.pdf", 200, 2)
    _isolated_registry.remove_document("a.pdf")

    docs = _isolated_registry.list_documents()
    assert "a.pdf" not in docs
    assert "b.pdf" in docs


def test_remove_nonexistent_document_does_not_raise(_isolated_registry):
    _isolated_registry.remove_document("never_existed.pdf")  # should not raise
    assert _isolated_registry.list_documents() == {}


def test_clear_all(_isolated_registry):
    _isolated_registry.register_document("a.pdf", 100, 1)
    _isolated_registry.clear_all()
    assert _isolated_registry.list_documents() == {}


def test_list_documents_survives_corrupt_json(_isolated_registry):
    _isolated_registry.REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _isolated_registry.REGISTRY_PATH.write_text("{not valid json")

    # A corrupted manifest shouldn't crash the sidebar on startup — it
    # should degrade to "no documents recorded" rather than raising.
    assert _isolated_registry.list_documents() == {}


def test_registry_persists_across_reload(_isolated_registry, tmp_path):
    _isolated_registry.register_document("persisted.pdf", 500, 2)
    raw = json.loads(_isolated_registry.REGISTRY_PATH.read_text())
    assert raw["persisted.pdf"]["size_bytes"] == 500
