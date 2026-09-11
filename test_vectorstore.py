"""
Tests for src/vectorstore.py.

src/vectorstore.py imports `langchain_chroma`, `langchain_ollama`, and
`chromadb` at module level — all of which either need a running Ollama
server or a multi-hundred-MB install. Neither should be required to run
`pytest`. Instead, we install lightweight fake modules into sys.modules
*before* src.vectorstore is imported, so the module-level imports resolve
to fakes we fully control, and we test the actual logic in vectorstore.py
(dedup-on-reindex, source counting, corpus reconstruction) against an
in-memory fake Chroma collection.
"""
import sys
import types
from unittest.mock import MagicMock

import pytest


class _FakeCollection:
    """Minimal in-memory stand-in for a Chroma collection's storage."""

    def __init__(self):
        self._by_id = {}  # id -> (document_text, metadata)
        self._counter = 0

    def add_documents(self, docs):
        for doc in docs:
            self._counter += 1
            self._by_id[str(self._counter)] = (doc.page_content, dict(doc.metadata))

    def get(self, where=None, include=None):
        ids, docs, metas = [], [], []
        for _id, (text, meta) in self._by_id.items():
            if where and not all(meta.get(k) == v for k, v in where.items()):
                continue
            ids.append(_id)
            docs.append(text)
            metas.append(meta)
        result = {"ids": ids}
        if include is None or "documents" in include:
            result["documents"] = docs
        if include is None or "metadatas" in include:
            result["metadatas"] = metas
        return result

    def delete(self, ids):
        for _id in ids:
            self._by_id.pop(_id, None)

    def delete_collection(self):
        self._by_id.clear()

    def count(self):
        return len(self._by_id)


@pytest.fixture
def fake_vectorstore_module(monkeypatch):
    """Install stub langchain_chroma / langchain_ollama / chromadb modules,
    then (re)import src.vectorstore against them."""
    fake_collection = _FakeCollection()

    class FakeChroma:
        def __init__(self, collection_name, embedding_function, persist_directory, client_settings=None):
            self.collection_name = collection_name
            self._collection = fake_collection

        def add_documents(self, docs):
            self._collection.add_documents(docs)

        def get(self, where=None, include=None):
            return self._collection.get(where=where, include=include)

        def delete(self, ids):
            self._collection.delete(ids)

        def delete_collection(self):
            self._collection.delete_collection()

    fake_chroma_mod = types.ModuleType("langchain_chroma")
    fake_chroma_mod.Chroma = FakeChroma

    fake_ollama_mod = types.ModuleType("langchain_ollama")
    fake_ollama_mod.OllamaEmbeddings = MagicMock()

    fake_chromadb_mod = types.ModuleType("chromadb")
    fake_chromadb_config_mod = types.ModuleType("chromadb.config")
    fake_chromadb_config_mod.Settings = MagicMock()
    fake_chromadb_mod.config = fake_chromadb_config_mod

    monkeypatch.setitem(sys.modules, "langchain_chroma", fake_chroma_mod)
    monkeypatch.setitem(sys.modules, "langchain_ollama", fake_ollama_mod)
    monkeypatch.setitem(sys.modules, "chromadb", fake_chromadb_mod)
    monkeypatch.setitem(sys.modules, "chromadb.config", fake_chromadb_config_mod)

    sys.modules.pop("src.vectorstore", None)
    import src.vectorstore as vectorstore
    yield vectorstore, fake_collection
    sys.modules.pop("src.vectorstore", None)


def _doc(text, source, page=None):
    from langchain_core.documents import Document
    meta = {"source": source}
    if page is not None:
        meta["page"] = page
    return Document(page_content=text, metadata=meta)


def test_add_and_list_indexed_sources(fake_vectorstore_module):
    vectorstore_mod, _ = fake_vectorstore_module
    vs = vectorstore_mod.get_vectorstore()
    vectorstore_mod.add_documents(vs, [_doc("chunk1", "a.pdf"), _doc("chunk2", "a.pdf"), _doc("chunk3", "b.pdf")])

    sources = vectorstore_mod.list_indexed_sources(vs)
    assert sources == {"a.pdf": 2, "b.pdf": 1}


def test_reindexing_replaces_not_duplicates(fake_vectorstore_module):
    """The exact bug the code comments call out: re-uploading a file must
    replace its chunks, not add a second copy that skews retrieval."""
    vectorstore_mod, _ = fake_vectorstore_module
    vs = vectorstore_mod.get_vectorstore()

    vectorstore_mod.add_documents(vs, [_doc("v1 chunk1", "a.pdf"), _doc("v1 chunk2", "a.pdf")])
    assert vectorstore_mod.list_indexed_sources(vs)["a.pdf"] == 2

    # Simulate re-indexing "a.pdf" with different chunk_size -> 3 chunks now
    vectorstore_mod.delete_by_source(vs, "a.pdf")
    vectorstore_mod.add_documents(vs, [_doc("v2 chunk1", "a.pdf"), _doc("v2 chunk2", "a.pdf"), _doc("v2 chunk3", "a.pdf")])

    assert vectorstore_mod.list_indexed_sources(vs)["a.pdf"] == 3  # not 5


def test_delete_by_source_only_removes_matching_file(fake_vectorstore_module):
    vectorstore_mod, _ = fake_vectorstore_module
    vs = vectorstore_mod.get_vectorstore()
    vectorstore_mod.add_documents(vs, [_doc("c1", "a.pdf"), _doc("c2", "b.pdf")])

    vectorstore_mod.delete_by_source(vs, "a.pdf")

    sources = vectorstore_mod.list_indexed_sources(vs)
    assert "a.pdf" not in sources
    assert sources["b.pdf"] == 1


def test_delete_by_source_on_empty_store_does_not_raise(fake_vectorstore_module):
    vectorstore_mod, _ = fake_vectorstore_module
    vs = vectorstore_mod.get_vectorstore()
    vectorstore_mod.delete_by_source(vs, "never_indexed.pdf")  # should not raise


def test_get_all_documents_reconstructs_corpus_for_bm25(fake_vectorstore_module):
    vectorstore_mod, _ = fake_vectorstore_module
    vs = vectorstore_mod.get_vectorstore()
    vectorstore_mod.add_documents(vs, [_doc("alpha", "a.pdf", page=0), _doc("beta", "a.pdf", page=1)])

    docs = vectorstore_mod.get_all_documents(vs)
    contents = {d.page_content for d in docs}
    assert contents == {"alpha", "beta"}
    assert all(d.metadata["source"] == "a.pdf" for d in docs)


def test_clear_vectorstore_empties_collection(fake_vectorstore_module):
    vectorstore_mod, _ = fake_vectorstore_module
    vs = vectorstore_mod.get_vectorstore()
    vectorstore_mod.add_documents(vs, [_doc("c1", "a.pdf")])
    assert vectorstore_mod.list_indexed_sources(vs) != {}

    vectorstore_mod.clear_vectorstore()
    vs2 = vectorstore_mod.get_vectorstore()
    assert vectorstore_mod.list_indexed_sources(vs2) == {}
