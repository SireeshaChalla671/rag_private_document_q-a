"""
Tests for src/rag_chain.py.

ChatOllama itself is stubbed out (no real LLM call), so these tests check
the *wiring* we actually wrote: retriever selection logic (hybrid vs.
vector-only fallback), candidate-pool sizing for the reranker, and the
chat-history-dict -> LangChain-message conversion used before every query.
"""
from unittest.mock import MagicMock, patch

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.rag_chain import _build_retriever, to_lc_messages


class _FakeVectorRetriever(BaseRetriever):
    """A minimal real Runnable so EnsembleRetriever's pydantic validation
    (which requires an actual Runnable, not a bare MagicMock) is satisfied."""

    def _get_relevant_documents(self, query, *, run_manager: CallbackManagerForRetrieverRun):
        return []


def _fake_vectorstore(corpus_size=10):
    vs = MagicMock()
    vs._collection.count.return_value = corpus_size
    retriever = _FakeVectorRetriever()
    vs.as_retriever.return_value = retriever
    return vs, retriever


def _docs(n):
    return [Document(page_content=f"chunk {i}", metadata={"source": f"doc{i}.pdf"}) for i in range(n)]


def test_build_retriever_uses_hybrid_when_corpus_available():
    vs, vector_retriever = _fake_vectorstore()
    result = _build_retriever(vs, all_docs=_docs(5), use_hybrid=True, candidate_k=4)
    # EnsembleRetriever wraps both a BM25Retriever and the vector retriever.
    assert hasattr(result, "retrievers")
    assert len(result.retrievers) == 2


def test_build_retriever_falls_back_to_vector_only_when_hybrid_disabled():
    vs, vector_retriever = _fake_vectorstore()
    result = _build_retriever(vs, all_docs=_docs(5), use_hybrid=False, candidate_k=4)
    assert result is vector_retriever


def test_build_retriever_falls_back_to_vector_only_with_empty_corpus():
    """A freshly cleared knowledge base has no docs to build a BM25 index
    from — must not crash, must degrade to vector search."""
    vs, vector_retriever = _fake_vectorstore()
    result = _build_retriever(vs, all_docs=[], use_hybrid=True, candidate_k=4)
    assert result is vector_retriever


def test_build_retriever_degrades_gracefully_if_bm25_build_fails():
    vs, vector_retriever = _fake_vectorstore()
    with patch("src.rag_chain.BM25Retriever.from_documents", side_effect=RuntimeError("boom")):
        result = _build_retriever(vs, all_docs=_docs(3), use_hybrid=True, candidate_k=4)
    assert result is vector_retriever


def test_build_retriever_vector_search_uses_mmr_with_wider_fetch_k():
    vs, vector_retriever = _fake_vectorstore()
    _build_retriever(vs, all_docs=[], use_hybrid=False, candidate_k=4)

    _, kwargs = vs.as_retriever.call_args
    assert kwargs["search_type"] == "mmr"
    assert kwargs["search_kwargs"]["fetch_k"] > kwargs["search_kwargs"]["k"]  # diversity pool > k


def test_to_lc_messages_converts_roles_and_skips_unknown():
    messages = [
        {"role": "user", "content": "What is the return window?"},
        {"role": "assistant", "content": "30 days.", "sources": []},
        {"role": "system", "content": "should be ignored"},
    ]
    lc_messages = to_lc_messages(messages)

    assert len(lc_messages) == 2
    assert lc_messages[0].content == "What is the return window?"
    assert lc_messages[0].type == "human"
    assert lc_messages[1].content == "30 days."
    assert lc_messages[1].type == "ai"


def test_to_lc_messages_empty_history():
    assert to_lc_messages([]) == []
