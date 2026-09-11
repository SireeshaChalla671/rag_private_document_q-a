"""
Tests for src/reranker.py.

The real cross-encoder (sentence-transformers) is never loaded here — it's
a ~90MB download and network-dependent, which has no business being a
prerequisite for `pytest`. Instead we monkeypatch `get_reranker` with a
fake model whose `.predict()` behaviour we control, so we can test the
*reranking logic* (sorting, truncation, metadata attachment, graceful
fallback) deterministically.
"""
from unittest.mock import patch

from langchain_core.documents import Document

from src.reranker import rerank


class _FakeCrossEncoder:
    """Scores each pair by how many words of the query appear in the doc —
    good enough to produce a deterministic, checkable ranking."""

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            q_words = set(query.lower().split())
            t_words = set(text.lower().split())
            scores.append(len(q_words & t_words))
        return scores


def _docs():
    return [
        Document(page_content="Refunds take 5-7 business days.", metadata={"source": "a"}),
        Document(page_content="Our office is located downtown.", metadata={"source": "b"}),
        Document(page_content="Refund policy allows returns within 30 days.", metadata={"source": "c"}),
    ]


def test_rerank_reorders_by_relevance_and_truncates():
    with patch("src.reranker.get_reranker", return_value=_FakeCrossEncoder()):
        result = rerank("refund policy", _docs(), top_n=2)

    assert len(result) == 2
    # The two refund-related docs should outrank the office-location doc.
    assert {d.metadata["source"] for d in result} == {"a", "c"}


def test_rerank_attaches_score_to_metadata():
    with patch("src.reranker.get_reranker", return_value=_FakeCrossEncoder()):
        result = rerank("refund policy", _docs(), top_n=3)

    for doc in result:
        assert "rerank_score" in doc.metadata
        assert isinstance(doc.metadata["rerank_score"], float)


def test_rerank_empty_input_returns_empty_list():
    with patch("src.reranker.get_reranker", return_value=_FakeCrossEncoder()):
        assert rerank("anything", [], top_n=5) == []


def test_rerank_degrades_gracefully_when_model_fails_to_load():
    """A missing/broken reranker dependency must never crash the chat —
    it should fall back to the original (unranked) order."""
    docs = _docs()
    with patch("src.reranker.get_reranker", side_effect=RuntimeError("model load failed")):
        result = rerank("refund policy", docs, top_n=2)

    assert result == docs[:2]


def test_rerank_top_n_larger_than_pool_returns_all():
    with patch("src.reranker.get_reranker", return_value=_FakeCrossEncoder()):
        result = rerank("refund policy", _docs(), top_n=10)
    assert len(result) == 3
