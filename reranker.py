"""
Cross-encoder reranking.

Vector/BM25 retrieval is a *bi-encoder* approach: the query and each chunk
are embedded independently, then compared by a cheap similarity metric.
That's fast but approximate — it has no way to look at the query and a
specific chunk *together*.

A cross-encoder does the opposite: it takes (query, chunk) pairs jointly
and scores how well each chunk actually answers that specific query. It's
too slow to run over an entire corpus, but run over a small candidate set
(the top ~15-20 from hybrid retrieval) it reliably reorders that shortlist
so the truly best chunks end up in the final top-k sent to the LLM. This is
the standard "retrieve many, rerank, keep few" pattern used in production
RAG systems, and is the single biggest lever left for answer precision on
top of hybrid retrieval.

The model is loaded lazily and cached (`st.cache_resource` in app.py wraps
`get_reranker`) since loading it is the expensive part, not scoring.
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document

from src.utils import logger

# A small, fast, well-established reranker. Swap for "BAAI/bge-reranker-base"
# (larger, slightly stronger, slower on CPU) if you have GPU/time to spare.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None


def is_reranker_available() -> bool:
    """Check the optional dependency exists without importing it eagerly."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def get_reranker():
    """Lazily load and cache the cross-encoder model."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading reranker model '%s'...", RERANKER_MODEL)
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, docs: Sequence[Document], top_n: int) -> list[Document]:
    """
    Score each (query, doc) pair with the cross-encoder and return the
    top_n documents in descending relevance order. The score is attached
    to each document's metadata as 'rerank_score' so the UI can show it.

    Falls back to returning the first top_n docs unchanged (with a logged
    warning) if the reranker can't be loaded — a missing optional
    dependency should degrade gracefully, not crash the chat.
    """
    if not docs:
        return []
    try:
        model = get_reranker()
        pairs = [[query, d.page_content] for d in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        result = []
        for doc, score in ranked[:top_n]:
            doc.metadata = {**doc.metadata, "rerank_score": round(float(score), 4)}
            result.append(doc)
        return result
    except Exception as e:  # noqa: BLE001 - degrade, don't break the chat
        logger.warning("Reranking failed (%s); falling back to un-reranked order", e)
        return list(docs[:top_n])
