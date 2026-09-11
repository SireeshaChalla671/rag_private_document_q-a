"""
Pure, dependency-free scoring functions used by evaluate.py.

Kept separate from evaluate.py deliberately: these functions take plain
Python data in and return plain Python data out, so they can be unit
tested (see tests/test_eval_metrics.py) without needing Ollama, Chroma,
or any network access. evaluate.py is the thin, harder-to-test layer that
wires these functions up to the real pipeline.
"""
from __future__ import annotations


def hit_at_k(retrieved_sources: list[str], expected_source: str | None) -> bool | None:
    """
    Did the expected source document appear anywhere in the retrieved
    chunks? Returns None (not applicable) for refusal cases that have no
    expected source, so they're excluded from the retrieval metric rather
    than counted as a miss.
    """
    if expected_source is None:
        return None
    return expected_source in retrieved_sources


def reciprocal_rank(retrieved_sources: list[str], expected_source: str | None) -> float | None:
    """
    1/rank of the first chunk from the expected source (1.0 = it was the
    very first retrieved chunk), or 0.0 if it never appears. None for
    cases with no expected source.
    """
    if expected_source is None:
        return None
    for i, src in enumerate(retrieved_sources, start=1):
        if src == expected_source:
            return 1.0 / i
    return 0.0


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    """
    Fraction of expected_keywords found (case-insensitive substring match)
    in the generated answer. A lightweight, deterministic proxy for
    "did the answer actually contain the right facts" that doesn't
    require an LLM judge or exact string matching against a reference
    answer LLMs rarely phrase identically.
    """
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def aggregate(results: list[dict]) -> dict:
    """
    Roll up per-question results into summary statistics.

    Each item in `results` is expected to have keys: hit, mrr (both may
    be None for refusal cases), keyword_coverage, latency_s.
    """
    retrieval_applicable = [r for r in results if r["hit"] is not None]
    n = len(results)

    return {
        "n_questions": n,
        "hit_rate": (
            sum(1 for r in retrieval_applicable if r["hit"]) / len(retrieval_applicable)
            if retrieval_applicable else None
        ),
        "mrr": (
            sum(r["mrr"] for r in retrieval_applicable) / len(retrieval_applicable)
            if retrieval_applicable else None
        ),
        "avg_keyword_coverage": sum(r["keyword_coverage"] for r in results) / n if n else 0.0,
        "avg_latency_s": sum(r["latency_s"] for r in results) / n if n else 0.0,
        "perfect_keyword_matches": sum(1 for r in results if r["keyword_coverage"] == 1.0),
    }
