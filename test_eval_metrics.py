import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import aggregate, hit_at_k, keyword_coverage, reciprocal_rank


def test_hit_at_k_true_when_source_present():
    assert hit_at_k(["a.pdf", "b.pdf"], "a.pdf") is True


def test_hit_at_k_false_when_source_absent():
    assert hit_at_k(["b.pdf", "c.pdf"], "a.pdf") is False


def test_hit_at_k_none_when_no_expected_source():
    assert hit_at_k(["a.pdf"], None) is None


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a.pdf", "b.pdf"], "a.pdf") == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["b.pdf", "c.pdf", "a.pdf"], "a.pdf") == pytest_approx(1 / 3)


def test_reciprocal_rank_zero_when_never_found():
    assert reciprocal_rank(["b.pdf", "c.pdf"], "a.pdf") == 0.0


def test_reciprocal_rank_none_when_no_expected_source():
    assert reciprocal_rank(["a.pdf"], None) is None


def test_keyword_coverage_full_match():
    assert keyword_coverage("Refunds take 5-7 business days.", ["5-7 business days"]) == 1.0


def test_keyword_coverage_partial_match():
    score = keyword_coverage("Refunds take 5-7 business days.", ["5-7 business days", "inspected"])
    assert score == 0.5


def test_keyword_coverage_case_insensitive():
    assert keyword_coverage("REFUNDS TAKE 5-7 BUSINESS DAYS", ["5-7 business days"]) == 1.0


def test_keyword_coverage_empty_keywords_is_trivially_full():
    assert keyword_coverage("anything at all", []) == 1.0


def test_keyword_coverage_no_match():
    assert keyword_coverage("Unrelated text.", ["5-7 business days"]) == 0.0


def test_aggregate_mixes_retrieval_applicable_and_refusal_cases():
    results = [
        {"hit": True, "mrr": 1.0, "keyword_coverage": 1.0, "latency_s": 1.0},
        {"hit": False, "mrr": 0.0, "keyword_coverage": 0.5, "latency_s": 2.0},
        {"hit": None, "mrr": None, "keyword_coverage": 1.0, "latency_s": 0.5},  # refusal case
    ]
    summary = aggregate(results)

    assert summary["n_questions"] == 3
    assert summary["hit_rate"] == 0.5  # only 2 of 3 cases are retrieval-applicable
    assert summary["mrr"] == 0.5
    assert summary["avg_keyword_coverage"] == pytest_approx((1.0 + 0.5 + 1.0) / 3)
    assert summary["perfect_keyword_matches"] == 2


def test_aggregate_all_refusal_cases_gives_none_retrieval_metrics():
    results = [{"hit": None, "mrr": None, "keyword_coverage": 1.0, "latency_s": 0.1}]
    summary = aggregate(results)
    assert summary["hit_rate"] is None
    assert summary["mrr"] is None


def test_aggregate_empty_results():
    summary = aggregate([])
    assert summary["n_questions"] == 0
    assert summary["avg_keyword_coverage"] == 0.0


def pytest_approx(value, rel=1e-6):
    import pytest
    return pytest.approx(value, rel=rel)
