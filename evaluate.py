"""
Automated evaluation harness for the RAG pipeline.

Runs eval/qa_testset.json against the live pipeline (real Ollama + real
Chroma - this needs `ollama serve` running and sample_docs/sample_return_policy.pdf
already indexed) and reports:

  - Hit Rate@k and MRR       - did retrieval find the right source document?
  - Keyword coverage         - did the generated answer contain the expected facts?
  - Latency                  - end-to-end seconds per question

It also supports comparing configurations, so you can show *why* the
hybrid+reranker design in this project beats a naive vector-only baseline,
instead of just asserting it in the README.

Usage:
    # 1. Make sure Ollama is running and the sample doc is indexed:
    ollama serve
    python eval/index_sample_doc.py

    # 2. Run the eval (baseline vector-only vs. this project's full pipeline):
    python eval/evaluate.py --compare

    # or just the current default config:
    python eval/evaluate.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import aggregate, hit_at_k, keyword_coverage, reciprocal_rank
from src.config import DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE, DEFAULT_TOP_K
from src.rag_chain import build_rag_chain, to_lc_messages
from src.utils import check_ollama_connection
from src.vectorstore import get_all_documents, get_vectorstore

TESTSET_PATH = Path(__file__).parent / "qa_testset.json"


def load_testset() -> list[dict]:
    data = json.loads(TESTSET_PATH.read_text())
    return data["cases"]


def run_config(cases: list[dict], *, use_hybrid: bool, use_reranker: bool, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    vectorstore = get_vectorstore()
    all_docs = get_all_documents(vectorstore) if use_hybrid else []

    chain = build_rag_chain(
        vectorstore, DEFAULT_LLM_MODEL, DEFAULT_TEMPERATURE, top_k,
        all_docs=all_docs, use_hybrid=use_hybrid, use_reranker=use_reranker,
    )

    results = []
    for case in cases:
        start = time.perf_counter()
        response = chain.invoke({"input": case["question"], "chat_history": []})
        latency = time.perf_counter() - start

        retrieved_sources = [d.metadata.get("source", "unknown") for d in response.get("context", [])]
        answer = response.get("answer", "")

        results.append({
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "answer": answer,
            "retrieved_sources": retrieved_sources,
            "hit": hit_at_k(retrieved_sources, case["expected_source"]),
            "mrr": reciprocal_rank(retrieved_sources, case["expected_source"]),
            "keyword_coverage": keyword_coverage(answer, case["expected_keywords"]),
            "latency_s": latency,
        })
    return results


def print_report(label: str, results: list[dict]) -> None:
    summary = aggregate(results)
    print(f"\n=== {label} ===")
    hit_rate = f"{summary['hit_rate']:.0%}" if summary["hit_rate"] is not None else "n/a"
    mrr = f"{summary['mrr']:.3f}" if summary["mrr"] is not None else "n/a"
    print(f"  Hit Rate@k          : {hit_rate}")
    print(f"  MRR                 : {mrr}")
    print(f"  Avg keyword coverage: {summary['avg_keyword_coverage']:.0%}")
    print(f"  Perfect matches     : {summary['perfect_keyword_matches']}/{summary['n_questions']}")
    print(f"  Avg latency         : {summary['avg_latency_s']:.2f}s")
    print()
    for r in results:
        flag = "✓" if (r["hit"] is not False and r["keyword_coverage"] >= 0.5) else "✗"
        cov = f"{r['keyword_coverage']:.0%}"
        print(f"  {flag} [{r['category']:<20}] {r['id']:<32} coverage={cov:>4}  hit={r['hit']}")


def write_markdown_report(path: Path, runs: dict[str, list[dict]]) -> None:
    lines = ["# RAG Evaluation Report\n"]
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Test set: `eval/qa_testset.json` ({len(next(iter(runs.values())))} questions)\n")

    lines.append("## Summary\n")
    lines.append("| Config | Hit Rate@k | MRR | Avg Keyword Coverage | Avg Latency |")
    lines.append("|---|---|---|---|---|")
    for label, results in runs.items():
        s = aggregate(results)
        hr = f"{s['hit_rate']:.0%}" if s["hit_rate"] is not None else "n/a"
        mrr = f"{s['mrr']:.3f}" if s["mrr"] is not None else "n/a"
        lines.append(f"| {label} | {hr} | {mrr} | {s['avg_keyword_coverage']:.0%} | {s['avg_latency_s']:.2f}s |")

    for label, results in runs.items():
        lines.append(f"\n## {label} — per-question detail\n")
        lines.append("| ID | Category | Coverage | Hit | Latency |")
        lines.append("|---|---|---|---|---|")
        for r in results:
            lines.append(
                f"| {r['id']} | {r['category']} | {r['keyword_coverage']:.0%} "
                f"| {r['hit']} | {r['latency_s']:.2f}s |"
            )

    path.write_text("\n".join(lines))
    print(f"\nWrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", action="store_true",
                         help="Also run a naive vector-only baseline (no hybrid, no reranker) for comparison")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results.md")
    args = parser.parse_args()

    is_up, msg = check_ollama_connection()
    if not is_up:
        print(f"Ollama is not reachable ({msg}). Start it with `ollama serve` and try again.", file=sys.stderr)
        sys.exit(1)

    cases = load_testset()
    runs: dict[str, list[dict]] = {}

    if args.compare:
        print("Running baseline (vector-only, no reranker)...")
        runs["Baseline (vector-only)"] = run_config(cases, use_hybrid=False, use_reranker=False, top_k=args.top_k)
        print_report("Baseline (vector-only)", runs["Baseline (vector-only)"])

    print("Running full pipeline (hybrid + reranker)...")
    runs["Full pipeline (hybrid + reranker)"] = run_config(cases, use_hybrid=True, use_reranker=True, top_k=args.top_k)
    print_report("Full pipeline (hybrid + reranker)", runs["Full pipeline (hybrid + reranker)"])

    write_markdown_report(args.out, runs)


if __name__ == "__main__":
    main()
