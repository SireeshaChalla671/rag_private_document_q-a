# Changes made to this project

This document summarizes what was added/fixed on top of the original codebase, and why — useful as a changelog for your report and as talking points for a viva.

## 1. Automated evaluation harness (`eval/`)

The README previously listed this as an explicit limitation: *"No automated evaluation harness (e.g. RAGAS) — answer quality is currently assessed via the 👍/👎 feedback buttons, not a labeled test set."* This closes that gap.

- **`eval/qa_testset.json`** — 10 labeled questions grounded in the actual content of `sample_docs/sample_return_policy.pdf`, covering distinct failure modes on purpose:
  - exact-number retrieval (e.g. "7 days" vs "10 days" — tests whether the system grabs the *right* nearby number, a classic RAG failure)
  - semantic paraphrase with no keyword overlap (tests vector search, not just BM25)
  - a "personal detail + general policy" question (mirrors the worked example already in your QA prompt)
  - a genuine out-of-scope question that *should* trigger the refusal path, not a hallucination
  - a multi-hop question requiring two separate clauses to be retrieved together
- **`eval/metrics.py`** — Hit Rate@k, MRR (Mean Reciprocal Rank) for retrieval quality, and keyword-coverage scoring for answer quality. Pure functions, fully unit tested, no LLM or network dependency.
- **`eval/evaluate.py`** — runs the test set through your actual pipeline (real Chroma + real Ollama) and writes a markdown report. Supports `--compare` to run a naive vector-only baseline *and* the full hybrid+reranker pipeline side by side, so you can show empirically — not just assert in the README — that hybrid retrieval and reranking improve Hit Rate/MRR over a naive baseline.
- **`eval/index_sample_doc.py`** — one-command helper to index the sample doc before running the eval.

**To run it:** `ollama serve`, then `python eval/index_sample_doc.py`, then `python eval/evaluate.py --compare`. This needs your local Ollama server, so it couldn't be executed in the sandbox this was built in — but every other piece of code touched here (58 tests) was actually run and verified, not just read over.

## 2. Test suite (`tests/`, 58 tests, all passing)

Covers `document_processor`, `document_registry`, `utils`, `reranker`, `vectorstore`, and `rag_chain`'s retriever-building logic. External dependencies (Ollama, Chroma, the cross-encoder model) are mocked at clean boundaries so the suite runs in under a second with no GPU, network, or running services required — the same pattern you'd use to test this professionally, since the model weights aren't your code's responsibility to verify.

Run with: `pip install -r requirements-dev.txt && pytest`

## 3. Real bug fixed

`load_and_split()` in `document_processor.py` checked `if not pages:` to catch unreadable files, but a loader can return a *non-empty* list containing an *empty* Document — e.g. a 0-byte `.txt` upload returns one Document with `page_content == ""`. That slipped through the check and would silently "index" zero chunks with no error shown. Fixed by checking that at least one page has non-whitespace content. Caught by `test_empty_txt_file_raises_value_error`, which failed before the fix and passes after.

## 4. Robustness / UX fixes in `app.py`

- **Temp upload cleanup**: uploaded files were saved to `temp_uploads/` and never deleted, even after being embedded and persisted to Chroma. They now get removed after successful indexing.
- **Confirm-before-destroy**: "Remove All Documents" was a single-click irreversible action. It now requires a confirm/cancel step.
- **Empty-state guidance**: a fresh install with no documents indexed showed a blank chat window with no hint of what to do. Added a one-line "get started" prompt.
- **Export chat transcript**: added a button to download the conversation (with citations) as markdown, so a good demo run isn't lost when you hit "Clear Chat."
- **File-type icons** in the document management list for quicker scanning.

## 5. Minor code-quality cleanup

`_build_retriever()` in `rag_chain.py` accepted a `top_k` parameter it never used (the retrieval width was actually controlled entirely by `candidate_k`). Removed the dead parameter and updated the one call site — small, but the kind of thing that reads as sloppy in a code review if left in.

## What wasn't changed

The core pipeline design — hybrid retrieval, cross-encoder reranking, history-aware query rewriting, honest per-format citations, clean re-indexing — was already sound and wasn't touched. That architecture is the strongest part of this project and is worth defending directly in a viva rather than apologizing for.
