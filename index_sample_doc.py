"""
One-time helper: index sample_docs/sample_return_policy.pdf so eval/evaluate.py
has something to query against. Safe to re-run (replaces, doesn't duplicate).

Usage:
    ollama serve                    # in one terminal
    python eval/index_sample_doc.py # in another
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.document_processor import load_and_split
from src.utils import check_ollama_connection
from src.vectorstore import add_documents, delete_by_source, get_vectorstore

SAMPLE_DOC = Path(__file__).resolve().parent.parent / "sample_docs" / "sample_return_policy.pdf"


def main():
    is_up, msg = check_ollama_connection()
    if not is_up:
        print(f"Ollama is not reachable ({msg}). Start it with `ollama serve` first.", file=sys.stderr)
        sys.exit(1)

    splits, page_counts = load_and_split([SAMPLE_DOC], chunk_size=1000, chunk_overlap=200)
    vectorstore = get_vectorstore()
    delete_by_source(vectorstore, SAMPLE_DOC.name)
    add_documents(vectorstore, splits)
    print(f"Indexed {len(splits)} chunk(s) from {SAMPLE_DOC.name} ({page_counts[SAMPLE_DOC.name]} pages).")


if __name__ == "__main__":
    main()
