"""
Document ingestion: load PDFs, DOCX files, and TXT files from disk and
split them into overlapping chunks suitable for embedding.

Supports multiple files in a single indexing pass, and stamps each chunk
with its originating filename + page number so citations are possible
later in the pipeline.
"""

from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils import logger


def load_and_split(
    file_paths: list[Path],
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[Document], dict[str, int]]:
    """
    Load one or more PDF, DOCX, or TXT files and split them into chunks.

    Each resulting chunk carries metadata:
      - source: original filename (for citations)
      - page:   0-indexed page number for PDFs only

    Returns:
        (chunks, page_counts) where page_counts maps filename -> page count,
        used to populate the document management sidebar.

    Raises:
        ValueError: if a file cannot be parsed (e.g. corrupt/encrypted PDF).
    """
    all_docs: list[Document] = []
    page_counts: dict[str, int] = {}

    for path in file_paths:
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(path))
                pages = loader.load()
            elif suffix == ".docx":
                loader = Docx2txtLoader(str(path))
                pages = loader.load()
            elif suffix == ".txt":
                loader = TextLoader(str(path), autodetect_encoding=True)
                pages = loader.load()
            else:
                raise ValueError("Supported formats are PDF, DOCX, and TXT.")
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Could not read '{path.name}': {e}") from e

        # A loader can return a non-empty list of Documents that are
        # themselves empty (e.g. a 0-byte .txt file yields one Document
        # with page_content == ""), so `not pages` alone doesn't catch
        # that case. Without this check the file would silently produce
        # zero chunks and get "indexed" with no error shown to the user.
        if not pages or not any(p.page_content.strip() for p in pages):
            raise ValueError(f"'{path.name}' contains no extractable text.")

        for page in pages:
            # Only PDFs have a reliable page number. The interface shows a
            # document-level citation for DOCX/TXT rather than inventing one.
            page.metadata["source"] = path.name
            page.metadata["file_type"] = suffix.lstrip(".").upper()
            if suffix != ".pdf":
                page.metadata.pop("page", None)
        all_docs.extend(pages)
        page_counts[path.name] = len(pages) if suffix == ".pdf" else 0

    # Ordered from "strongest" to "weakest" boundary. Documents like policies
    # and manuals are usually structured as numbered clauses / bullets, so
    # splitting on those first (before falling back to sentence/word breaks)
    # keeps a clause and its content in the same chunk far more often.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "\n- ", "\n• ", ". ", "; ", ", ", " ", ""],
    )
    splits = splitter.split_documents(all_docs)
    logger.info("Split %d page(s) into %d chunk(s)", len(all_docs), len(splits))
    return splits, page_counts
