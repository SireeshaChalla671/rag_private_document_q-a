"""
Persistent vector store management (Chroma).

The previous version of this app rebuilt an in-memory Chroma store from
scratch on every indexing click, and lost everything on page refresh.
Here the store is persisted to disk, so:
  - documents survive a Streamlit restart
  - multiple documents can be added incrementally across sessions
  - the knowledge base can be inspected and explicitly cleared
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from chromadb.config import Settings

from src.config import COLLECTION_NAME, EMBEDDING_MODEL, PERSIST_DIR
from src.utils import logger


def get_vectorstore() -> Chroma:
    """Load (or lazily create) the persistent Chroma collection."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        # Anonymized telemetry is opt-out and, on some chromadb/posthog
        # version combos, throws a noisy (harmless) error on every call.
        # Turning it off avoids that log spam entirely.
        client_settings=Settings(anonymized_telemetry=False),
    )


def add_documents(vectorstore: Chroma, docs: list[Document]) -> None:
    """Embed and persist a batch of chunks."""
    vectorstore.add_documents(docs)
    logger.info("Added %d chunk(s) to the vector store", len(docs))


def delete_by_source(vectorstore: Chroma, filename: str) -> None:
    """
    Remove every chunk previously indexed for a given filename.

    Called before re-adding a file so that re-indexing (e.g. after changing
    chunk size) *replaces* its chunks instead of duplicating them. Duplicate
    chunks silently skew retrieval toward whatever got indexed twice, which
    is a subtle cause of biased/wrong answers.
    """
    try:
        existing = vectorstore.get(where={"source": filename}, include=[])
        ids = existing.get("ids", [])
        if ids:
            vectorstore.delete(ids=ids)
            logger.info("Removed %d existing chunk(s) for '%s' before re-indexing", len(ids), filename)
    except Exception:  # noqa: BLE001 - nothing to delete on a fresh collection
        pass


def get_all_documents(vectorstore: Chroma) -> list[Document]:
    """
    Reconstruct every indexed chunk as a Document (content + metadata).

    Used to build the BM25 keyword index for hybrid retrieval. BM25 needs
    the raw corpus in memory; Chroma's own get() already stores everything
    needed, so no separate copy of the documents has to be kept around.
    """
    try:
        data = vectorstore.get(include=["documents", "metadatas"])
    except Exception:  # noqa: BLE001 - empty/uninitialized collection
        return []

    texts = data.get("documents", []) or []
    metas = data.get("metadatas", []) or []
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(texts, metas)
    ]


def list_indexed_sources(vectorstore: Chroma) -> dict[str, int]:
    """
    Return {filename: chunk_count} for everything currently indexed.
    Used to populate the "Indexed Documents" panel in the sidebar.
    """
    try:
        data = vectorstore.get(include=["metadatas"])
    except Exception:  # noqa: BLE001 - empty/uninitialized collection
        return {}

    counts: dict[str, int] = {}
    for meta in data.get("metadatas", []) or []:
        source = meta.get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def clear_vectorstore() -> None:
    """Clear the collection through Chroma rather than deleting live DB files.

    Removing the persistence directory while Chroma has an active SQLite
    connection can leave Windows file locks behind or produce a half-deleted
    database. Deleting the collection is atomic from Chroma's perspective;
    the next ``get_vectorstore`` call recreates an empty collection.
    """
    try:
        get_vectorstore().delete_collection()
        logger.info("Vector store collection cleared")
    except Exception as e:  # noqa: BLE001 - a missing collection is already empty
        logger.warning("Could not clear vector store collection: %s", e)
