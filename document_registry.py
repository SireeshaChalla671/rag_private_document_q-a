"""
Persisted metadata about indexed documents (size, page count, upload time).

The vector store itself only tracks chunks, not per-file bookkeeping, so
this keeps a small JSON manifest alongside it. It's deliberately simple
(one JSON file, no locking) — appropriate for a single-user local app,
not a concurrent multi-user service.
"""

import json
from datetime import datetime
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "chroma_db" / "document_registry.json"


def _load() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(data, indent=2))


def register_document(filename: str, size_bytes: int, num_pages: int) -> None:
    """Record/overwrite metadata for a file at index time."""
    data = _load()
    data[filename] = {
        "size_bytes": size_bytes,
        "num_pages": num_pages,
        "file_type": Path(filename).suffix.lstrip(".").upper() or "FILE",
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
    }
    _save(data)


def remove_document(filename: str) -> None:
    data = _load()
    data.pop(filename, None)
    _save(data)


def list_documents() -> dict:
    """Return {filename: {size_bytes, num_pages, uploaded_at}}."""
    return _load()


def clear_all() -> None:
    _save({})
