"""
Small utility helpers: Ollama connectivity checks and misc formatting.

Checking connectivity up front (instead of letting a raw connection error
bubble up mid-chain) is what makes the app feel "production-minded" rather
than a happy-path demo.
"""

import logging

import requests

from src.config import OLLAMA_BASE_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag_app")


def check_ollama_connection() -> tuple[bool, str]:
    """
    Verify that an Ollama server is reachable.

    Returns (is_up, message).
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            return True, "Connected"
        return False, f"Ollama responded with status {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot reach Ollama. Is it running? Try: ollama serve"
    except requests.exceptions.Timeout:
        return False, "Ollama connection timed out."
    except Exception as e:  # noqa: BLE001 - surfaced to the user, not swallowed
        return False, f"Unexpected error contacting Ollama: {e}"


def get_installed_models() -> list[str]:
    """Return the list of model names currently pulled in the local Ollama instance."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"].split(":")[0] for m in data.get("models", [])]
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not fetch installed models: %s", e)
        return []


def export_chat_as_markdown(messages: list[dict]) -> str:
    """
    Render the session's chat history (including citations) as a markdown
    document, so a conversation worth keeping doesn't disappear the moment
    'Clear Chat' is pressed or the browser tab is closed.
    """
    if not messages:
        return "_No conversation yet._\n"

    lines = ["# Chat Transcript", ""]
    for msg in messages:
        speaker = "**You**" if msg["role"] == "user" else "**Assistant**"
        lines.append(f"{speaker}: {msg['content']}")
        sources = msg.get("sources") or []
        if sources:
            lines.append("")
            lines.append("<details><summary>Sources</summary>")
            lines.append("")
            for i, s in enumerate(sources, start=1):
                lines.append(f"{i}. `{s['source']}` — {s['location']}")
            lines.append("")
            lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


def format_bytes(num_bytes: int) -> str:
    """Human-readable file size, e.g. 1.4 MB."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
