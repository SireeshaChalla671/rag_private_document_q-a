"""Tests for src/utils.py — connectivity checks and formatting helpers."""
from unittest.mock import Mock, patch

import requests

from src.utils import check_ollama_connection, export_chat_as_markdown, format_bytes, get_installed_models


def test_format_bytes_scales_units():
    assert format_bytes(500) == "500.0 B"
    assert format_bytes(1536) == "1.5 KB"
    assert format_bytes(5 * 1024 * 1024) == "5.0 MB"
    assert format_bytes(2 * 1024 * 1024 * 1024) == "2.0 GB"


def test_check_ollama_connection_success():
    fake_resp = Mock(status_code=200)
    with patch("src.utils.requests.get", return_value=fake_resp):
        is_up, msg = check_ollama_connection()
    assert is_up is True


def test_check_ollama_connection_non_200_is_treated_as_down():
    fake_resp = Mock(status_code=500)
    with patch("src.utils.requests.get", return_value=fake_resp):
        is_up, msg = check_ollama_connection()
    assert is_up is False
    assert "500" in msg


def test_check_ollama_connection_refused_gives_actionable_message():
    with patch("src.utils.requests.get", side_effect=requests.exceptions.ConnectionError()):
        is_up, msg = check_ollama_connection()
    assert is_up is False
    assert "ollama serve" in msg.lower()


def test_check_ollama_connection_timeout():
    with patch("src.utils.requests.get", side_effect=requests.exceptions.Timeout()):
        is_up, msg = check_ollama_connection()
    assert is_up is False
    assert "timed out" in msg.lower()


def test_get_installed_models_parses_names_without_tag():
    fake_resp = Mock()
    fake_resp.json.return_value = {"models": [{"name": "llama3.2:latest"}, {"name": "phi3:latest"}]}
    fake_resp.raise_for_status = Mock()
    with patch("src.utils.requests.get", return_value=fake_resp):
        models = get_installed_models()
    assert models == ["llama3.2", "phi3"]


def test_get_installed_models_degrades_to_empty_list_on_error():
    with patch("src.utils.requests.get", side_effect=requests.exceptions.ConnectionError()):
        models = get_installed_models()
    assert models == []


def test_export_chat_empty_history():
    md = export_chat_as_markdown([])
    assert "No conversation yet" in md


def test_export_chat_includes_speakers_and_content():
    messages = [
        {"role": "user", "content": "What is the return window?"},
        {"role": "assistant", "content": "10 days for most items.", "sources": []},
    ]
    md = export_chat_as_markdown(messages)
    assert "**You**: What is the return window?" in md
    assert "**Assistant**: 10 days for most items." in md


def test_export_chat_includes_citations_when_present():
    messages = [
        {
            "role": "assistant",
            "content": "10 days.",
            "sources": [{"source": "policy.pdf", "location": "Page 1"}],
        }
    ]
    md = export_chat_as_markdown(messages)
    assert "policy.pdf" in md
    assert "Page 1" in md
    assert "<details>" in md


def test_export_chat_omits_source_block_when_no_sources():
    messages = [{"role": "assistant", "content": "Answer.", "sources": []}]
    md = export_chat_as_markdown(messages)
    assert "<details>" not in md
