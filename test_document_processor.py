"""Tests for src/document_processor.py — loading and chunking."""
import pytest

from src.document_processor import load_and_split


def test_pdf_chunks_carry_source_and_page_metadata(sample_pdf_path):
    splits, page_counts = load_and_split([sample_pdf_path], chunk_size=500, chunk_overlap=50)

    assert len(splits) > 0
    assert page_counts[sample_pdf_path.name] >= 1
    for chunk in splits:
        assert chunk.metadata["source"] == sample_pdf_path.name
        assert chunk.metadata["file_type"] == "PDF"
        assert isinstance(chunk.metadata["page"], int)  # PDFs must have a real page number


def test_txt_chunks_have_no_fabricated_page_number(sample_txt_path):
    splits, page_counts = load_and_split([sample_txt_path], chunk_size=200, chunk_overlap=20)

    assert len(splits) > 0
    assert page_counts[sample_txt_path.name] == 0  # TXT/DOCX never claim a page count
    for chunk in splits:
        assert chunk.metadata["source"] == sample_txt_path.name
        assert chunk.metadata["file_type"] == "TXT"
        # Critical honesty check: the app must not invent a page number for
        # formats that don't have one, or citations would be misleading.
        assert "page" not in chunk.metadata


def test_chunk_size_is_respected_within_reason(sample_txt_path):
    splits, _ = load_and_split([sample_txt_path], chunk_size=100, chunk_overlap=0)
    # RecursiveCharacterTextSplitter can slightly exceed chunk_size when a
    # single "atomic" separator unit is longer than chunk_size, so allow
    # some slack rather than asserting a hard <= chunk_size.
    assert all(len(c.page_content) <= 100 * 1.5 for c in splits)


def test_multiple_files_in_one_call_are_all_indexed(sample_pdf_path, sample_txt_path):
    splits, page_counts = load_and_split(
        [sample_pdf_path, sample_txt_path], chunk_size=500, chunk_overlap=50
    )
    sources = {c.metadata["source"] for c in splits}
    assert sources == {sample_pdf_path.name, sample_txt_path.name}
    assert set(page_counts) == {sample_pdf_path.name, sample_txt_path.name}


def test_unsupported_extension_raises_value_error(tmp_path):
    bad_file = tmp_path / "data.csv"
    bad_file.write_text("a,b,c\n1,2,3")
    with pytest.raises(ValueError, match="Supported formats"):
        load_and_split([bad_file], chunk_size=500, chunk_overlap=50)


def test_empty_txt_file_raises_value_error(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")
    with pytest.raises(ValueError):
        load_and_split([empty_file], chunk_size=500, chunk_overlap=50)


def test_nonexistent_file_raises_value_error_not_crash(tmp_path):
    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises(ValueError, match="Could not read"):
        load_and_split([missing], chunk_size=500, chunk_overlap=50)
