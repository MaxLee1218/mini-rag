from pathlib import Path

from docx import Document as DocxDocument

from app import document_loader


def test_txt_collapses_spaces_and_tabs_but_preserves_newlines(tmp_path):
    raw_file = tmp_path / "sample.txt"
    raw_file.write_text("  alpha\t\tbeta  \nsecond\t line  ", encoding="utf-8")

    documents = document_loader.load_documents(str(tmp_path))

    assert documents == [
        {
            "text": "alpha beta \nsecond line",
            "source": "sample.txt",
        }
    ]


def test_empty_txt_file_is_skipped(tmp_path):
    raw_file = tmp_path / "empty.txt"
    raw_file.write_text(" \t  ", encoding="utf-8")

    documents = document_loader.load_documents(str(tmp_path))

    assert documents == []


def test_docx_paragraphs_are_joined_with_newlines(tmp_path):
    raw_file = tmp_path / "notes.docx"
    doc = DocxDocument()
    doc.add_paragraph("First   paragraph")
    doc.add_paragraph("Second\tparagraph")
    doc.save(raw_file)

    documents = document_loader.load_documents(str(tmp_path))

    assert documents == [
        {
            "text": "First paragraph\nSecond paragraph",
            "source": "notes.docx",
        }
    ]


def test_pdf_returns_one_document_and_skips_empty_pages(tmp_path, monkeypatch):
    raw_file = tmp_path / "paper.pdf"
    raw_file.write_bytes(b"fake pdf content")

    class FakePage:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class FakePdfReader:
        def __init__(self, file_path):
            assert Path(file_path) == raw_file
            self.pages = [
                FakePage(" First   page "),
                FakePage(None),
                FakePage("   "),
                FakePage("Second\tpage"),
            ]

    monkeypatch.setattr(document_loader, "PdfReader", FakePdfReader, raising=False)

    documents = document_loader.load_documents(str(tmp_path))

    assert documents == [
        {
            "text": "First page\nSecond page",
            "source": "paper.pdf",
        }
    ]


def test_recursive_scan_uses_posix_relative_source_paths(tmp_path):
    first = tmp_path / "first" / "same.txt"
    second = tmp_path / "second" / "same.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first file", encoding="utf-8")
    second.write_text("second file", encoding="utf-8")

    documents = document_loader.load_documents(str(tmp_path))

    assert {"text": "first file", "source": "first/same.txt"} in documents
    assert {"text": "second file", "source": "second/same.txt"} in documents


def test_unsupported_files_are_ignored(tmp_path):
    raw_file = tmp_path / "ignored.md"
    raw_file.write_text("ignored", encoding="utf-8")

    documents = document_loader.load_documents(str(tmp_path))

    assert documents == []


def test_failed_file_load_prints_warning_and_is_skipped(tmp_path, capsys):
    raw_file = tmp_path / "broken.pdf"
    raw_file.write_bytes(b"not a valid pdf")

    documents = document_loader.load_documents(str(tmp_path))

    captured = capsys.readouterr()
    assert documents == []
    assert "[Loader Warning] Failed to load" in captured.out
    assert "broken.pdf" in captured.out


def test_missing_folder_returns_empty_list(tmp_path):
    missing_folder = tmp_path / "missing"

    documents = document_loader.load_documents(str(missing_folder))

    assert documents == []
