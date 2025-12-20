from __future__ import annotations

from pathlib import Path

from src.ingestion.text_file_parser import TextFileParser


def test_text_file_parser_txt_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    content = "Hello world.\n\nThis is a test.\n"
    path.write_text(content, encoding="utf-8")

    parser = TextFileParser()
    doc1 = parser.parse_file(path)
    doc2 = parser.parse_file(path)

    assert doc1.document_id == doc2.document_id
    assert doc1.metadata["filename"] == "notes.txt"
    assert doc1.metadata["checksum"]
    assert doc1.raw_text == content
    assert doc1.structure["sections"] == []


def test_text_file_parser_md_extracts_headings(tmp_path: Path) -> None:
    path = tmp_path / "readme.md"
    content = "# Title\n" "\n" "Intro.\n" "\n" "## Sub\n" "\n" "Details.\n"
    path.write_text(content, encoding="utf-8")

    parser = TextFileParser()
    doc = parser.parse_file(path)

    sections = doc.structure["sections"]
    assert len(sections) == 1
    assert sections[0].title == "Title"
    assert "Intro." in sections[0].content
    assert len(sections[0].subsections) == 1
    assert sections[0].subsections[0].title == "Sub"
    assert "Details." in sections[0].subsections[0].content
