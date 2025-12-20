"""Text/Markdown document parsing without Docling.

This module supports lightweight ingestion for plain text documents (e.g. .txt, .md)
by producing a `ParsedDocument` compatible with the existing chunker/pipeline.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from loguru import logger

from src.ingestion.pdf_parser import ParsedDocument, Section


class TextFileParser:
    """Parse text-based documents into a `ParsedDocument`.

    - `.txt`: treated as unstructured text (no sections by default)
    - `.md`/`.markdown`: basic heading extraction for `#` and `##`
    """

    SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}

    def parse_file(self, path: Path | str) -> ParsedDocument:
        file_path = Path(path)
        suffix = file_path.suffix.lower()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if suffix not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported text document type: {suffix}")

        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_text = file_path.read_text(encoding="utf-8", errors="replace")

        checksum = self._sha256_text(raw_text)
        document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(checksum)))

        title = self._infer_title(file_path, raw_text)
        structure = {"sections": self._extract_sections(file_path, raw_text)}

        metadata = {
            "filename": file_path.name,
            "file_path": str(file_path.absolute()),
            "file_size": file_path.stat().st_size,
            "checksum": checksum,
            "title": title,
            "source_type": "text",
            "file_suffix": suffix,
        }

        return ParsedDocument(
            document_id=document_id,
            metadata=metadata,
            structure=structure,
            raw_text=raw_text,
            tables=[],
            figures=[],
            page_count=0,
            error=None,
        )

    def _infer_title(self, file_path: Path, raw_text: str) -> str:
        suffix = file_path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            for line in raw_text.splitlines():
                stripped = line.strip()
                if stripped.startswith("# "):
                    return stripped[2:].strip()[:200] or file_path.stem
        for line in raw_text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:200]
        return file_path.stem

    def _extract_sections(self, file_path: Path, raw_text: str) -> list[Section]:
        suffix = file_path.suffix.lower()
        if suffix not in {".md", ".markdown"}:
            return []

        sections: list[Section] = []
        current_section: Section | None = None
        current_subsection: Section | None = None
        section_counter = 0
        subsection_counter = 0

        def flush_subsection() -> None:
            nonlocal current_subsection
            if current_subsection is not None and current_section is not None:
                current_section.subsections.append(current_subsection)
                current_subsection = None

        def flush_section() -> None:
            nonlocal current_section, current_subsection
            if current_section is None:
                return
            flush_subsection()
            sections.append(current_section)
            current_section = None
            current_subsection = None

        for line in raw_text.splitlines():
            stripped = line.rstrip()
            if stripped.startswith("# "):
                flush_section()
                section_counter += 1
                subsection_counter = 0
                current_section = Section(
                    level=1,
                    title=stripped[2:].strip(),
                    content="",
                    start_page=1,
                    end_page=1,
                    hierarchy_path=str(section_counter),
                )
                continue

            if stripped.startswith("## "):
                if current_section is None:
                    section_counter += 1
                    current_section = Section(
                        level=1,
                        title=file_path.stem,
                        content="",
                        start_page=1,
                        end_page=1,
                        hierarchy_path=str(section_counter),
                    )
                flush_subsection()
                subsection_counter += 1
                current_subsection = Section(
                    level=2,
                    title=stripped[3:].strip(),
                    content="",
                    start_page=1,
                    end_page=1,
                    hierarchy_path=f"{section_counter}.{subsection_counter}",
                )
                continue

            target = current_subsection if current_subsection is not None else current_section
            if target is None:
                # No headings encountered yet; treat as unstructured.
                continue
            target.content += stripped + "\n"

        flush_section()

        if not sections:
            logger.debug("No markdown headings detected for {}", file_path.name)

        # Normalize whitespace: chunker handles paragraph splits on blank lines.
        for section in sections:
            section.content = section.content.strip() + "\n"
            for subsection in section.subsections:
                subsection.content = subsection.content.strip() + "\n"

        return sections

    def _sha256_text(self, content: str) -> str:
        hasher = hashlib.sha256()
        hasher.update(content.encode("utf-8", errors="replace"))
        return hasher.hexdigest()
