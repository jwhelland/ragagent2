from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from src.ingestion.chunker import HierarchicalChunker
from src.ingestion.text_cleaner import TextCleaner
from src.pipeline.ingestion_pipeline import IngestionPipeline
from src.utils.config import Config


class _FakeEmbeddings:
    def generate(self, texts: List[str]) -> List[np.ndarray]:
        return [np.zeros(8, dtype=np.float32) for _ in texts]


class _FailingPDFParser:
    def parse_pdf(self, _: Path) -> None:  # pragma: no cover
        raise AssertionError("PDF parser should not be called for .txt/.md ingestion")


def test_process_document_txt_bypasses_pdf_parser(tmp_path: Path) -> None:
    doc_path = tmp_path / "note.txt"
    doc_path.write_text("Hello from text.\n\nSecond paragraph.\n", encoding="utf-8")

    cfg = Config.from_yaml("config/config.yaml")
    cfg.extraction.enable_llm = False
    cfg.pipeline.enable_checkpointing = False

    pipeline = IngestionPipeline(cfg)

    def _init_components() -> None:
        pipeline.pdf_parser = _FailingPDFParser()
        pipeline.text_cleaner = TextCleaner(cfg.ingestion.text_cleaning)
        pipeline.chunker = HierarchicalChunker(cfg.ingestion.chunking)
        pipeline.embeddings = _FakeEmbeddings()
        pipeline.neo4j_manager = object()
        pipeline.qdrant_manager = object()

    pipeline.initialize_components = _init_components  # type: ignore[method-assign]

    pipeline._upsert_document_status = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    pipeline._store_document_and_chunks = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    pipeline._store_entity_candidates = lambda *_args, **_kwargs: 0  # type: ignore[method-assign]
    pipeline._store_relationship_candidates = lambda *_args, **_kwargs: 0  # type: ignore[method-assign]
    pipeline._merge_entities = lambda *_args, **_kwargs: 0  # type: ignore[method-assign]

    result = pipeline.process_document(doc_path)

    assert result.success is True
    assert result.chunks_created > 0
