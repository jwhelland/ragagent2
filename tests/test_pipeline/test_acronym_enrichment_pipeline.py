"""Unit tests for ingestion pipeline acronym enrichment (Phase 3 Task 3.3)."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from src.normalization.acronym_resolver import AcronymResolver
from src.normalization.string_normalizer import StringNormalizer
from src.pipeline.ingestion_pipeline import IngestionPipeline
from src.utils.config import Config


class _IngestChunk(BaseModel):
    chunk_id: str
    document_id: str
    level: int
    parent_chunk_id: str | None = None
    child_chunk_ids: List[str] = Field(default_factory=list)
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: int = 0


def test_pipeline_enriches_candidate_aliases_with_expansion() -> None:
    cfg = Config.from_yaml("config/config.yaml")
    cfg.normalization.enable_acronym_resolution = True

    pipeline = IngestionPipeline(cfg)
    pipeline.string_normalizer = StringNormalizer(cfg.normalization)
    pipeline.acronym_resolver = AcronymResolver(
        config=cfg.normalization, normalizer=pipeline.string_normalizer
    )

    chunk = _IngestChunk(
        chunk_id="c1",
        document_id="doc1",
        level=2,
        content="Telemetry and Command (T&C) subsystem handles uplink commands.",
        metadata={
            "merged_entities": [
                {
                    "canonical_name": "T&C",
                    "canonical_normalized": "t&c",
                    "type": "SYSTEM",
                    "confidence": 0.9,
                    "aliases": ["T&C subsystem"],
                    "description": "",
                    "mention_count": 1,
                    "conflicting_types": [],
                    "provenance": [],
                }
            ]
        },
    )

    pipeline._update_acronym_dictionary([chunk])
    pipeline._enrich_merged_entities_with_acronyms([chunk])

    merged = chunk.metadata["merged_entities"][0]
    assert "Telemetry and Command" in merged["aliases"]
    assert "Telemetry and Command subsystem" in merged["aliases"]
