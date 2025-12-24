"""Regex-based relationship extractor using Hearst-like patterns."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger

from src.extraction.models import ExtractedRelationship


class PatternRelationshipExtractor:
    """Extracts relationships using regex patterns defined in configuration."""

    def __init__(self, patterns_path: str | Path = "config/relationship_patterns.yaml") -> None:
        self.patterns_path = Path(patterns_path)
        self.patterns = self._load_patterns(self.patterns_path)

        logger.info(
            f"Initialized PatternRelationshipExtractor with {len(self.patterns)} pattern groups",
            path=str(self.patterns_path)
        )

    def extract_relationships(
        self,
        chunk: Any,
        *,
        document_context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractedRelationship]:
        """Extract relationships from a chunk using regex patterns."""
        text = getattr(chunk, "content", None)
        if text is None and isinstance(chunk, dict):
            text = chunk.get("content")

        if not text:
            return []

        chunk_id = getattr(chunk, "chunk_id", None) or getattr(chunk, "id", None) or (chunk.get("chunk_id") if isinstance(chunk, dict) else None)
        document_id = getattr(chunk, "document_id", None) or (chunk.get("document_id") if isinstance(chunk, dict) else None)

        relationships: List[ExtractedRelationship] = []

        for group in self.patterns:
            rel_type = group.get("relationship_type", "RELATED_TO")
            regex_list = group.get("patterns", [])

            for pattern_str in regex_list:
                try:
                    # Using case-insensitive matching
                    matches = re.finditer(pattern_str, text, re.IGNORECASE)
                    for match in matches:
                        groups = match.groupdict()
                        source = groups.get("source", "").strip()
                        target = groups.get("target", "").strip()

                        if source and target and source.lower() != target.lower():
                            # Basic filtering: ignore if source/target are too long (likely false positive)
                            if len(source) > 50 or len(target) > 50:
                                continue

                            relationships.append(
                                ExtractedRelationship(
                                    source=source,
                                    target=target,
                                    type=rel_type,
                                    description=f"Extracted via pattern: {pattern_str}",
                                    confidence=0.8, # High confidence for explicit patterns
                                    bidirectional=False,
                                    chunk_id=chunk_id,
                                    document_id=document_id,
                                    source_extractor="regex_patterns",
                                    raw={"match": match.group(0), "pattern": pattern_str}
                                )
                            )
                except re.error as e:
                    logger.warning(f"Invalid regex pattern: {pattern_str}", error=str(e))

        return relationships

    def _load_patterns(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            logger.warning(f"Relationship patterns file not found: {path}")
            return []

        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return data.get("patterns", [])
        except Exception as e:
            logger.error(f"Failed to load relationship patterns: {e}")
            return []
