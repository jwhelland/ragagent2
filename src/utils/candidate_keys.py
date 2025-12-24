"""Helpers for building deterministic candidate keys.

Candidate keys are used for aggregation/upserts in Neo4j and to efficiently find related
RelationshipCandidate rows. Keep this logic centralized so ingestion and curation stay aligned.
"""

from __future__ import annotations

import re

from src.normalization.string_normalizer import StringNormalizer


def normalize_candidate_key_fragment(
    value: str, *, normalizer: StringNormalizer | None = None
) -> str:
    """Normalize a string for use inside candidate_key fragments.

    This is intentionally a little more permissive than the normalization table key
    (it keeps alphanumerics and converts everything else to underscores).
    """
    normalized = ""
    if normalizer is not None:
        try:
            normalized = normalizer.normalize(value).normalized
        except Exception:  # noqa: BLE001
            normalized = ""

    if not normalized:
        normalized = (value or "").strip().lower()

    return re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
