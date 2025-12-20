"""Fuzzy string matching utilities for entity normalization (Task 3.2)."""

from __future__ import annotations

from typing import Callable, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict
from rapidfuzz import fuzz, process

from src.normalization.string_normalizer import StringNormalizer
from src.utils.config import NormalizationConfig


class FuzzyMatchCandidate(BaseModel):
    """Represents a fuzzy match suggestion."""

    model_config = ConfigDict(frozen=True)

    source: str
    target: str
    source_normalized: str
    target_normalized: str
    score: float  # 0-1
    confidence: float  # 0-1
    threshold: float  # 0-1
    passed: bool
    entity_type: str | None = None


class FuzzyMatcher:
    """RapidFuzz-based matcher for detecting entity variants and typos."""

    def __init__(
        self,
        config: NormalizationConfig | None = None,
        normalizer: StringNormalizer | None = None,
        threshold_overrides: Mapping[str, float] | None = None,
        scorer: Callable[[str, str], float] | None = None,
    ) -> None:
        self.config = config or NormalizationConfig()
        self.normalizer = normalizer or StringNormalizer(config=self.config)
        self.scorer: Callable[[str, str], float] = scorer or fuzz.WRatio
        self.threshold_overrides: MutableMapping[str, float] = {}
        for key, value in (self.config.fuzzy_threshold_overrides or {}).items():
            self.threshold_overrides[key.upper()] = value
        for key, value in (threshold_overrides or {}).items():
            self.threshold_overrides[key.upper()] = value

        logger.info(
            "Initialized FuzzyMatcher with base threshold {:.2f}", self.config.fuzzy_threshold
        )

    def match_pair(
        self, source: str, target: str, entity_type: str | None = None
    ) -> FuzzyMatchCandidate:
        """Score a single pair of strings."""
        source_norm = self._normalize(source)
        target_norm = self._normalize(target)
        threshold = self._threshold_for(entity_type, source_norm, target_norm)

        score = self._score(source_norm, target_norm)
        confidence = self._confidence(score, threshold, source_norm, target_norm)
        passed = score >= threshold

        return FuzzyMatchCandidate(
            source=source,
            target=target,
            source_normalized=source_norm,
            target_normalized=target_norm,
            score=score,
            confidence=confidence,
            threshold=threshold,
            passed=passed,
            entity_type=entity_type.upper() if entity_type else None,
        )

    def match_batch(
        self,
        sources: Sequence[str],
        choices: Sequence[str],
        entity_type: str | None = None,
        limit: int = 3,
    ) -> List[FuzzyMatchCandidate]:
        """Batch matching for efficiency across many entities."""
        if not sources or not choices:
            return []

        normalized_sources = [self._normalize(value) for value in sources]
        normalized_choices = [self._normalize(value) for value in choices]
        effective_limit = max(1, min(limit, len(choices)))

        try:
            score_matrix = process.cdist(
                normalized_sources,
                normalized_choices,
                scorer=self.scorer,
                processor=None,
            )
        except Exception:  # pragma: no cover - fallback for unexpected RapidFuzz backends
            score_matrix = None

        results: List[FuzzyMatchCandidate] = []
        for source_index, (source_raw, source_norm) in enumerate(
            zip(sources, normalized_sources, strict=False)
        ):
            if score_matrix is None:
                matches = process.extract(
                    source_norm,
                    normalized_choices,
                    scorer=self.scorer,
                    processor=None,
                    limit=effective_limit,
                )
                candidate_indices = [int(match[2]) for match in matches]
                candidate_scores = [float(match[1]) for match in matches]
            else:
                row = np.asarray(score_matrix[source_index])
                top_indices = np.argpartition(-row, kth=effective_limit - 1)[:effective_limit]
                top_indices = top_indices[np.argsort(row[top_indices])[::-1]]
                candidate_indices = [int(idx) for idx in top_indices.tolist()]
                candidate_scores = [float(row[idx]) for idx in candidate_indices]

            for target_index, raw_score in zip(candidate_indices, candidate_scores, strict=False):
                target_norm = normalized_choices[target_index]
                target_raw = choices[target_index]

                score = raw_score / 100.0
                threshold = self._threshold_for(entity_type, source_norm, target_norm)
                confidence = self._confidence(score, threshold, source_norm, target_norm)
                passed = score >= threshold

                results.append(
                    FuzzyMatchCandidate(
                        source=source_raw,
                        target=target_raw,
                        source_normalized=source_norm,
                        target_normalized=target_norm,
                        score=score,
                        confidence=confidence,
                        threshold=threshold,
                        passed=passed,
                        entity_type=entity_type.upper() if entity_type else None,
                    )
                )

        return results

    def generate_candidates(
        self,
        values: Iterable[str],
        entity_type: str | None = None,
        limit: int = 3,
        only_passed: bool = True,
    ) -> List[FuzzyMatchCandidate]:
        """Create candidate pairs by matching each string against the set."""
        pool = list(values)
        if len(pool) < 2:
            return []

        candidates = self.match_batch(pool, pool, entity_type=entity_type, limit=limit)

        filtered = [candidate for candidate in candidates if candidate.source != candidate.target]
        if only_passed:
            filtered = [candidate for candidate in filtered if candidate.passed]

        deduped: dict[tuple[str, str, str | None], FuzzyMatchCandidate] = {}
        for candidate in filtered:
            left, right = sorted([candidate.source_normalized, candidate.target_normalized])
            key = (left, right, candidate.entity_type)
            if key not in deduped or candidate.confidence > deduped[key].confidence:
                deduped[key] = candidate

        return sorted(deduped.values(), key=lambda item: item.confidence, reverse=True)

    def _normalize(self, value: str) -> str:
        result = self.normalizer.normalize(value)
        return result.normalized

    def _threshold_for(self, entity_type: str | None, source: str, target: str) -> float:
        threshold = self.config.fuzzy_threshold
        if entity_type:
            threshold = self.threshold_overrides.get(entity_type.upper(), threshold)

        shortest = min(len(source), len(target))
        if shortest == 0:
            return 1.0
        if shortest < 4:
            threshold = max(threshold, 0.96)
        elif shortest < 8:
            threshold = max(threshold, threshold + 0.02)

        return min(threshold, 0.99)

    def _score(self, source_norm: str, target_norm: str) -> float:
        return self.scorer(source_norm, target_norm) / 100.0

    def _confidence(self, score: float, threshold: float, source: str, target: str) -> float:
        normalized_score = max(0.0, min(1.0, score))
        if not source or not target:
            return 0.0

        length_penalty = min(abs(len(source) - len(target)) / max(len(source), len(target)), 0.4)
        length_penalty *= 0.25

        margin_bonus = max(0.0, normalized_score - threshold)
        confidence = normalized_score - length_penalty + margin_bonus
        return max(0.0, min(1.0, confidence))
