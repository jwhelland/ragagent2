"""Merge entity candidates from spaCy and LLM extractors.

Merging happens *within a chunk* (chunk_id-scoped) and is based on a normalized name
match across:
- The surface form (spaCy `text`, LLM `name`)
- LLM-provided `aliases`
- Any aliases accumulated during merging

Type conflict resolution:
- Each extractor contributes a weighted confidence vote per raw type label
  (`spacy_weight`, `llm_weight`).
- The resolved type is the label with the highest total weighted score; remaining
  non-zero labels are recorded as `conflicting_types`.

Confidence combination:
- Weighted confidences are combined with a probabilistic OR:
  `combined = 1 - Π(1 - conf_i)`.
- If evidence comes from more than one extractor, a small bonus (+0.05, capped at 1.0)
  is applied to reflect cross-source confirmation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.extraction.llm_extractor import LLMExtractedEntity
from src.extraction.spacy_extractor import ExtractedEntity
from src.normalization import StringNormalizer


class SourceAttribution(BaseModel):
    """Evidence for a merged entity from a specific extractor."""

    model_config = ConfigDict(extra="allow")

    extractor: str
    raw_type: str
    confidence: float
    text: str
    sentence: str | None = None
    context: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MergedEntityCandidate(BaseModel):
    """Merged representation of an entity across extractors."""

    model_config = ConfigDict(extra="forbid")

    canonical_name: str
    canonical_normalized: str = Field(
        default="", description="Normalized canonical form for matching"
    )
    resolved_type: str
    combined_confidence: float
    chunk_id: str | None
    document_id: str | None
    mention_count: int
    provenance: List[SourceAttribution]
    aliases: List[str] = Field(default_factory=list)
    description: str = ""
    conflicting_types: List[str] = Field(default_factory=list)


class _CandidateAccumulator:
    """Internal accumulator used during merging."""

    def __init__(
        self,
        canonical_name: str,
        chunk_id: str | None,
        document_id: str | None,
        normalizer: StringNormalizer,
    ) -> None:
        self.normalizer = normalizer

        canonical_norm = self.normalizer.normalize(canonical_name)
        self.canonical_name = canonical_norm.display or canonical_name
        self.canonical_norm = canonical_norm.normalized or self.canonical_name.lower()
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.aliases: set[str] = {self.canonical_name}
        self.alias_norms: set[str] = {self.canonical_norm}
        self.description: str = ""
        self.type_scores: Dict[str, float] = defaultdict(float)
        self.type_sources: Dict[str, set[str]] = defaultdict(set)
        self.weighted_confidences: List[Tuple[str, float]] = []
        self.provenance: List[SourceAttribution] = []

    def add_spacy(self, entity: ExtractedEntity, *, weight: float) -> None:
        """Add spaCy evidence."""
        self._ensure_document(entity.document_id)
        self._add_alias(entity.text)
        self._record_type(entity.label, weight * _clamp(entity.confidence), source="spacy")
        self._record_confidence(source="spacy", confidence=weight * _clamp(entity.confidence))

        self.provenance.append(
            SourceAttribution(
                extractor="spacy",
                raw_type=entity.label,
                confidence=_clamp(entity.confidence),
                text=entity.text,
                sentence=entity.sentence,
                context=entity.context,
                start_char=entity.start_char,
                end_char=entity.end_char,
                metadata=entity.metadata or {},
            )
        )

    def add_llm(self, entity: LLMExtractedEntity, *, weight: float) -> None:
        """Add LLM evidence."""
        self._ensure_document(entity.document_id)
        self._prefer_canonical(entity.name)
        self._add_alias(entity.name)
        self._add_aliases(entity.aliases)
        if entity.description and not self.description:
            self.description = entity.description

        self._record_type(entity.type, weight * _clamp(entity.confidence), source="llm")
        self._record_confidence(source="llm", confidence=weight * _clamp(entity.confidence))

        self.provenance.append(
            SourceAttribution(
                extractor="llm",
                raw_type=entity.type,
                confidence=_clamp(entity.confidence),
                text=entity.name,
                metadata=entity.raw or {},
            )
        )

    def to_candidate(self) -> MergedEntityCandidate:
        """Finalize accumulator into a merged candidate."""
        resolved_type, conflicting = self._resolve_type()
        combined_conf = _combine_confidences(self.weighted_confidences)

        return MergedEntityCandidate(
            canonical_name=self.canonical_name,
            canonical_normalized=self.canonical_norm,
            resolved_type=resolved_type,
            combined_confidence=combined_conf,
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            mention_count=len(self.provenance),
            provenance=self.provenance,
            aliases=sorted(self.aliases),
            description=self.description,
            conflicting_types=conflicting,
        )

    def _record_type(self, raw_type: str, weighted_confidence: float, *, source: str) -> None:
        type_label = raw_type.upper()
        self.type_scores[type_label] += weighted_confidence
        self.type_sources[type_label].add(source)

    def _record_confidence(self, *, source: str, confidence: float) -> None:
        if confidence <= 0:
            return
        self.weighted_confidences.append((source, confidence))

    def _add_alias(self, alias: str) -> None:
        if not alias:
            return
        norm = self.normalizer.normalize(alias)
        alias_display = norm.display or alias
        self.aliases.add(alias_display)
        if norm.normalized:
            self.alias_norms.add(norm.normalized)

    def _add_aliases(self, aliases: Iterable[str]) -> None:
        for alias in aliases:
            self._add_alias(alias)

    def _prefer_canonical(self, candidate_name: str) -> None:
        """Prefer LLM-provided names as canonical when available."""
        if not candidate_name:
            return
        norm = self.normalizer.normalize(candidate_name)
        self.canonical_name = norm.display or candidate_name
        self.canonical_norm = norm.normalized or self.canonical_name.lower()
        self.alias_norms.add(self.canonical_norm)
        self.aliases.add(self.canonical_name)

    def _ensure_document(self, document_id: str | None) -> None:
        if not self.document_id:
            self.document_id = document_id

    def _resolve_type(self) -> Tuple[str, List[str]]:
        if not self.type_scores:
            return "UNKNOWN", []
        sorted_types = sorted(self.type_scores.items(), key=lambda item: item[1], reverse=True)
        resolved = sorted_types[0][0]
        conflicting = [label for label, score in sorted_types[1:] if score > 0]
        return resolved, conflicting


class EntityMerger:
    """Merge entities across spaCy and LLM extractors with conflict resolution."""

    def __init__(
        self,
        allowed_types: Optional[Sequence[str]] = None,
        *,
        spacy_weight: float = 0.9,
        llm_weight: float = 1.0,
        normalizer: StringNormalizer | None = None,
    ) -> None:
        self.allowed_types = {t.upper() for t in allowed_types} if allowed_types else None
        self.source_weights = {"spacy": spacy_weight, "llm": llm_weight}
        self.normalizer = normalizer or StringNormalizer()

    def merge(
        self,
        spacy_entities: Optional[Iterable[ExtractedEntity]] = None,
        llm_entities: Optional[Iterable[LLMExtractedEntity]] = None,
    ) -> List[MergedEntityCandidate]:
        """Merge entities from both extractors into unified candidates."""
        accumulators: Dict[Tuple[str, str], _CandidateAccumulator] = {}

        for ent in spacy_entities or []:
            if self._is_filtered(ent.label):
                continue
            key = self._match_key(accumulators, ent.text, ent.chunk_id, alias_norms=set())
            acc = accumulators.setdefault(
                key,
                _CandidateAccumulator(ent.text, ent.chunk_id, ent.document_id, self.normalizer),
            )
            acc.add_spacy(ent, weight=self.source_weights["spacy"])

        for ent in llm_entities or []:
            if self._is_filtered(ent.type):
                continue
            alias_norms = {self._normalize_name(a) for a in ent.aliases}
            alias_norms.discard("")
            name_norm = self._normalize_name(ent.name)
            if name_norm:
                alias_norms.add(name_norm)
            key = self._match_key(accumulators, ent.name, ent.chunk_id, alias_norms=alias_norms)
            acc = accumulators.setdefault(
                key,
                _CandidateAccumulator(ent.name, ent.chunk_id, ent.document_id, self.normalizer),
            )
            acc.add_llm(ent, weight=self.source_weights["llm"])

        merged = [acc.to_candidate() for acc in accumulators.values()]
        logger.debug("Merged entities", total=len(merged))
        return merged

    def _match_key(
        self,
        accumulators: Dict[Tuple[str, str], _CandidateAccumulator],
        name: str,
        chunk_id: str | None,
        *,
        alias_norms: set[str],
    ) -> Tuple[str, str]:
        chunk_key = chunk_id or "__no_chunk__"
        normalized = self._normalize_name(name)
        exact_key = (chunk_key, normalized)
        if exact_key in accumulators:
            return exact_key

        for key, candidate in accumulators.items():
            if key[0] != chunk_key:
                continue
            if normalized == key[1]:
                return key
            if normalized in candidate.alias_norms:
                return key
            if alias_norms & candidate.alias_norms:
                return key

        return exact_key

    def _is_filtered(self, entity_type: str) -> bool:
        return bool(self.allowed_types) and entity_type.upper() not in self.allowed_types

    def _normalize_name(self, name: str) -> str:
        norm = self.normalizer.normalize(name)
        if norm.normalized:
            return norm.normalized
        if name:
            return " ".join(name.split()).lower()
        return ""


def _clamp(value: float) -> float:
    return max(0.0, min(value, 1.0))


def _combine_confidences(confidences: List[Tuple[str, float]]) -> float:
    if not confidences:
        return 0.0

    distinct_sources = {source for source, _ in confidences}
    prob_not = 1.0
    for _, conf in confidences:
        prob_not *= 1 - _clamp(conf)

    combined = 1 - prob_not
    if len(distinct_sources) > 1:
        combined = min(1.0, combined + 0.05)

    return combined
