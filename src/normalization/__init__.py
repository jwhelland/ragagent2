"""Normalization package."""

from src.normalization.acronym_resolver import (
    AcronymDefinition,
    AcronymDictionaryEntry,
    AcronymResolution,
    AcronymResolver,
)
from src.normalization.entity_deduplicator import (
    DeduplicationResult,
    EntityCluster,
    EntityDeduplicator,
    EntityRecord,
    MergeSuggestion,
)
from src.normalization.fuzzy_matcher import FuzzyMatchCandidate, FuzzyMatcher
from src.normalization.normalization_table import (
    NormalizationEntry,
    NormalizationMethod,
    NormalizationRecord,
    NormalizationTable,
)
from src.normalization.string_normalizer import (
    NormalizationResult,
    NormalizationRules,
    StringNormalizer,
)

__all__ = [
    "AcronymDefinition",
    "AcronymDictionaryEntry",
    "AcronymResolution",
    "AcronymResolver",
    "DeduplicationResult",
    "EntityCluster",
    "EntityDeduplicator",
    "EntityRecord",
    "FuzzyMatchCandidate",
    "FuzzyMatcher",
    "MergeSuggestion",
    "NormalizationEntry",
    "NormalizationMethod",
    "NormalizationRecord",
    "NormalizationResult",
    "NormalizationRules",
    "NormalizationTable",
    "StringNormalizer",
]
