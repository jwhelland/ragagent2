"""Relationship validation module for filtering low-quality relationships."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger
from pydantic import BaseModel
from rapidfuzz import fuzz, process

from src.normalization.string_normalizer import StringNormalizer
from src.utils.config import RelationshipValidationConfig


class ValidationResult(BaseModel):
    """Result of relationship validation."""

    valid: bool
    reason: Optional[str] = None


class RelationshipValidator:
    """Validates extracted relationships to filter low-quality candidates."""

    def __init__(
        self,
        config: RelationshipValidationConfig,
        normalizer: StringNormalizer,
    ) -> None:
        """Initialize the validator.

        Args:
            config: Validation configuration
            normalizer: String normalizer for entity name comparison
        """
        self.config = config
        self.normalizer = normalizer

        # Common adjectives that shouldn't be IS_A targets
        self.adjective_indicators = {
            "critical",
            "important",
            "essential",
            "primary",
            "secondary",
            "active",
            "passive",
            "normal",
            "abnormal",
            "high",
            "low",
            "major",
            "minor",
            "key",
            "main",
            "significant",
            "trivial",
            "basic",
            "advanced",
        }

        logger.info(
            "Initialized RelationshipValidator "
            f"(confidence>={config.min_confidence_threshold}, "
            f"fuzzy_threshold={config.fuzzy_entity_match_threshold}, "
            f"validate_entity_existence={config.validate_entity_existence})"
        )

    def validate_relationship(
        self,
        relationship: Dict[str, Any],
        known_entities: Sequence[Dict[str, Any]],
    ) -> ValidationResult:
        """Validate a single relationship.

        Args:
            relationship: Relationship dict with source, target, type, confidence
            known_entities: List of entity dicts with name/canonical_name

        Returns:
            ValidationResult indicating if valid and reason if not
        """
        # Extract relationship data
        source = str(relationship.get("source") or "").strip()
        target = str(relationship.get("target") or "").strip()
        rel_type = str(relationship.get("type") or "").strip()
        confidence = float(relationship.get("confidence") or 0.0)

        # Check basic fields
        if not source or not target or not rel_type:
            return ValidationResult(
                valid=False,
                reason="Missing required field (source, target, or type)",
            )

        # Check confidence threshold
        if confidence < self.config.min_confidence_threshold:
            return ValidationResult(
                valid=False,
                reason=f"Confidence {confidence:.2f} below threshold "
                f"{self.config.min_confidence_threshold}",
            )

        # Check entity name length
        if len(source) > self.config.max_entity_name_length:
            return ValidationResult(
                valid=False,
                reason=f"Source name too long ({len(source)} > "
                f"{self.config.max_entity_name_length})",
            )

        if len(target) > self.config.max_entity_name_length:
            return ValidationResult(
                valid=False,
                reason=f"Target name too long ({len(target)} > "
                f"{self.config.max_entity_name_length})",
            )

        # Check IS_A semantic validity
        if rel_type == "IS_A":
            target_lower = target.lower()
            if any(word in target_lower for word in self.adjective_indicators):
                return ValidationResult(
                    valid=False,
                    reason=f"IS_A target '{target}' appears to be adjective, not type",
                )

        # Check entity existence with fuzzy matching
        if self.config.validate_entity_existence:
            entity_names = []
            for ent in known_entities:
                name = ""
                if isinstance(ent, dict):
                    name = str(
                        ent.get("name") or ent.get("canonical_name") or ent.get("text") or ""
                    )
                else:
                    name = str(
                        getattr(ent, "name", None)
                        or getattr(ent, "canonical_name", None)
                        or getattr(ent, "text", None)
                        or ""
                    )

                if name.strip():
                    entity_names.append(name.strip())

            if not entity_names:
                return ValidationResult(
                    valid=False,
                    reason=f"No known entities found in context to validate '{source}' or '{target}'",
                )
            else:
                # Normalize entity names for comparison
                normalized_entities = [
                    self.normalizer.normalize(name).normalized for name in entity_names
                ]
                source_norm = self.normalizer.normalize(source).normalized
                target_norm = self.normalizer.normalize(target).normalized

                # Check if source exists (fuzzy match)
                source_match = process.extractOne(
                    source_norm,
                    normalized_entities,
                    scorer=fuzz.ratio,
                    score_cutoff=self.config.fuzzy_entity_match_threshold
                    * 100,  # rapidfuzz uses 0-100
                )
                if not source_match:
                    return ValidationResult(
                        valid=False,
                        reason=f"Source entity '{source}' not found in known entities",
                    )

                # Check if target exists (fuzzy match)
                target_match = process.extractOne(
                    target_norm,
                    normalized_entities,
                    scorer=fuzz.ratio,
                    score_cutoff=self.config.fuzzy_entity_match_threshold
                    * 100,  # rapidfuzz uses 0-100
                )
                if not target_match:
                    return ValidationResult(
                        valid=False,
                        reason=f"Target entity '{target}' not found in known entities",
                    )

        return ValidationResult(valid=True)

    def filter_relationships(
        self,
        relationships: Sequence[Dict[str, Any]],
        known_entities: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], str]]]:
        """Filter relationships, returning valid and rejected lists.

        Args:
            relationships: List of relationship dicts
            known_entities: List of entity dicts

        Returns:
            Tuple of (valid_relationships, rejected_with_reasons)
            where rejected_with_reasons is a list of (relationship, reason) tuples
        """
        valid: List[Dict[str, Any]] = []
        rejected: List[Tuple[Dict[str, Any], str]] = []

        for rel in relationships:
            result = self.validate_relationship(rel, known_entities)
            if result.valid:
                valid.append(rel)
            else:
                rejected.append((rel, result.reason or "Unknown validation failure"))

        if rejected:
            logger.debug(
                f"Filtered {len(rejected)}/{len(relationships)} relationships "
                f"(kept {len(valid)})"
            )

        return valid, rejected
