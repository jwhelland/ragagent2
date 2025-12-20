"""Tests for FuzzyMatcher (Phase 3 Task 3.2)."""

from __future__ import annotations

import pytest

from src.normalization.fuzzy_matcher import FuzzyMatcher
from src.utils.config import NormalizationConfig


def test_matches_variants_with_high_confidence() -> None:
    matcher = FuzzyMatcher()

    result = matcher.match_pair("Power Subsystem", "power sub-system", entity_type="system")

    assert result.passed is True
    assert result.score >= 0.9
    assert result.confidence > 0.85


def test_short_strings_require_stricter_threshold() -> None:
    matcher = FuzzyMatcher()

    result = matcher.match_pair("EPS", "EPC")

    assert result.threshold >= 0.96
    assert result.passed is False


def test_batch_matching_returns_top_candidates() -> None:
    matcher = FuzzyMatcher()

    sources = ["Power subsystem", "Attitude Control System"]
    choices = ["power sub-system", "attitude controls system", "thermal control"]

    matches = matcher.match_batch(sources, choices, limit=2)

    power_matches = [m for m in matches if m.source.startswith("Power")]
    attitude_matches = [m for m in matches if m.source.startswith("Attitude")]

    assert any(match.target == "power sub-system" and match.passed for match in power_matches)
    assert any(match.target == "attitude controls system" and match.passed for match in attitude_matches)


def test_threshold_override_applies_by_entity_type() -> None:
    matcher = FuzzyMatcher(threshold_overrides={"system": 0.82})

    result = matcher.match_pair(
        "Command and Data Handling", "Command & Data Handling", entity_type="system"
    )

    assert result.threshold == pytest.approx(0.82)
    assert result.passed is True


def test_config_threshold_override_applies_by_entity_type() -> None:
    config = NormalizationConfig(fuzzy_threshold_overrides={"SYSTEM": 0.81})
    matcher = FuzzyMatcher(config=config)

    result = matcher.match_pair("Solar Array Drive", "Solar Array Drive Assembly", entity_type="system")

    assert result.threshold == pytest.approx(0.81)


def test_generate_candidates_excludes_self_and_dedupes() -> None:
    matcher = FuzzyMatcher()

    candidates = matcher.generate_candidates(["Power subsystem", "power sub-system", "Thermal control"])

    assert all(candidate.source != candidate.target for candidate in candidates)
    assert len({(c.source_normalized, c.target_normalized) for c in candidates}) == len(candidates)
