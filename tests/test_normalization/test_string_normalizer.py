"""Tests for StringNormalizer (Phase 3 Task 3.1)."""

from __future__ import annotations

from pathlib import Path

from src.normalization.string_normalizer import NormalizationRules, StringNormalizer


def _write_rules(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "lowercase: true",
                "unicode_form: NFKC",
                "collapse_whitespace: true",
                "preserve_case_terms:",
                "  - \"C++\"",
                "  - \"GPU\"",
                "punctuation_replacements:",
                "  \"–\": \"-\"",
                "  \"—\": \"-\"",
                "  \"“\": '\"'",
                "  \"”\": '\"'",
                "strip_characters:",
                "  - \"\\u200b\"",
                "keep_symbols: ['+', '#', '-', '/', '_', '.', ':', \"'\"]",
                "tighten_around_symbols: ['+', '#', '/', '-']",
            ]
        ),
        encoding="utf-8",
    )


def test_normalizes_and_preserves_display_terms(tmp_path: Path) -> None:
    rules_file = tmp_path / "rules.yaml"
    _write_rules(rules_file)

    normalizer = StringNormalizer(rules_path=rules_file, rules=None)

    result = normalizer.normalize('  “NASA” GPU systems — C++  ')

    assert result.normalized == "nasa gpu systems - c++"
    assert result.display == "NASA GPU systems - C++"


def test_tightens_symbols_and_collapses_whitespace(tmp_path: Path) -> None:
    rules_file = tmp_path / "rules.yaml"
    _write_rules(rules_file)

    normalizer = StringNormalizer(rules_path=rules_file, rules=None)

    result = normalizer.normalize("GPU / CPU   v2")

    assert result.normalized == "gpu/cpu v2"
    assert result.display == "GPU/CPU v2"


def test_handles_empty_and_none() -> None:
    rules = NormalizationRules()
    normalizer = StringNormalizer(rules=rules)

    empty_result = normalizer.normalize("   ")
    none_result = normalizer.normalize(None)

    assert empty_result.normalized == ""
    assert empty_result.display == ""
    assert none_result.normalized == ""
    assert none_result.display == ""
