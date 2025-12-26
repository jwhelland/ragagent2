"""String normalization utilities for entity canonicalization (Task 3.1)."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import yaml
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.utils.config import NormalizationConfig


class PreservedTerm(BaseModel):
    """Term that should keep its original casing/symbols."""

    model_config = ConfigDict(frozen=True)

    placeholder: str
    original: str
    normalized: str


class NormalizationResult(BaseModel):
    """Result of a normalization call."""

    model_config = ConfigDict(frozen=True)

    original: str
    normalized: str
    display: str
    preserved_terms: Mapping[str, PreservedTerm]

    def to_display(self) -> str:
        """Return the display-form string (alias for display)."""
        return self.display


class NormalizationRules(BaseModel):
    """Normalization rule set loaded from YAML."""

    model_config = ConfigDict(extra="ignore")

    lowercase: bool = True
    unicode_form: str = "NFKC"
    collapse_whitespace: bool = True
    preserve_case_terms: List[str] = Field(default_factory=list)
    preserve_case_patterns: List[str] = Field(
        default_factory=lambda: [
            r"\b[A-Z]{2,}\b",  # acronyms
            r"\b[A-Z]\d{1,}\b",  # mixed letter-number (e.g., L2)
            r"\bC\+\+\b",
        ]
    )
    punctuation_replacements: Dict[str, str] = Field(
        default_factory=lambda: {
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "–": "-",
            "—": "-",
            "−": "-",
            "·": "-",
            "•": "-",
            "…": "...",
            "×": "x",
        }
    )
    strip_characters: List[str] = Field(default_factory=lambda: ["\u200b", "\ufeff"])
    strip_articles: List[str] = Field(default_factory=lambda: ["the", "a", "an"])
    enable_article_stripping: bool = True
    keep_symbols: List[str] = Field(
        default_factory=lambda: ["+", "#", "-", "/", "_", ".", ":", "'"]
    )
    tighten_around_symbols: List[str] = Field(default_factory=lambda: ["+", "#", "/", "-"])

    @classmethod
    def from_yaml(cls, rules_file: Path | None) -> NormalizationRules:
        """Load rules from YAML, merging with defaults."""
        base = cls()

        if rules_file is None:
            return base

        if not rules_file.exists():
            raise FileNotFoundError(f"Normalization rules file not found: {rules_file}")

        loaded = yaml.safe_load(rules_file.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Normalization rules must be a mapping/dict.")

        merged = base.model_dump()
        for key, value in loaded.items():
            if key not in merged:
                continue
            if isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value

        return cls(**merged)

    @field_validator("unicode_form")
    @classmethod
    def _validate_unicode_form(cls, value: str) -> str:
        valid_forms = {"NFC", "NFD", "NFKC", "NFKD"}
        upper_value = value.upper()
        if upper_value not in valid_forms:
            raise ValueError(f"Invalid unicode_form '{value}'. Must be one of {valid_forms}.")
        return upper_value


class StringNormalizer:
    """Normalize entity strings while keeping technical meaning intact."""

    def __init__(
        self,
        config: NormalizationConfig | None = None,
        rules_path: str | Path | None = None,
        rules: NormalizationRules | None = None,
    ) -> None:
        self.config = config or NormalizationConfig()

        rules_file = Path(rules_path) if rules_path else Path(self.config.rules_file)
        self.rules = rules or NormalizationRules.from_yaml(rules_file)

        self._preserve_patterns: List[re.Pattern[str]] = [
            re.compile(pattern) for pattern in self.rules.preserve_case_patterns
        ]
        self._punctuation_translation = {
            ord(src): dest for src, dest in self.rules.punctuation_replacements.items()
        }
        tighten_symbols = "".join(re.escape(sym) for sym in self.rules.tighten_around_symbols)
        self._tighten_symbols_re = (
            re.compile(rf"\s*([{tighten_symbols}])\s*") if tighten_symbols else None
        )
        allowed_symbols = "".join(re.escape(sym) for sym in self.rules.keep_symbols)
        self._unwanted_punct_re = re.compile(rf"[^\w\s{allowed_symbols}]")
        self._whitespace_re = re.compile(r"\s+")

        # Leading articles regex
        if self.rules.strip_articles:
            articles_pattern = "|".join(re.escape(a) for a in self.rules.strip_articles)
            self._articles_re = re.compile(rf"^({articles_pattern})\s+", re.IGNORECASE)
        else:
            self._articles_re = None

        logger.info(f"Loaded normalization rules from {rules_file}")

    def normalize(self, text: str | None) -> NormalizationResult:
        """Normalize a single string."""
        if text is None:
            return NormalizationResult(original="", normalized="", display="", preserved_terms={})

        working = text.strip()
        if not working:
            return NormalizationResult(original=text, normalized="", display="", preserved_terms={})

        preserved, protected_text = self._protect_terms(working)

        normalized_body = self._apply_rules(protected_text, lowercase=self.rules.lowercase)
        normalized = self._restore_preserved(normalized_body, preserved, use_original=False)

        display_body = self._apply_rules(protected_text, lowercase=False)
        display = self._restore_preserved(display_body, preserved, use_original=True)

        return NormalizationResult(
            original=text, normalized=normalized, display=display, preserved_terms=preserved
        )

    def normalize_batch(self, texts: Sequence[str | None]) -> List[NormalizationResult]:
        """Normalize a batch of strings."""
        return [self.normalize(text) for text in texts]

    def _apply_rules(self, text: str, lowercase: bool) -> str:
        text = self._normalize_unicode(text)
        text = self._strip_characters(text)
        text = self._apply_punctuation_replacements(text)
        text = self._remove_unwanted_punctuation(text)
        text = self._tighten_technical_symbols(text)
        text = self._collapse_whitespace(text)

        if self.rules.enable_article_stripping:
            text = self._strip_leading_articles(text)

        if lowercase and self.rules.lowercase:
            text = text.lower()

        return text.strip()

    def _strip_leading_articles(self, text: str) -> str:
        if not self._articles_re:
            return text

        # Strip recursively in case of multiple articles (unlikely but safe)
        while True:
            new_text = self._articles_re.sub("", text)
            if new_text == text or not new_text.strip():
                break
            text = new_text
        return text

    def _normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize(self.rules.unicode_form, text)

    def _strip_characters(self, text: str) -> str:
        for ch in self.rules.strip_characters:
            text = text.replace(ch, "")
        return text

    def _apply_punctuation_replacements(self, text: str) -> str:
        if not self._punctuation_translation:
            return text
        return text.translate(self._punctuation_translation)

    def _remove_unwanted_punctuation(self, text: str) -> str:
        return self._unwanted_punct_re.sub(" ", text)

    def _tighten_technical_symbols(self, text: str) -> str:
        if not self._tighten_symbols_re:
            return text
        return self._tighten_symbols_re.sub(self._tighten_replacer, text)

    def _tighten_replacer(self, match: re.Match[str]) -> str:
        symbol = match.group(1)
        if symbol == "-":
            return " - "
        return symbol

    def _collapse_whitespace(self, text: str) -> str:
        return self._whitespace_re.sub(" ", text) if self.rules.collapse_whitespace else text

    def _protect_terms(self, text: str) -> Tuple[MutableMapping[str, PreservedTerm], str]:
        spans: List[Tuple[int, int, str]] = []

        for term in self.rules.preserve_case_terms:
            for match in re.finditer(re.escape(term), text, flags=re.IGNORECASE):
                spans.append((match.start(), match.end(), match.group(0)))

        for pattern in self._preserve_patterns:
            for match in pattern.finditer(text):
                spans.append((match.start(), match.end(), match.group(0)))

        spans = self._dedupe_spans(spans)
        if not spans:
            return {}, text

        preserved: MutableMapping[str, PreservedTerm] = {}
        pieces: List[str] = []
        cursor = 0

        for idx, (start, end, value) in enumerate(spans):
            if start < cursor:
                continue

            placeholder = f"__preserved_{idx}__"
            pieces.append(text[cursor:start])
            pieces.append(placeholder)

            preserved[placeholder] = PreservedTerm(
                placeholder=placeholder,
                original=value,
                normalized=self._safe_lower(value),
            )
            cursor = end

        pieces.append(text[cursor:])
        protected_text = "".join(pieces)
        return preserved, protected_text

    def _dedupe_spans(self, spans: Iterable[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
        sorted_spans = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
        deduped: List[Tuple[int, int, str]] = []

        for start, end, value in sorted_spans:
            if any(
                existing_start <= start < existing_end
                for existing_start, existing_end, _ in deduped
            ):
                continue
            deduped.append((start, end, value))

        return deduped

    def _restore_preserved(
        self, text: str, preserved: Mapping[str, PreservedTerm], use_original: bool
    ) -> str:
        for placeholder, term in preserved.items():
            replacement = term.original if use_original else term.normalized
            text = text.replace(placeholder, replacement)
        return text

    def _safe_lower(self, value: str) -> str:
        return value if not self.rules.lowercase else value.lower()
