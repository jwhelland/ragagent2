"""Acronym resolution utilities (Task 3.3)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import yaml  # type: ignore[import-untyped]
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz

from src.ingestion.chunker import Chunk
from src.normalization.string_normalizer import StringNormalizer
from src.utils.config import NormalizationConfig


class AcronymDefinition(BaseModel):
    """A single acronym definition extracted from text."""

    model_config = ConfigDict(frozen=True)

    acronym: str
    expansion: str
    context: str | None = None
    source: str = "pattern"
    document_id: str | None = None
    chunk_id: str | None = None


class AcronymResolution(BaseModel):
    """Resolved acronym with confidence and normalized form."""

    model_config = ConfigDict(frozen=True)

    acronym: str
    expansion: str
    normalized_expansion: str
    confidence: float
    method: str
    aliases: List[str] = Field(default_factory=list)


class AcronymDictionaryEntry(BaseModel):
    """Aggregated dictionary entry for a single acronym."""

    model_config = ConfigDict(extra="forbid")

    acronym: str
    expansions: Dict[str, int] = Field(default_factory=dict)
    contexts: Dict[str, List[str]] = Field(default_factory=dict)

    def add(self, expansion: str, context: str | None = None) -> None:
        """Add an expansion/context occurrence."""
        cleaned = expansion.strip()
        if not cleaned:
            return
        self.expansions[cleaned] = self.expansions.get(cleaned, 0) + 1
        if context:
            bucket = self.contexts.setdefault(cleaned, [])
            if len(bucket) < 10:
                bucket.append(context.strip())

    def frequency(self, expansion: str) -> float:
        """Relative frequency for a candidate expansion."""
        total = sum(self.expansions.values())
        if not total:
            return 0.0
        return self.expansions.get(expansion, 0) / total


class AcronymResolver:
    """Builds and uses an acronym dictionary with context-aware resolution."""

    _definition_patterns: Sequence[re.Pattern[str]] = (
        re.compile(
            r"(?P<expansion>[A-Za-z][^().\n]{1,80}?)"
            r"\s*\(\s*(?P<acronym>[A-Z][A-Z0-9&/+\-]{1,10})\s*\)"
        ),
        re.compile(
            r"(?P<acronym>[A-Z][A-Z0-9&/+\-]{1,10})"
            r"\s*\(\s*(?P<expansion>[A-Za-z][^().\n]{3,80}?)\s*\)"
        ),
    )
    _acronym_pattern = re.compile(r"\b[A-Z][A-Z0-9&/\-]{1,10}\b")

    def __init__(
        self,
        config: NormalizationConfig | None = None,
        normalizer: StringNormalizer | None = None,
        overrides_path: str | Path | None = None,
        storage_path: str | Path | None = None,
    ) -> None:
        self.config = config or NormalizationConfig()
        self.normalizer = normalizer or StringNormalizer(config=self.config)
        self.overrides_path = Path(overrides_path or self.config.acronym_overrides_file)
        self.storage_path = Path(storage_path or self.config.acronym_storage_path)
        self.overrides: Mapping[str, List[str]] = self._load_overrides(self.overrides_path)
        self.dictionary: MutableMapping[str, AcronymDictionaryEntry] = {}

        logger.info(
            "Initialized AcronymResolver with overrides={} storage={}",
            self.overrides_path,
            self.storage_path,
        )

    def extract_definitions(
        self, text: str, *, document_id: str | None = None, chunk_id: str | None = None
    ) -> List[AcronymDefinition]:
        """Extract acronym definitions like 'Natural Language Processing (NLP)'."""
        if not text:
            return []

        definitions: List[AcronymDefinition] = []
        for pattern in self._definition_patterns:
            for match in pattern.finditer(text):
                acronym = match.group("acronym").strip()
                expansion = self._clean_expansion(match.group("expansion"))
                if not self._is_valid_acronym(acronym) or len(expansion) < 2:
                    continue

                context = self._sentence_context(text, match.start(), match.end())
                definitions.append(
                    AcronymDefinition(
                        acronym=acronym,
                        expansion=expansion,
                        context=context,
                        source="pattern",
                        document_id=document_id,
                        chunk_id=chunk_id,
                    )
                )

        return definitions

    def build_dictionary_from_texts(
        self, texts: Iterable[str]
    ) -> Mapping[str, AcronymDictionaryEntry]:
        """Build acronym dictionary from raw text corpus."""
        self.dictionary.clear()
        for text in texts:
            for definition in self.extract_definitions(text):
                self._add_definition(definition)
        return dict(self.dictionary)

    def update_dictionary_from_texts(self, texts: Iterable[str]) -> int:
        """Update (aggregate) acronym dictionary from additional texts.

        Returns:
            Number of definition occurrences added.
        """
        added = 0
        for text in texts:
            for definition in self.extract_definitions(text):
                self._add_definition(definition)
                added += 1
        return added

    def build_dictionary_from_chunks(
        self, chunks: Iterable[Chunk]
    ) -> Mapping[str, AcronymDictionaryEntry]:
        """Build acronym dictionary from chunk objects."""
        self.dictionary.clear()
        for chunk in chunks:
            for definition in self.extract_definitions(
                chunk.content, document_id=chunk.document_id, chunk_id=chunk.chunk_id
            ):
                self._add_definition(definition)
        return dict(self.dictionary)

    def update_dictionary_from_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Update (aggregate) acronym dictionary from additional chunks.

        Returns:
            Number of definition occurrences added.
        """
        added = 0
        for chunk in chunks:
            for definition in self.extract_definitions(
                chunk.content, document_id=chunk.document_id, chunk_id=chunk.chunk_id
            ):
                self._add_definition(definition)
                added += 1
        return added

    def resolve(self, acronym: str, context: str | None = None) -> AcronymResolution | None:
        """Resolve a single acronym using overrides, dictionary, and context."""
        if not acronym:
            return None
        normalized = acronym.strip().upper()
        if not self._is_valid_acronym(normalized):
            return None

        context_norm = self._normalize_context(context)

        if normalized in self.overrides:
            expansion, confidence = self._select_best(
                self.overrides[normalized], context_norm, entry=None
            )
            return self._to_resolution(
                acronym=normalized,
                expansion=expansion,
                confidence=max(confidence, 0.95),
                method="override",
            )

        entry = self.dictionary.get(normalized)
        if not entry:
            return None

        expansion, confidence = self._select_best(
            list(entry.expansions.keys()), context_norm, entry=entry
        )
        return self._to_resolution(
            acronym=normalized, expansion=expansion, confidence=confidence, method="dictionary"
        )

    def resolve_in_text(self, text: str) -> List[AcronymResolution]:
        """Resolve all acronyms present in a text block using text as context."""
        if not text:
            return []

        acronyms = {match.group(0) for match in self._acronym_pattern.finditer(text)}
        resolutions: List[AcronymResolution] = []
        for acronym in sorted(acronyms):
            resolution = self.resolve(acronym, context=text)
            if resolution:
                resolutions.append(resolution)
        return resolutions

    def store_mappings(self, path: str | Path | None = None) -> Path:
        """Persist acronym mappings to YAML for downstream use."""
        target = Path(path) if path else self.storage_path
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "overrides": self.overrides,
            "dictionary": {
                acronym: {
                    "expansions": entry.expansions,
                    "contexts": entry.contexts,
                }
                for acronym, entry in sorted(self.dictionary.items())
            },
        }

        target.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
        logger.info("Stored acronym mappings to {}", target)
        return target

    def load_mappings(self, path: str | Path | None = None) -> None:
        """Load mappings previously stored by `store_mappings`."""
        source = Path(path) if path else self.storage_path
        if not source.exists():
            raise FileNotFoundError(f"Acronym mappings file not found: {source}")

        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("Acronym mappings file must contain a mapping/dict at the root.")

        loaded_overrides = raw.get("overrides", {})
        if isinstance(loaded_overrides, dict):
            overrides: Dict[str, List[str]] = dict(self.overrides)
            for key, value in loaded_overrides.items():
                if not isinstance(key, str):
                    continue
                if isinstance(value, str):
                    overrides[key.upper()] = [value]
                elif isinstance(value, list):
                    cleaned = [str(item) for item in value if str(item).strip()]
                    if cleaned:
                        overrides[key.upper()] = cleaned
            self.overrides = overrides

        loaded_dict = raw.get("dictionary", {})
        if not isinstance(loaded_dict, dict):
            raise ValueError("Acronym mappings 'dictionary' must be a mapping/dict.")

        dictionary: Dict[str, AcronymDictionaryEntry] = {}
        for acronym, payload in loaded_dict.items():
            if not isinstance(acronym, str) or not isinstance(payload, dict):
                continue
            expansions = payload.get("expansions", {})
            contexts = payload.get("contexts", {})
            if not isinstance(expansions, dict) or not isinstance(contexts, dict):
                continue
            dictionary[acronym.upper()] = AcronymDictionaryEntry(
                acronym=acronym.upper(),
                expansions={str(k): int(v) for k, v in expansions.items()},
                contexts={
                    str(key): [str(item) for item in value]
                    for key, value in contexts.items()
                    if isinstance(value, list)
                },
            )

        self.dictionary = dictionary
        logger.info("Loaded acronym mappings from {}", source)

    def _add_definition(self, definition: AcronymDefinition) -> None:
        acronym = definition.acronym.upper()
        entry = self.dictionary.get(acronym)
        if entry is None:
            entry = AcronymDictionaryEntry(acronym=acronym)
            self.dictionary[acronym] = entry
        entry.add(definition.expansion, context=definition.context)

    def _select_best(
        self,
        candidates: Sequence[str],
        context_norm: str | None,
        entry: AcronymDictionaryEntry | None,
    ) -> tuple[str, float]:
        if not candidates:
            return "", 0.0
        if len(candidates) == 1:
            return candidates[0], 1.0

        best_candidate = candidates[0]
        best_score = -1.0

        for candidate in candidates:
            base_score = entry.frequency(candidate) if entry else 1.0 / len(candidates)
            context_score = self._context_similarity(
                context_norm, entry.contexts.get(candidate, []) if entry else []
            )
            combined = 0.6 * context_score + 0.4 * base_score if context_norm else base_score
            if combined > best_score:
                best_score = combined
                best_candidate = candidate

        confidence = max(0.2, min(1.0, best_score))
        return best_candidate, confidence

    def _context_similarity(self, context_norm: str | None, contexts: Sequence[str]) -> float:
        if not context_norm or not contexts:
            return 0.0

        best = 0.0
        for candidate_context in contexts:
            normalized_candidate = self._normalize_context(candidate_context)
            if not normalized_candidate:
                continue
            score = fuzz.token_set_ratio(context_norm, normalized_candidate) / 100.0
            best = max(best, score)
        return best

    def _to_resolution(
        self, acronym: str, expansion: str, confidence: float, method: str
    ) -> AcronymResolution:
        normalized = self.normalizer.normalize(expansion)
        aliases = [acronym, expansion]
        if normalized.display and normalized.display not in aliases:
            aliases.append(normalized.display)
        return AcronymResolution(
            acronym=acronym,
            expansion=expansion,
            normalized_expansion=normalized.normalized or expansion.lower(),
            confidence=max(0.0, min(1.0, confidence)),
            method=method,
            aliases=aliases,
        )

    def _clean_expansion(self, expansion: str) -> str:
        cleaned = " ".join(expansion.split()).strip(" -;:,")
        for prefix in ("The ", "A ", "An "):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :]
                break
        return cleaned.strip()

    def _normalize_context(self, context: str | None) -> str | None:
        if not context:
            return None
        normalized = self.normalizer.normalize(context)
        return normalized.normalized or context.lower()

    def _sentence_context(self, text: str, start: int, end: int) -> str:
        left = max(
            text.rfind(".", 0, start),
            text.rfind(";", 0, start),
            text.rfind("\n", 0, start),
        )
        right_period = text.find(".", end)
        right_semicolon = text.find(";", end)
        right_newline = text.find("\n", end)

        rights = [idx for idx in [right_period, right_semicolon, right_newline] if idx != -1]
        right = min(rights) if rights else len(text)

        left_bound = left + 1 if left != -1 else 0
        right_bound = right if right != -1 else len(text)
        return text[left_bound:right_bound].strip()

    def _is_valid_acronym(self, acronym: str) -> bool:
        return bool(acronym) and len(acronym) > 1 and acronym.isupper()

    def _load_overrides(self, path: Path) -> Mapping[str, List[str]]:
        if not path.exists():
            logger.info("Acronym overrides file not found at {}; using empty overrides", path)
            return {}

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            logger.warning("Overrides file must be a mapping of acronym -> expansion(s)")
            return {}

        overrides: Dict[str, List[str]] = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, str):
                overrides[key.upper()] = [value]
            elif isinstance(value, list):
                cleaned = [str(item) for item in value if str(item).strip()]
                if cleaned:
                    overrides[key.upper()] = cleaned
        return overrides
