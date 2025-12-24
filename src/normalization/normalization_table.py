"""Normalization table manager (Task 3.5)."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, MutableMapping, Sequence

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.normalization.string_normalizer import StringNormalizer
from src.utils.config import NormalizationConfig


class NormalizationMethod(str, Enum):
    """Method used to derive a normalization mapping."""

    EXACT = "exact"
    FUZZY = "fuzzy"
    ACRONYM = "acronym"
    EMBEDDING = "embedding"
    MANUAL = "manual"


class NormalizationEntry(BaseModel):
    """Input payload for creating/updating mappings."""

    raw_text: str
    canonical_id: str
    canonical_name: str
    entity_type: str | None = None
    method: NormalizationMethod = NormalizationMethod.MANUAL
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    notes: str = ""
    source: str | None = None
    status: str = "pending"


class NormalizationRecord(BaseModel):
    """Stored normalization mapping with audit metadata."""

    model_config = ConfigDict(extra="ignore")

    raw_variants: List[str] = Field(default_factory=list)
    normalized_form: str
    canonical_id: str
    canonical_name: str
    entity_type: str | None = None
    method: NormalizationMethod = NormalizationMethod.MANUAL
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    notes: str = ""
    source: str | None = None
    status: str = "pending"

    def summary(self) -> str:
        """Short, human-readable description for CLI display."""
        return (
            f"{self.canonical_name} [{self.canonical_id}] via {self.method.value} "
            f"(confidence={self.confidence:.2f}, variants={len(self.raw_variants)})"
        )


class NormalizationTableState(BaseModel):
    """Serialized normalization table state."""

    model_config = ConfigDict(extra="ignore")

    version: int = 1
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    records: Dict[str, NormalizationRecord] = Field(default_factory=dict)


class NormalizationTable:
    """Manage mappings from raw mentions to canonical entity identifiers."""

    def __init__(
        self,
        *,
        table_path: str | Path | None = None,
        config: NormalizationConfig | None = None,
        normalizer: StringNormalizer | None = None,
    ) -> None:
        self.config = config or NormalizationConfig()
        self.table_path = (
            Path(table_path) if table_path else Path(self.config.normalization_table_path)
        )
        self._normalizer = normalizer or StringNormalizer(self.config)

        self._state = NormalizationTableState()
        if self.table_path.exists():
            self._state = self._load_state(self.table_path)
            logger.info(
                "Loaded normalization table with {} records from {}",
                len(self._state.records),
                self.table_path,
            )

    # CRUD operations
    def upsert(self, entry: NormalizationEntry) -> NormalizationRecord:
        """Create or update a mapping for the given raw text."""
        record = self._apply_upsert(entry)
        self._bump_version()
        self._persist()
        return record

    def bulk_upsert(self, entries: Sequence[NormalizationEntry]) -> List[NormalizationRecord]:
        """Apply multiple upserts and persist once."""
        if not entries:
            return []

        records = [self._apply_upsert(entry) for entry in entries]
        self._bump_version()
        self._persist()
        return records

    def remove(self, raw_text: str) -> bool:
        """Delete a mapping by raw text mention."""
        key = self._normalize_key(raw_text)
        if key not in self._state.records:
            return False
        del self._state.records[key]
        self._bump_version()
        self._persist()
        return True

    # Lookup operations
    def lookup(self, raw_text: str) -> NormalizationRecord | None:
        """Resolve a raw mention to its canonical mapping."""
        key = self._normalize_key(raw_text)
        return self._state.records.get(key)

    def normalize_key(self, raw_text: str) -> str:
        """Return the normalized key used for internal storage/lookup."""
        return self._normalize_key(raw_text)

    def lookup_many(self, raw_texts: Sequence[str]) -> Dict[str, NormalizationRecord | None]:
        """Resolve a batch of mentions."""
        return {text: self.lookup(text) for text in raw_texts}

    # Export/import
    def export_json(self, path: str | Path | None = None) -> Path:
        """Export normalization table to JSON."""
        target = Path(path) if path else self.table_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self._state.model_dump(mode="json")
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        logger.info(
            "Exported normalization table ({} records) to {}",
            len(self._state.records),
            target,
        )
        return target

    def export_csv(self, path: str | Path) -> Path:
        """Export normalization table to CSV for manual editing."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "normalized_form",
            "canonical_id",
            "canonical_name",
            "entity_type",
            "method",
            "confidence",
            "status",
            "version",
            "created_at",
            "updated_at",
            "raw_variants",
            "notes",
            "source",
        ]

        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in self._state.records.values():
                writer.writerow(
                    {
                        "normalized_form": record.normalized_form,
                        "canonical_id": record.canonical_id,
                        "canonical_name": record.canonical_name,
                        "entity_type": record.entity_type or "",
                        "method": record.method.value,
                        "confidence": f"{record.confidence:.3f}",
                        "status": record.status,
                        "version": record.version,
                        "created_at": record.created_at.isoformat(),
                        "updated_at": record.updated_at.isoformat(),
                        "raw_variants": " | ".join(record.raw_variants),
                        "notes": record.notes,
                        "source": record.source or "",
                    }
                )

        logger.info("Exported normalization table to CSV at {}", target)
        return target

    def import_json(self, path: str | Path) -> None:
        """Import table contents from JSON, replacing current state."""
        imported_state = self._load_state(Path(path))
        imported_state.version += 1
        imported_state.updated_at = datetime.now(UTC)
        self._state = imported_state
        self._persist()
        logger.info(
            "Imported normalization table ({} records) from {}",
            len(imported_state.records),
            path,
        )

    def import_csv(self, path: str | Path) -> None:
        """Import table contents from CSV, replacing current state."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Normalization CSV not found: {path}")

        records: MutableMapping[str, NormalizationRecord] = {}
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                normalized_form = row.get("normalized_form", "").strip()
                if not normalized_form:
                    continue

                created_value = row.get("created_at") or datetime.now(UTC).isoformat()
                updated_value = row.get("updated_at") or datetime.now(UTC).isoformat()

                record = NormalizationRecord(
                    normalized_form=normalized_form,
                    canonical_id=row.get("canonical_id", "").strip(),
                    canonical_name=row.get("canonical_name", "").strip(),
                    entity_type=(row.get("entity_type") or "").strip() or None,
                    method=NormalizationMethod(row.get("method", NormalizationMethod.MANUAL)),
                    confidence=float(row.get("confidence") or 1.0),
                    status=(row.get("status") or "pending"),
                    version=int(row.get("version") or 1),
                    created_at=self._parse_datetime(created_value),
                    updated_at=self._parse_datetime(updated_value),
                    raw_variants=self._split_variants(row.get("raw_variants", "")),
                    notes=row.get("notes", ""),
                    source=(row.get("source") or "").strip() or None,
                )
                records[normalized_form] = record

        self._state = NormalizationTableState(
            version=self._state.version + 1,
            updated_at=datetime.now(UTC),
            records=dict(records),
        )
        self._persist()
        logger.info("Imported normalization table from CSV at {}", path)

    def restore_record(self, record: NormalizationRecord) -> NormalizationRecord:
        """Restore a previous record version (used for undo)."""
        key = record.normalized_form
        self._state.records[key] = record
        self._bump_version()
        self._persist()
        return record

    # Helpers
    def all_records(self) -> List[NormalizationRecord]:
        """Return all records sorted by canonical name."""
        return sorted(self._state.records.values(), key=lambda r: r.canonical_name.lower())

    def _apply_upsert(self, entry: NormalizationEntry) -> NormalizationRecord:
        raw_text = entry.raw_text.strip()
        if not raw_text:
            raise ValueError("raw_text cannot be empty")

        key = self._normalize_key(raw_text)
        now = datetime.now(UTC)
        existing = self._state.records.get(key)
        variants = set(existing.raw_variants if existing else [])
        variants.add(raw_text)

        record = NormalizationRecord(
            raw_variants=sorted(variants),
            normalized_form=key,
            canonical_id=entry.canonical_id,
            canonical_name=entry.canonical_name,
            entity_type=entry.entity_type,
            method=entry.method,
            confidence=entry.confidence,
            status=entry.status,
            version=(existing.version + 1) if existing else 1,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            notes=entry.notes or (existing.notes if existing else ""),
            source=entry.source or (existing.source if existing else None),
        )

        self._state.records[key] = record
        return record

    def _normalize_key(self, raw_text: str) -> str:
        result = self._normalizer.normalize(raw_text)
        if not result.normalized:
            raise ValueError("Normalization produced empty key")
        return result.normalized

    def _split_variants(self, value: str) -> List[str]:
        parts = [part.strip() for part in value.split("|") if part.strip()]
        return sorted(set(parts))

    def _bump_version(self) -> None:
        self._state.version += 1
        self._state.updated_at = datetime.now(UTC)

    def _persist(self) -> None:
        self.table_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._state.model_dump(mode="json")
        self.table_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _load_state(self, path: Path) -> NormalizationTableState:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return NormalizationTableState.model_validate(payload)

    def _parse_datetime(self, value: str) -> datetime:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed
