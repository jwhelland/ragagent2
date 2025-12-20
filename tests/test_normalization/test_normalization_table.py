"""Tests for NormalizationTable (Phase 3 Task 3.5)."""

from __future__ import annotations

import json
from pathlib import Path

from src.normalization.normalization_table import (
    NormalizationEntry,
    NormalizationMethod,
    NormalizationTable,
)


def test_upsert_and_lookup_round_trip(tmp_path: Path) -> None:
    table_path = tmp_path / "normalization.json"
    table = NormalizationTable(table_path=table_path)

    entry = NormalizationEntry(
        raw_text="Power Subsystem",
        canonical_id="SYS-1",
        canonical_name="Power Subsystem",
        method=NormalizationMethod.EXACT,
        confidence=0.93,
        status="approved",
    )

    table.upsert(entry)
    lookup = table.lookup("power subsystem")

    assert lookup is not None
    assert lookup.canonical_id == "SYS-1"
    assert "Power Subsystem" in lookup.raw_variants
    assert lookup.status == "approved"


def test_bulk_upsert_increments_version_once(tmp_path: Path) -> None:
    table_path = tmp_path / "normalization.json"
    table = NormalizationTable(table_path=table_path)

    entries = [
        NormalizationEntry(
            raw_text="Electric Power Subsystem",
            canonical_id="SYS-1",
            canonical_name="Electric Power Subsystem",
            method=NormalizationMethod.FUZZY,
            confidence=0.88,
            status="pending",
        ),
        NormalizationEntry(
            raw_text="Attitude Control System",
            canonical_id="SYS-2",
            canonical_name="Attitude Control System",
            method=NormalizationMethod.EMBEDDING,
            confidence=0.9,
            status="pending",
        ),
    ]

    table.bulk_upsert(entries)
    export_payload = json.loads(table.export_json().read_text())

    assert export_payload["version"] == 2  # initial version 1 + bulk update
    assert len(export_payload["records"]) == 2


def test_import_export_csv_round_trip(tmp_path: Path) -> None:
    table_path = tmp_path / "normalization.json"
    csv_path = tmp_path / "normalization.csv"

    table = NormalizationTable(table_path=table_path)
    table.upsert(
        NormalizationEntry(
            raw_text="EPS",
            canonical_id="SYS-1",
            canonical_name="Electric Power Subsystem",
            method=NormalizationMethod.ACRONYM,
            confidence=0.97,
            status="pending",
        )
    )

    table.export_csv(csv_path)

    restored = NormalizationTable(table_path=tmp_path / "restored.json")
    restored.import_csv(csv_path)

    record = restored.lookup("eps")
    assert record is not None
    assert record.method == NormalizationMethod.ACRONYM
    assert "EPS" in record.raw_variants
