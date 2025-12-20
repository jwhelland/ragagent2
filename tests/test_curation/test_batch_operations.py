"""Batch curation operation tests (Task 3.8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.curation.batch_operations import BatchCurationService
from src.curation.entity_approval import EntityCurationService
from src.normalization.normalization_table import NormalizationTable
from src.storage.schemas import CandidateStatus, EntityCandidate, EntityType
from src.utils.config import Config, CurationConfig


class _FakeManager:
    def __init__(self) -> None:
        self.status_updates: List[Tuple[str, CandidateStatus]] = []
        self.entity_upserts: List[Dict[str, Any]] = []
        self.deleted_entities: List[str] = []
        self.relationship_candidate_rows: List[Dict[str, Any]] = []

    def upsert_entity(self, entity) -> str:  # type: ignore[override]
        self.entity_upserts.append(entity.model_dump())
        return entity.id

    def delete_entity(self, entity_id: str) -> bool:
        self.deleted_entities.append(entity_id)
        return True

    def update_entity_candidate_status(self, identifier: str, status: CandidateStatus) -> bool:
        self.status_updates.append((identifier, status))
        return True

    def update_entity_candidate(self, identifier: str, properties: Dict[str, Any]) -> bool:
        return True

    def get_relationship_candidates_involving_keys(
        self, keys: List[str], *, status: str = "pending", limit: int = 200
    ) -> List[Dict[str, Any]]:
        return list(self.relationship_candidate_rows)

    def update_relationship_candidate_status(
        self, identifier: str, status: CandidateStatus
    ) -> bool:
        return True

    def upsert_relationship(self, relationship) -> str:  # type: ignore[override]
        return relationship.id


def _candidate(key: str, confidence: float) -> EntityCandidate:
    return EntityCandidate(
        id=None,
        candidate_key=key,
        canonical_name=key,
        candidate_type=EntityType.SYSTEM,
        confidence_score=confidence,
    )


def test_batch_approve_respects_threshold(tmp_path: Path) -> None:
    manager = _FakeManager()
    table = NormalizationTable(table_path=tmp_path / "norm.json")
    service = EntityCurationService(
        manager, table, Config(), undo_stack_path=tmp_path / "undo.json"
    )
    batch_service = BatchCurationService(service, CurationConfig(auto_approve_threshold=0.8))

    candidates = [
        _candidate("cand-high", 0.9),
        _candidate("cand-low", 0.5),
    ]

    preview = batch_service.preview_batch_approve(candidates)
    assert preview.to_approve == ["cand-high"]
    assert "cand-low" in preview.skipped

    result = batch_service.batch_approve(candidates, dry_run=False)
    assert len(result.approved_entities) == 1
    assert manager.status_updates[-1] == ("cand-high", CandidateStatus.APPROVED)


def test_batch_merge_clusters_handles_multiple_groups(tmp_path: Path) -> None:
    manager = _FakeManager()
    table = NormalizationTable(table_path=tmp_path / "norm.json")
    service = EntityCurationService(
        manager, table, Config(), undo_stack_path=tmp_path / "undo.json"
    )
    batch_service = BatchCurationService(service, CurationConfig())

    clusters = [
        [_candidate("primary-1", 0.9), _candidate("dup-1", 0.7)],
        [_candidate("primary-2", 0.95), _candidate("dup-2", 0.6)],
    ]

    result = batch_service.batch_merge_clusters(clusters)
    assert len(result.merged_entities) == 2
    assert manager.status_updates.count(("primary-1", CandidateStatus.APPROVED)) == 1
    assert manager.status_updates.count(("dup-1", CandidateStatus.REJECTED)) == 1
    assert manager.status_updates.count(("primary-2", CandidateStatus.APPROVED)) == 1
