"""Unit tests for undo behavior on relationship candidate actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.curation.entity_approval import EntityCurationService
from src.normalization.normalization_table import (
    NormalizationEntry,
    NormalizationMethod,
    NormalizationTable,
)
from src.storage.schemas import CandidateStatus, RelationshipCandidate
from src.utils.config import Config


class _FakeManager:
    def __init__(self) -> None:
        self.relationship_candidate_rows: List[Dict[str, Any]] = []
        self.relationship_status_updates: List[Tuple[str, CandidateStatus]] = []
        self.relationship_upserts: List[Dict[str, Any]] = []
        self.deleted_relationships: List[str] = []
        self.entities: Dict[str, Dict[str, Any]] = {}

    def get_entity(self, entity_id: str) -> Dict[str, Any] | None:
        return self.entities.get(entity_id)

    def get_relationship(self, relationship_id: str) -> Dict[str, Any] | None:
        return None

    def get_relationship_candidates_involving_keys(
        self, keys: List[str], *, statuses: List[str] | None = None, limit: int = 200
    ) -> List[Dict[str, Any]]:
        return list(self.relationship_candidate_rows)

    def update_relationship_candidate_status(
        self, identifier: str, status: CandidateStatus
    ) -> bool:
        self.relationship_status_updates.append((identifier, status))
        return True

    def upsert_relationship(self, relationship) -> str:  # type: ignore[override]
        payload = relationship.model_dump()
        self.relationship_upserts.append(payload)
        return payload["id"]

    def delete_relationship(self, relationship_id: str) -> bool:
        self.deleted_relationships.append(relationship_id)
        return True


def test_approve_relationship_candidate_can_be_undone(tmp_path: Path) -> None:
    manager = _FakeManager()
    table = NormalizationTable(table_path=tmp_path / "norm.json")
    service = EntityCurationService(
        manager=manager,
        normalization_table=table,
        config=Config(),
        undo_stack_path=tmp_path / "undo.json",
    )

    table.upsert(
        NormalizationEntry(
            raw_text="A",
            canonical_id="ent-a",
            canonical_name="A",
            entity_type="SYSTEM",
            method=NormalizationMethod.MANUAL,
            confidence=1.0,
            status="approved",
        )
    )
    table.upsert(
        NormalizationEntry(
            raw_text="B",
            canonical_id="ent-b",
            canonical_name="B",
            entity_type="SYSTEM",
            method=NormalizationMethod.MANUAL,
            confidence=1.0,
            status="approved",
        )
    )
    manager.entities["ent-a"] = {"id": "ent-a"}
    manager.entities["ent-b"] = {"id": "ent-b"}

    manager.relationship_candidate_rows = [
        {
            "id": "rc-1",
            "candidate_key": "a:RELATED_TO:b",
            "source": "A",
            "target": "B",
            "type": "RELATED_TO",
            "description": "",
            "confidence_score": 0.9,
            "status": "pending",
            "mention_count": 1,
            "source_documents": [],
            "chunk_ids": [],
            "provenance_events": [],
        }
    ]

    candidate = RelationshipCandidate.model_validate(manager.relationship_candidate_rows[0])
    service.approve_relationship_candidate(candidate)

    assert manager.relationship_upserts, "expected a promoted Relationship upsert"
    created_relationship_id = manager.relationship_upserts[0]["id"]

    assert service.undo_last_operation() is True
    assert manager.deleted_relationships == [created_relationship_id]
    assert manager.relationship_status_updates[-1] == ("rc-1", CandidateStatus.PENDING)


def test_reject_relationship_candidate_can_be_undone(tmp_path: Path) -> None:
    manager = _FakeManager()
    table = NormalizationTable(table_path=tmp_path / "norm.json")
    service = EntityCurationService(
        manager=manager,
        normalization_table=table,
        config=Config(),
        undo_stack_path=tmp_path / "undo.json",
    )

    candidate = RelationshipCandidate(
        id="rc-2",
        candidate_key="a:RELATED_TO:b",
        source="A",
        target="B",
        type="RELATED_TO",
        status=CandidateStatus.PENDING,
    )
    service.reject_relationship_candidate(candidate, reason="noise")
    assert manager.relationship_status_updates[-1] == ("rc-2", CandidateStatus.REJECTED)

    assert service.undo_last_operation() is True
    assert manager.relationship_status_updates[-1] == ("rc-2", CandidateStatus.PENDING)
