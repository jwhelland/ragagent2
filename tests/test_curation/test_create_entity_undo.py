"""Unit tests for create_entity and its undo behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.curation.entity_approval import EntityCurationService
from src.normalization.normalization_table import (
    NormalizationEntry,
    NormalizationMethod,
    NormalizationTable,
)
from src.storage.schemas import CandidateStatus, EntityType, RelationshipCandidate
from src.utils.config import Config


class _FakeManager:
    def __init__(self) -> None:
        self.entity_upserts: List[Dict[str, Any]] = []
        self.deleted_entities: List[str] = []
        self.relationship_candidate_rows: List[Dict[str, Any]] = []
        self.relationship_status_updates: List[Tuple[str, CandidateStatus]] = []
        self.relationship_upserts: List[Dict[str, Any]] = []
        self.deleted_relationships: List[str] = []

    def upsert_entity(self, entity) -> str:  # type: ignore[override]
        self.entity_upserts.append(entity.model_dump())
        return entity.id

    def delete_entity(self, entity_id: str) -> bool:
        self.deleted_entities.append(entity_id)
        return True

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
        self.relationship_upserts.append(relationship.model_dump())
        return relationship.id

    def delete_relationship(self, relationship_id: str) -> bool:
        self.deleted_relationships.append(relationship_id)
        return True
    
    def get_relationship(self, relationship_id: str) -> Dict[str, Any] | None:
        return None


def test_create_entity_and_undo(tmp_path: Path) -> None:
    manager = _FakeManager()
    table = NormalizationTable(table_path=tmp_path / "norm.json")
    service = EntityCurationService(
        manager=manager,
        normalization_table=table,
        config=Config(),
        undo_stack_path=tmp_path / "undo.json",
    )

    # 1. Create Entity
    entity_id = service.create_entity(
        name="New System",
        entity_type=EntityType.SYSTEM,
        description="A manually created system",
    )

    assert manager.entity_upserts, "Entity should be upserted"
    assert manager.entity_upserts[0]["canonical_name"] == "new_system"
    assert manager.entity_upserts[0]["entity_type"] == "SYSTEM"
    
    # Check normalization
    record = table.lookup("New System")
    assert record is not None
    assert record.canonical_id == entity_id
    assert record.status == "approved"

    # 2. Undo
    assert service.undo_last_operation() is True

    # Verify entity deletion
    assert manager.deleted_entities == [entity_id]
    
    # Verify normalization rollback
    record = table.lookup("New System")
    assert record is None


def test_create_entity_promotes_relationships_and_undo(tmp_path: Path) -> None:
    manager = _FakeManager()
    table = NormalizationTable(table_path=tmp_path / "norm.json")
    service = EntityCurationService(
        manager=manager,
        normalization_table=table,
        config=Config(),
        undo_stack_path=tmp_path / "undo.json",
    )

    # Pre-existing approved entity
    table.upsert(
        NormalizationEntry(
            raw_text="Other System",
            canonical_id="ent-other",
            canonical_name="Other System",
            entity_type="SYSTEM",
            method=NormalizationMethod.MANUAL,
            confidence=1.0,
            status="approved",
        )
    )

    # Pending relationship
    manager.relationship_candidate_rows = [
        {
            "id": "relcand-1",
            "candidate_key": "new_system:DEPENDS_ON:other_system",
            "source": "New System",
            "target": "Other System",
            "type": "DEPENDS_ON",
            "description": "",
            "confidence_score": 0.8,
            "status": "pending",
            "mention_count": 1,
            "source_documents": [],
            "chunk_ids": [],
            "provenance_events": [],
        }
    ]

    # 1. Create Entity (should trigger promotion)
    entity_id = service.create_entity(
        name="New System",
        entity_type=EntityType.SYSTEM,
    )

    # Check relationship promotion
    assert manager.relationship_upserts, "Relationship should be promoted"
    assert manager.relationship_status_updates[-1] == ("relcand-1", CandidateStatus.APPROVED)
    
    created_rel_id = manager.relationship_upserts[0]["id"]

    # 2. Undo
    assert service.undo_last_operation() is True

    # Verify relationship deletion and status revert
    assert manager.deleted_relationships == [created_rel_id]
    
    # The undo logic calls update_relationship_candidate_status to restore PREVIOUS status
    # Check if we have an update setting it back to pending
    # The last update should be to PENDING
    assert manager.relationship_status_updates[-1] == ("relcand-1", CandidateStatus.PENDING)
