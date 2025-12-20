"""Batch curation operations (Task 3.8)."""

from __future__ import annotations

from typing import Callable, List, Sequence

from loguru import logger
from pydantic import BaseModel, Field

from src.curation.entity_approval import EntityCurationService
from src.storage.schemas import CandidateStatus, EntityCandidate
from src.utils.config import CurationConfig


class BatchOperationPreview(BaseModel):
    """Preview of a batch operation."""

    to_approve: List[str] = Field(default_factory=list)
    skipped: List[str] = Field(default_factory=list)
    threshold: float
    total_candidates: int


class BatchOperationResult(BaseModel):
    """Result of a batch operation."""

    approved_entities: List[str] = Field(default_factory=list)
    merged_entities: List[str] = Field(default_factory=list)
    skipped: List[str] = Field(default_factory=list)
    failed: List[str] = Field(default_factory=list)
    preview_only: bool = False
    rolled_back: bool = False


class BatchCurationService:
    """Execute batch curation actions with preview/rollback support."""

    def __init__(
        self,
        curation_service: EntityCurationService,
        config: CurationConfig | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.curation_service = curation_service
        self.config = config or CurationConfig()
        self.progress_callback = progress_callback

    def preview_batch_approve(
        self, candidates: Sequence[EntityCandidate], *, threshold: float | None = None
    ) -> BatchOperationPreview:
        threshold = threshold or self.config.auto_approve_threshold
        to_approve: List[str] = []
        skipped: List[str] = []

        for candidate in candidates:
            if candidate.status != CandidateStatus.PENDING:
                skipped.append(candidate.candidate_key)
                continue
            if candidate.confidence_score >= threshold:
                to_approve.append(candidate.candidate_key)
            else:
                skipped.append(candidate.candidate_key)

        return BatchOperationPreview(
            to_approve=to_approve,
            skipped=skipped,
            threshold=threshold,
            total_candidates=len(candidates),
        )

    def batch_approve(
        self,
        candidates: Sequence[EntityCandidate],
        *,
        threshold: float | None = None,
        dry_run: bool = False,
    ) -> BatchOperationResult:
        preview = self.preview_batch_approve(candidates, threshold=threshold)
        if dry_run:
            logger.info("Dry-run: would approve {} candidates", len(preview.to_approve))
            return BatchOperationResult(
                approved_entities=[],
                skipped=preview.skipped,
                preview_only=True,
            )

        checkpoint = self.curation_service.undo_checkpoint()
        approved_entities: List[str] = []
        try:
            for idx, candidate in enumerate(candidates):
                self._tick(f"Approving {idx + 1}/{len(candidates)}")
                if candidate.candidate_key not in preview.to_approve:
                    continue
                entity_id = self.curation_service.approve_candidate(candidate)
                approved_entities.append(entity_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Batch approve failed, rolling back: {}", exc)
            self.curation_service.rollback_to_checkpoint(checkpoint)
            return BatchOperationResult(
                approved_entities=[],
                skipped=preview.skipped,
                failed=[str(exc)],
                rolled_back=True,
            )

        return BatchOperationResult(
            approved_entities=approved_entities,
            skipped=preview.skipped,
        )

    def batch_merge_clusters(
        self,
        clusters: Sequence[Sequence[EntityCandidate]],
        *,
        dry_run: bool = False,
    ) -> BatchOperationResult:
        if dry_run:
            merged = [cluster[0].candidate_key for cluster in clusters if cluster]
            logger.info("Dry-run: would merge {} clusters", len(merged))
            return BatchOperationResult(merged_entities=[], preview_only=True)

        checkpoint = self.curation_service.undo_checkpoint()
        merged_entities: List[str] = []
        failed: List[str] = []

        try:
            for idx, cluster in enumerate(clusters):
                if not cluster:
                    continue
                self._tick(f"Merging cluster {idx + 1}/{len(clusters)}")
                primary, *duplicates = cluster
                entity_id = self.curation_service.merge_candidates(primary, duplicates)
                merged_entities.append(entity_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Batch merge failed, rolling back: {}", exc)
            self.curation_service.rollback_to_checkpoint(checkpoint)
            failed.append(str(exc))
            return BatchOperationResult(
                merged_entities=[],
                failed=failed,
                rolled_back=True,
            )

        return BatchOperationResult(merged_entities=merged_entities, failed=failed)

    def _tick(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)
        else:
            logger.debug(message)
