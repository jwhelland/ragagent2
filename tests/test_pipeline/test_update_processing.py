"""Unit tests for the update pipeline processing (Task 5.2)."""

from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.ingestion_pipeline import IngestionResult
from src.pipeline.update_pipeline import ChangeReport, FileState, UpdatePipeline


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.database = MagicMock()
    return config


@pytest.fixture
def update_pipeline(mock_config):
    with (
        patch("src.pipeline.update_pipeline.Neo4jManager"),
        patch("src.pipeline.update_pipeline.QdrantManager"),
        patch("src.pipeline.update_pipeline.IngestionPipeline"),
    ):
        pipeline = UpdatePipeline(mock_config)
        return pipeline


def test_process_report_new_files(update_pipeline):
    report = ChangeReport(new_files=[FileState(path="new.pdf", checksum="abc", status="new")])

    update_pipeline.ingestion_pipeline.process_document.return_value = IngestionResult(
        document_id="doc1", success=True
    )

    stats = update_pipeline.process_report(report)

    assert stats["new"] == 1
    update_pipeline.ingestion_pipeline.process_document.assert_called_once_with("new.pdf")


def test_process_report_modified_files(update_pipeline):
    report = ChangeReport(
        modified_files=[
            FileState(path="mod.pdf", checksum="def", status="modified", document_id="doc1")
        ]
    )

    update_pipeline.ingestion_pipeline.process_document.return_value = IngestionResult(
        document_id="doc1", success=True
    )

    stats = update_pipeline.process_report(report)

    assert stats["modified"] == 1
    # Should call with force_reingest=True
    update_pipeline.ingestion_pipeline.process_document.assert_called_once_with(
        "mod.pdf", force_reingest=True
    )


def test_process_report_deleted_files(update_pipeline):
    report = ChangeReport(
        deleted_files=[
            FileState(path="del.pdf", checksum="ghi", status="deleted", document_id="doc1")
        ]
    )

    stats = update_pipeline.process_report(report)

    assert stats["deleted"] == 1
    update_pipeline.qdrant_manager.delete_chunks_by_document.assert_called_once_with("doc1")
    update_pipeline.neo4j_manager.delete_document.assert_called_once_with("doc1")


def test_process_report_mixed(update_pipeline):
    report = ChangeReport(
        new_files=[FileState(path="new.pdf", checksum="abc", status="new")],
        modified_files=[
            FileState(path="mod.pdf", checksum="def", status="modified", document_id="doc2")
        ],
        deleted_files=[
            FileState(path="del.pdf", checksum="ghi", status="deleted", document_id="doc3")
        ],
    )

    update_pipeline.ingestion_pipeline.process_document.return_value = IngestionResult(
        document_id="any", success=True
    )

    stats = update_pipeline.process_report(report)

    assert stats["new"] == 1
    assert stats["modified"] == 1
    assert stats["deleted"] == 1
    assert stats["failed"] == 0
