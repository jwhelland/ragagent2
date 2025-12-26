from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from src.curation import review_interface
from src.curation.entity_approval import NeighborhoodIssue
from src.storage.schemas import CandidateStatus, EntityCandidate, EntityType, RelationshipCandidate

runner = CliRunner()


@pytest.fixture
def mock_dependencies():
    with (
        patch("src.curation.review_interface.create_candidate_store") as mock_store_factory,
        patch("src.curation.review_interface._create_curation_service") as mock_service_factory,
        patch("src.curation.review_interface.get_neighborhood_issues") as mock_get_issues,
    ):

        mock_store = MagicMock()
        mock_store_factory.return_value = mock_store

        mock_service = MagicMock()
        mock_manager = MagicMock()
        mock_service_factory.return_value = (mock_service, MagicMock(), mock_manager)

        yield mock_store, mock_service, mock_get_issues


def test_approve_recursive_flow(mock_dependencies):
    mock_store, mock_service, mock_get_issues = mock_dependencies

    # Setup Candidates
    main_candidate = EntityCandidate(
        candidate_key="main",
        canonical_name="Main",
        candidate_type=EntityType.SYSTEM,
        status=CandidateStatus.PENDING,
    )
    peer_candidate = EntityCandidate(
        candidate_key="peer_a",
        canonical_name="Peer A",
        candidate_type=EntityType.COMPONENT,
        status=CandidateStatus.PENDING,
    )

    mock_store.get_candidate.side_effect = lambda k: {
        "main": main_candidate,
        "peer_a": peer_candidate,
    }.get(k)

    # Setup Issues
    # 1. Resolvable (Peer A)
    # 2. Missing (Peer B)
    rc1 = RelationshipCandidate(
        candidate_key="rel1",
        source="Main",
        target="Peer A",
        type="CONNECTS",
        status=CandidateStatus.PENDING,
    )
    rc2 = RelationshipCandidate(
        candidate_key="rel2",
        source="Main",
        target="Peer B",
        type="HAS_PART",
        status=CandidateStatus.PENDING,
    )

    # First call for "Main" returns [Issue 1, Issue 2]
    # Second call for "Peer A" returns []
    # Third call for "Peer B" returns []
    mock_get_issues.side_effect = [
        [
            NeighborhoodIssue(
                relationship_candidate=rc1,
                peer_name="Peer A",
                issue_type="resolvable",
                peer_candidate_key="peer_a",
            ),
            NeighborhoodIssue(relationship_candidate=rc2, peer_name="Peer B", issue_type="missing"),
        ],
        [],  # For Peer A
        [],  # For Peer B
    ]

    # Run Command
    # Inputs:
    # 1. "y" to approve Peer A
    # 2. "y" to create Peer B
    # 3. "CONCEPT" for Peer B type
    result = runner.invoke(
        review_interface.app,
        ["approve", "main", "--config", "config/config.yaml"],
        input="y\ny\nCONCEPT\n",
    )

    # Verifications
    assert result.exit_code == 0

    # 1. Main Entity Approved
    mock_service.approve_candidate.assert_any_call(main_candidate)

    # 2. Peer A Approved (Recursive)
    mock_service.approve_candidate.assert_any_call(peer_candidate)

    # 3. Peer B Created and Approved
    mock_service.create_entity.assert_any_call(
        "Peer B",
        EntityType.CONCEPT,
        description="Created during neighborhood approval",
        source_documents=rc2.source_documents,
        chunk_ids=rc2.chunk_ids,
    )

    # Verify Output
    assert "Approved Main" in result.stdout
    assert "Found 2 pending relationship(s)" in result.stdout
    assert "Approve pending candidate 'Peer A'?" in result.stdout
    assert "Create new entity for 'Peer B'?" in result.stdout
