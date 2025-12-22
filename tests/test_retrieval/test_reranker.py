"""Tests for reranker (Task 4.5)."""

import pytest

from src.retrieval.models import HybridChunk
from src.retrieval.reranker import Reranker
from src.utils.config import Config


@pytest.fixture
def config() -> Config:
    """Create test configuration."""
    return Config.from_yaml()


@pytest.fixture
def reranker(config: Config) -> Reranker:
    """Create reranker with mock configuration."""
    return Reranker(config=config)


class TestRerankerInitialization:
    """Tests for Reranker initialization."""

    def test_init_with_config(self, config: Config) -> None:
        """Test initialization with configuration."""
        reranker = Reranker(config=config)
        assert reranker.config is config
        assert reranker.reranking_config is config.retrieval.reranking

    def test_init_with_defaults(self) -> None:
        """Test initialization with default configuration."""
        reranker = Reranker()
        assert reranker.config is not None
        assert reranker.reranking_config is not None


class TestScoreFusion:
    """Tests for score fusion and reranking."""

    def test_apply_score_fusion_with_both_scores(self, reranker: Reranker) -> None:
        """Test score fusion with both vector and graph scores."""
        chunks = [
            HybridChunk(
                chunk_id="chunk_1",
                document_id="doc_001",
                content="Test content",
                level=3,
                vector_score=0.9,
                graph_score=0.7,
                entity_coverage_score=0.8,
                confidence_score=0.85,
                diversity_score=0.0,
                final_score=0.0,  # Will be computed
                rank=1,
                source="hybrid",
            )
        ]

        fused = reranker._apply_score_fusion(chunks)

        assert len(fused) == 1
        assert fused[0].final_score > 0.0
        assert fused[0].final_score <= 1.0

    def test_apply_score_fusion_vector_only(self, reranker: Reranker) -> None:
        """Test score fusion with only vector score."""
        chunks = [
            HybridChunk(
                chunk_id="chunk_1",
                document_id="doc_001",
                content="Test content",
                level=3,
                vector_score=0.9,
                graph_score=None,
                entity_coverage_score=0.0,
                confidence_score=0.5,
                diversity_score=0.0,
                final_score=0.0,
                rank=1,
                source="vector",
            )
        ]

        fused = reranker._apply_score_fusion(chunks)

        assert len(fused) == 1
        assert fused[0].final_score > 0.0

    def test_apply_score_fusion_graph_only(self, reranker: Reranker) -> None:
        """Test score fusion with only graph score."""
        chunks = [
            HybridChunk(
                chunk_id="chunk_1",
                document_id="doc_001",
                content="Test content",
                level=3,
                vector_score=None,
                graph_score=0.8,
                entity_coverage_score=0.6,
                confidence_score=0.7,
                diversity_score=0.0,
                final_score=0.0,
                rank=1,
                source="graph",
            )
        ]

        fused = reranker._apply_score_fusion(chunks)

        assert len(fused) == 1
        assert fused[0].final_score > 0.0


class TestDiversityRanking:
    """Tests for diversity-aware ranking."""

    def test_apply_diversity_ranking(self, reranker: Reranker) -> None:
        """Test diversity ranking with similar content."""
        chunks = [
            HybridChunk(
                chunk_id=f"chunk_{i}",
                document_id="doc_001",
                content=(
                    "power system thermal control management"
                    if i < 2
                    else "battery voltage monitoring"
                ),
                level=3,
                vector_score=0.9 - (i * 0.05),
                graph_score=0.8,
                entity_coverage_score=0.7,
                confidence_score=0.8,
                diversity_score=0.0,
                final_score=0.9 - (i * 0.05),
                rank=i + 1,
                source="hybrid",
            )
            for i in range(4)
        ]

        diverse = reranker._apply_diversity_ranking(chunks)

        assert len(diverse) == 4
        # Check that diversity scores were computed
        assert all(chunk.diversity_score >= 0.0 for chunk in diverse)

    def test_content_similarity(self, reranker: Reranker) -> None:
        """Test content similarity calculation."""
        text1 = "power system thermal control"
        text2 = "power system management control"
        text3 = "battery voltage monitoring"

        sim_12 = reranker._content_similarity(text1, text2)
        sim_13 = reranker._content_similarity(text1, text3)

        # text1 and text2 should be more similar than text1 and text3
        assert sim_12 > sim_13
        assert 0.0 <= sim_12 <= 1.0
        assert 0.0 <= sim_13 <= 1.0


class TestRerankMethod:
    """Tests for the main rerank method."""

    def test_rerank_full_flow(self, reranker: Reranker) -> None:
        """Test the full reranking flow."""
        chunks = [
            HybridChunk(
                chunk_id=f"chunk_{i}",
                document_id="doc_001",
                content=f"Content {i}",
                level=3,
                vector_score=0.5 + (i * 0.1),
                graph_score=0.5 + (i * 0.1),
                entity_coverage_score=0.5,
                confidence_score=0.8,
                diversity_score=0.0,
                final_score=0.0,
                rank=i + 1,
                source="hybrid",
            )
            for i in range(5)
        ]

        # Rerank and limit to top 3
        top_k = 3
        reranked = reranker.rerank(chunks, top_k=top_k)

        assert len(reranked) == top_k
        # Should be sorted by final_score descending
        assert reranked[0].final_score >= reranked[1].final_score
        assert reranked[1].final_score >= reranked[2].final_score
        # Ranks should be assigned correctly
        assert [c.rank for c in reranked] == [1, 2, 3]

    def test_rerank_empty_list(self, reranker: Reranker) -> None:
        """Test reranking with empty list."""
        assert reranker.rerank([]) == []
