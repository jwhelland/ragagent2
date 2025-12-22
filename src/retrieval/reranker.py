"""Reranking system for improving retrieval results (Task 4.5).

This module implements reranking logic to improve the ordering of retrieved chunks.
It combines multiple signals including:
- Vector similarity score
- Graph relevance score
- Entity coverage score
- Confidence score
- Diversity score

It also supports MMR-like diversity ranking to ensure results cover different aspects.
"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger

from src.retrieval.models import HybridChunk
from src.utils.config import Config, RerankingConfig


class Reranker:
    """Reranker for optimizing retrieval results."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize reranker.

        Args:
            config: Configuration object
        """
        self.config = config or Config.from_yaml()
        self.reranking_config: RerankingConfig = self.config.retrieval.reranking

        logger.info(
            "Initialized Reranker",
            enabled=self.reranking_config.enabled,
            weights=self.reranking_config.weights,
        )

    def rerank(self, chunks: List[HybridChunk], top_k: Optional[int] = None) -> List[HybridChunk]:
        """Rerank a list of chunks using multi-signal scoring.

        Args:
            chunks: List of HybridChunk objects to rerank
            top_k: Number of results to return (defaults to config limit)

        Returns:
            Reranked and sorted list of chunks
        """
        if not chunks:
            return []

        # Use config limit if top_k not specified
        top_k = top_k or self.reranking_config.max_results

        # Apply score fusion
        reranked_chunks = self._apply_score_fusion(chunks)

        # Apply diversity ranking if enabled and weight > 0
        if self.reranking_config.enabled and self.reranking_config.weights.get("diversity", 0.0) > 0:
            reranked_chunks = self._apply_diversity_ranking(reranked_chunks)

        # Sort by final score descending
        reranked_chunks.sort(key=lambda c: c.final_score, reverse=True)

        # Limit to top_k and assign ranks
        final_chunks = reranked_chunks[:top_k]
        for rank, chunk in enumerate(final_chunks, 1):
            chunk.rank = rank

        return final_chunks

    def _apply_score_fusion(self, chunks: List[HybridChunk]) -> List[HybridChunk]:
        """Apply weighted score fusion to chunks.

        Args:
            chunks: List of HybridChunk objects

        Returns:
            Chunks with computed final_score
        """
        if not self.reranking_config.enabled:
            # Simple average of available scores if reranking disabled
            for chunk in chunks:
                scores = []
                if chunk.vector_score is not None:
                    scores.append(chunk.vector_score)
                if chunk.graph_score is not None:
                    scores.append(chunk.graph_score)
                chunk.final_score = sum(scores) / len(scores) if scores else 0.0
            return chunks

        # Weighted fusion using config weights
        weights = self.reranking_config.weights

        for chunk in chunks:
            score = 0.0
            total_weight = 0.0

            # Vector similarity score
            if chunk.vector_score is not None:
                weight = weights.get("vector_similarity", 0.4)
                score += chunk.vector_score * weight
                total_weight += weight

            # Graph relevance score
            if chunk.graph_score is not None:
                weight = weights.get("graph_relevance", 0.3)
                score += chunk.graph_score * weight
                total_weight += weight

            # Entity coverage score
            weight = weights.get("entity_coverage", 0.15)
            score += chunk.entity_coverage_score * weight
            total_weight += weight

            # Confidence score
            weight = weights.get("confidence", 0.10)
            score += chunk.confidence_score * weight
            total_weight += weight

            # Normalize by total weight
            chunk.final_score = score / total_weight if total_weight > 0 else 0.0

        return chunks

    def _apply_diversity_ranking(self, chunks: List[HybridChunk]) -> List[HybridChunk]:
        """Apply diversity-aware reranking using MMR-like approach.

        Args:
            chunks: List of HybridChunk objects (sorted by score)

        Returns:
            Reranked chunks with diversity scores
        """
        if len(chunks) <= 1:
            return chunks

        diversity_weight = self.reranking_config.weights.get("diversity", 0.05)

        # Sort initially by score to get best candidates first
        chunks.sort(key=lambda c: c.final_score, reverse=True)

        selected: List[HybridChunk] = []
        remaining = chunks.copy()

        # Select first chunk (highest score)
        selected.append(remaining.pop(0))
        selected[0].diversity_score = 1.0

        # Iteratively select remaining chunks
        while remaining:
            best_idx = 0
            best_score = float("-inf")

            for idx, candidate in enumerate(remaining):
                # Calculate diversity (dissimilarity to selected chunks)
                max_similarity = 0.0
                for selected_chunk in selected:
                    similarity = self._content_similarity(candidate.content, selected_chunk.content)
                    max_similarity = max(max_similarity, similarity)

                # Diversity score (1 - max similarity)
                diversity = 1.0 - max_similarity
                candidate.diversity_score = diversity

                # Combined score with diversity bonus
                # We blend the original score with the diversity score
                # Note: This changes the final_score to include diversity
                combined = (
                    candidate.final_score * (1 - diversity_weight) + diversity * diversity_weight
                )

                if combined > best_score:
                    best_score = combined
                    best_idx = idx

            # Move best candidate to selected
            # Update final score to reflect the diversity-adjusted score
            best_candidate = remaining.pop(best_idx)
            best_candidate.final_score = best_score
            selected.append(best_candidate)

        return selected

    def _content_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple content similarity using Jaccard on words.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0

        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0
