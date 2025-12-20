"""Embedding-based entity deduplication utilities (Task 3.4)."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.utils.config import DatabaseConfig, NormalizationConfig
from src.utils.embeddings import EmbeddingGenerator


class EntityRecord(BaseModel):
    """Lightweight representation of an entity for deduplication."""

    model_config = ConfigDict(frozen=True)

    entity_id: str
    name: str
    entity_type: str = "UNKNOWN"
    description: str = ""
    aliases: List[str] = Field(default_factory=list)
    mention_count: int = 1

    def embedding_text(self, alias_limit: int = 3) -> str:
        """Concatenate fields into a single text for embedding."""
        pieces: list[str] = [self.name]
        if self.aliases:
            pieces.append(" / ".join(self.aliases[:alias_limit]))
        if self.description:
            pieces.append(self.description)
        return " ".join(piece for piece in pieces if piece)


class EntityCluster(BaseModel):
    """Cluster of semantically similar entities."""

    model_config = ConfigDict(frozen=True)

    cluster_id: int
    entity_ids: List[str]
    representative_id: str
    average_similarity: float


class MergeSuggestion(BaseModel):
    """Represents a proposed merge between similar entities."""

    model_config = ConfigDict(frozen=True)

    cluster_id: int
    source_id: str
    target_id: str
    entity_type: str
    similarity: float
    confidence: float
    auto_merge: bool
    reason: str


class DeduplicationResult(BaseModel):
    """Aggregate result of a deduplication run."""

    model_config = ConfigDict(frozen=True)

    clusters: List[EntityCluster]
    merge_suggestions: List[MergeSuggestion]


class _SimilarityContext(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    entities: Sequence[EntityRecord]
    similarity_matrix: np.ndarray
    clusters: list[list[int]]


class EntityDeduplicator:
    """Identify semantically similar entities using embeddings."""

    def __init__(
        self,
        config: NormalizationConfig | None = None,
        embedder: EmbeddingGenerator | None = None,
        database_config: DatabaseConfig | None = None,
    ) -> None:
        self.config = config or NormalizationConfig()
        self._embedder = embedder
        self._database_config = database_config or DatabaseConfig()

    def deduplicate(self, entities: Sequence[EntityRecord]) -> DeduplicationResult:
        """Cluster entities and produce ranked merge suggestions."""
        if len(entities) < 2:
            return DeduplicationResult(clusters=[], merge_suggestions=[])

        logger.info("Running entity deduplication for {} entities", len(entities))

        embeddings = self._embed_entities(entities)
        similarity_matrix = self._compute_similarity_matrix(embeddings)
        cluster_indices = self._build_clusters(similarity_matrix, entities)

        context = _SimilarityContext(
            entities=entities,
            similarity_matrix=similarity_matrix,
            clusters=cluster_indices,
        )

        clusters = self._to_cluster_models(context)
        suggestions = self._rank_merge_suggestions(context)

        return DeduplicationResult(clusters=clusters, merge_suggestions=suggestions)

    def _embed_entities(self, entities: Sequence[EntityRecord]) -> List[np.ndarray]:
        embedder = self._ensure_embedder()
        texts = [entity.embedding_text() for entity in entities]
        embeddings = embedder.generate(texts)
        if len(embeddings) != len(entities):
            raise ValueError(
                f"Embedding count mismatch: {len(embeddings)} for {len(entities)} entities"
            )
        return [self._normalize_vector(embedding) for embedding in embeddings]

    def _compute_similarity_matrix(self, embeddings: Sequence[np.ndarray]) -> np.ndarray:
        if not embeddings:
            return np.zeros((0, 0), dtype=np.float32)
        stacked = np.vstack(embeddings)
        similarities = np.clip(stacked @ stacked.T, -1.0, 1.0)
        np.fill_diagonal(similarities, 1.0)
        return similarities

    def _build_clusters(
        self, similarity_matrix: np.ndarray, entities: Sequence[EntityRecord]
    ) -> list[list[int]]:
        num_entities = similarity_matrix.shape[0]
        visited = set[int]()
        clusters: list[list[int]] = []
        threshold = self.config.embedding_similarity_threshold

        for idx in range(num_entities):
            if idx in visited:
                continue
            neighbors = self._neighbors(idx, similarity_matrix, threshold, entities=entities)
            if not neighbors:
                visited.add(idx)
                continue
            cluster: list[int] = []
            queue = [idx]
            visited.add(idx)

            while queue:
                current = queue.pop()
                cluster.append(current)
                for neighbor in self._neighbors(current, similarity_matrix, threshold, entities=entities):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(cluster) > 1:
                clusters.append(sorted(cluster))

        return clusters

    def _neighbors(
        self,
        index: int,
        similarity_matrix: np.ndarray,
        threshold: float,
        *,
        entities: Sequence[EntityRecord] | None,
    ) -> list[int]:
        row = similarity_matrix[index]
        result: list[int] = []
        for i, score in enumerate(row):
            if i == index or score < threshold:
                continue
            if entities and not self._types_compatible(entities[index], entities[int(i)]):
                continue
            result.append(int(i))
        return result

    def _to_cluster_models(self, context: _SimilarityContext) -> List[EntityCluster]:
        clusters: List[EntityCluster] = []
        for cluster_id, member_indices in enumerate(context.clusters):
            representative_idx = self._representative_index(member_indices, context.entities)
            avg_similarity = self._average_similarity(member_indices, context.similarity_matrix)
            clusters.append(
                EntityCluster(
                    cluster_id=cluster_id,
                    entity_ids=[context.entities[i].entity_id for i in member_indices],
                    representative_id=context.entities[representative_idx].entity_id,
                    average_similarity=avg_similarity,
                )
            )
        return clusters

    def _rank_merge_suggestions(self, context: _SimilarityContext) -> List[MergeSuggestion]:
        suggestions: List[MergeSuggestion] = []
        for cluster_id, member_indices in enumerate(context.clusters):
            representative_idx = self._representative_index(member_indices, context.entities)
            representative = context.entities[representative_idx]
            for other_idx in member_indices:
                if other_idx == representative_idx:
                    continue
                other = context.entities[other_idx]
                if not self._types_compatible(representative, other):
                    continue
                similarity = float(context.similarity_matrix[representative_idx, other_idx])
                if similarity < self.config.embedding_similarity_threshold:
                    continue

                type_alignment = self._type_alignment(representative.entity_type, other.entity_type)
                mention_support = self._mention_support(representative, other)
                confidence = self._confidence_score(similarity, type_alignment, mention_support)
                auto_merge = self._should_auto_merge(similarity, type_alignment)
                reason = (
                    f"similarity={similarity:.2f}; type={representative.entity_type} vs "
                    f"{other.entity_type}; mentions={representative.mention_count}/{other.mention_count}"
                )

                suggestions.append(
                    MergeSuggestion(
                        cluster_id=cluster_id,
                        source_id=representative.entity_id,
                        target_id=other.entity_id,
                        entity_type=representative.entity_type,
                        similarity=similarity,
                        confidence=confidence,
                        auto_merge=auto_merge,
                        reason=reason,
                    )
                )

        return sorted(suggestions, key=lambda item: item.confidence, reverse=True)

    def _representative_index(self, indices: list[int], entities: Sequence[EntityRecord]) -> int:
        ranked = sorted(
            indices,
            key=lambda idx: (
                entities[idx].mention_count,
                len(entities[idx].description),
                len(entities[idx].name),
            ),
            reverse=True,
        )
        return ranked[0]

    def _average_similarity(self, indices: list[int], similarity_matrix: np.ndarray) -> float:
        if len(indices) < 2:
            return 1.0
        pairs: list[float] = []
        for i, left in enumerate(indices):
            for right in indices[i + 1 :]:
                pairs.append(float(similarity_matrix[left, right]))
        return float(np.mean(pairs)) if pairs else 1.0

    def _type_alignment(self, left_type: str | None, right_type: str | None) -> float:
        left = (left_type or "").upper()
        right = (right_type or "").upper()
        if not left or not right or "UNKNOWN" in {left, right}:
            return 0.5
        if left == right:
            return 1.0
        return 0.2

    def _types_compatible(self, left: EntityRecord, right: EntityRecord) -> bool:
        left_type = (left.entity_type or "UNKNOWN").upper()
        right_type = (right.entity_type or "UNKNOWN").upper()
        return left_type == right_type or "UNKNOWN" in {left_type, right_type}

    def _mention_support(self, left: EntityRecord, right: EntityRecord) -> float:
        min_mentions = min(left.mention_count, right.mention_count)
        normalized = min_mentions / max(1, self.config.min_mention_count)
        return float(max(0.0, min(1.0, normalized)))

    def _confidence_score(
        self, similarity: float, type_alignment: float, mention_support: float
    ) -> float:
        confidence = (
            similarity * 0.7 + type_alignment * 0.2 + mention_support * 0.1
        )
        return float(max(0.0, min(1.0, confidence)))

    def _should_auto_merge(self, similarity: float, type_alignment: float) -> bool:
        return similarity >= self.config.auto_merge_threshold and type_alignment >= 0.9

    def _normalize_vector(self, embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return (embedding / norm).astype(np.float32)

    def _ensure_embedder(self) -> EmbeddingGenerator:
        if self._embedder is None:
            self._embedder = EmbeddingGenerator(self._database_config)
        return self._embedder
