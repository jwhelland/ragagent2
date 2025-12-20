"""Unit tests for the discovery pipeline core analysis (Task 3.9)."""

from __future__ import annotations

from src.pipeline.discovery_pipeline import (
    DiscoveryCandidate,
    cluster_cooccurrence_graph,
    compute_cooccurrence_edges,
    compute_entity_type_suggestions,
    generate_fuzzy_merge_suggestions,
)
from src.utils.config import NormalizationConfig


def test_cooccurrence_edges_count_and_pmi() -> None:
    candidates = [
        DiscoveryCandidate(
            candidate_key="A",
            canonical_name="Alpha",
            candidate_type="SYSTEM",
            chunk_ids=["c1", "c2"],
        ),
        DiscoveryCandidate(
            candidate_key="B",
            canonical_name="Beta",
            candidate_type="SYSTEM",
            chunk_ids=["c1"],
        ),
        DiscoveryCandidate(
            candidate_key="C",
            canonical_name="Gamma",
            candidate_type="SYSTEM",
            chunk_ids=["c2"],
        ),
    ]

    edges, chunk_freq, total_chunks = compute_cooccurrence_edges(
        candidates,
        min_cooccurrence=1,
        max_edges=10,
        max_entities_per_chunk=50,
    )

    assert total_chunks == 2
    assert chunk_freq["A"] == 2
    assert chunk_freq["B"] == 1
    assert chunk_freq["C"] == 1

    by_pair = {(edge.left_key, edge.right_key): edge for edge in edges}
    assert by_pair[("A", "B")].count == 1
    assert by_pair[("A", "C")].count == 1

    # PMI should be finite in this setup (non-zero probabilities).
    assert by_pair[("A", "B")].pmi > float("-inf")


def test_cooccurrence_clusters_connected_components() -> None:
    candidates = [
        DiscoveryCandidate(
            candidate_key="A",
            canonical_name="Alpha",
            candidate_type="SYSTEM",
            chunk_ids=["c1"],
        ),
        DiscoveryCandidate(
            candidate_key="B",
            canonical_name="Beta",
            candidate_type="SYSTEM",
            chunk_ids=["c1"],
        ),
        DiscoveryCandidate(
            candidate_key="C",
            canonical_name="Gamma",
            candidate_type="SYSTEM",
            chunk_ids=["c2"],
        ),
        DiscoveryCandidate(
            candidate_key="D",
            canonical_name="Delta",
            candidate_type="SYSTEM",
            chunk_ids=["c2"],
        ),
    ]

    edges, _, _ = compute_cooccurrence_edges(
        candidates,
        min_cooccurrence=1,
        max_edges=10,
        max_entities_per_chunk=50,
    )
    clusters = cluster_cooccurrence_graph(edges, min_edge_count=1, max_clusters=10)

    cluster_sets = [set(cluster.entity_keys) for cluster in clusters]
    assert {"A", "B"} in cluster_sets
    assert {"C", "D"} in cluster_sets


def test_entity_type_suggestions_ignores_known_types() -> None:
    candidates = [
        DiscoveryCandidate(
            candidate_key="A",
            canonical_name="Alpha",
            candidate_type="SYSTEM",
            conflicting_types=["SYSTEM", "NEW_KIND"],
        ),
        DiscoveryCandidate(
            candidate_key="B",
            canonical_name="Beta",
            candidate_type="SYSTEM",
            conflicting_types=["new_kind", "OTHER_KIND"],
        ),
    ]

    suggestions = compute_entity_type_suggestions(
        candidates,
        known_types=["SYSTEM", "COMPONENT"],
        top_k=10,
    )

    labels = {s.label for s in suggestions}
    assert "NEW_KIND" in labels
    assert "OTHER_KIND" in labels
    assert "SYSTEM" not in labels


def test_fuzzy_merge_suggestions_returns_high_confidence_pairs() -> None:
    config = NormalizationConfig(fuzzy_threshold=0.80)
    candidates = [
        DiscoveryCandidate(
            candidate_key="A",
            canonical_name="Attitude Control System",
            candidate_type="SYSTEM",
        ),
        DiscoveryCandidate(
            candidate_key="B",
            canonical_name="Attitude Control Subsystem",
            candidate_type="SYSTEM",
        ),
        DiscoveryCandidate(
            candidate_key="C",
            canonical_name="Thermal Control System",
            candidate_type="SYSTEM",
        ),
    ]

    suggestions = generate_fuzzy_merge_suggestions(
        candidates,
        config=config,
        max_suggestions=10,
        block_prefix=3,
    )

    assert any(
        suggestion.source_key in {"A", "B"} and suggestion.target_key in {"A", "B"}
        for suggestion in suggestions
    )
    assert all(suggestion.method == "fuzzy" for suggestion in suggestions)
