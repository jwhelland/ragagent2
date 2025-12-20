"""Corpus-wide discovery pipeline (Task 3.9).

This pipeline analyzes the extracted EntityCandidate corpus and produces a discovery report:
- frequency stats by type/status
- chunk-level co-occurrence analysis (co-mentions within the same chunk)
- lightweight clustering over the co-occurrence graph
- merge suggestions (fuzzy by default; optional semantic via embeddings)
- suggestions for new entity types (based on unknown conflicting type labels)

Outputs are written as JSON + Markdown, with an optional GraphViz DOT co-occurrence visualization.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.normalization.entity_deduplicator import EntityDeduplicator, EntityRecord
from src.normalization.fuzzy_matcher import FuzzyMatcher
from src.normalization.string_normalizer import StringNormalizer
from src.storage.neo4j_manager import Neo4jManager
from src.utils.config import Config, DatabaseConfig, NormalizationConfig


class DiscoveryCandidate(BaseModel):
    """EntityCandidate payload used by discovery analysis."""

    model_config = ConfigDict(extra="ignore")

    candidate_key: str
    canonical_name: str
    candidate_type: str
    status: str = "pending"
    description: str = ""
    aliases: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    mention_count: int = 0
    source_documents: List[str] = Field(default_factory=list)
    chunk_ids: List[str] = Field(default_factory=list)
    conflicting_types: List[str] = Field(default_factory=list)

    @property
    def chunks_seen(self) -> int:
        return len(set(self.chunk_ids))


class CooccurrenceEdge(BaseModel):
    """Co-mention edge between two candidates."""

    model_config = ConfigDict(frozen=True)

    left_key: str
    right_key: str
    count: int
    pmi: float


class CooccurrenceCluster(BaseModel):
    """Connected component over the co-occurrence graph."""

    model_config = ConfigDict(frozen=True)

    cluster_id: int
    entity_keys: List[str]
    size: int
    cohesion: float


class EntityTypeSuggestion(BaseModel):
    """Suggested new entity type labels found in candidate metadata."""

    model_config = ConfigDict(frozen=True)

    label: str
    occurrences: int
    example_candidates: List[str] = Field(default_factory=list)


class DiscoveryMergeSuggestion(BaseModel):
    """Proposed merge suggestion for review."""

    model_config = ConfigDict(frozen=True)

    method: str  # "fuzzy" | "semantic"
    source_key: str
    target_key: str
    entity_type: str
    score: float
    confidence: float
    reason: str


class DiscoveryReport(BaseModel):
    """Serialized discovery report."""

    model_config = ConfigDict(extra="ignore")

    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    parameters: Dict[str, Any] = Field(default_factory=dict)
    totals: Dict[str, int] = Field(default_factory=dict)
    by_type: List[Dict[str, Any]] = Field(default_factory=list)
    by_status: List[Dict[str, Any]] = Field(default_factory=list)
    top_entities: List[Dict[str, Any]] = Field(default_factory=list)
    cooccurrence_edges: List[CooccurrenceEdge] = Field(default_factory=list)
    cooccurrence_clusters: List[CooccurrenceCluster] = Field(default_factory=list)
    merge_suggestions: List[DiscoveryMergeSuggestion] = Field(default_factory=list)
    entity_type_suggestions: List[EntityTypeSuggestion] = Field(default_factory=list)
    artifacts: Dict[str, str] = Field(default_factory=dict)

    def to_markdown(self, *, candidate_names: Mapping[str, str] | None = None) -> str:
        candidate_names = candidate_names or {}
        lines: list[str] = []
        lines.append("# Entity Discovery Report")
        lines.append("")
        lines.append(f"Generated at: `{self.generated_at.isoformat()}`")
        if self.parameters:
            lines.append("")
            lines.append("## Parameters")
            lines.append("")
            for key in sorted(self.parameters):
                lines.append(f"- `{key}`: `{self.parameters[key]}`")
        lines.append("")
        lines.append("## Corpus Stats")
        lines.append("")
        for key in sorted(self.totals):
            lines.append(f"- **{key}**: {self.totals[key]}")
        if self.by_type:
            lines.append("")
            lines.append("### By Type")
            lines.append("")
            for row in self.by_type[:25]:
                lines.append(f"- `{row.get('candidate_type')}`: {row.get('count')}")
        if self.by_status:
            lines.append("")
            lines.append("### By Status")
            lines.append("")
            for row in self.by_status:
                lines.append(f"- `{row.get('status')}`: {row.get('count')}")
        if self.top_entities:
            lines.append("")
            lines.append("## Top Entities")
            lines.append("")
            for row in self.top_entities[:25]:
                key = row.get("candidate_key", "")
                name = row.get("canonical_name") or candidate_names.get(key) or key
                lines.append(
                    f"- `{name}` ({row.get('candidate_type')}): mentions={row.get('mention_count')}, "
                    f"chunks={row.get('chunks_seen')}, conf={row.get('confidence_score'):.2f}"
                )
        if self.cooccurrence_edges:
            lines.append("")
            lines.append("## Co-occurrence (Top Pairs)")
            lines.append("")
            for edge in self.cooccurrence_edges[:25]:
                left = candidate_names.get(edge.left_key, edge.left_key)
                right = candidate_names.get(edge.right_key, edge.right_key)
                lines.append(f"- `{left}` ↔ `{right}`: count={edge.count}, pmi={edge.pmi:.2f}")
        if self.cooccurrence_clusters:
            lines.append("")
            lines.append("## Co-occurrence Clusters")
            lines.append("")
            for cluster in self.cooccurrence_clusters[:15]:
                members = [candidate_names.get(key, key) for key in cluster.entity_keys[:10]]
                suffix = "..." if len(cluster.entity_keys) > 10 else ""
                lines.append(
                    f"- Cluster {cluster.cluster_id} (size={cluster.size}, cohesion={cluster.cohesion:.2f}): "
                    + ", ".join(f"`{m}`" for m in members)
                    + suffix
                )
        if self.merge_suggestions:
            lines.append("")
            lines.append("## Merge Suggestions")
            lines.append("")
            for suggestion in self.merge_suggestions[:50]:
                left = candidate_names.get(suggestion.source_key, suggestion.source_key)
                right = candidate_names.get(suggestion.target_key, suggestion.target_key)
                lines.append(
                    f"- [{suggestion.method}] `{left}` → `{right}` ({suggestion.entity_type}): "
                    f"score={suggestion.score:.2f}, conf={suggestion.confidence:.2f} ({suggestion.reason})"
                )
        if self.entity_type_suggestions:
            lines.append("")
            lines.append("## Suggested New Entity Types")
            lines.append("")
            for suggestion in self.entity_type_suggestions[:25]:
                examples = ", ".join(f"`{c}`" for c in suggestion.example_candidates[:5])
                lines.append(f"- `{suggestion.label}`: {suggestion.occurrences} (e.g. {examples})")
        if self.artifacts:
            lines.append("")
            lines.append("## Artifacts")
            lines.append("")
            for key in sorted(self.artifacts):
                lines.append(f"- `{key}`: `{self.artifacts[key]}`")
        lines.append("")
        return "\n".join(lines)


@dataclass(frozen=True)
class DiscoveryParameters:
    min_confidence: float = 0.0
    statuses: Tuple[str, ...] = ("pending", "approved", "rejected")
    candidate_types: Tuple[str, ...] = ()
    max_candidates: int = 2000
    max_entities_per_chunk: int = 50
    min_cooccurrence: int = 2
    max_edges: int = 500
    max_clusters: int = 50
    enable_semantic_merge: bool = False
    max_merge_suggestions: int = 100
    fuzzy_block_prefix: int = 4


def _safe_log2(value: float) -> float:
    if value <= 0:
        return float("-inf")
    return math.log(value, 2)


def compute_entity_type_suggestions(
    candidates: Sequence[DiscoveryCandidate],
    *,
    known_types: Iterable[str],
    max_examples: int = 5,
    top_k: int = 25,
) -> List[EntityTypeSuggestion]:
    known = {t.upper() for t in known_types}
    counts: Counter[str] = Counter()
    examples: DefaultDict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        for label in candidate.conflicting_types:
            if not label:
                continue
            normalized = str(label).strip()
            if not normalized:
                continue
            upper = normalized.upper()
            if upper in known:
                continue
            counts[upper] += 1
            if len(examples[upper]) < max_examples:
                examples[upper].append(candidate.canonical_name)

    suggestions = [
        EntityTypeSuggestion(label=label, occurrences=count, example_candidates=examples[label])
        for label, count in counts.most_common(top_k)
    ]
    return suggestions


def build_chunk_index(
    candidates: Sequence[DiscoveryCandidate], *, max_entities_per_chunk: int
) -> Dict[str, List[str]]:
    chunk_to_keys: DefaultDict[str, list[str]] = defaultdict(list)
    for candidate in candidates:
        if not candidate.chunk_ids:
            continue
        for chunk_id in set(candidate.chunk_ids):
            chunk_to_keys[chunk_id].append(candidate.candidate_key)

    if max_entities_per_chunk <= 0:
        return {chunk_id: list(keys) for chunk_id, keys in chunk_to_keys.items()}

    trimmed: Dict[str, List[str]] = {}
    for chunk_id, keys in chunk_to_keys.items():
        unique = list(dict.fromkeys(keys))
        if len(unique) > max_entities_per_chunk:
            unique = unique[:max_entities_per_chunk]
        trimmed[chunk_id] = unique

    return trimmed


def compute_cooccurrence_edges(
    candidates: Sequence[DiscoveryCandidate],
    *,
    min_cooccurrence: int,
    max_edges: int,
    max_entities_per_chunk: int,
) -> Tuple[List[CooccurrenceEdge], Dict[str, int], int]:
    chunk_index = build_chunk_index(candidates, max_entities_per_chunk=max_entities_per_chunk)
    total_chunks = len(chunk_index)
    if total_chunks == 0:
        return [], {}, 0

    chunk_freq: Dict[str, int] = {}
    for candidate in candidates:
        if not candidate.chunk_ids:
            continue
        chunk_freq[candidate.candidate_key] = len(set(candidate.chunk_ids))

    pair_counts: Counter[Tuple[str, str]] = Counter()
    for keys in chunk_index.values():
        if len(keys) < 2:
            continue
        sorted_keys = sorted(set(keys))
        for i in range(len(sorted_keys)):
            for j in range(i + 1, len(sorted_keys)):
                pair_counts[(sorted_keys[i], sorted_keys[j])] += 1

    edges: list[CooccurrenceEdge] = []
    for (left, right), count in pair_counts.most_common():
        if count < min_cooccurrence:
            break
        freq_left = chunk_freq.get(left, 0)
        freq_right = chunk_freq.get(right, 0)
        if freq_left == 0 or freq_right == 0:
            continue
        p_xy = count / total_chunks
        p_x = freq_left / total_chunks
        p_y = freq_right / total_chunks
        pmi = _safe_log2(p_xy / (p_x * p_y)) if p_x > 0 and p_y > 0 else float("-inf")
        edges.append(CooccurrenceEdge(left_key=left, right_key=right, count=count, pmi=pmi))
        if len(edges) >= max_edges:
            break

    return edges, chunk_freq, total_chunks


def cluster_cooccurrence_graph(
    edges: Sequence[CooccurrenceEdge],
    *,
    min_edge_count: int,
    max_clusters: int,
) -> List[CooccurrenceCluster]:
    if not edges:
        return []

    parent: Dict[str, str] = {}
    rank: Dict[str, int] = {}
    adjacency: DefaultDict[str, list[CooccurrenceEdge]] = defaultdict(list)

    def find(item: str) -> str:
        if item not in parent:
            parent[item] = item
            rank[item] = 0
            return item
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        left_rank = rank.get(left_root, 0)
        right_rank = rank.get(right_root, 0)
        if left_rank < right_rank:
            parent[left_root] = right_root
        elif left_rank > right_rank:
            parent[right_root] = left_root
        else:
            parent[right_root] = left_root
            rank[left_root] = left_rank + 1

    for edge in edges:
        if edge.count < min_edge_count:
            continue
        union(edge.left_key, edge.right_key)
        adjacency[edge.left_key].append(edge)
        adjacency[edge.right_key].append(edge)

    components: DefaultDict[str, set[str]] = defaultdict(set)
    for node in parent:
        components[find(node)].add(node)

    clusters: list[CooccurrenceCluster] = []
    for cluster_id, nodes in enumerate(
        sorted(components.values(), key=lambda group: len(group), reverse=True)
    ):
        if len(nodes) < 2:
            continue
        cohesion = _cluster_cohesion(nodes, adjacency)
        clusters.append(
            CooccurrenceCluster(
                cluster_id=cluster_id,
                entity_keys=sorted(nodes),
                size=len(nodes),
                cohesion=cohesion,
            )
        )
        if len(clusters) >= max_clusters:
            break

    return clusters


def _cluster_cohesion(
    nodes: Set[str], adjacency: Mapping[str, Sequence[CooccurrenceEdge]]
) -> float:
    weights: list[float] = []
    for node in nodes:
        for edge in adjacency.get(node, []):
            other = edge.right_key if edge.left_key == node else edge.left_key
            if other in nodes:
                weights.append(float(edge.count))
    if not weights:
        return 0.0
    return sum(weights) / len(weights)


def generate_fuzzy_merge_suggestions(
    candidates: Sequence[DiscoveryCandidate],
    *,
    config: NormalizationConfig | None = None,
    max_suggestions: int,
    block_prefix: int,
) -> List[DiscoveryMergeSuggestion]:
    if len(candidates) < 2:
        return []

    matcher = FuzzyMatcher(config=config)
    normalizer: StringNormalizer = matcher.normalizer

    by_block: DefaultDict[Tuple[str, str], list[Tuple[DiscoveryCandidate, str]]] = defaultdict(list)
    for candidate in candidates:
        normalized = normalizer.normalize(candidate.canonical_name).normalized
        prefix = normalized[: max(1, block_prefix)] if normalized else ""
        block = (candidate.candidate_type, prefix)
        by_block[block].append((candidate, normalized))

    proposals: list[DiscoveryMergeSuggestion] = []
    seen: set[Tuple[str, str, str]] = set()
    for (entity_type, _), items in by_block.items():
        if len(items) < 2:
            continue
        for i in range(len(items)):
            left, _ = items[i]
            for j in range(i + 1, len(items)):
                right, _ = items[j]
                if left.candidate_key == right.candidate_key:
                    continue

                match = matcher.match_pair(
                    left.canonical_name, right.canonical_name, entity_type=entity_type
                )
                if not match.passed:
                    continue
                source_key, target_key = sorted([left.candidate_key, right.candidate_key])
                dedup_key = (source_key, target_key, entity_type)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                proposals.append(
                    DiscoveryMergeSuggestion(
                        method="fuzzy",
                        source_key=source_key,
                        target_key=target_key,
                        entity_type=entity_type,
                        score=float(match.score),
                        confidence=float(match.confidence),
                        reason=f"threshold={match.threshold:.2f}",
                    )
                )

    proposals.sort(key=lambda item: (item.confidence, item.score), reverse=True)
    return proposals[:max_suggestions]


def generate_semantic_merge_suggestions(
    candidates: Sequence[DiscoveryCandidate],
    *,
    config: NormalizationConfig | None = None,
    database_config: DatabaseConfig | None = None,
    max_suggestions: int,
) -> List[DiscoveryMergeSuggestion]:
    if len(candidates) < 2:
        return []

    deduplicator = EntityDeduplicator(
        config=config,
        database_config=database_config or DatabaseConfig(),
    )
    records = [
        EntityRecord(
            entity_id=c.candidate_key,
            name=c.canonical_name,
            entity_type=c.candidate_type,
            description=c.description,
            aliases=c.aliases,
            mention_count=max(1, int(c.mention_count)),
        )
        for c in candidates
    ]

    try:
        result = deduplicator.deduplicate(records)
    except Exception as exc:  # pragma: no cover - depends on embedding runtime availability
        logger.warning("Semantic merge suggestions failed: {}", exc)
        return []

    suggestions: list[DiscoveryMergeSuggestion] = []
    for suggestion in result.merge_suggestions:
        suggestions.append(
            DiscoveryMergeSuggestion(
                method="semantic",
                source_key=suggestion.source_id,
                target_key=suggestion.target_id,
                entity_type=suggestion.entity_type,
                score=float(suggestion.similarity),
                confidence=float(suggestion.confidence),
                reason=suggestion.reason,
            )
        )

    suggestions.sort(key=lambda item: (item.confidence, item.score), reverse=True)
    return suggestions[:max_suggestions]


def write_graphviz_dot(
    edges: Sequence[CooccurrenceEdge],
    *,
    candidate_names: Mapping[str, str],
    output_path: Path,
    max_nodes: int = 75,
) -> Path:
    counts: Counter[str] = Counter()
    for edge in edges:
        counts[edge.left_key] += edge.count
        counts[edge.right_key] += edge.count
    keep = {key for key, _ in counts.most_common(max_nodes)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["graph cooccurrence {"]
    lines.append("  graph [overlap=false, splines=true];")
    for key in sorted(keep):
        label = candidate_names.get(key, key).replace('"', '\\"')
        lines.append(f'  "{key}" [label="{label}"];')

    for edge in edges:
        if edge.left_key not in keep or edge.right_key not in keep:
            continue
        penwidth = max(1.0, min(8.0, edge.count / 2))
        lines.append(
            f'  "{edge.left_key}" -- "{edge.right_key}" '
            f'[label="{edge.count}", penwidth={penwidth:.2f}];'
        )
    lines.append("}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


class DiscoveryPipeline:
    """Analyze EntityCandidate corpus and write discovery artifacts."""

    def __init__(
        self,
        config: Config,
        *,
        neo4j_manager: Neo4jManager | None = None,
    ) -> None:
        self.config = config
        self.neo4j_manager = neo4j_manager or Neo4jManager(self.config.database)

    def run(
        self,
        *,
        output_dir: str | Path,
        parameters: DiscoveryParameters | None = None,
        create_visualization: bool = True,
    ) -> DiscoveryReport:
        params = parameters or DiscoveryParameters()

        if not getattr(self.neo4j_manager, "_connected", False):
            self.neo4j_manager.connect()
        candidates = self._load_candidates(params)
        candidate_names = {c.candidate_key: c.canonical_name for c in candidates}

        stats = self._compute_stats(candidates)
        edges, _chunk_freq, total_chunks = compute_cooccurrence_edges(
            candidates,
            min_cooccurrence=params.min_cooccurrence,
            max_edges=params.max_edges,
            max_entities_per_chunk=params.max_entities_per_chunk,
        )
        clusters = cluster_cooccurrence_graph(
            edges, min_edge_count=params.min_cooccurrence, max_clusters=params.max_clusters
        )

        type_suggestions = compute_entity_type_suggestions(
            candidates,
            known_types=self.config.extraction.entity_types,
        )

        merge_suggestions = generate_fuzzy_merge_suggestions(
            candidates,
            config=self.config.normalization,
            max_suggestions=params.max_merge_suggestions,
            block_prefix=params.fuzzy_block_prefix,
        )
        if params.enable_semantic_merge:
            merge_suggestions = [
                *merge_suggestions,
                *generate_semantic_merge_suggestions(
                    candidates,
                    config=self.config.normalization,
                    database_config=self.config.database,
                    max_suggestions=params.max_merge_suggestions,
                ),
            ]

        report = DiscoveryReport(
            parameters={
                "min_confidence": params.min_confidence,
                "statuses": list(params.statuses),
                "candidate_types": list(params.candidate_types),
                "max_candidates": params.max_candidates,
                "max_entities_per_chunk": params.max_entities_per_chunk,
                "min_cooccurrence": params.min_cooccurrence,
                "max_edges": params.max_edges,
                "max_clusters": params.max_clusters,
                "enable_semantic_merge": params.enable_semantic_merge,
                "max_merge_suggestions": params.max_merge_suggestions,
                "total_chunks_indexed": total_chunks,
            },
            totals=stats["totals"],
            by_type=stats["by_type"],
            by_status=stats["by_status"],
            top_entities=stats["top_entities"],
            cooccurrence_edges=edges,
            cooccurrence_clusters=clusters,
            merge_suggestions=merge_suggestions,
            entity_type_suggestions=type_suggestions,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "discovery_report.json"
        markdown_path = output_dir / "discovery_report.md"

        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        markdown_path.write_text(
            report.to_markdown(candidate_names=candidate_names), encoding="utf-8"
        )
        report.artifacts = {
            "report_json": str(json_path),
            "report_markdown": str(markdown_path),
        }

        if create_visualization and edges:
            dot_path = write_graphviz_dot(
                edges,
                candidate_names=candidate_names,
                output_path=output_dir / "cooccurrence.dot",
            )
            report.artifacts["cooccurrence_dot"] = str(dot_path)

        (output_dir / "candidate_name_map.json").write_text(
            json.dumps(candidate_names, indent=2, sort_keys=True), encoding="utf-8"
        )
        report.artifacts["candidate_name_map"] = str(output_dir / "candidate_name_map.json")

        logger.info(
            "Discovery report written to {} ({} candidates, {} edges, {} clusters)",
            output_dir,
            len(candidates),
            len(edges),
            len(clusters),
        )
        return report

    def _load_candidates(self, params: DiscoveryParameters) -> List[DiscoveryCandidate]:
        status_set = {s.lower() for s in params.statuses}
        type_set = {t.upper() for t in params.candidate_types} if params.candidate_types else None

        rows: list[dict[str, Any]] = []
        offset = 0
        batch_size = 500
        while len(rows) < params.max_candidates:
            limit = min(batch_size, params.max_candidates - len(rows))
            batch = self.neo4j_manager.get_entity_candidates(
                status=None,
                candidate_types=None,
                min_confidence=params.min_confidence,
                limit=limit,
                offset=offset,
            )
            if not batch:
                break
            offset += len(batch)
            for row in batch:
                status = str(row.get("status", "pending")).lower()
                candidate_type = str(row.get("candidate_type", "")).upper()
                if status not in status_set:
                    continue
                if type_set is not None and candidate_type not in type_set:
                    continue
                rows.append(row)
                if len(rows) >= params.max_candidates:
                    break

        candidates = [DiscoveryCandidate(**row) for row in rows]
        logger.info("Loaded {} candidates for discovery analysis", len(candidates))
        return candidates

    def _compute_stats(self, candidates: Sequence[DiscoveryCandidate]) -> Dict[str, Any]:
        totals = {
            "candidates": len(candidates),
            "unique_chunks": len({chunk for c in candidates for chunk in set(c.chunk_ids)}),
            "unique_documents": len({doc for c in candidates for doc in set(c.source_documents)}),
        }

        by_type_counts: Counter[str] = Counter()
        by_status_counts: Counter[str] = Counter()
        for candidate in candidates:
            by_type_counts[str(candidate.candidate_type).upper()] += 1
            by_status_counts[str(candidate.status).lower()] += 1

        by_type = [
            {"candidate_type": entity_type, "count": count}
            for entity_type, count in by_type_counts.most_common()
        ]
        by_status = [
            {"status": status, "count": count} for status, count in by_status_counts.items()
        ]

        top_entities = sorted(
            [
                {
                    "candidate_key": c.candidate_key,
                    "canonical_name": c.canonical_name,
                    "candidate_type": c.candidate_type,
                    "mention_count": int(c.mention_count),
                    "chunks_seen": int(c.chunks_seen),
                    "confidence_score": float(c.confidence_score),
                }
                for c in candidates
            ],
            key=lambda row: (row["mention_count"], row["confidence_score"]),
            reverse=True,
        )[:50]

        return {
            "totals": totals,
            "by_type": by_type,
            "by_status": by_status,
            "top_entities": top_entities,
        }
