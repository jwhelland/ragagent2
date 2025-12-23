# Phase 2.1: Non-LLM Relationship Extraction

## Overview
Currently, relationship extraction in `ragagent2` is exclusively handled by LLMs. When LLM extraction is disabled, the knowledge graph loses its semantic connectivity. This phase implements high-precision, low-cost alternatives using rule-based and syntactic methods to ensure a baseline graph is always generated.

## Comparative Analysis of Extraction Methods

| Method | Complexity | Cost | Precision | Recall | Best For... |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dependency Parsing** | Low | Low | High | Low | Explicitly structured tech specs. |
| **Hearst Patterns** | Low | Low | Very High | Very Low | Defining hierarchies ("is a", "part of"). |
| **Small Models (REBEL)**| High (Setup) | Medium (Compute)| Medium | High | General purpose "AI" extraction without API costs. |
| **Co-occurrence** | Very Low | Zero | Low | High | Ensuring graph connectivity (baseline). |

---

## Tasks

### 2.1.1: Regex-based Extraction (Hearst Patterns)
**Priority:** High
**Description:** Implement extraction based on specific linguistic patterns commonly used in technical documentation to define hierarchies and compositions.

**Subtasks:**
1. Create `config/relationship_patterns.yaml` to define regex/token patterns:
   - *Hierarchy*: "X such as Y", "X including Y", "Y is a type of X" -> `IS_A`
   - *Composition*: "X comprises Y", "X consists of Y", "X contains Y" -> `PART_OF` / `CONTAINS`
2. Implement a `PatternRelationshipExtractor` that scans text chunks for these patterns.
3. Support evidence capture by storing the exact sentence containing the match.

### 2.1.2: Syntactic Dependency Extraction
**Priority:** High
**Description:** Use spaCy's dependency parser to identify Subject-Verb-Object (SVO) triples where both Subject and Object are recognized entities.

**Subtasks:**
1. Define a list of "Technical Action Verbs" (e.g., "controls", "monitors", "powers", "triggers", "connects to").
2. Implement a `DependencyRelationshipExtractor` using spaCy's `DependencyMatcher`.
3. Create grammar patterns:
   - `[Entity A] (nsubj) <- [Verb] -> (dobj/pobj) -> [Entity B]`
4. Map verbs to relationship types (e.g., "powers" -> `PROVIDES_POWER_TO`).

### 2.1.3: Pipeline Integration
**Priority:** Critical
**Description:** Integrate new extractors into the `IngestionPipeline` as standard components.

**Subtasks:**
1. Update `IngestionPipeline` to initialize non-LLM extractors during `initialize_components`.
2. Modify the ingestion flow to run these extractors regardless of the `enable_llm` setting.
3. Update `RelationshipCandidate` schema to track provenance correctly:
   - `source_extractor`: "spacy_dependency" or "regex_patterns"
4. Ensure the curation TUI displays these non-LLM candidates for review.

### 2.1.4: Statistical Co-occurrence (Baseline/Fallback)
**Priority:** Medium
**Description:** Provide a fallback relationship for highly correlated entities within the same context.

**Subtasks:**
1. Implement logic to create `RELATED_TO` candidates for entities appearing in the same sentence or paragraph.
2. Assign a lower heuristic confidence (e.g., 0.4) to distinguish them from semantic extractions.

---

## Success Criteria
- [ ] Semantic relationships are created in Neo4j even when LLM extraction is disabled.
- [ ] System hierarchies (Part-Of) are automatically extracted from definitions.
- [ ] Functional relationships (Controls/Monitors) are extracted from technical descriptions.
- [ ] Minimal performance impact on ingestion (<100ms per chunk).
- [ ] Provenance clearly identifies which rule or pattern triggered the extraction.
