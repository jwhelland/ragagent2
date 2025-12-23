# Developer Guide - Graph RAG System

This guide provides deep-dive architectural information for developers working on the Graph RAG system.

## Data Flow

### 1. Ingestion Pipeline
`ParsedDocument` → `TextCleaner` → `HierarchicalChunker` → `EmbeddingGenerator` → `Neo4jManager` & `QdrantManager`

### 2. Extraction Pipeline
`Chunk` → `SpacyExtractor` & `LLMExtractor` & `PatternRelationshipExtractor` → `EntityMerger` → `EntityCandidate` & `RelationshipCandidate`

### 3. Retrieval Pipeline
`Query` → `QueryParser` → `HybridRetriever` (Vector + Graph) → `Reranker` → `ResponseGenerator` → `Answer`

## Core Components

### Neo4j Schema
The system uses several labels and relationship types:
- **Nodes**: `Document`, `Chunk`, `Entity` (with subtypes like `SYSTEM`, `COMPONENT`), `EntityCandidate`, `RelationshipCandidate`.
- **Relationships**: `PARENT_CHUNK`, `MENTIONED_IN`, and semantic types like `PART_OF`, `CONTROLS`, `DEPENDS_ON`.

### Storage Managers
- `Neo4jManager`: Handles all Cypher queries, constraints, and data promotion from candidates to entities.
- `QdrantManager`: Manages vector collections, payloads, and similarity searches.

### Custom Extraction Rules
Rules-based extraction is used to supplement LLMs:
- **spaCy patterns**: Defined in `config/entity_patterns.jsonl`.
- **Regex patterns**: Defined in `config/relationship_patterns.yaml`.
- **Dependency parsing**: Hardcoded logic in `src/extraction/dependency_extractor.py` for SVO triples.

## Extending the System

### Adding a New Extractor
1. Create a new class in `src/extraction/`.
2. Ensure it returns `ExtractedEntity` or `ExtractedRelationship` from `src.extraction.models`.
3. Integrate it into `IngestionPipeline` in `src/pipeline/ingestion_pipeline.py`.

### Implementing a New Retrieval Strategy
1. Add the strategy to `RetrievalStrategy` enum in `src/retrieval/models.py`.
2. Implement the retrieval logic in `HybridRetriever._retrieve_<name>`.
3. Update `HybridRetriever._select_strategy` if it should be auto-selected.

## Database Migrations
We currently don't use a formal migration tool. Schema changes should be implemented in `scripts/setup_databases.py` in an idempotent way.

## Troubleshooting Tools

### Debug Logs
Change `logging.level` to `DEBUG` in `config/config.yaml` to see detailed execution traces.

### Database Web UIs
- **Neo4j Browser**: http://localhost:7474 (Cypher exploration)
- **Qdrant Dashboard**: http://localhost:6333/dashboard (Vector inspection)

### Parsing Inspection
Check `data/processed/` for intermediate JSON outputs of the PDF parser to debug extraction issues.
