# GEMINI.md - Context & Instructions

## Project Overview
**ragagent2** is a Graph RAG (Retrieval-Augmented Generation) system for technical documents. It combines vector similarity search (Qdrant) with knowledge graph traversal (Neo4j) to enable complex queries. It supports PDF ingestion, hierarchical chunking, dual entity extraction (spaCy + LLM), and manual entity curation.

## Technology Stack
- **Language:** Python 3.12+
- **Dependency Manager:** `uv`
- **Graph DB:** Neo4j 5.x
- **Vector DB:** Qdrant
- **Parsing:** Docling (OCR support)
- **NLP/Extraction:** spaCy, Anthropic/OpenAI LLMs
- **CLI:** Typer, Rich

## Environment Setup
1.  **Dependencies:** Managed via `uv`. Run `uv sync` to install.
2.  **Infrastructure:** `docker-compose up -d` starts Neo4j and Qdrant.
3.  **Configuration:**
    -   Copy `.env.example` to `.env` (configure API keys, DB creds).
    -   Main config: `config/config.yaml`.
    -   Cleaning/Prompts: `config/cleaning_patterns.yaml`, `config/extraction_prompts.yaml`.

## Key Commands
All commands should be run using `uv run`.

### Application CLI
-   **Setup:** `uv run ragagent-setup` (Init DBs). Add `--recreate-qdrant` to reset vectors.
-   **Ingest:** `uv run ragagent-ingest --directory data/raw` (Parse/embed PDFs).
-   **Discover:** `uv run ragagent-discover` (Find entity candidates).
-   **Review:** `uv run ragagent-review` (Interactive TUI for curation).
-   **Interactive Review (Alternative):** `uv run ragagent-review-interactive`

### Development
-   **Test:** `uv run pytest` (Runs all tests with coverage).
-   **Format:** `uv run black src/ tests/`
-   **Lint:** `uv run ruff check src/ tests/`
-   **Type Check:** `uv run mypy src/`

## Project Structure
-   **`src/`**: Core logic.
    -   `ingestion/`: PDF parsing, chunking, cleaning.
    -   `extraction/`: Entity/relationship extraction (spaCy, LLM).
    -   `normalization/`: Deduplication, fuzzy matching, acronyms.
    -   `storage/`: Neo4j and Qdrant managers.
    -   `retrieval/`: Query parsing, hybrid retrieval strategies.
    -   `curation/`: Review interface logic.
    -   `pipeline/`: Orchestration (ingestion, discovery pipelines).
    -   `utils/`: Config, logging, embeddings.
-   **`scripts/`**: CLI entry points (mapped in `pyproject.toml`).
-   **`config/`**: YAML/JSONL configuration files.
-   **`data/`**:
    -   `raw/`: Input PDFs.
    -   `processed/`: Intermediate parsed files.
    -   `entities/`: Extracted candidate entities.
    -   `normalization/`: Approved entity tables.
-   **`tests/`**: Pytest suite (mirrors `src/` structure).
-   **`plans/`**: Architecture documentation and task tracking.

## Development Conventions
-   **Style:** Follow Black/Ruff standards.
-   **Typing:** Strict type hints (`mypy` enabled). Use Pydantic models for data structures.
-   **Testing:** Mock external services (Neo4j, Qdrant, LLM) in unit tests. Coverage is tracked.
-   **Architecture:** Logic should be modular and injectable. Avoid side effects in core logic modules.
-   **Imports:** Use absolute imports (e.g., `from src.utils import ...`).

## Core Workflows
1.  **Ingestion:** PDF -> Text -> Chunks -> Embeddings -> Vector DB.
2.  **Extraction:** Text -> LLM/spaCy -> Entity Candidates -> JSONL.
3.  **Curation:** Candidates -> Human Review (CLI) -> Normalized Knowledge Graph.
4.  **Retrieval:** Query -> Vector Search + Graph Traversal -> Reranking -> Answer.
