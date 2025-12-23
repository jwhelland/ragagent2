# Contributing to Graph RAG System

Thank you for your interest in contributing to the Graph RAG System! This document provides guidelines and instructions for developers who want to help improve the project.

## Development Setup

### 1. Environment Requirements

- **Python 3.12.1+**
- **uv** (astral.sh/uv) for dependency management
- **Docker & Docker Compose** for local infrastructure (Neo4j, Qdrant)

### 2. Initial Setup

```bash
# Clone the repository
git clone <repository-url>
cd ragagent2

# Create virtual environment and install dependencies
uv sync

# Install spaCy model
uv run spacy download en_core_web_lg
uv run spacy download en_core_web_sm

# Start databases
docker-compose up -d

# Initialize databases
uv run ragagent-setup
```

## Workflow and Standards

### Code Style

We follow strict code quality standards to maintain a clean and maintainable codebase:

- **Formatting**: We use [Black](https://github.com/psf/black).
- **Linting**: We use [Ruff](https://github.com/astral-sh/ruff) for fast linting and import sorting.
- **Typing**: Use Python type hints for all public functions and class members. We use [Pydantic](https://docs.pydantic.dev/) for data models and configuration.

```bash
# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/
```

### Testing

All new features and bug fixes must include tests. We aim for >80% code coverage.

- **Unit Tests**: Place in `tests/` mirroring the `src/` structure.
- **Integration Tests**: Verify interactions between components.
- **End-to-End Tests**: Run full pipelines with sample data.

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src --cov-report=term-missing
```

See [docs/TESTING.md](TESTING.md) for detailed information on VCR cassettes and mocking.

## Project Architecture

The system is organized into modular components:

- `src/ingestion/`: Document parsing, cleaning, and hierarchical chunking.
- `src/extraction/`: Entity and relationship extraction logic (spaCy + LLM + Rules).
- `src/normalization/`: Deduplication, fuzzy matching, and acronym resolution.
- `src/storage/`: Database managers for Neo4j and Qdrant.
- `src/retrieval/`: Query parsing, hybrid search, and reranking.
- `src/curation/`: Interactive review tools and curation logic.
- `src/pipeline/`: Orchestration logic for various system workflows.

## Configuration Management

Configuration should be additive and backward compatible:
1. Update `src/utils/config.py` Pydantic models.
2. Update `config/config.yaml` with default values.
3. Update `.env.example` if new environment variables are required.

## Adding New Features

1. **Plan**: Open a discussion or issue to describe the proposed change.
2. **Implement**: Create a feature branch and implement your changes.
3. **Document**: Update docstrings and relevant Markdown documentation in `docs/`.
4. **Test**: Ensure all tests pass and coverage is maintained.
5. **Verify**: Run the system end-to-end to ensure no regressions.

## Documentation Guidelines

- Use Google-style docstrings for functions and classes.
- Update `README.md` if CLI commands or installation steps change.
- Keep `plans/` updated if project milestones change.
- Use clear, professional language in all documentation.

## Feedback and Support

- Report bugs via GitHub Issues.
- Provide clear reproduction steps for any reported issues.
- Contributions via Pull Requests are welcome!
