# Testing Guide

This project uses `pytest` for testing. The test suite includes unit tests, integration tests, and end-to-end (E2E) tests.

## Running Tests

To run the full test suite (including E2E tests):

```bash
uv run pytest
```

### Excluding E2E Tests
E2E tests can be slow and require running databases. To run only unit and integration tests:

```bash
uv run pytest -m "not e2e"
```

### Running Only E2E Tests
To run only the end-to-end suite:

```bash
uv run pytest -m e2e
```

To run a specific test file:

```bash
uv run pytest tests/test_e2e.py
```

## End-to-End (E2E) Tests

The E2E tests (`tests/test_e2e.py`) verify the complete pipeline using the sample documents in `examples/sample_docs`.

**Prerequisites:**
- Neo4j and Qdrant must be running (`docker-compose up -d`).
- `.env` file must be configured with API keys.

The E2E test performs the following steps:
1.  **Ingestion:** Processes sample Markdown files, extracting entities and generating embeddings.
2.  **Discovery:** Runs the discovery pipeline to analyze the extracted corpus.
3.  **Retrieval:** Executes the queries defined in `examples/demo_queries.json` and verifies that expected entities are found in the retrieved context.

## VCR Cassettes (Mocking LLM Calls)

The E2E tests use `pytest-recording` (based on `vcrpy`) to record and replay HTTP interactions with external LLM APIs (OpenAI/Anthropic). This ensures tests are:
- **Fast:** No network latency for API calls.
- **Deterministic:** The LLM's response is frozen, preventing flaky tests.
- **Free:** No API costs for subsequent runs.

### How it works
1.  **First Run:** If no cassette exists, the test makes **real** API calls. The responses are saved to `tests/cassettes/`.
2.  **Subsequent Runs:** The test intercepts the requests and returns the saved responses from the cassette file. Local database traffic (Neo4j/Qdrant) is **not** recorded and always hits the local containers.

### Re-recording Cassettes
If you modify prompts, change model parameters, or update the sample documents, the cached requests will no longer match. You need to re-record the cassettes:

```bash
# Option 1: Delete the cassette file
rm tests/cassettes/test_e2e/test_e2e_pipeline.yaml
uv run pytest tests/test_e2e.py

# Option 2: Use the rewrite flag
uv run pytest tests/test_e2e.py --record-mode=rewrite
```

### Security
The configuration is set to automatically filter sensitive headers like `Authorization` and `x-api-key` from the cassettes. 

**Note:** The `tests/cassettes/` directory is included in `.gitignore` because the recorded responses contain project-specific document content. Developers running these tests for the first time will generate their own local cassettes.

## Troubleshooting

- **Neo4j connection failed:** Ensure the container is running and the password in `.env` (or `config.yaml`) matches the docker container (default: `ragagent2024`).
- **Embedding dimension mismatch:** Ensure `EMBEDDING_DIMENSION` in `.env` matches the model (e.g., 384 for `bge-small`).
