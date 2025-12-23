"""End-to-end tests using sample documents and demo queries.

This test suite executes the full RAG pipeline:
1. Ingestion of 'examples/sample_docs'
2. Entity Discovery
3. Hybrid Retrieval using 'examples/demo_queries.json'

It uses VCR to record LLM API calls, allowing deterministic replay.
Database interactions (Neo4j/Qdrant) are NOT recorded and run against the live local services.
"""

import json
import os
from pathlib import Path

import pytest
from loguru import logger

from src.pipeline.discovery_pipeline import DiscoveryPipeline
from src.pipeline.ingestion_pipeline import IngestionPipeline
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_parser import QueryParser
from src.storage.neo4j_manager import Neo4jManager
from src.utils.config import load_config

# Skip if vital env vars are missing (only strictly needed if not using cassettes)
# But since we need DBs anyway, we check DBs.
pytestmark = pytest.mark.e2e


def scrub_response_headers(response):
    """Remove specific sensitive headers from the response."""
    headers_to_remove = {"openai-organization", "openai-project", "set-cookie"}
    if "headers" in response:
        # Create a list of keys to remove to avoid runtime error during iteration
        keys_to_remove = [
            k for k in response["headers"] if k.lower() in headers_to_remove
        ]
        for k in keys_to_remove:
            del response["headers"][k]
    return response


@pytest.fixture(scope="module")
def vcr_config():
    """Configure VCR to ignore local DB traffic and filter secrets."""
    return {
        "filter_headers": [
            "authorization",
            "x-api-key",
        ],
        "ignore_localhost": True,  # Don't record Qdrant/Neo4j interactions
        "record_mode": "once",  # Record if missing, replay otherwise
        "before_record_response": scrub_response_headers,
    }


@pytest.fixture(scope="module")
def sample_data_path():
    """Return path to sample docs."""
    root = Path(__file__).parent.parent
    path = root / "examples" / "sample_docs"
    if not path.exists():
        pytest.skip("examples/sample_docs not found")
    return path


@pytest.fixture(scope="module")
def demo_queries_path():
    """Return path to demo queries."""
    root = Path(__file__).parent.parent
    path = root / "examples" / "demo_queries.json"
    if not path.exists():
        pytest.skip("examples/demo_queries.json not found")
    return path


@pytest.fixture(scope="module")
def config():
    """Load default config, patching env to ensure valid embedding settings."""
    # Patch env to avoid mismatch if user has bad .env
    old_dim = os.environ.get("EMBEDDING_DIMENSION")
    os.environ["EMBEDDING_DIMENSION"] = "384"

    try:
        cfg = load_config("config/config.yaml")
        # Ensure it matches what we want for the test
        cfg.database.embedding_model = "BAAI/bge-small-en-v1.5"
        cfg.database.embedding_dimension = 384

        # FORCE password to match docker-compose default for E2E test
        # This fixes local env mismatches where .env might have "password" but docker has "ragagent2024"
        cfg.database.neo4j_password = "ragagent2024"

        return cfg
    finally:
        if old_dim is None:
            del os.environ["EMBEDDING_DIMENSION"]
        else:
            os.environ["EMBEDDING_DIMENSION"] = old_dim


@pytest.fixture(scope="module")
def check_services(config):
    """Ensure databases are reachable before running."""
    # Check Neo4j
    try:
        neo4j = Neo4jManager(config.database)
        neo4j.connect()
        neo4j.close()
    except Exception as e:
        pytest.fail(f"Neo4j connection failed with error: {e}")

    # Check Qdrant
    if not config.database.qdrant_host:
        pytest.skip("Qdrant Host not configured")


@pytest.mark.vcr
def test_e2e_pipeline(config, sample_data_path, demo_queries_path, check_services, tmp_path):
    """Run the full ingestion -> discovery -> retrieval loop."""

    # 1. Ingestion
    logger.info("Starting E2E Ingestion...")
    ingestion = IngestionPipeline(config)

    # Collect files
    supported_suffixes = {".pdf", ".md", ".txt", ".markdown"}
    files = [
        p for p in sample_data_path.rglob("*") if p.suffix.lower() in supported_suffixes and p.is_file()
    ]

    results = ingestion.process_batch(files, force_reingest=True)

    assert len(results) > 0, "No documents ingested"
    assert all(r.success for r in results), "Some document ingestions failed"

    # 2. Discovery
    logger.info("Starting E2E Discovery...")
    discovery = DiscoveryPipeline(config)
    discovery_output = tmp_path / "discovery"
    report = discovery.run(output_dir=discovery_output)

    assert report.totals["candidates"] > 0, "No entity candidates found"
    assert (discovery_output / "discovery_report.md").exists()

    # 3. Retrieval
    logger.info("Starting E2E Retrieval...")
    retriever = HybridRetriever(config)
    parser = QueryParser(config)

    with open(demo_queries_path) as f:
        queries = json.load(f)

    for q in queries:
        query_text = q["query"]
        expected_entities = q.get("expected_entities", [])

        logger.info(f"Testing query: {query_text}")

        # Parse query first
        parsed_query = parser.parse(query_text)
        context = retriever.retrieve(parsed_query)

        # Basic validation: ensure we got some context
        assert (
            len(context.chunks) > 0 or len(context.graph_data) > 0
        ), f"No context retrieved for: {query_text}"

        # Loose check for expected entities in the retrieved chunks
        combined_text = "\n".join([c.content for c in context.chunks]).lower()
        found_count = 0
        for entity in expected_entities:
            if entity.lower() in combined_text:
                found_count += 1

        # We expect at least *some* of the expected entities to be present
        if expected_entities:
            assert (
                found_count > 0
            ), f"None of {expected_entities} found in context for '{query_text}'"
