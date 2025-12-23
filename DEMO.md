# RAGAgent2 Demo Walkthrough

This guide walks you through the complete lifecycle of a Graph RAG system using the provided sample documents.

## 1. Setup

First, ensure your databases (Neo4j and Qdrant) are running via Docker Compose:

```bash
docker-compose up -d
```

Initialize the databases. **Warning:** This will clear existing data if you use `--recreate-qdrant`.

```bash
uv run ragagent-setup --recreate-qdrant
```

## 2. Ingestion

Ingest the sample documents located in `examples/sample_docs`. This process will parse the markdown files, chunk them, generate embeddings, and store them in the vector and graph databases. It also performs initial entity and relationship extraction.

```bash
uv run ragagent-ingest --directory examples/sample_docs --include-text
```

## 3. Discovery

Analyze the extracted entity candidates to identify frequency statistics, clusters, and potential merges. This produces a discovery report in `data/discovery/<timestamp>/`.

```bash
uv run ragagent-discover
```

Check the generated report (Markdown or HTML) to see what entities were found.

## 4. Curation (Interactive)

Launch the interactive TUI to review, approve, merge, or reject entity candidates. This is a crucial step to ensure the knowledge graph is clean.

```bash
uv run ragagent-review-interactive
```

### Specific Curation Examples for this Demo:

When you run the review tool on the sample docs, you'll likely see candidates like these:

*   **Approve (`a`):**
    *   `NASA` (Type: ORG)
    *   `SpaceX` (Type: ORG)
    *   `Mars` (Type: LOCATION/PLANET)
    *   `Perseverance` (Type: MISSION/ROVER)
    *   `Elon Musk` (Type: PERSON)
*   **Merge (`m`):**
    *   You might see `Jet Propulsion Laboratory` and `JPL`. Select both and press `m` to merge them into a single entity (choose `Jet Propulsion Laboratory` as the primary name).
    *   `Starship` and `Starship spacecraft` might appear separately. Merge them.
*   **Edit (`e`):**
    *   If `Mastcam-Z` is categorized as `ORG`, press `e` to change its type to `INSTRUMENT` or `EQUIPMENT`.
*   **Reject (`r`):**
    *   Reject generic terms that might have been picked up as entities, such as `civilian space program` or `Red Planet` (unless you want to treat it as an alias for Mars).

After reviewing, press `q` to exit. Your approved entities are now persisted in the graph as formal `Entity` nodes, while unreviewed ones remain as `EntityCandidate`.

## 5. Querying

Now that you have an approved knowledge graph, you can run hybrid RAG queries.

```bash
uv run ragagent-query
```

**Example Queries:**
- "What are the main goals of the Perseverance rover?"
- "How is SpaceX planning to explore Mars?"
- "Which organizations are involved in Mars exploration?"
- "What is Starship and who developed it?"

## 6. Visualization (Optional)

You can view the resulting graph in the Neo4j Browser (usually at http://localhost:7474). Use the following Cypher query to see your entities:

```cypher
MATCH (e:Entity)-[r]->(target) RETURN e, r, target LIMIT 50
```
