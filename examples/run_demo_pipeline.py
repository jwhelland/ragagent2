"""Automated demo pipeline for RAGAgent2.

This script runs the ingestion and discovery stages automatically using the sample documents.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], description: str):
    print(f"\n>>> {description}...")
    print(f"Running: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during {description}: {e}")
        sys.exit(1)


def main():
    root_dir = Path(__file__).parent.parent
    sample_docs = root_dir / "examples" / "sample_docs"

    if not sample_docs.exists():
        print(f"Error: Sample documents directory not found at {sample_docs}")
        sys.exit(1)

    # 1. Setup
    run_command(["uv", "run", "ragagent-setup", "--recreate-qdrant"], "Initializing databases")

    # 2. Ingestion
    run_command(
        ["uv", "run", "ragagent-ingest", "--directory", str(sample_docs), "--include-text"],
        "Ingesting sample documents",
    )

    # 3. Discovery
    run_command(["uv", "run", "ragagent-discover"], "Running entity discovery")

    print("\n" + "=" * 50)
    print("Demo Pipeline Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Review the discovery report in data/discovery/")
    print("2. Run the interactive curation tool: uv run ragagent-review-interactive")
    print("3. Start querying the system: uv run ragagent-query")


if __name__ == "__main__":
    main()
