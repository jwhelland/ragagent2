"""CLI script for incremental document updates."""

import argparse
import sys

from loguru import logger

from src.pipeline.update_pipeline import UpdatePipeline
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Update documents in the Graph RAG system.")
    parser.add_argument(
        "paths", nargs="+", help="Paths to files or directories to scan for updates"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be updated without making changes"
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".pdf", ".txt", ".md"],
        help="File extensions to include",
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Initialize update pipeline
    pipeline = UpdatePipeline(config)

    logger.info(f"Starting document update for paths: {args.paths}")

    # 1. Detect changes
    report = pipeline.detect_changes(args.paths, extensions=args.extensions)

    logger.info(report.summary())

    if not report.has_changes:
        logger.info("No changes detected. System is up to date.")
        return

    # 2. Process changes
    if not args.dry_run:
        logger.info("Processing changes...")
        stats = pipeline.process_report(report, dry_run=False)
    else:
        logger.info("Running in DRY RUN mode...")
        stats = pipeline.process_report(report, dry_run=True)

    # 3. Final summary
    logger.success("Update process completed.")
    logger.info(
        f"Summary: {stats.get('new', 0)} new, {stats.get('modified', 0)} modified, "
        f"{stats.get('deleted', 0)} deleted, {stats.get('failed', 0)} failed, "
        f"{stats.get('unchanged', 0)} unchanged."
    )


if __name__ == "__main__":
    main()
