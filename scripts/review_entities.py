#!/usr/bin/env python3
"""CLI entrypoint for reviewing normalization mappings."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.curation.review_interface import run  # noqa: E402


def main() -> None:
    """Launch the review interface."""
    run()


if __name__ == "__main__":
    main()
