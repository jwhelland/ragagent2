#!/usr/bin/env python3
"""Entry point for the interactive entity candidate review application.

This script launches the Textual TUI for reviewing entity candidates.
"""

# ruff: noqa: E402

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

# Prevent Loguru's default stderr sink from writing into the alternate screen buffer
# while the TUI is running (this can manifest as a brief "flash" at the top).
logger.remove()

from src.curation.interactive import ReviewApp
from src.curation.interactive.tui_logging import setup_tui_logging


def main() -> None:
    """Main entry point for interactive review CLI."""
    setup_tui_logging()
    app = ReviewApp()
    app.run()


if __name__ == "__main__":
    main()
