"""Logging configuration for the Textual interactive review app.

The interactive TUI runs in an alternate screen buffer; any writes to stdout/stderr
from background threads (e.g., Loguru's default sink) will briefly "flash" in the
terminal above the UI. To avoid that, we disable console logging while the TUI is
running and write logs to a file instead.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger


def setup_tui_logging(*, log_path: str = "logs/interactive_tui.log") -> None:
    """Configure Loguru to avoid writing to stdout/stderr during the TUI session."""
    # Allow developers to opt out while debugging.
    if os.getenv("RAGAGENT_TUI_DISABLE_LOG_RECONFIG") == "1":
        return

    level = os.getenv("RAGAGENT_TUI_LOG_LEVEL", "INFO").upper()
    keep_console = os.getenv("RAGAGENT_TUI_CONSOLE_LOGS") == "1"

    logger.remove()

    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(target),
        level=level,
        rotation="10 MB",
        retention=3,
        compression="zip",
    )

    if keep_console:
        # Useful for local debugging, but will re-introduce flicker in the TUI.
        logger.add(lambda msg: print(msg, end=""), level=level)
