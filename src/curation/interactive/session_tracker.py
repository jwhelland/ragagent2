"""Session tracking for entity review progress and statistics."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

from loguru import logger
from pydantic import BaseModel, Field


class SessionStats(BaseModel):
    """Statistics for a review session."""

    start_time: datetime = Field(default_factory=datetime.now)
    # Entity statistics
    approved_count: int = 0
    rejected_count: int = 0
    edited_count: int = 0
    flagged_count: int = 0
    undo_count: int = 0
    merged_count: int = 0
    # Relationship statistics
    relationship_approved_count: int = 0
    relationship_rejected_count: int = 0
    # Total across all types
    total_processed: int = 0

    @property
    def elapsed_time(self) -> timedelta:
        """Calculate elapsed time since session start."""
        return datetime.now() - self.start_time

    @property
    def velocity(self) -> float:
        """Calculate candidates processed per minute."""
        minutes = self.elapsed_time.total_seconds() / 60
        if minutes == 0:
            return 0.0
        return self.total_processed / minutes

    @property
    def formatted_elapsed(self) -> str:
        """Format elapsed time as HH:MM:SS."""
        total_seconds = int(self.elapsed_time.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class SessionTracker:
    """Tracks review session progress and statistics."""

    DEFAULT_SESSION_DIR = Path.home() / ".ragagent"
    DEFAULT_SESSION_FILE = "current_session.json"

    def __init__(self, session_file: Path | None = None) -> None:
        """Initialize session tracker.

        Args:
            session_file: Optional path to persist session data. If None, uses default location.
        """
        if session_file is None:
            self.session_dir = self.DEFAULT_SESSION_DIR
            self.session_file = self.session_dir / self.DEFAULT_SESSION_FILE
        else:
            self.session_file = session_file
            self.session_dir = session_file.parent

        self.stats = SessionStats()
        self._ensure_session_dir()

    def _ensure_session_dir(self) -> None:
        """Ensure session directory exists."""
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def record_approval(self) -> None:
        """Record an approval action."""
        self.stats.approved_count += 1
        self.stats.total_processed += 1

    def record_rejection(self) -> None:
        """Record a rejection action."""
        self.stats.rejected_count += 1
        self.stats.total_processed += 1

    def record_edit(self) -> None:
        """Record an edit action."""
        self.stats.edited_count += 1

    def record_flag(self) -> None:
        """Record a flag action."""
        self.stats.flagged_count += 1

    def record_undo(self) -> None:
        """Record an undo action."""
        self.stats.undo_count += 1
        # Note: We don't decrement total_processed to keep a count of all operations

    def record_merge(self) -> None:
        """Record a merge action."""
        self.stats.merged_count += 1

    def record_relationship_approval(self) -> None:
        """Record a relationship approval action."""
        self.stats.relationship_approved_count += 1
        self.stats.total_processed += 1

    def record_relationship_rejection(self) -> None:
        """Record a relationship rejection action."""
        self.stats.relationship_rejected_count += 1
        self.stats.total_processed += 1

    def estimate_time_remaining(self, candidates_remaining: int) -> str:
        """Estimate time remaining based on current velocity.

        Args:
            candidates_remaining: Number of candidates left to review

        Returns:
            Formatted time estimate (e.g., "2h 15m") or "calculating..." if too early
        """
        if self.stats.velocity == 0 or candidates_remaining == 0:
            return "calculating..."

        minutes_remaining = candidates_remaining / self.stats.velocity
        if minutes_remaining < 1:
            return "<1m"

        hours = int(minutes_remaining // 60)
        mins = int(minutes_remaining % 60)

        if hours > 0:
            return f"{hours}h {mins}m"
        return f"{mins}m"

    def get_summary(self) -> Dict[str, any]:
        """Get session summary dictionary.

        Returns:
            Dictionary with session statistics and metrics
        """
        return {
            "elapsed_time": self.stats.formatted_elapsed,
            "entities_approved": self.stats.approved_count,
            "entities_rejected": self.stats.rejected_count,
            "entities_edited": self.stats.edited_count,
            "entities_merged": self.stats.merged_count,
            "relationships_approved": self.stats.relationship_approved_count,
            "relationships_rejected": self.stats.relationship_rejected_count,
            "flagged": self.stats.flagged_count,
            "total_processed": self.stats.total_processed,
            "velocity": f"{self.stats.velocity:.1f}/min",
        }

    def reset(self) -> None:
        """Reset session statistics."""
        self.stats = SessionStats()

    def save_session(self) -> bool:
        """Save current session to file.

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.session_file, "w") as f:
                json.dump(
                    self.stats.model_dump(mode="json"),
                    f,
                    indent=2,
                    default=str,  # Handle datetime serialization
                )
            logger.debug(f"Saved session to {self.session_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save session to {self.session_file}: {e}")
            return False

    def load_session(self) -> bool:
        """Load session from file.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.session_file.exists():
            logger.debug(f"No session file found at {self.session_file}")
            return False

        try:
            with open(self.session_file) as f:
                data = json.load(f)

            self.stats = SessionStats.model_validate(data)
            logger.info(f"Loaded session from {self.session_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to load session from {self.session_file}: {e}")
            return False

    def clear_session_file(self) -> bool:
        """Delete the session file.

        Returns:
            True if deleted or doesn't exist, False on error
        """
        try:
            if self.session_file.exists():
                self.session_file.unlink()
                logger.debug(f"Deleted session file {self.session_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session file: {e}")
            return False
