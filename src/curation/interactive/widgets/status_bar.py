"""Status bar widget for displaying session progress and statistics."""

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from src.curation.interactive.review_mode import ReviewMode


class StatusBar(Widget):
    """Status bar displaying session progress and statistics."""

    # Reactive attributes for real-time updates
    elapsed_time: reactive[str] = reactive("00:00:00")
    review_mode: reactive[ReviewMode] = reactive(ReviewMode.ENTITY)
    # Entity stats
    approved_count: reactive[int] = reactive(0)
    rejected_count: reactive[int] = reactive(0)
    # Relationship stats
    relationship_approved_count: reactive[int] = reactive(0)
    relationship_rejected_count: reactive[int] = reactive(0)
    # Universal stats
    velocity: reactive[float] = reactive(0.0)
    time_remaining: reactive[str] = reactive("--")
    total_candidates: reactive[int] = reactive(0)

    def compose(self) -> ComposeResult:
        """Create status bar widgets."""
        yield Static(id="status-content")

    def watch_elapsed_time(self) -> None:
        """Update display when elapsed time changes."""
        self._update_display()

    def watch_review_mode(self) -> None:
        """Update display when review mode changes."""
        self._update_display()

    def watch_approved_count(self) -> None:
        """Update display when approved count changes."""
        self._update_display()

    def watch_rejected_count(self) -> None:
        """Update display when rejected count changes."""
        self._update_display()

    def watch_relationship_approved_count(self) -> None:
        """Update display when relationship approved count changes."""
        self._update_display()

    def watch_relationship_rejected_count(self) -> None:
        """Update display when relationship rejected count changes."""
        self._update_display()

    def watch_velocity(self) -> None:
        """Update display when velocity changes."""
        self._update_display()

    def watch_time_remaining(self) -> None:
        """Update display when time remaining changes."""
        self._update_display()

    def watch_total_candidates(self) -> None:
        """Update display when total changes."""
        self._update_display()

    def _update_display(self) -> None:
        """Update the status bar display with current statistics."""
        try:
            content = self.query_one("#status-content", Static)

            # Get counts based on review mode
            if self.review_mode == ReviewMode.ENTITY:
                approved = self.approved_count
                rejected = self.rejected_count
            else:  # ReviewMode.RELATIONSHIP
                approved = self.relationship_approved_count
                rejected = self.relationship_rejected_count

            # Build status text
            total_processed = approved + rejected
            progress_pct = (
                (total_processed / self.total_candidates * 100) if self.total_candidates > 0 else 0
            )

            status_text = (
                f"⏱ {self.elapsed_time}  |  "
                f"✓ {approved}  |  "
                f"✗ {rejected}  |  "
                f"📊 {total_processed}/{self.total_candidates} ({progress_pct:.0f}%)  |  "
                f"⚡ {self.velocity:.1f}/min  |  "
                f"⏳ {self.time_remaining} remaining"
            )

            content.update(status_text)
        except Exception:
            # Widget might not be fully initialized yet
            pass


# CSS for StatusBar
STATUS_BAR_CSS = """
StatusBar {
    dock: bottom;
    height: 1;
    background: $panel;
    color: $text;
    padding: 0 1;
}

StatusBar #status-content {
    width: 100%;
    content-align: left middle;
}
"""
