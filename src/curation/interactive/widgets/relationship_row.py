"""Row widget for displaying relationship candidates."""

from rich.text import Text
from textual.widgets import Static

from src.storage.schemas import RelationshipCandidate


class RelationshipRow(Static):
    """A single row displaying a relationship candidate's information."""

    def __init__(
        self,
        candidate: RelationshipCandidate,
        index: int,
        is_selected: bool = False,
        is_checked: bool = False,
    ) -> None:
        """Initialize relationship row.

        Args:
            candidate: The relationship candidate to display
            index: Row index in the list
            is_selected: Whether this row is currently selected (navigation cursor)
            is_checked: Whether this row is checked for batch operations
        """
        super().__init__()
        self.candidate = candidate
        self.index = index
        self.is_selected = is_selected
        self.is_checked = is_checked
        self.update_content()

    def update_content(self) -> None:
        """Update the row's display content."""
        # Format confidence score with color
        conf = self.candidate.confidence_score
        if conf >= 0.9:
            conf_color = "green"
        elif conf >= 0.7:
            conf_color = "yellow"
        else:
            conf_color = "red"

        # Build the row text
        text = Text()

        # Checkbox indicator for batch selection
        if self.is_checked:
            text.append("[✓] ", style="bold green")
        else:
            text.append("[ ] ", style="dim")

        # Navigation cursor indicator
        if self.is_selected:
            text.append("► ", style="bold cyan")
        else:
            text.append("  ")

        # Index
        text.append(f"[{self.index + 1:>3}] ", style="dim")

        # Source entity name (truncate if too long)
        source = self.candidate.source
        if len(source) > 25:
            source = source[:22] + "..."
        text.append(f"{source:<25} ", style="bold" if self.is_selected else "")

        # Arrow separator
        text.append("→ ", style="dim")

        # Target entity name (truncate if too long)
        target = self.candidate.target
        if len(target) > 25:
            target = target[:22] + "..."
        text.append(f"{target:<25} ", style="bold" if self.is_selected else "")

        # Relationship type
        rel_type = self.candidate.type
        if len(rel_type) > 18:
            rel_type = rel_type[:15] + "..."
        text.append(f" {rel_type:<18}", style="cyan")

        # Confidence
        text.append(f" {conf:.2f} ", style=f"bold {conf_color}")

        # Mentions
        text.append(f"({self.candidate.mention_count:>3} mentions)", style="dim")

        # Status indicator
        if self.candidate.status.value == "approved":
            text.append(" ✓", style="green")
        elif self.candidate.status.value == "rejected":
            text.append(" ✗", style="red")

        self.update(text)

    def set_selected(self, selected: bool) -> None:
        """Update navigation selection state and refresh display.

        Args:
            selected: Whether this row should be selected
        """
        self.is_selected = selected
        self.update_content()

    def set_checked(self, checked: bool) -> None:
        """Update batch selection state and refresh display.

        Args:
            checked: Whether this row should be checked
        """
        self.is_checked = checked
        self.update_content()
