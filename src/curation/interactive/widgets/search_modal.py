"""Search and filter modal for entity candidates."""

from typing import Callable, Optional

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select

from src.storage.schemas import EntityType


class SearchFilters:
    """Container for search and filter criteria."""

    def __init__(
        self,
        search_text: str = "",
        status: str = "pending",
        entity_type: Optional[str] = None,
        min_confidence: float = 0.0,
    ):
        """Initialize search filters.

        Args:
            search_text: Text to search for in candidate names
            status: Status filter (pending/approved/rejected/all)
            entity_type: Entity type filter (or None for all)
            min_confidence: Minimum confidence threshold
        """
        self.search_text = search_text
        self.status = status
        self.entity_type = entity_type
        self.min_confidence = min_confidence


class SearchModalScreen(ModalScreen[Optional[SearchFilters]]):
    """Modal screen for searching and filtering candidates."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("enter", "search", "Search", show=True),
    ]

    CSS = """
    SearchModalScreen {
        align: center middle;
    }

    #search-dialog {
        width: 70;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #search-dialog-title {
        width: 100%;
        content-align: center middle;
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    .field-label {
        width: 100%;
        margin-top: 1;
        color: $text-muted;
    }

    .field-input {
        width: 100%;
        margin-bottom: 1;
    }

    .field-select {
        width: 100%;
        margin-bottom: 1;
    }

    #button-container {
        width: 100%;
        height: auto;
        margin-top: 1;
        align: center middle;
    }

    #button-container Horizontal {
        width: auto;
        height: auto;
    }

    Button {
        margin: 0 1;
    }

    .help-text {
        color: $text-muted;
        text-style: italic;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        current_filters: Optional[SearchFilters] = None,
        on_search: Optional[Callable[[SearchFilters], None]] = None,
    ) -> None:
        """Initialize search modal.

        Args:
            current_filters: Current filter state to pre-populate
            on_search: Optional callback to call when search is executed
        """
        super().__init__()
        self.current_filters = current_filters or SearchFilters()
        self.on_search_callback = on_search

    def compose(self) -> ComposeResult:
        """Create modal dialog widgets."""
        with Container(id="search-dialog"):
            yield Label("Search & Filter Candidates", id="search-dialog-title")

            # Search text field
            yield Label("Search Text (name, aliases):", classes="field-label")
            yield Label(
                "Fuzzy match - try partial names or acronyms",
                classes="help-text",
            )
            yield Input(
                value=self.current_filters.search_text,
                placeholder="e.g., 'power', 'EPS', 'thermal'",
                id="search-input",
                classes="field-input",
            )

            # Status filter
            yield Label("Status:", classes="field-label")
            status_options = [
                ("All Statuses", "all"),
                ("Pending", "pending"),
                ("Approved", "approved"),
                ("Rejected", "rejected"),
            ]
            yield Select(
                options=status_options,
                value=self.current_filters.status,
                id="status-select",
                classes="field-select",
            )

            # Entity type filter
            yield Label("Entity Type:", classes="field-label")
            type_options = [("All Types", "all")] + [(t.value, t.value) for t in EntityType]
            current_type = self.current_filters.entity_type or "all"
            yield Select(
                options=type_options,
                value=current_type,
                id="type-select",
                classes="field-select",
            )

            # Confidence threshold
            yield Label("Minimum Confidence (0.0 - 1.0):", classes="field-label")
            yield Input(
                value=str(self.current_filters.min_confidence),
                placeholder="0.0",
                id="confidence-input",
                classes="field-input",
            )

            # Buttons
            with Vertical(id="button-container"):
                with Horizontal():
                    yield Button("Search", variant="primary", id="search-button")
                    yield Button("Clear", variant="default", id="clear-button")
                    yield Button("Cancel", variant="default", id="cancel-button")

    def action_cancel(self) -> None:
        """Handle cancel action (Esc key)."""
        self.dismiss(None)

    def action_search(self) -> None:
        """Handle search action (Enter key)."""
        self._execute_search()

    @on(Button.Pressed, "#search-button")
    def on_search_button(self) -> None:
        """Handle search button click."""
        self._execute_search()

    @on(Button.Pressed, "#clear-button")
    def on_clear_button(self) -> None:
        """Handle clear button click - reset all filters."""
        # Reset all inputs to defaults
        search_input = self.query_one("#search-input", Input)
        status_select = self.query_one("#status-select", Select)
        type_select = self.query_one("#type-select", Select)
        confidence_input = self.query_one("#confidence-input", Input)

        search_input.value = ""
        status_select.value = "pending"
        type_select.value = "all"
        confidence_input.value = "0.0"

    @on(Button.Pressed, "#cancel-button")
    def on_cancel_button(self) -> None:
        """Handle cancel button click."""
        self.dismiss(None)

    def _execute_search(self) -> None:
        """Execute search with current filter values."""
        # Get input values
        search_input = self.query_one("#search-input", Input)
        status_select = self.query_one("#status-select", Select)
        type_select = self.query_one("#type-select", Select)
        confidence_input = self.query_one("#confidence-input", Input)

        # Validate confidence
        try:
            min_confidence = float(confidence_input.value or "0.0")
            if not (0.0 <= min_confidence <= 1.0):
                raise ValueError("Confidence out of range")
        except ValueError:
            self.notify("Confidence must be between 0.0 and 1.0", severity="error")
            confidence_input.focus()
            return

        # Build filters
        filters = SearchFilters(
            search_text=search_input.value.strip(),
            status=status_select.value,
            entity_type=type_select.value if type_select.value != "all" else None,
            min_confidence=min_confidence,
        )

        # Call the search callback if provided
        if self.on_search_callback:
            self.on_search_callback(filters)

        # Dismiss modal with filters
        self.dismiss(filters)
