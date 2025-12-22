"""Edit modal for modifying entity candidate fields."""

from typing import Callable, Optional

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.validation import Number, ValidationResult, Validator
from textual.widgets import Button, Input, Label, Select, TextArea

from src.storage.schemas import EntityCandidate, EntityType


class AliasValidator(Validator):
    """Validator for comma-separated aliases."""

    def validate(self, value: str) -> ValidationResult:
        """Validate aliases format.

        Args:
            value: The input value to validate

        Returns:
            ValidationResult indicating success or failure
        """
        # Allow empty or whitespace-only (no aliases is valid)
        if not value or value.strip() == "":
            return self.success()

        # Check that all aliases are non-empty after splitting
        aliases = [a.strip() for a in value.split(",")]
        if all(alias for alias in aliases):
            return self.success()

        return self.failure("Aliases must be comma-separated, non-empty values")


class EditModalScreen(ModalScreen[Optional[EntityCandidate]]):
    """Modal screen for editing entity candidate fields."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("ctrl+s", "save", "Save", show=True),
    ]

    CSS = """
    EditModalScreen {
        align: center middle;
    }

    #edit-dialog {
        width: 80;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #edit-dialog-title {
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

    .field-textarea {
        width: 100%;
        height: 6;
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
    """

    def __init__(
        self,
        candidate: EntityCandidate,
        on_save: Optional[Callable[[EntityCandidate], None]] = None,
    ) -> None:
        """Initialize edit modal.

        Args:
            candidate: The entity candidate to edit
            on_save: Optional callback to call after successful save
        """
        super().__init__()
        self.candidate = candidate
        self.on_save_callback = on_save

    def compose(self) -> ComposeResult:
        """Create modal dialog widgets."""
        with Container(id="edit-dialog"):
            yield Label("Edit Entity Candidate", id="edit-dialog-title")

            # Canonical name field
            yield Label("Canonical Name:", classes="field-label")
            yield Input(
                value=self.candidate.canonical_name,
                placeholder="Enter canonical name",
                id="canonical-name-input",
                classes="field-input",
            )

            # Type selector
            yield Label("Entity Type:", classes="field-label")
            type_options = [(t.value, t.value) for t in EntityType]
            yield Select(
                options=type_options,
                value=self.candidate.candidate_type.value,
                id="type-select",
                classes="field-input",
            )

            # Aliases field (comma-separated)
            yield Label("Aliases (comma-separated):", classes="field-label")
            aliases_str = ", ".join(self.candidate.aliases)
            yield Input(
                value=aliases_str,
                placeholder="alias1, alias2, alias3",
                id="aliases-input",
                classes="field-input",
                validators=[AliasValidator()],
            )

            # Confidence score field
            yield Label("Confidence Score (0.0 - 1.0):", classes="field-label")
            yield Input(
                value=str(self.candidate.confidence_score),
                placeholder="0.0 - 1.0",
                id="confidence-input",
                classes="field-input",
                validators=[Number(minimum=0.0, maximum=1.0)],
            )

            # Description field (multiline)
            yield Label("Description:", classes="field-label")
            yield TextArea(
                text=self.candidate.description,
                id="description-textarea",
                classes="field-textarea",
            )

            # Buttons
            with Vertical(id="button-container"):
                with Horizontal():
                    yield Button("Save", variant="primary", id="save-button")
                    yield Button("Cancel", variant="default", id="cancel-button")

    def action_cancel(self) -> None:
        """Handle cancel action (Esc key)."""
        self.dismiss(None)

    def action_save(self) -> None:
        """Handle save action (Ctrl+S key)."""
        self._save_changes()

    @on(Button.Pressed, "#save-button")
    def on_save_button(self) -> None:
        """Handle save button click."""
        self._save_changes()

    @on(Button.Pressed, "#cancel-button")
    def on_cancel_button(self) -> None:
        """Handle cancel button click."""
        self.dismiss(None)

    def _save_changes(self) -> None:
        """Validate and save changes to candidate."""
        # Get input widgets
        canonical_name_input = self.query_one("#canonical-name-input", Input)
        type_select = self.query_one("#type-select", Select)
        aliases_input = self.query_one("#aliases-input", Input)
        confidence_input = self.query_one("#confidence-input", Input)
        description_textarea = self.query_one("#description-textarea", TextArea)

        # Validate all fields
        if not canonical_name_input.is_valid:
            self.notify("Invalid canonical name", severity="error")
            canonical_name_input.focus()
            return

        if not aliases_input.is_valid:
            self.notify("Invalid aliases format", severity="error")
            aliases_input.focus()
            return

        if not confidence_input.is_valid:
            self.notify("Confidence must be between 0.0 and 1.0", severity="error")
            confidence_input.focus()
            return

        # Extract values
        canonical_name = canonical_name_input.value.strip()
        if not canonical_name:
            self.notify("Canonical name is required", severity="error")
            canonical_name_input.focus()
            return

        # Parse aliases (comma-separated)
        aliases_str = aliases_input.value.strip()
        aliases = []
        if aliases_str:
            aliases = [a.strip() for a in aliases_str.split(",") if a.strip()]

        # Parse confidence
        try:
            confidence = float(confidence_input.value)
            if not (0.0 <= confidence <= 1.0):
                raise ValueError("Confidence out of range")
        except ValueError:
            self.notify("Invalid confidence value", severity="error")
            confidence_input.focus()
            return

        # Get entity type
        entity_type = EntityType(type_select.value)

        # Get description
        description = description_textarea.text.strip()

        # Create updated candidate
        updated_candidate = self.candidate.model_copy(
            update={
                "canonical_name": canonical_name,
                "candidate_type": entity_type,
                "aliases": aliases,
                "confidence_score": confidence,
                "description": description,
            }
        )

        # Call the save callback if provided
        if self.on_save_callback:
            self.on_save_callback(updated_candidate)

        # Dismiss modal with updated candidate
        self.dismiss(updated_candidate)
