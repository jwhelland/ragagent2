"""Batch operation modal for previewing and confirming batch actions."""

from typing import List

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from src.storage.schemas import EntityCandidate


class BatchOperationModal(ModalScreen[bool]):
    """Modal screen for previewing and confirming batch operations."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("enter", "confirm", "Confirm", show=True),
    ]

    CSS = """
    BatchOperationModal {
        align: center middle;
    }

    #batch-dialog {
        width: 90;
        height: 35;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #batch-dialog-title {
        width: 100%;
        content-align: center middle;
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    #batch-operation-type {
        width: 100%;
        content-align: center middle;
        margin-bottom: 1;
    }

    #batch-summary {
        width: 100%;
        margin-bottom: 1;
        padding: 1;
        background: $panel;
        color: $text;
    }

    #candidate-preview {
        width: 100%;
        height: 1fr;
        border: solid $secondary;
        margin-bottom: 1;
    }

    .candidate-item {
        padding: 0 1;
    }

    #button-container {
        width: 100%;
        height: auto;
        align: center middle;
    }

    #button-container Horizontal {
        width: auto;
        height: auto;
    }

    Button {
        margin: 0 1;
    }

    .confirm-button {
        background: $success;
    }

    .cancel-button {
        background: $error;
    }
    """

    def __init__(
        self,
        operation: str,
        candidates: List[EntityCandidate],
    ) -> None:
        """Initialize batch operation modal.

        Args:
            operation: The operation to perform ("approve" or "reject")
            candidates: List of candidates to operate on
        """
        super().__init__()
        self.operation = operation
        self.candidates = candidates

    def compose(self) -> ComposeResult:
        """Create modal widgets."""
        with Container(id="batch-dialog"):
            yield Label("Batch Operation Preview", id="batch-dialog-title")

            # Operation type indicator
            operation_text = f"Operation: {self.operation.upper()}"
            if self.operation == "approve":
                operation_text += " ✓"
            elif self.operation == "reject":
                operation_text += " ✗"

            yield Static(operation_text, id="batch-operation-type", markup=True)

            # Summary
            summary_text = (
                f"This will {self.operation} {len(self.candidates)} candidates.\n"
                f"This action can be undone with 'u' (undo)."
            )
            yield Static(summary_text, id="batch-summary")

            # Candidate preview list
            with VerticalScroll(id="candidate-preview"):
                yield Label("Candidates to be affected:", classes="candidate-item")
                for idx, candidate in enumerate(self.candidates, 1):
                    candidate_text = (
                        f"{idx}. {candidate.canonical_name} "
                        f"({candidate.candidate_type.value}) "
                        f"- {candidate.confidence_score:.2f}"
                    )
                    yield Static(candidate_text, classes="candidate-item")

            # Buttons
            with Container(id="button-container"):
                with Horizontal():
                    yield Button(
                        f"Confirm {self.operation.capitalize()}",
                        id="confirm-button",
                        variant="success",
                    )
                    yield Button("Cancel", id="cancel-button", variant="error")

    @on(Button.Pressed, "#confirm-button")
    def action_confirm(self) -> None:
        """Handle confirm button press."""
        self.dismiss(True)

    @on(Button.Pressed, "#cancel-button")
    def action_cancel(self) -> None:
        """Handle cancel button press."""
        self.dismiss(False)
