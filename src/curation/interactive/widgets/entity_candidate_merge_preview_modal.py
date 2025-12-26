"""Modal screen for previewing merge of a candidate into an existing entity."""

from typing import Any, Dict

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from src.storage.schemas import EntityCandidate


class EntityCandidateMergePreviewModal(ModalScreen[bool]):
    """Modal screen for previewing merge of candidate into existing entity."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("enter", "confirm", "Confirm", show=True),
    ]

    CSS = """
    EntityCandidateMergePreviewModal {
        align: center middle;
    }

    #merge-preview-dialog {
        width: 90;
        height: auto;
        max-height: 45;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #merge-preview-title {
        width: 100%;
        content-align: center middle;
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    .section-title {
        text-style: bold;
        color: $accent;
        margin-top: 1;
        margin-bottom: 0;
    }

    .entity-marker {
        color: $success;
        text-style: bold;
    }

    .candidate-marker {
        color: $secondary;
        text-style: bold;
    }

    #merged-result-preview {
        width: 100%;
        border: solid $secondary;
        padding: 1;
        margin-top: 1;
        margin-bottom: 1;
        background: $panel;
    }

    .preview-field {
        margin-bottom: 0;
    }

    .preview-value {
        margin-left: 2;
        margin-bottom: 1;
        color: $text;
    }

    #preview-scroll {
        width: 100%;
        height: 1fr;
        margin-bottom: 1;
    }

    #button-container {
        width: 100%;
        height: auto;
        align: center middle;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        entity_data: Dict[str, Any],
        candidate: EntityCandidate,
    ) -> None:
        super().__init__()
        self.entity_data = entity_data
        self.candidate = candidate

    def compose(self) -> ComposeResult:
        with Container(id="merge-preview-dialog"):
            yield Label("Merge Candidate into Existing Entity", id="merge-preview-title")

            with VerticalScroll(id="preview-scroll"):
                # Target Entity section
                yield Static("Target Entity (to be updated):", classes="section-title")
                entity_text = (
                    f"► {self.entity_data.get('canonical_name')} "
                    f"({self.entity_data.get('entity_type')})"
                )
                yield Static(entity_text, classes="entity-marker")

                # Candidate section
                yield Static(
                    "\nCandidate to merge (will be marked rejected):", classes="section-title"
                )
                candidate_text = (
                    f"◄ {self.candidate.canonical_name} "
                    f"({self.candidate.candidate_type.value}) - "
                    f"{self.candidate.confidence_score:.2f} conf"
                )
                yield Static(candidate_text, classes="candidate-marker")

                # Merged result preview
                yield Static("\nResulting Changes:", classes="section-title")

                with Container(id="merged-result-preview"):
                    # Aliases (union)
                    existing_aliases = set(self.entity_data.get("aliases", []))
                    new_aliases = set(self.candidate.aliases) | {self.candidate.canonical_name}
                    added_aliases = new_aliases - existing_aliases

                    yield Static("New Aliases to be added:", classes="preview-field")
                    if added_aliases:
                        for alias in sorted(added_aliases):
                            yield Static(f"  + {alias}", classes="preview-value")
                    else:
                        yield Static("  (none)", classes="preview-value")

                    # Mention count
                    yield Static("Mention Count update:", classes="preview-field")
                    current_mentions = self.entity_data.get("mention_count", 0)
                    new_mentions = current_mentions + self.candidate.mention_count
                    yield Static(f"{current_mentions} -> {new_mentions}", classes="preview-value")

                    # Documents (union)
                    yield Static("Source Documents:", classes="preview-field")
                    existing_docs = set(self.entity_data.get("source_documents", []))
                    new_docs = set(self.candidate.source_documents)
                    total_docs = len(existing_docs | new_docs)
                    added_docs = len(new_docs - existing_docs)
                    yield Static(
                        f"Total: {total_docs} (+{added_docs} new)", classes="preview-value"
                    )

            # Buttons
            with Horizontal(id="button-container"):
                yield Button("Confirm Merge", id="confirm-btn", variant="success")
                yield Button("Cancel", id="cancel-btn", variant="error")

    @on(Button.Pressed, "#confirm-btn")
    def on_confirm(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel(self) -> None:
        self.dismiss(False)

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
