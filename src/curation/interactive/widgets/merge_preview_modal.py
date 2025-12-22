"""Merge preview modal for showing detailed preview before merging candidates."""

from typing import List

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from src.storage.schemas import EntityCandidate


class MergePreviewModal(ModalScreen[str]):
    """Modal screen for previewing merge operation before execution.

    Returns:
        'confirm' to proceed with merge
        'change_primary' to go back and change primary
        None to cancel
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("enter", "confirm", "Confirm", show=True),
    ]

    CSS = """
    MergePreviewModal {
        align: center middle;
    }

    #merge-preview-dialog {
        width: 100;
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

    .primary-marker {
        color: $success;
        text-style: bold;
    }

    .duplicate-item {
        margin-left: 2;
        color: $text-muted;
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
        primary: EntityCandidate,
        duplicates: List[EntityCandidate],
    ) -> None:
        """Initialize merge preview modal.

        Args:
            primary: The primary candidate (will become the entity)
            duplicates: The duplicate candidates (will be rejected)
        """
        super().__init__()
        self.primary = primary
        self.duplicates = duplicates
        self.all_candidates = [primary] + duplicates

    def compose(self) -> ComposeResult:
        """Create modal widgets."""
        with Container(id="merge-preview-dialog"):
            yield Label("Merge Preview", id="merge-preview-title")

            with VerticalScroll(id="preview-scroll"):
                # Primary section
                yield Static("Primary Candidate (will become entity):", classes="section-title")
                primary_text = (
                    f"► [•] {self.primary.canonical_name} "
                    f"({self.primary.candidate_type.value}) - "
                    f"{self.primary.confidence_score:.2f} conf"
                )
                yield Static(primary_text, classes="primary-marker")

                # Duplicates section
                if self.duplicates:
                    yield Static(
                        "\nCandidates to merge (will be marked rejected):",
                        classes="section-title",
                    )
                    for dup in self.duplicates:
                        dup_text = (
                            f"[ ] {dup.canonical_name} "
                            f"({dup.candidate_type.value}) - "
                            f"{dup.confidence_score:.2f}"
                        )
                        yield Static(dup_text, classes="duplicate-item")

                # Merged result preview
                yield Static("\nMerged Result Preview:", classes="section-title")

                with Container(id="merged-result-preview"):
                    # Name
                    yield Static("Canonical Name:", classes="preview-field")
                    yield Static(self.primary.canonical_name, classes="preview-value")

                    # Type
                    yield Static("Entity Type:", classes="preview-field")
                    yield Static(self.primary.candidate_type.value, classes="preview-value")

                    # Aliases (union of all)
                    yield Static("Aliases (union of all):", classes="preview-field")
                    all_aliases = self._collect_all_aliases()
                    if all_aliases:
                        for alias in sorted(all_aliases):
                            yield Static(f"  • {alias}", classes="preview-value")
                    else:
                        yield Static("  (none)", classes="preview-value")

                    # Description
                    yield Static("Description:", classes="preview-field")
                    description = self._get_merged_description()
                    if description:
                        desc_preview = description[:150]
                        if len(description) > 150:
                            desc_preview += "..."
                        yield Static(desc_preview, classes="preview-value")
                    else:
                        yield Static("  (none)", classes="preview-value")

                    # Confidence (max)
                    yield Static("Confidence (maximum):", classes="preview-field")
                    max_conf = max(c.confidence_score for c in self.all_candidates)
                    yield Static(f"{max_conf:.2f}", classes="preview-value")

                    # Mention count (sum)
                    yield Static("Mention Count (sum):", classes="preview-field")
                    total_mentions = sum(c.mention_count for c in self.all_candidates)
                    yield Static(str(total_mentions), classes="preview-value")

                    # Documents (union)
                    yield Static("Source Documents (union):", classes="preview-field")
                    all_docs = self._collect_all_documents()
                    yield Static(f"{len(all_docs)} documents", classes="preview-value")

            # Buttons
            with Container(id="button-container"):
                with Horizontal():
                    yield Button("Change Primary", id="change-primary-button", variant="default")
                    yield Button("Confirm Merge", id="confirm-button", variant="success")
                    yield Button("Cancel", id="cancel-button", variant="error")

    def _collect_all_aliases(self) -> set[str]:
        """Collect all unique aliases from all candidates."""
        aliases = set()
        for candidate in self.all_candidates:
            aliases.add(candidate.canonical_name)
            aliases.update(candidate.aliases)
        return aliases

    def _get_merged_description(self) -> str:
        """Get the description for the merged entity (primary's description with fallback)."""
        if self.primary.description:
            return self.primary.description

        # Fallback to first duplicate with description
        for dup in self.duplicates:
            if dup.description:
                return dup.description

        return ""

    def _collect_all_documents(self) -> set[str]:
        """Collect all unique source documents from all candidates."""
        documents = set()
        for candidate in self.all_candidates:
            documents.update(candidate.source_documents)
        return documents

    @on(Button.Pressed, "#confirm-button")
    def action_confirm(self) -> None:
        """Handle confirm button press."""
        self.dismiss("confirm")

    @on(Button.Pressed, "#change-primary-button")
    def action_change_primary(self) -> None:
        """Handle change primary button press."""
        self.dismiss("change_primary")

    @on(Button.Pressed, "#cancel-button")
    def action_cancel(self) -> None:
        """Handle cancel button press."""
        self.dismiss(None)
