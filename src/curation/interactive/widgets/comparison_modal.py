"""Side-by-side comparison modal for comparing two entity candidates."""

from typing import Callable, Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Center, Grid, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from src.storage.schemas import EntityCandidate


class CandidateComparisonView(Vertical):
    """Widget showing a single candidate's details in comparison view."""

    def __init__(
        self,
        candidate: EntityCandidate,
        label: str,
        highlight_differences: bool = True,
        **kwargs,
    ) -> None:
        """Initialize comparison view.

        Args:
            candidate: Candidate to display
            label: Label for this candidate (e.g., "Candidate A", "Primary")
            highlight_differences: Whether to highlight different fields
            **kwargs: Additional widget arguments
        """
        super().__init__(**kwargs)
        self.candidate = candidate
        self.label = label
        self.highlight_differences = highlight_differences

    def compose(self) -> ComposeResult:
        """Compose the comparison view."""
        c = self.candidate

        # Header
        yield Static(f"[bold]{self.label}[/bold]", classes="comparison-header")

        # Name
        yield Static(f"[cyan]Name:[/cyan] [bold]{c.canonical_name}[/bold]")

        # Type
        yield Static(f"[cyan]Type:[/cyan] {c.candidate_type.value}")

        # Status
        status_color = "green" if c.status.value == "approved" else "yellow"
        yield Static(f"[cyan]Status:[/cyan] [{status_color}]{c.status.value}[/{status_color}]")

        # Confidence
        conf_bar = "█" * int(c.confidence_score * 10)
        yield Static(f"[cyan]Confidence:[/cyan] {c.confidence_score:.2f} [green]{conf_bar}[/green]")

        # Mentions
        yield Static(f"[cyan]Mentions:[/cyan] {c.mention_count}")

        # Aliases
        if c.aliases:
            aliases_text = ", ".join(c.aliases[:5])
            if len(c.aliases) > 5:
                aliases_text += f" (+{len(c.aliases) - 5} more)"
            yield Static(f"[cyan]Aliases:[/cyan] {aliases_text}")
        else:
            yield Static("[cyan]Aliases:[/cyan] [dim](none)[/dim]")

        # Description
        if c.description:
            desc_preview = (
                c.description[:150] + "..." if len(c.description) > 150 else c.description
            )
            yield Static(f"[cyan]Description:[/cyan]\n{desc_preview}")
        else:
            yield Static("[cyan]Description:[/cyan] [dim](none)[/dim]")


class ComparisonModalScreen(ModalScreen):
    """Modal screen for side-by-side comparison of two candidates."""

    DEFAULT_CSS = """
    ComparisonModalScreen {
        align: center middle;
    }

    #comparison-dialog {
        width: 90%;
        max-width: 140;
        height: auto;
        max-height: 90%;
        background: $surface;
        border: thick $primary;
        padding: 1;
    }

    .comparison-title {
        text-align: center;
        width: 100%;
        padding: 1;
        background: $primary;
        color: $text;
    }

    #comparison-grid {
        grid-size: 2;
        grid-gutter: 1;
        height: auto;
        margin: 1 0;
    }

    CandidateComparisonView {
        border: solid $accent;
        padding: 1;
        height: auto;
        overflow-y: auto;
    }

    .comparison-header {
        background: $accent;
        padding: 0 1;
        margin-bottom: 1;
    }

    .shared-section {
        background: $surface-darken-1;
        padding: 1;
        margin: 1 0;
        border: solid $success;
    }

    .merged-preview {
        background: $surface-darken-1;
        padding: 1;
        margin: 1 0;
        border: solid $warning;
    }

    #button-bar {
        dock: bottom;
        width: 100%;
        height: auto;
        padding: 1;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        candidate_a: EntityCandidate,
        candidate_b: EntityCandidate,
        on_merge: Optional[Callable[[EntityCandidate, EntityCandidate], None]] = None,
        on_select_different: Optional[Callable[[], None]] = None,
        **kwargs,
    ) -> None:
        """Initialize comparison modal.

        Args:
            candidate_a: First candidate to compare
            candidate_b: Second candidate to compare
            on_merge: Callback when user chooses to merge (receives primary, duplicate)
            on_select_different: Callback to select different candidates
            **kwargs: Additional screen arguments
        """
        super().__init__(**kwargs)
        self.candidate_a = candidate_a
        self.candidate_b = candidate_b
        self.on_merge = on_merge
        self.on_select_different = on_select_different

    def compose(self) -> ComposeResult:
        """Compose the comparison modal."""
        with Vertical(id="comparison-dialog"):
            yield Static("🔄 Side-by-Side Comparison", classes="comparison-title")

            # Side-by-side comparison
            with Grid(id="comparison-grid"):
                yield CandidateComparisonView(self.candidate_a, "Candidate A")
                yield CandidateComparisonView(self.candidate_b, "Candidate B")

            # Shared aliases section
            shared_aliases = self._get_shared_aliases()
            if shared_aliases:
                with Vertical(classes="shared-section"):
                    yield Static("[bold green]✓ Shared Aliases:[/bold green]")
                    yield Static(", ".join(shared_aliases))

            # Similarity analysis
            with Vertical(classes="shared-section"):
                yield Static("[bold cyan]📊 Similarity Analysis:[/bold cyan]")
                yield Static(self._get_similarity_analysis())

            # Merged preview
            with Vertical(classes="merged-preview"):
                yield Static("[bold yellow]🔀 Merged Result Preview:[/bold yellow]")
                yield Static(self._get_merge_preview())

            # Buttons
            with Center(id="button-bar"):
                with Horizontal():
                    yield Button("Merge A → B", id="merge_a_b", variant="success")
                    yield Button("Merge B → A", id="merge_b_a", variant="success")
                    yield Button("Select Different", id="select_different", variant="default")
                    yield Button("Cancel", id="cancel", variant="error")

    def _get_shared_aliases(self) -> list[str]:
        """Get aliases shared between both candidates."""
        aliases_a = {a.lower() for a in self.candidate_a.aliases}
        aliases_b = {a.lower() for a in self.candidate_b.aliases}
        return list(aliases_a & aliases_b)

    def _get_similarity_analysis(self) -> str:
        """Get similarity analysis text."""
        from difflib import SequenceMatcher

        # Name similarity
        name_sim = SequenceMatcher(
            None,
            self.candidate_a.canonical_name.lower(),
            self.candidate_b.canonical_name.lower(),
        ).ratio()

        # Type match
        type_match = self.candidate_a.candidate_type == self.candidate_b.candidate_type

        # Shared aliases
        shared_aliases = len(self._get_shared_aliases())

        analysis = []
        analysis.append(f"• Name similarity: {name_sim:.0%}")
        analysis.append(f"• Type match: {'✓ Yes' if type_match else '✗ No (WARNING)'}")
        analysis.append(f"• Shared aliases: {shared_aliases}")
        analysis.append(
            f"• Confidence: A={self.candidate_a.confidence_score:.2f}, "
            f"B={self.candidate_b.confidence_score:.2f}"
        )

        return "\n".join(analysis)

    def _get_merge_preview(self) -> str:
        """Get preview of what the merged result would look like."""
        # Primary is the one with higher confidence
        if self.candidate_a.confidence_score >= self.candidate_b.confidence_score:
            primary = self.candidate_a
            duplicate = self.candidate_b
            label = "A"
        else:
            primary = self.candidate_b
            duplicate = self.candidate_a
            label = "B"

        # Merged aliases
        all_aliases = list(set(primary.aliases) | set(duplicate.aliases))

        # Merged mentions
        total_mentions = primary.mention_count + duplicate.mention_count

        preview = []
        preview.append(f"• Primary: Candidate {label} ({primary.canonical_name})")
        preview.append(f"• Name: {primary.canonical_name}")
        preview.append(f"• Type: {primary.candidate_type.value}")
        preview.append(f"• Aliases: {len(all_aliases)} total ({', '.join(all_aliases[:3])}...)")
        preview.append(f"• Mentions: {total_mentions} (combined)")
        preview.append(
            f"• Confidence: {max(primary.confidence_score, duplicate.confidence_score):.2f} (max)"
        )

        return "\n".join(preview)

    @on(Button.Pressed, "#merge_a_b")
    def handle_merge_a_b(self) -> None:
        """Handle merge A into B (B is primary)."""
        if self.on_merge:
            self.on_merge(self.candidate_b, self.candidate_a)
        self.dismiss("merge_a_b")

    @on(Button.Pressed, "#merge_b_a")
    def handle_merge_b_a(self) -> None:
        """Handle merge B into A (A is primary)."""
        if self.on_merge:
            self.on_merge(self.candidate_a, self.candidate_b)
        self.dismiss("merge_b_a")

    @on(Button.Pressed, "#select_different")
    def handle_select_different(self) -> None:
        """Handle select different candidates."""
        if self.on_select_different:
            self.on_select_different()
        self.dismiss("select_different")

    @on(Button.Pressed, "#cancel")
    def handle_cancel(self) -> None:
        """Handle cancel button."""
        self.dismiss(None)
