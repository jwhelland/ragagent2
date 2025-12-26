"""Detail panel widget for displaying full relationship candidate information."""

from typing import Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from src.storage.schemas import RelationshipCandidate


class DetailSection(Static):
    """A section within the detail panel."""

    def __init__(self, title: str, content: str, *args, **kwargs) -> None:
        """Initialize detail section.

        Args:
            title: Section title
            content: Section content
        """
        super().__init__(*args, **kwargs)
        self.section_title = title
        self.section_content = content
        self.update_display()

    def update_display(self) -> None:
        """Update the section display."""
        text = Text()
        text.append(f"{self.section_title}\n", style="bold cyan")
        text.append(self.section_content, style="")
        self.update(text)


class RelationshipDetailPanel(Widget):
    """Panel displaying detailed information about the selected relationship candidate."""

    # Reactive attribute for the current candidate
    candidate: reactive[Optional[RelationshipCandidate]] = reactive(None, recompose=True)

    def __init__(self, candidate: Optional[RelationshipCandidate] = None) -> None:
        """Initialize relationship detail panel.

        Args:
            candidate: Optional initial relationship candidate to display
        """
        super().__init__()
        if candidate:
            self.candidate = candidate

    def compose(self) -> ComposeResult:
        """Create child widgets for the detail panel."""
        if not self.candidate:
            yield Static("No relationship candidate selected", classes="empty-detail")
        else:
            with VerticalScroll():
                # Header with relationship source → type → target
                yield self._render_header()

                # Core information
                yield self._render_core_info()

                # Description
                if self.candidate.description:
                    yield self._render_description()

                # Source documents
                if self.candidate.source_documents:
                    yield self._render_source_documents()

                # Chunk IDs
                if self.candidate.chunk_ids:
                    yield self._render_chunk_ids()

                # Provenance events
                if self.candidate.provenance_events:
                    yield self._render_provenance()

    def _render_header(self) -> Static:
        """Render the header with relationship triple."""
        text = Text()
        text.append("\n", style="")
        # Source entity
        text.append(f"{self.candidate.source}", style="bold yellow")
        text.append(" → ", style="dim")
        # Relationship type
        text.append(f"{self.candidate.type}", style="bold cyan")
        text.append(" → ", style="dim")
        # Target entity
        text.append(f"{self.candidate.target}\n", style="bold yellow")

        text.append(f"Key: {self.candidate.candidate_key}\n", style="dim")
        if self.candidate.id:
            text.append(f"ID: {self.candidate.id}\n", style="dim")

        return Static(text, classes="detail-header")

    def _render_core_info(self) -> DetailSection:
        """Render core candidate information."""
        conf = self.candidate.confidence_score
        if conf >= 0.9:
            conf_color = "green"
        elif conf >= 0.7:
            conf_color = "yellow"
        else:
            conf_color = "red"

        text = Text()

        text.append("Relationship Type: ", style="bold")
        text.append(f"{self.candidate.type}\n", style="cyan")

        text.append("Status: ", style="bold")
        status_style = {
            "pending": "yellow",
            "approved": "green",
            "rejected": "red",
        }.get(self.candidate.status.value, "")
        text.append(f"{self.candidate.status.value.upper()}\n", style=status_style)

        text.append("Confidence: ", style="bold")
        text.append(f"{conf:.3f}\n", style=f"bold {conf_color}")

        text.append("Mentions: ", style="bold")
        text.append(f"{self.candidate.mention_count}\n")

        if self.candidate.first_seen:
            text.append("First Seen: ", style="bold")
            text.append(f"{self.candidate.first_seen.strftime('%Y-%m-%d %H:%M')}\n", style="dim")

        if self.candidate.last_seen:
            text.append("Last Seen: ", style="bold")
            text.append(f"{self.candidate.last_seen.strftime('%Y-%m-%d %H:%M')}\n", style="dim")

        return DetailSection("Core Information", text.plain)

    def _render_description(self) -> DetailSection:
        """Render description section."""
        # Wrap long descriptions
        desc = self.candidate.description
        if len(desc) > 500:
            desc = desc[:497] + "..."

        return DetailSection("Description", desc, classes="detail-description")

    def _render_source_documents(self) -> DetailSection:
        """Render source documents section."""
        docs_text = "\n".join(f"  • {doc}" for doc in self.candidate.source_documents[:10])
        if len(self.candidate.source_documents) > 10:
            docs_text += f"\n  ... and {len(self.candidate.source_documents) - 10} more"

        return DetailSection(
            f"Source Documents ({len(self.candidate.source_documents)})",
            docs_text,
            classes="detail-sources",
        )

    def _render_chunk_ids(self) -> DetailSection:
        """Render chunk IDs section."""
        # Show first 5 chunk IDs
        chunks_text = "\n".join(f"  • {chunk}" for chunk in self.candidate.chunk_ids[:5])
        if len(self.candidate.chunk_ids) > 5:
            chunks_text += f"\n  ... and {len(self.candidate.chunk_ids) - 5} more"

        return DetailSection(
            f"Chunk IDs ({len(self.candidate.chunk_ids)})",
            chunks_text,
            classes="detail-chunks",
        )

    def _render_provenance(self) -> DetailSection:
        """Render provenance events section."""
        # Parse provenance events (assuming they're JSON strings)
        import json

        provenance_text = ""
        try:
            for i, event in enumerate(self.candidate.provenance_events[:5]):
                if isinstance(event, str):
                    event_data = json.loads(event)
                else:
                    event_data = event

                provenance_text += f"  Event {i + 1}:\n"
                if isinstance(event_data, dict):
                    for key, value in event_data.items():
                        provenance_text += f"    {key}: {value}\n"
                else:
                    provenance_text += f"    {event_data}\n"
                provenance_text += "\n"

            if len(self.candidate.provenance_events) > 5:
                provenance_text += (
                    f"  ... and {len(self.candidate.provenance_events) - 5} more events"
                )

        except (json.JSONDecodeError, TypeError) as e:
            provenance_text = f"Error parsing provenance: {e}"

        return DetailSection(
            f"Provenance ({len(self.candidate.provenance_events)} events)",
            provenance_text,
            classes="detail-provenance",
        )
