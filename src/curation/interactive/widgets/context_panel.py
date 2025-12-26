"""Context panel combining duplicate suggestions and neighborhood issues."""

from __future__ import annotations

from typing import List, Optional

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Static

from src.curation.entity_approval import (
    EntityCurationService,
    NeighborhoodIssue,
    get_neighborhood_issues,
)
from src.curation.interactive.widgets.duplicate_suggestions import DuplicateSuggestion
from src.storage.schemas import EntityCandidate, RelationshipCandidate


class NeighborhoodIssueRow(Static):
    """A single neighborhood issue with inline action buttons."""

    class Action(Message):
        """Action requested on an issue."""

        def __init__(self, issue: NeighborhoodIssue, action_type: str) -> None:
            super().__init__()
            self.issue = issue
            self.action_type = action_type  # "promote", "approve_peer", "create_entity"

    def __init__(self, issue: NeighborhoodIssue, **kwargs) -> None:
        super().__init__(**kwargs)
        self.issue = issue

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(
                f"[bold]{self.issue.peer_name}[/bold] ({self.issue.relationship_candidate.type})"
            )

            action_text = ""
            btn_label = ""
            btn_id = ""

            if self.issue.issue_type == "promotable":
                action_text = "Peer is APPROVED. Ready to promote."
                btn_label = "Promote"
                btn_id = "promote-btn"
            elif self.issue.issue_type == "resolvable":
                action_text = "Peer is PENDING. Approve it too?"
                btn_label = "Approve Peer"
                btn_id = "approve-peer-btn"
            elif self.issue.issue_type == "missing":
                action_text = "Peer UNKNOWN. Create missing entity?"
                btn_label = "Create Entity"
                btn_id = "create-entity-btn"

            yield Static(f"[italic]{action_text}[/italic]", classes="issue-status")
            yield Button(btn_label, id=btn_id, variant="primary")

    @on(Button.Pressed, "#promote-btn")
    def on_promote(self) -> None:
        self.post_message(self.Action(self.issue, "promote"))

    @on(Button.Pressed, "#approve-peer-btn")
    def on_approve_peer(self) -> None:
        self.post_message(self.Action(self.issue, "approve_peer"))

    @on(Button.Pressed, "#create-entity-btn")
    def on_create_entity(self) -> None:
        self.post_message(self.Action(self.issue, "create_entity"))


class ContextPanel(VerticalScroll):
    """Combined panel for Duplicates and Neighborhood Context."""

    DEFAULT_CSS = """
    ContextPanel {
        height: 100%;
        border: solid $accent;
        padding: 0;
    }

    .section-title {
        background: $accent;
        color: $text;
        padding: 0 1;
        width: 100%;
    }

    .empty-msg {
        color: $text-muted;
        text-align: center;
        padding: 1;
    }

    NeighborhoodIssueRow {
        margin: 1 1;
        padding: 1;
        background: $surface-darken-1;
        border: solid $secondary;
    }

    .issue-status {
        margin-bottom: 1;
    }

    DuplicateSuggestion {
        margin: 1 1;
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary-lighten-1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.current_candidate: Optional[EntityCandidate] = None
        self.issues: List[NeighborhoodIssue] = []
        self.duplicates: List[tuple[EntityCandidate, float, str]] = []

    def compose(self) -> ComposeResult:
        yield Static("🔍 Duplicates", classes="section-title")
        yield Static("No candidate selected", classes="empty-msg", id="dups-empty")
        yield Static("🌐 Neighborhood", classes="section-title")
        yield Static("No candidate selected", classes="empty-msg", id="neighborhood-empty")

    @work(thread=True)
    def update_context(
        self,
        candidate: Optional[EntityCandidate | RelationshipCandidate],
        all_candidates: List[EntityCandidate | RelationshipCandidate],
        service: EntityCurationService,
    ) -> None:
        self.current_candidate = candidate
        if not candidate:
            self.app.call_from_thread(self._clear_ui)
            return

        # Skip duplicate/neighborhood checks for RelationshipCandidates
        # (these are entity-specific features)
        if isinstance(candidate, RelationshipCandidate):
            self.app.call_from_thread(self._clear_ui_for_relationships)
            return

        # 1. Compute Duplicates (sync)

        # Reuse logic from DuplicateSuggestionsPanel if possible, but here we just re-impl minimal
        dups = self._find_duplicates(candidate, all_candidates)  # type: ignore

        # 2. Compute Neighborhood (requires IO)
        issues = []
        try:
            issues = get_neighborhood_issues(service, candidate.canonical_name, candidate.aliases)
        except Exception:
            pass

        self.app.call_from_thread(self._apply_updates, dups, issues)

    def _apply_updates(self, dups, issues) -> None:
        self.duplicates = dups
        self.issues = issues
        self._refresh_ui()

    def _clear_ui(self) -> None:
        self.remove_children()
        self.mount(Static("🔍 Duplicates", classes="section-title"))
        self.mount(Static("No candidate selected", classes="empty-msg"))
        self.mount(Static("🌐 Neighborhood", classes="section-title"))
        self.mount(Static("No candidate selected", classes="empty-msg"))

    def _clear_ui_for_relationships(self) -> None:
        """Clear UI for relationship mode (no duplicates/neighborhood features)."""
        self.remove_children()
        self.mount(Static("🔗 Relationship Context", classes="section-title"))
        self.mount(Static("Relationship-specific context coming soon", classes="empty-msg"))

    def _refresh_ui(self) -> None:
        self.remove_children()

        # Duplicates Section
        self.mount(Static("🔍 Duplicates", classes="section-title"))
        if not self.duplicates:
            self.mount(Static("✓ No duplicates detected", classes="empty-msg"))
        else:
            for cand, score, reason in self.duplicates:
                self.mount(DuplicateSuggestion(cand, score, reason))

        # Neighborhood Section
        self.mount(Static("🌐 Neighborhood", classes="section-title"))
        if not self.issues:
            self.mount(Static("✓ No pending issues", classes="empty-msg"))
        else:
            for issue in self.issues:
                self.mount(NeighborhoodIssueRow(issue))

    def _find_duplicates(self, current: EntityCandidate, all_candidates: List[EntityCandidate]):
        # Stub logic similar to duplicate_suggestions.py
        from difflib import SequenceMatcher

        suggestions = []
        c_name = current.canonical_name.lower()
        for c in all_candidates:
            if c.id == current.id or c.status.value != "pending":
                continue
            sim = SequenceMatcher(None, c_name, c.canonical_name.lower()).ratio()
            if sim > 0.7:
                suggestions.append((c, sim, f"Fuzzy name ({sim:.0%})"))
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:5]
