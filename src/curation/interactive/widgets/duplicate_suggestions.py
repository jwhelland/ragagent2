"""Duplicate detection suggestions widget."""

from typing import List, Optional

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from src.storage.schemas import EntityCandidate


class DuplicateSuggestion(Static):
    """A single duplicate suggestion row."""

    def __init__(
        self,
        candidate: EntityCandidate,
        similarity_score: float,
        reason: str,
        **kwargs,
    ) -> None:
        """Initialize duplicate suggestion.

        Args:
            candidate: The suggested duplicate candidate
            similarity_score: Similarity score (0-1)
            reason: Reason for suggestion (e.g., "Similar name", "Shared alias")
            **kwargs: Additional widget arguments
        """
        super().__init__(**kwargs)
        self.candidate = candidate
        self.similarity_score = similarity_score
        self.reason = reason

    def render(self) -> str:
        """Render the suggestion."""
        confidence_bar = "█" * int(self.candidate.confidence_score * 10)
        similarity_bar = "▓" * int(self.similarity_score * 10)

        return (
            f"[bold]{self.candidate.canonical_name}[/bold]\n"
            f"  Type: {self.candidate.candidate_type.value} | "
            f"Conf: {self.candidate.confidence_score:.2f} {confidence_bar}\n"
            f"  Similarity: {self.similarity_score:.2f} {similarity_bar}\n"
            f"  Reason: {self.reason}\n"
            f"  Aliases: {', '.join(self.candidate.aliases[:3])}"
        )


class DuplicateSuggestionsPanel(VerticalScroll):
    """Panel showing duplicate detection suggestions for the current candidate."""

    DEFAULT_CSS = """
    DuplicateSuggestionsPanel {
        height: 100%;
        border: solid $accent;
        padding: 1;
    }

    DuplicateSuggestionsPanel > .title {
        background: $accent;
        color: $text;
        padding: 0 1;
        dock: top;
    }

    DuplicateSuggestionsPanel > .empty {
        color: $text-muted;
        text-align: center;
        padding-top: 5;
    }

    DuplicateSuggestion {
        margin: 1 0;
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary-lighten-1;
    }

    DuplicateSuggestion:hover {
        background: $surface;
        border: solid $primary;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize duplicate suggestions panel."""
        super().__init__(**kwargs)
        self.current_candidate: Optional[EntityCandidate] = None
        self.suggestions: List[tuple[EntityCandidate, float, str]] = []

    def compose(self) -> ComposeResult:
        """Compose the widget."""
        yield Static("🔍 Duplicate Suggestions", classes="title")
        if not self.suggestions:
            yield Static("No duplicates detected", classes="empty")
        else:
            for candidate, score, reason in self.suggestions:
                yield DuplicateSuggestion(candidate, score, reason)

    def update_suggestions(
        self,
        current_candidate: Optional[EntityCandidate],
        all_candidates: List[EntityCandidate],
    ) -> None:
        """Update suggestions based on current candidate.

        Args:
            current_candidate: The candidate to find duplicates for
            all_candidates: List of all available candidates to check

        Note:
            TODO (Phase 5 Task 5.6.6a): Also check against existing APPROVED entities
            in the graph, not just other candidates. This will help prevent approving
            duplicates of already-approved entities and enable merge-into-entity workflow.
        """
        self.current_candidate = current_candidate
        self.suggestions = []

        if not current_candidate:
            self.remove_children()
            self.mount(Static("🔍 Duplicate Suggestions", classes="title"))
            self.mount(Static("No candidate selected", classes="empty"))
            return

        # Find potential duplicates
        suggestions = self._find_duplicates(current_candidate, all_candidates)
        self.suggestions = suggestions

        # Update UI
        self.remove_children()
        self.mount(Static("🔍 Duplicate Suggestions", classes="title"))

        if not suggestions:
            self.mount(Static("✓ No duplicates detected", classes="empty"))
        else:
            for candidate, score, reason in suggestions:
                self.mount(DuplicateSuggestion(candidate, score, reason))

    def _find_duplicates(
        self,
        current: EntityCandidate,
        all_candidates: List[EntityCandidate],
    ) -> List[tuple[EntityCandidate, float, str]]:
        """Find potential duplicate candidates.

        Args:
            current: Current candidate
            all_candidates: All candidates to check against

        Returns:
            List of (candidate, similarity_score, reason) tuples
        """
        from difflib import SequenceMatcher

        suggestions = []
        current_name_lower = current.canonical_name.lower()
        current_aliases_lower = {a.lower() for a in current.aliases}

        for candidate in all_candidates:
            # Skip self and non-PENDING candidates
            if candidate.id == current.id or candidate.status.value != "pending":
                continue

            # Check for exact alias matches
            candidate_aliases_lower = {a.lower() for a in candidate.aliases}
            shared_aliases = current_aliases_lower & candidate_aliases_lower
            if shared_aliases:
                suggestions.append(
                    (
                        candidate,
                        1.0,
                        f"Shared alias: {', '.join(list(shared_aliases)[:2])}",
                    )
                )
                continue

            # Check for fuzzy name similarity
            candidate_name_lower = candidate.canonical_name.lower()
            similarity = SequenceMatcher(None, current_name_lower, candidate_name_lower).ratio()

            if similarity > 0.7:
                suggestions.append(
                    (
                        candidate,
                        similarity,
                        f"Similar name (Fuzzy: {similarity:.0%})",
                    )
                )
                continue

            # Check if current name appears in candidate aliases or vice versa
            if current_name_lower in candidate_aliases_lower:
                suggestions.append(
                    (
                        candidate,
                        0.9,
                        f"Name '{current.canonical_name}' in candidate aliases",
                    )
                )
                continue

            if candidate_name_lower in current_aliases_lower:
                suggestions.append(
                    (
                        candidate,
                        0.9,
                        f"Candidate name '{candidate.canonical_name}' in current aliases",
                    )
                )
                continue

            # Check dedup_suggestions if available
            if current.dedup_suggestions:
                for suggestion in current.dedup_suggestions:
                    if suggestion.candidate_key == candidate.candidate_key:
                        suggestions.append(
                            (
                                candidate,
                                suggestion.similarity,
                                f"Embedding similarity: {suggestion.similarity:.0%}",
                            )
                        )
                        break

        # Sort by similarity score (highest first) and limit to top 5
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:5]
