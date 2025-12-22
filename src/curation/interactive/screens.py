"""Screen definitions for the interactive review application."""

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer, Header, Static


class ReviewScreen(Screen):
    """Main review screen for entity candidates.

    This screen will contain the candidate list, detail panel, and status bar.
    Will be fully implemented in Tasks 3.5.2-3.5.7.
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("?", "help", "Help"),
    ]

    def compose(self) -> ComposeResult:
        """Create child widgets for the review screen."""
        yield Header()
        yield Static("Review Screen - Coming in Task 3.5.2+", classes="placeholder")
        yield Footer()

    def on_mount(self) -> None:
        """Handle screen mount event."""
        self.app.title = "Entity Candidate Review"
        self.app.sub_title = "Interactive Mode"


class HelpScreen(Screen):
    """Help screen showing keyboard shortcuts and commands.

    Will be implemented in Task 3.5.3+.
    """

    BINDINGS = [
        ("escape", "pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        """Create child widgets for the help screen."""
        yield Header()
        yield Static(
            """
            Keyboard Shortcuts (Coming Soon):

            Navigation:
            ↑/↓   - Move selection up/down
            PgUp/PgDn - Page up/down
            Home/End - Go to first/last

            Actions:
            a - Approve candidate
            r - Reject candidate
            e - Edit candidate
            f - Flag for later
            u - Undo last action

            Other:
            / - Search
            ? - Show this help
            q - Quit
            """,
            classes="help-content",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Handle screen mount event."""
        self.app.title = "Help"
        self.app.sub_title = "Keyboard Shortcuts"
