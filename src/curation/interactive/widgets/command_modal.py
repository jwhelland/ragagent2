"""Command mode modal for vim-style commands."""

from typing import Callable, Optional

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from src.curation.interactive.command_parser import CommandHistory, CommandParser, ParsedCommand


class CommandModalScreen(ModalScreen[Optional[ParsedCommand]]):
    """Modal screen for entering vim-style commands."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("enter", "execute", "Execute", show=True),
        Binding("up", "history_prev", "Previous", show=False),
        Binding("down", "history_next", "Next", show=False),
    ]

    CSS = """
    CommandModalScreen {
        align: center middle;
    }

    #command-dialog {
        width: 80;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    #command-dialog-title {
        width: 100%;
        content-align: center middle;
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    #command-input-container {
        width: 100%;
        margin-bottom: 1;
    }

    #command-input-label {
        width: auto;
        color: $accent;
        text-style: bold;
    }

    #command-input {
        width: 1fr;
    }

    #command-help {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    .help-line {
        color: $text-muted;
        text-style: italic;
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

    #command-error {
        width: 100%;
        height: auto;
        color: $error;
        margin-bottom: 1;
    }
    """

    def __init__(
        self,
        history: CommandHistory,
        on_execute: Optional[Callable[[ParsedCommand], None]] = None,
    ) -> None:
        """Initialize command modal.

        Args:
            history: Command history manager
            on_execute: Optional callback to call when command is executed
        """
        super().__init__()
        self.history = history
        self.on_execute_callback = on_execute
        self.parser = CommandParser()

    def compose(self) -> ComposeResult:
        """Create modal dialog widgets."""
        with Container(id="command-dialog"):
            yield Label("Command Mode", id="command-dialog-title")

            # Command input with ':' prefix
            with Horizontal(id="command-input-container"):
                yield Label(":", id="command-input-label")
                yield Input(
                    placeholder="Enter command (e.g., filter type=SYSTEM, batch-approve >0.9)",
                    id="command-input",
                )

            # Help text
            with Vertical(id="command-help"):
                yield Static("Available commands:", classes="help-line")
                yield Static("  filter [key=value...]  - Apply filters", classes="help-line")
                yield Static("  sort <field>           - Sort by field", classes="help-line")
                yield Static(
                    "  batch-approve <threshold> - Approve high confidence", classes="help-line"
                )
                yield Static(
                    "  batch-reject <threshold>  - Reject low confidence", classes="help-line"
                )
                yield Static("  export <file>          - Export results", classes="help-line")
                yield Static("  help                   - Show help", classes="help-line")
                yield Static("  quit, q                - Quit application", classes="help-line")
                yield Static("\nUse ↑↓ arrows to navigate history", classes="help-line")

            # Error display (hidden by default)
            yield Static("", id="command-error")

            # Buttons
            with Vertical(id="button-container"):
                with Horizontal():
                    yield Button("Execute", variant="primary", id="execute-button")
                    yield Button("Cancel", variant="default", id="cancel-button")

    def on_mount(self) -> None:
        """Focus command input on mount."""
        command_input = self.query_one("#command-input", Input)
        command_input.focus()
        # Reset history index
        self.history.reset_index()

    def action_cancel(self) -> None:
        """Handle cancel action (Esc key)."""
        self.dismiss(None)

    def action_execute(self) -> None:
        """Handle execute action (Enter key)."""
        self._execute_command()

    def action_history_prev(self) -> None:
        """Navigate to previous command in history (↑ key)."""
        prev_command = self.history.previous()
        if prev_command is not None:
            command_input = self.query_one("#command-input", Input)
            command_input.value = prev_command

    def action_history_next(self) -> None:
        """Navigate to next command in history (↓ key)."""
        next_command = self.history.next()
        if next_command is not None:
            command_input = self.query_one("#command-input", Input)
            command_input.value = next_command

    @on(Button.Pressed, "#execute-button")
    def on_execute_button(self) -> None:
        """Handle execute button click."""
        self._execute_command()

    @on(Button.Pressed, "#cancel-button")
    def on_cancel_button(self) -> None:
        """Handle cancel button click."""
        self.dismiss(None)

    def _show_error(self, message: str) -> None:
        """Display an error message.

        Args:
            message: Error message to display
        """
        error_label = self.query_one("#command-error", Static)
        error_label.update(f"❌ Error: {message}")

    def _clear_error(self) -> None:
        """Clear any displayed error message."""
        error_label = self.query_one("#command-error", Static)
        error_label.update("")

    def _execute_command(self) -> None:
        """Execute the current command."""
        command_input = self.query_one("#command-input", Input)
        command_str = command_input.value.strip()

        # Clear any previous error
        self._clear_error()

        if not command_str:
            self._show_error("Empty command")
            return

        try:
            # Parse command
            parsed = self.parser.parse(command_str)

            # Add to history
            self.history.add(command_str)

            # Call the execute callback if provided
            if self.on_execute_callback:
                self.on_execute_callback(parsed)

            # Dismiss modal with parsed command
            self.dismiss(parsed)

        except ValueError as e:
            # Show error and keep modal open
            self._show_error(str(e))
            command_input.focus()
