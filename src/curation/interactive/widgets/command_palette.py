"""Unified command palette for search, filters, and curation actions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Union

from loguru import logger
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static

from src.storage.neo4j_manager import Neo4jManager
from src.storage.schemas import CandidateStatus, EntityCandidate, EntityType, EntityStatus
from src.utils.config import Config


class PaletteResult(ListItem):
    """Base class for results in the command palette."""
    def __init__(self, item: Any, label: str, sublabel: str = "", metadata: Dict[str, Any] | None = None) -> None:
        super().__init__()
        self.item = item
        self.label = label
        self.sublabel = sublabel
        self.metadata = metadata or {}

    def compose(self) -> ComposeResult:
        yield Label(f"[bold]{self.label}[/bold]")
        if self.sublabel:
            yield Label(f"[italic]{self.sublabel}[/italic]", classes="sublabel")


class CandidateResult(PaletteResult):
    """A candidate result in the palette."""
    def __init__(self, candidate: EntityCandidate, index: int) -> None:
        super().__init__(
            item=candidate,
            label=candidate.canonical_name,
            sublabel=f"Candidate [{index+1}] ({candidate.candidate_type.value}) - {candidate.status.value}",
            metadata={"type": "candidate", "index": index}
        )


class EntityResult(PaletteResult):
    """An approved entity result in the palette."""
    def __init__(self, entity_data: Dict[str, Any]) -> None:
        super().__init__(
            item=entity_data,
            label=entity_data.get("canonical_name", "Unknown"),
            sublabel=f"Entity ({entity_data.get('entity_type', 'CONCEPT')}) - Approved",
            metadata={"type": "entity", "id": entity_data.get("id")}
        )


class CommandResult(PaletteResult):
    """A command action result in the palette."""
    def __init__(self, command: str, description: str) -> None:
        super().__init__(
            item=command,
            label=f":{command}",
            sublabel=description,
            metadata={"type": "command"}
        )


class CommandPalette(ModalScreen[Optional[Dict[str, Any]]]):
    """Unified palette for search, filter, entity lookup, and actions."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "select_item", "Select", show=True),
    ]

    CSS = """
    CommandPalette {
        align: center top;
        padding-top: 4;
    }

    #palette-container {
        width: 80;
        max-height: 30;
        background: $surface;
        border: thick $primary;
        padding: 0;
    }

    #palette-input {
        width: 100%;
        border: none;
        background: $surface;
        padding: 1 2;
    }

    #results-list {
        height: auto;
        max-height: 20;
        border-top: solid $primary;
        background: $surface;
    }

    .sublabel {
        color: $text-muted;
        margin-left: 2;
        font-size: 0.9em;
    }

    #palette-footer {
        height: auto;
        padding: 0 2;
        background: $surface-darken-1;
        color: $text-muted;
        font-size: 0.8em;
    }
    """

    def __init__(
        self, 
        candidates: List[EntityCandidate], 
        config: Config,
        initial_text: str = ""
    ) -> None:
        super().__init__()
        self.candidates = candidates
        self.config = config
        self.initial_text = initial_text
        self._neo4j_manager: Optional[Neo4jManager] = None

    def compose(self) -> ComposeResult:
        with Container(id="palette-container"):
            yield Input(
                placeholder="Search candidates, entities, or type :command...",
                id="palette-input",
                value=self.initial_text
            )
            yield ListView(id="results-list")
            with Horizontal(id="palette-footer"):
                yield Label("↑↓ to navigate • Enter to select • Esc to close")

    def on_mount(self) -> None:
        self._neo4j_manager = Neo4jManager(self.config.database)
        self._neo4j_manager.connect()
        self.query_one("#palette-input").focus()
        if self.initial_text:
            self._update_results(self.initial_text)

    def on_unmount(self) -> None:
        if self._neo4j_manager:
            self._neo4j_manager.close()

    @on(Input.Changed, "#palette-input")
    def on_input_changed(self, event: Input.Changed) -> None:
        self._update_results(event.value)

    def _update_results(self, text: str) -> None:
        text = text.strip()
        list_view = self.query_one("#results-list", ListView)
        list_view.clear()

        if not text:
            # Show default commands when empty
            self._add_default_commands(list_view)
            return

        if text.startswith(":"):
            self._filter_commands(text[1:], list_view)
        else:
            self._search_candidates(text, list_view)
            self._search_entities_debounced(text)

    def _add_default_commands(self, list_view: ListView) -> None:
        commands = [
            ("approve", "Approve current candidate"),
            ("reject", "Reject current candidate"),
            ("edit", "Edit current candidate"),
            ("merge", "Merge selected candidates (multi-selection)"),
            ("merge-into", "Merge current candidate into existing entity"),
            ("filter", "Set status/type filters"),
            ("undo", "Undo last operation"),
            ("quit", "Exit application"),
        ]
        for cmd, desc in commands:
            list_view.append(CommandResult(cmd, desc))

    def _filter_commands(self, query: str, list_view: ListView) -> None:
        commands = [
            ("approve", "Approve current candidate"),
            ("reject", "Reject current candidate"),
            ("edit", "Edit current candidate"),
            ("merge", "Merge selected candidates (multi-selection)"),
            ("merge-into", "Merge current candidate into existing entity"),
            ("filter", "Set status/type filters"),
            ("undo", "Undo last operation"),
            ("quit", "Exit application"),
        ]
        for cmd, desc in commands:
            if query.lower() in cmd.lower():
                list_view.append(CommandResult(cmd, desc))

    def _search_candidates(self, query: str, list_view: ListView) -> None:
        query = query.lower()
        matches = []
        for i, c in enumerate(self.candidates):
            if query in c.canonical_name.lower() or any(query in a.lower() for a in c.aliases):
                matches.append(CandidateResult(c, i))
            
            if len(matches) >= 10:
                break
        
        for m in matches:
            list_view.append(m)

    @work(thread=True, exclusive=True)
    def _search_entities_debounced(self, query: str) -> None:
        if len(query) < 2 or query.startswith(":"):
            return

        import time
        time.sleep(0.1)  # Minimal debounce

        if not self._neo4j_manager:
            return

        try:
            results = self._neo4j_manager.search_entities(
                query=query,
                status=EntityStatus.APPROVED,
                limit=10
            )
            self.app.call_from_thread(self._append_entity_results, results)
        except Exception as e:
            logger.error(f"Palette entity search failed: {e}")

    def _append_entity_results(self, results: List[Dict[str, Any]]) -> None:
        list_view = self.query_one("#results-list", ListView)
        # Avoid duplicates if they already exist from candidates (though unlikely to overlap perfectly)
        for res in results:
            list_view.append(EntityResult(res))

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_cursor_up(self) -> None:
        self.query_one("#results-list", ListView).action_cursor_up()

    def action_cursor_down(self) -> None:
        self.query_one("#results-list", ListView).action_cursor_down()

    def action_select_item(self) -> None:
        list_view = self.query_one("#results-list", ListView)
        if list_view.highlighted_child:
            item = list_view.highlighted_child
            if isinstance(item, PaletteResult):
                self.dismiss({
                    "type": item.metadata["type"],
                    "item": item.item,
                    "metadata": item.metadata
                })

    @on(ListView.Selected)
    def on_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, PaletteResult):
            self.dismiss({
                "type": event.item.metadata["type"],
                "item": event.item.item,
                "metadata": event.item.metadata
            })
