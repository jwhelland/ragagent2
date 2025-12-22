"""User preferences and session state persistence."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field


class KeybindingPreferences(BaseModel):
    """Custom keybinding preferences."""

    # Action keys
    approve: str = "a"
    reject: str = "r"
    edit: str = "e"
    flag: str = "f"
    undo: str = "u"

    # Navigation keys (cannot be customized - reserved for Textual)
    # up, down, pageup, pagedown, home, end

    # Search and filter
    search: str = "/"
    next_result: str = "n"
    prev_result: str = "N"
    command_mode: str = ":"

    # Batch operations
    selection_mode: str = "v"
    batch_approve: str = "A"
    batch_reject: str = "R"
    merge: str = "M"

    # Misc
    help: str = "?"
    quit: str = "q"


class SessionState(BaseModel):
    """Last session state for resume functionality."""

    last_updated: datetime = Field(default_factory=datetime.now)
    filter_status: str = "pending"
    current_index: int = 0
    search_text: Optional[str] = None
    entity_type_filter: Optional[str] = None
    min_confidence: float = 0.0


class UIPreferences(BaseModel):
    """UI display preferences."""

    # Layout
    list_width_percent: int = 60  # Percentage of screen width for candidate list

    # Display options
    show_confidence_bar: bool = True
    show_mention_count: bool = True
    show_aliases_in_list: bool = True

    # Pagination
    candidates_per_page: int = 50

    # Notifications
    notification_duration: int = 3  # seconds
    milestone_notifications: bool = True


class UserPreferences(BaseModel):
    """Complete user preferences."""

    version: str = "1.0"
    keybindings: KeybindingPreferences = Field(default_factory=KeybindingPreferences)
    session_state: SessionState = Field(default_factory=SessionState)
    ui: UIPreferences = Field(default_factory=UIPreferences)


class PreferencesManager:
    """Manages user preferences and session state persistence."""

    DEFAULT_PREFS_DIR = Path.home() / ".ragagent"
    DEFAULT_PREFS_FILE = "review_preferences.json"

    def __init__(self, prefs_path: Optional[Path] = None) -> None:
        """Initialize preferences manager.

        Args:
            prefs_path: Optional path to preferences file. If None, uses default location.
        """
        if prefs_path is None:
            self.prefs_dir = self.DEFAULT_PREFS_DIR
            self.prefs_path = self.prefs_dir / self.DEFAULT_PREFS_FILE
        else:
            self.prefs_path = prefs_path
            self.prefs_dir = prefs_path.parent

        self.preferences = UserPreferences()
        self._ensure_prefs_dir()

    def _ensure_prefs_dir(self) -> None:
        """Ensure preferences directory exists."""
        self.prefs_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> UserPreferences:
        """Load preferences from file.

        Returns:
            UserPreferences object (either loaded or default)
        """
        if not self.prefs_path.exists():
            logger.info(f"No preferences file found at {self.prefs_path}, using defaults")
            return self.preferences

        try:
            with open(self.prefs_path) as f:
                data = json.load(f)

            self.preferences = UserPreferences.model_validate(data)
            logger.info(f"Loaded preferences from {self.prefs_path}")
            return self.preferences

        except Exception as e:
            logger.error(f"Failed to load preferences from {self.prefs_path}: {e}")
            logger.info("Using default preferences")
            return self.preferences

    def save(self, preferences: Optional[UserPreferences] = None) -> bool:
        """Save preferences to file.

        Args:
            preferences: Optional preferences to save. If None, saves current preferences.

        Returns:
            True if successful, False otherwise
        """
        if preferences is not None:
            self.preferences = preferences

        try:
            # Update last updated timestamp
            self.preferences.session_state.last_updated = datetime.now()

            # Write to file
            with open(self.prefs_path, "w") as f:
                json.dump(
                    self.preferences.model_dump(mode="json"),
                    f,
                    indent=2,
                    default=str,  # Handle datetime serialization
                )

            logger.debug(f"Saved preferences to {self.prefs_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save preferences to {self.prefs_path}: {e}")
            return False

    def update_session_state(
        self,
        filter_status: Optional[str] = None,
        current_index: Optional[int] = None,
        search_text: Optional[str] = None,
        entity_type_filter: Optional[str] = None,
        min_confidence: Optional[float] = None,
    ) -> None:
        """Update session state and save.

        Args:
            filter_status: Current filter status
            current_index: Current candidate index
            search_text: Current search text
            entity_type_filter: Current entity type filter
            min_confidence: Current minimum confidence filter
        """
        if filter_status is not None:
            self.preferences.session_state.filter_status = filter_status
        if current_index is not None:
            self.preferences.session_state.current_index = current_index
        if search_text is not None:
            self.preferences.session_state.search_text = search_text
        if entity_type_filter is not None:
            self.preferences.session_state.entity_type_filter = entity_type_filter
        if min_confidence is not None:
            self.preferences.session_state.min_confidence = min_confidence

        self.save()

    def reset_to_defaults(self) -> UserPreferences:
        """Reset preferences to defaults.

        Returns:
            Default UserPreferences object
        """
        self.preferences = UserPreferences()
        logger.info("Reset preferences to defaults")
        return self.preferences

    def get_keybinding(self, action: str) -> str:
        """Get keybinding for an action.

        Args:
            action: Action name (e.g., 'approve', 'reject')

        Returns:
            Key binding string (e.g., 'a', 'r')
        """
        try:
            return getattr(self.preferences.keybindings, action)
        except AttributeError:
            logger.warning(f"Unknown action: {action}, returning default")
            return ""

    def set_keybinding(self, action: str, key: str) -> bool:
        """Set keybinding for an action.

        Args:
            action: Action name
            key: Key to bind

        Returns:
            True if successful, False otherwise
        """
        try:
            setattr(self.preferences.keybindings, action, key)
            self.save()
            logger.info(f"Set keybinding for {action} to {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to set keybinding: {e}")
            return False

    def get_session_age_minutes(self) -> float:
        """Get age of last session in minutes.

        Returns:
            Minutes since last session update
        """
        now = datetime.now()
        delta = now - self.preferences.session_state.last_updated
        return delta.total_seconds() / 60

    def should_resume_session(self, max_age_minutes: int = 60) -> bool:
        """Check if session should be resumed based on age.

        Args:
            max_age_minutes: Maximum session age to consider for resume (default: 60 minutes)

        Returns:
            True if session should be resumed, False otherwise
        """
        age = self.get_session_age_minutes()
        return age < max_age_minutes
