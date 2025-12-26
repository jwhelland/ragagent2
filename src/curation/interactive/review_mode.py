"""Review mode definitions for entity vs relationship curation."""

from enum import Enum
from typing import Dict

from pydantic import BaseModel


class ReviewMode(str, Enum):
    """Review mode for the interactive TUI."""

    ENTITY = "ENTITY"
    RELATIONSHIP = "RELATIONSHIP"


class ModeConfig(BaseModel):
    """Configuration for a review mode."""

    display_name: str
    icon: str
    keybindings_hint: str


# Mode configuration mapping
_MODE_CONFIGS: Dict[ReviewMode, ModeConfig] = {
    ReviewMode.ENTITY: ModeConfig(
        display_name="Entities",
        icon="🔷",
        keybindings_hint="a:approve r:reject e:edit m:merge",
    ),
    ReviewMode.RELATIONSHIP: ModeConfig(
        display_name="Relationships",
        icon="🔗",
        keybindings_hint="a:approve r:reject",
    ),
}


def get_mode_config(mode: ReviewMode) -> ModeConfig:
    """Get configuration for a review mode.

    Args:
        mode: Review mode

    Returns:
        Configuration for the mode
    """
    return _MODE_CONFIGS[mode]
