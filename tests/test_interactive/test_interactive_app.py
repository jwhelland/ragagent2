#!/usr/bin/env python3
"""Test script to verify the interactive app can start (non-interactive mode)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from unittest.mock import MagicMock, patch

from src.curation.interactive import ReviewApp
from src.utils.config import Config


def test_app_initialization():
    """Test that the app initializes correctly."""
    print("Testing ReviewApp initialization...")

    # Load config
    with patch("src.utils.config.load_config") as mock_load:
        mock_config = MagicMock(spec=Config)
        mock_config.curation = MagicMock()
        mock_config.curation.batch_size = 10
        mock_config.curation.enable_audit_trail = False
        mock_config.database = MagicMock()
        mock_config.database.qdrant_location = ""
        mock_config.database.qdrant_api_key = "test"
        mock_load.return_value = mock_config

        # Create app instance
        with (
            patch("src.curation.interactive.app.Neo4jManager"),
            patch("src.curation.interactive.app.NormalizationTable"),
        ):
            app = ReviewApp()
            print("✓ ReviewApp instance created")

    # Verify reactive attributes
    assert hasattr(app, "candidates"), "Missing candidates attribute"
    assert hasattr(app, "current_index"), "Missing current_index attribute"
    assert hasattr(app, "approved_count"), "Missing approved_count attribute"
    assert hasattr(app, "rejected_count"), "Missing rejected_count attribute"
    print("✓ All reactive attributes present")

    # Verify action methods
    assert hasattr(app, "action_approve_current"), "Missing approve action"
    assert hasattr(app, "action_reject_current"), "Missing reject action"
    assert hasattr(app, "action_undo_last"), "Missing undo action"
    assert hasattr(app, "action_flag_current"), "Missing flag action"
    print("✓ All action methods present")

    # Verify keybindings
    assert len(app.BINDINGS) > 0, "No keybindings defined"
    print(f"✓ {len(app.BINDINGS)} keybindings configured")

    # Verify session tracker
    assert app.session_tracker is not None, "Session tracker not initialized"
    print("✓ Session tracker initialized")

    print("\n✅ All initialization tests passed!")
    print("\nReady to launch interactive mode.")
    print("\nTo start the interactive TUI, run:")
    print("  uv run ragagent-review-interactive")
    print("\nOr:")
    print("  uv run python scripts/review_entities_interactive.py")


if __name__ == "__main__":
    try:
        test_app_initialization()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
