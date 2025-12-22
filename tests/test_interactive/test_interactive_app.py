#!/usr/bin/env python3
"""Test script to verify the interactive app can start (non-interactive mode)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.curation.interactive import ReviewApp
from src.utils.config import load_config


def test_app_initialization():
    """Test that the app initializes correctly."""
    print("Testing ReviewApp initialization...")

    # Load config
    config = load_config()
    print(f"✓ Configuration loaded")

    # Create app instance
    app = ReviewApp()
    print(f"✓ ReviewApp instance created")

    # Verify reactive attributes
    assert hasattr(app, 'candidates'), "Missing candidates attribute"
    assert hasattr(app, 'current_index'), "Missing current_index attribute"
    assert hasattr(app, 'approved_count'), "Missing approved_count attribute"
    assert hasattr(app, 'rejected_count'), "Missing rejected_count attribute"
    print(f"✓ All reactive attributes present")

    # Verify action methods
    assert hasattr(app, 'action_approve_current'), "Missing approve action"
    assert hasattr(app, 'action_reject_current'), "Missing reject action"
    assert hasattr(app, 'action_undo_last'), "Missing undo action"
    assert hasattr(app, 'action_flag_current'), "Missing flag action"
    print(f"✓ All action methods present")

    # Verify keybindings
    assert len(app.BINDINGS) > 0, "No keybindings defined"
    print(f"✓ {len(app.BINDINGS)} keybindings configured")

    # Verify session tracker
    assert app.session_tracker is not None, "Session tracker not initialized"
    print(f"✓ Session tracker initialized")

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
