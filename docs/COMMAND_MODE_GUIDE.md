# Vim-Style Command Mode Guide

## Overview

The interactive review TUI now supports vim-style command mode for power users. Press `:` (colon) to open the command modal and enter commands for advanced operations.

## Opening Command Mode

- **Keyboard**: Press `:` (colon key)
- **Footer**: The `:` binding appears in the footer as "Command"

## Available Commands

### Filter Commands

Apply filters to the candidate list:

```
:filter type=SYSTEM
:filter status=pending
:filter type=COMPONENT status=approved
:filter confidence=0.8
:filter type=SYSTEM status=pending confidence=0.9
```

**Filter Parameters:**
- `type=<TYPE>` - Filter by entity type (SYSTEM, COMPONENT, SUBSYSTEM, etc.)
- `status=<STATUS>` - Filter by status (pending, approved, rejected, all)
- `confidence=<0.0-1.0>` - Minimum confidence threshold

### Sort Commands

Sort the candidate list by field (feature coming soon):

```
:sort confidence
:sort name
:sort type
:sort mentions
```

**Valid Sort Fields:**
- `confidence` - Sort by confidence score
- `name` - Sort alphabetically by name
- `type` - Sort by entity type
- `mentions` - Sort by mention count

### Batch Operations

Approve or reject candidates based on confidence thresholds:

```
:batch-approve >0.9
:batch-approve >=0.85
:batch-reject <0.5
:batch-reject <=0.6
```

**Operators:**
- `>` - Greater than
- `>=` - Greater than or equal
- `<` - Less than
- `<=` - Less than or equal
- `==` - Equal to

**Behavior:**
- Commands filter PENDING candidates by confidence
- Shows confirmation modal before executing
- Supports undo with `u` key after operation

### Export Commands

Export candidate data to a file (feature coming soon):

```
:export results.json
:export flagged.json
:export approved_candidates.json
```

### Utility Commands

```
:help         # Show command help
:quit         # Quit application (same as 'q' key)
:q            # Short alias for quit
```

## Command History

Command mode maintains a history of previous commands for easy re-use:

- **↑ (Up Arrow)** - Navigate to previous command
- **↓ (Down Arrow)** - Navigate to next command
- History persists during the session (max 100 commands)
- Duplicate consecutive commands are not added to history

## Examples

### Example 1: Review High-Confidence Systems

1. Press `:` to open command mode
2. Type: `filter type=SYSTEM confidence=0.8`
3. Press Enter
4. Review filtered candidates
5. Optionally: `:batch-approve >0.9` to bulk approve high confidence

### Example 2: Clean Up Low-Confidence Candidates

1. Press `:` to open command mode
2. Type: `filter confidence=0.0`
3. Review low-confidence candidates
4. Press `:` again
5. Type: `batch-reject <0.5`
6. Confirm rejection in modal

### Example 3: Review Specific Type

1. Press `:` to open command mode
2. Type: `filter type=COMPONENT status=pending`
3. Press Enter
4. Review all pending component candidates
5. Use `a`/`r`/`e` keys for individual actions

## Tips

- Command mode is case-insensitive (`:Filter` = `:filter`)
- Tab completion is not yet implemented (coming soon)
- Press `Esc` to cancel command entry
- Command history is cleared when you quit the application
- Commands are validated before execution - you'll see error messages for invalid input

## Comparison to Search Modal

| Feature | Search Modal (`/`) | Command Mode (`:`) |
|---------|-------------------|-------------------|
| **Purpose** | Interactive form-based search | Text-based vim-style commands |
| **Input** | Form fields with dropdowns | Text command with arguments |
| **History** | No | Yes (↑↓ arrows) |
| **Speed** | Slower (requires mouse/tab) | Faster for power users |
| **Learning Curve** | Easy (visual) | Steeper (requires memorizing syntax) |
| **Best For** | New users, complex filters | Power users, quick operations |

## Keyboard Shortcuts Summary

| Key | Action |
|-----|--------|
| `:` | Open command mode |
| `Enter` | Execute command |
| `Esc` | Cancel command |
| `↑` | Previous command in history |
| `↓` | Next command in history |

## Implementation Details

**Architecture:**
- `command_parser.py` - Command parsing and validation
- `command_modal.py` - TUI modal for command input
- `CommandHistory` - History management (max 100 entries)
- `CommandParser` - Parse and validate command syntax

**Testing:**
- 27 unit tests covering parser and history
- See `tests/test_interactive/test_command_parser.py`

## Future Enhancements (Task 3.5.10)

- Tab completion for commands and arguments
- Command aliases (`:f` for `:filter`)
- Saved command presets
- Export functionality implementation
- Sort functionality implementation
- Range-based batch operations (`:10,20 approve`)
- Regex search in command mode

---

**Related Documentation:**
- [Interactive TUI Guide](INTERACTIVE_TUI_GUIDE.md)
- [Phase 3.5 Enhanced Review](plans/phase-3.5-enhanced-review.md)
- [Developer Tasks](plans/developer-tasks.md)
