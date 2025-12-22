# Command Mode Quick Demo

## How to Access

Press `:` (colon) from the main review screen to open command mode.

## Demo Workflow

### 1. Filter by Entity Type
```
Press: :
Type:  filter type=SYSTEM
Press: Enter
Result: Shows only SYSTEM entities
```

### 2. Filter with Multiple Criteria
```
Press: :
Type:  filter type=COMPONENT status=pending confidence=0.8
Press: Enter
Result: Shows pending COMPONENT entities with confidence ≥ 0.8
```

### 3. Batch Approve High Confidence
```
Press: :
Type:  batch-approve >0.9
Press: Enter
Result: Shows confirmation modal for candidates with confidence > 0.9
```

### 4. Batch Reject Low Confidence
```
Press: :
Type:  batch-reject <0.5
Press: Enter
Result: Shows confirmation modal for candidates with confidence < 0.5
```

### 5. Get Help
```
Press: :
Type:  help
Press: Enter
Result: Shows detailed help text with examples
```

### 6. Use Command History
```
Press: :
Press: ↑ (up arrow)
Result: Shows previous command
Press: ↓ (down arrow)
Result: Shows next command (or clears if at end)
```

### 7. Quit Application
```
Press: :
Type:  quit
Press: Enter
Result: Shows session summary and quits
```

## Visual Layout

```
╭──────────────────────────────────────────────────────────────╮
│                       Command Mode                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ : filter type=SYSTEM confidence=0.8_                        │
│                                                              │
│ Available commands:                                          │
│   filter [key=value...]  - Apply filters                    │
│   sort <field>           - Sort by field                    │
│   batch-approve <threshold> - Approve high confidence       │
│   batch-reject <threshold>  - Reject low confidence         │
│   export <file>          - Export results                   │
│   help                   - Show help                        │
│   quit, q                - Quit application                 │
│                                                              │
│ Use ↑↓ arrows to navigate history                          │
│                                                              │
│ ❌ Error: [any errors show here]                            │
│                                                              │
│              [Execute]  [Cancel]                            │
╰──────────────────────────────────────────────────────────────╯
```

## Error Examples

### Invalid Entity Type
```
Input:  :filter type=INVALID
Error:  Invalid entity type 'INVALID'. Valid: SYSTEM, SUBSYSTEM, COMPONENT, ...
```

### Invalid Confidence Range
```
Input:  :filter confidence=1.5
Error:  Confidence must be between 0.0 and 1.0
```

### Invalid Threshold Format
```
Input:  :batch-approve 0.9
Error:  Invalid threshold format '0.9'. Expected format: >0.9, >=0.8, etc.
```

### Missing Required Argument
```
Input:  :export
Error:  export requires a filename (e.g., export results.json)
```

## Tips

1. **Start Simple**: Try `:help` first to see all commands
2. **Use History**: Press ↑ to recall and modify previous commands
3. **Check Errors**: Parse errors keep the modal open for correction
4. **Combine Operations**: Filter first, then batch operate on filtered results
5. **Quick Quit**: `:q` is faster than pressing the 'q' key on main screen

## Comparison to Interactive Search

| Feature | Command Mode (`:`) | Search Modal (`/`) |
|---------|-------------------|-------------------|
| Speed | ⚡ Fast | 🐢 Slower |
| Learning Curve | 📚 Higher | ✅ Easy |
| History | ✅ Yes | ❌ No |
| Batch Ops | ✅ Yes | ❌ No |
| Best For | Power users | First-time users |

## Try It Now!

```bash
# Start the interactive TUI
uv run ragagent-review-interactive

# Press ':' when the app loads
# Type 'help' and press Enter
# Explore the available commands!
```

---

**More Info:**
- [Full Command Guide](COMMAND_MODE_GUIDE.md)
- [Implementation Summary](TASK_3.5.9_SUMMARY.md)
- [Interactive TUI Guide](INTERACTIVE_TUI_GUIDE.md)
