# Interactive TUI Testing Guide

## Current Status

**Completed Features (Tasks 3.5.1 - 3.5.4):**
- ✅ Interactive TUI Foundation
- ✅ Candidate List Widget with keyboard navigation
- ✅ Detail Panel showing full candidate information
- ✅ Single-key actions (approve, reject, undo, flag)

**Available Data:**
- **Total candidates:** 386
- **Pending review:** 338
- **Already approved:** 48
- **Already rejected:** 0

## How to Launch

```bash
# Option 1: Using the CLI command
uv run ragagent-review-interactive

# Option 2: Direct script execution
uv run python scripts/review_entities_interactive.py
```

## Keyboard Shortcuts

### Navigation
- **↑/↓** - Move selection up/down
- **PgUp/PgDn** - Page up/down (10 items)
- **Home/End** - Jump to first/last candidate

### Actions
- **a** - Approve current candidate
- **r** - Reject current candidate
- **f** - Flag for later review (placeholder)
- **u** - Undo last operation
- **e** - Edit candidate (coming in Task 3.5.5)

### Other
- **/** - Search (coming in Task 3.5.6)
- **?** - Help
- **q** - Quit

## What to Test

### 1. Basic Navigation
- [ ] Launch the app
- [ ] Use arrow keys to move up/down through the list
- [ ] Try PgUp/PgDn for faster navigation
- [ ] Press Home to jump to first, End to jump to last
- [ ] Verify the detail panel updates as you navigate

### 2. Visual Elements
- [ ] Candidate list shows: index, name, type, confidence, mentions
- [ ] Confidence scores are color-coded (green ≥0.9, yellow ≥0.7, red <0.7)
- [ ] Selected row has a `►` indicator
- [ ] Detail panel shows all candidate information
- [ ] Status bar shows keybindings at bottom

### 3. Approve Action
- [ ] Navigate to a PENDING candidate
- [ ] Press `a` to approve
- [ ] Should see notification: "✓ Approved: [name] (ID: ...)"
- [ ] Candidate should be removed from pending list
- [ ] List should reload automatically

### 4. Reject Action
- [ ] Navigate to a PENDING candidate
- [ ] Press `r` to reject
- [ ] Should see notification: "✗ Rejected: [name]"
- [ ] Candidate should be removed from pending list
- [ ] List should reload automatically

### 5. Undo Action
- [ ] After approving or rejecting a candidate
- [ ] Press `u` to undo
- [ ] Should see notification: "↶ Undo successful"
- [ ] Candidate should return to pending list
- [ ] List should reload automatically

### 6. Error Handling
- [ ] Try to approve an already-approved candidate
- [ ] Should see warning: "Already approved"
- [ ] Try to reject an already-rejected candidate
- [ ] Should see warning: "Already rejected"

### 7. Status Bar and Footer
- [ ] Title should show "Entity Candidate Review"
- [ ] Subtitle should show status filter and total count
- [ ] Footer should show available keybindings

### 8. Performance
- [ ] Navigation should feel responsive (<50ms)
- [ ] Actions should complete quickly (<200ms for notification)
- [ ] No UI freezing during approve/reject operations

## Expected Layout

```
╭─────────────────────────────────────────────────────────────────────╮
│ Entity Candidate Review                         Status: pending | 338│
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ CANDIDATE LIST (60%)        │ DETAIL PANEL (40%)                    │
│                              │                                        │
│ ► [  1] power_subsystem ... │ Power Subsystem                       │
│   [  2] thermal_control ... │ ──────────────                        │
│   [  3] attitude_determ ... │ Core Information:                     │
│   ...                        │   Type: SYSTEM                        │
│                              │   Status: PENDING                     │
│                              │   Confidence: 0.950                   │
│                              │   Mentions: 45                        │
│                              │                                        │
│                              │ Aliases:                              │
│                              │   • EPS                               │
│                              │   • Electrical Power System           │
│                              │                                        │
│                              │ Description:                          │
│                              │   Manages electrical power...         │
│                              │                                        │
├─────────────────────────────────────────────────────────────────────┤
│ a Approve  r Reject  u Undo  f Flag  ? Help  q Quit                 │
╰─────────────────────────────────────────────────────────────────────╯
```

## Known Limitations (Future Tasks)

- ⏳ Edit modal not yet implemented (Task 3.5.5)
- ⏳ Search/filter not yet implemented (Task 3.5.6)
- ⏳ Progress tracking not yet visible (Task 3.5.7)
- ⏳ Batch operations not yet implemented (Task 3.5.8)

## Troubleshooting

### App won't start
```bash
# Check if Neo4j is running
docker-compose ps

# Restart if needed
docker-compose restart neo4j

# Verify connectivity
curl http://localhost:7474
```

### No candidates showing
```bash
# Check if you have candidates
uv run python -c "from src.storage.neo4j_manager import Neo4jManager; from src.utils.config import load_config; config = load_config(); mgr = Neo4jManager(config.database); mgr.connect(); print(mgr.get_entity_candidate_statistics()); mgr.close()"
```

### Error messages
- All errors should show as notifications in the UI
- Check `logs/ragagent2.log` for detailed error logs

## Feedback

After testing, note:
- What works well?
- What feels slow or unresponsive?
- Any bugs or unexpected behavior?
- Suggestions for improvements?

---

**Ready to test!** Run: `uv run ragagent-review-interactive`
