# Implementation Plan: Enhanced Interactive CLI for Entity Review (Phase 3.5, Option 1)

## Executive Summary

Building an interactive TUI (Text User Interface) using the `textual` framework to replace the current command-based review interface. This will enable keyboard-driven navigation, single-key actions, real-time progress tracking, and efficient batch operations for reviewing entity candidates at scale.

**Target:** Transform the review workflow from ~10 candidates/hour to 30+ candidates/hour
**Timeline:** 7 days (following recommended implementation order from phase plan)
**Technology:** Textual + Rich (already used) + Typer (keep for backward compatibility)

## Current State Analysis

### Existing Infrastructure (Well-Built Foundation)

**Services (Ready to Reuse):**
- `EntityCurationService` - Approve/reject/merge/edit/undo operations with checkpoint/rollback
- `BatchCurationService` - Batch operations with preview and dry-run support
- `Neo4jCandidateStore` - Data access layer for entity candidates with filtering/sorting
- `NormalizationTable` - Canonical name management
- Undo stack with full rollback support

**Data Models:**
- `EntityCandidate` (Pydantic) - Complete with status, confidence, aliases, provenance
- `CandidateStatus` enum - PENDING/APPROVED/REJECTED
- Well-defined EntityType and RelationshipType enums

**Current CLI:**
- Located: `src/curation/review_interface.py` (682 lines)
- Uses: Typer + Rich for table output
- Commands: queue, show, search, stats, approve, reject, edit, merge, undo, batch-*
- Pattern: Each command creates fresh service instances, performs operation, exits

### Gap Analysis

**What's Missing for Interactive TUI:**
1. ❌ Textual dependency (not in pyproject.toml)
2. ❌ Persistent application state across actions
3. ❌ Keyboard navigation between candidates
4. ❌ Real-time UI updates after actions
5. ❌ Session tracking (progress, statistics)
6. ❌ Interactive widgets (scrollable list, detail panel, modals)
7. ❌ Single-key action handlers

## Architecture Design

### Component Structure

```
src/curation/interactive/
├── __init__.py
├── app.py                    # Main Textual application
├── screens.py                # ReviewScreen definition
├── session_tracker.py        # Progress/statistics tracking
├── queue_manager.py          # Filtering, sorting, pagination
├── keybindings.py            # Keyboard shortcut definitions
└── widgets/
    ├── __init__.py
    ├── candidate_list.py     # Scrollable list widget
    ├── detail_panel.py       # Candidate detail view
    ├── status_bar.py         # Status and progress bar
    ├── edit_modal.py         # Edit form modal
    ├── search_modal.py       # Search overlay
    └── help_modal.py         # Help overlay
```

### Data Flow

```
User Keyboard Input
    ↓
Textual Event Handler (keybinding)
    ↓
Application State Update (Reactive)
    ↓
Service Call (EntityCurationService via @work thread)
    ↓
Neo4j Database Update
    ↓
UI Widget Update (reactive)
    ↓
Visual Feedback to User
```

### State Management (VALIDATED APPROACH)

**Use Textual's Reactive Attributes:**

```python
class ReviewApp(App):
    # Reactive attributes (auto-update UI on change)
    current_index: reactive[int] = reactive(0)
    candidates: reactive[list[EntityCandidate]] = reactive([])
    filter_status: reactive[str] = reactive("pending")
    approved_count: reactive[int] = reactive(0)
    rejected_count: reactive[int] = reactive(0)

    def watch_filter_status(self, old: str, new: str) -> None:
        """Auto-reload candidates when filter changes."""
        self.reload_candidates()
```

**SessionTracker class:**
- Start time
- Actions performed (by type)
- Velocity calculation (candidates/minute)
- Time remaining estimation

**QueueManager class:**
- Fetch candidates with filters
- Handle pagination (50 items/page)
- Apply sorting
- Manage search results

## Implementation Plan (7 Days)

### Days 1-2: Foundation + Navigation (Tasks 3.5.1, 3.5.2) ⚠️ CRITICAL

#### Task 3.5.1: Interactive TUI Foundation
**Goal:** Basic Textual app that launches and displays content

**Steps:**
1. Add `textual>=0.47.0` and `pytest-asyncio>=0.21.0` to pyproject.toml
2. Run `uv sync` to install
3. Create directory structure: `src/curation/interactive/`
4. Create `app.py` with basic Textual app (with reactive attributes)
5. Create `session_tracker.py` with SessionTracker
6. Create `screens.py` with ReviewScreen
7. Create entry point script: `scripts/review_entities_interactive.py`
8. Add to pyproject.toml: `ragagent-review-interactive`
9. Test basic launch and quit
10. Implement `@work(thread=True)` pattern for service calls

**Deliverable:** App launches, shows placeholder, quits cleanly with 'q'

#### Task 3.5.2: Candidate List Widget ⚠️ CRITICAL
**Goal:** Scrollable list with keyboard navigation

**Steps:**
1. Create `widgets/candidate_list.py` with CandidateListWidget
2. Integrate with Neo4jCandidateStore via `@work(thread=True)`
3. Implement row rendering with formatting (confidence, type, name, mentions)
4. Add keyboard navigation (↑↓, PgUp/PgDn, Home/End)
5. Implement selection highlighting
6. Wire up to App reactive attributes for selected index
7. Add pagination support (50 items/page, fetch more on scroll)
8. Test with real Neo4j data

**Deliverable:** Scrollable list showing candidates, responsive navigation (<50ms)

---

### Days 3-4: Actions + Details (Tasks 3.5.3, 3.5.4, 3.5.5)

#### Task 3.5.3: Detail Panel Widget
**Goal:** Show full candidate information in detail pane

**Steps:**
1. Create `widgets/detail_panel.py` with DetailPanel
2. Display: canonical_name, type, status, confidence, aliases, description
3. Show source documents and chunk IDs
4. Display provenance events
5. Add dynamic resizing
6. Wire up to selection changes via reactive attributes
7. Test layout with long text

**Deliverable:** Detail panel updates when selection changes

#### Task 3.5.4: Single-Key Actions ⚠️ CRITICAL
**Goal:** Execute approve/reject/flag with single keypress

**Steps:**
1. Create `keybindings.py` with action key definitions
2. Implement approve action ('a' key) using `@work(thread=True)`
3. Implement reject action ('r' key)
4. Implement flag action ('f' key) - marks for later review
5. Implement undo action ('u' key)
6. Add confirmation prompts for destructive actions
7. Update UI immediately after actions (reactive attributes)
8. Add visual feedback (Textual notifications)
9. Handle errors gracefully (try/finally with manager.close())
10. Test action execution and state updates

**Example Pattern:**
```python
@work(thread=True)
async def approve_candidate(self, candidate: EntityCandidate):
    service = self.get_curation_service()
    try:
        entity_id = service.approve_candidate(candidate)
        self.approved_count += 1
        self.notify(f"Approved: {candidate.canonical_name}", severity="success")
    except Exception as e:
        self.notify(f"Error: {e}", severity="error")
    finally:
        service.manager.close()
```

**Deliverable:** Single-key actions working with immediate UI updates (<200ms)

#### Task 3.5.5: Edit Modal
**Goal:** Modal dialog for editing candidate fields

**Steps:**
1. Create `widgets/edit_modal.py` with EditModalScreen
2. Add form fields: canonical_name, aliases (multi-value), description, type, confidence
3. Implement field validation
4. Add save/cancel buttons
5. Keyboard shortcuts (Esc = cancel, Ctrl+S = save)
6. Call EntityCurationService.edit_candidate() via `@work(thread=True)`
7. Update UI after successful edit
8. Test validation and save flow

**Deliverable:** Edit modal opens, validates, saves changes

---

### Day 5: Search/Filter + Progress (Tasks 3.5.6, 3.5.7)

#### Task 3.5.6: Search and Filter System
**Goal:** Search modal and filter controls

**Steps:**
1. Create `widgets/search_modal.py` with SearchModal
2. Implement search input with fuzzy matching
3. Create `queue_manager.py` with QueueManager class
4. Add filter builder (status, type, confidence threshold)
5. Implement search result navigation ('n'/'N' keys)
6. Add filter status display in UI
7. Wire up to CandidateStore.search() and list_candidates()
8. Test search and filter combinations

**Deliverable:** Search and filter working, results update list

#### Task 3.5.7: Progress Tracking and Statistics
**Goal:** Real-time progress display

**Steps:**
1. Create `widgets/status_bar.py` with StatusBar widget
2. Integrate SessionTracker for statistics
3. Display: progress bar, approved count, rejected count, session time
4. Calculate velocity (candidates/minute)
5. Estimate time remaining
6. Add milestone notifications ("100 candidates reviewed!")
7. Create session summary on exit
8. Test statistics accuracy

**Deliverable:** Status bar shows real-time progress and stats

---

### Days 6-7: Batch + Polish (Tasks 3.5.8, 3.5.9, 3.5.10)

#### Task 3.5.8: Batch Operations UI
**Goal:** Multi-select and batch operations

**Steps:**
1. Implement visual selection mode ('v' key)
2. Add multi-select UI (checkboxes or highlights)
3. Create batch operation preview modal
4. Implement batch approve selected
5. Implement batch reject selected
6. Add merge workflow UI (select multiple, confirm primary)
7. Wire up to BatchCurationService via `@work(thread=True)`
8. Add rollback support with visual feedback
9. Test batch operations with checkpoint/rollback

**Deliverable:** Can select multiple, preview, and execute batch operations

#### Task 3.5.9: Vim-Style Command Mode (Optional)
**Goal:** Advanced users can use ':' commands

**Steps:**
1. Implement command mode (':' key)
2. Add command parser
3. Support commands: `:filter type=SYSTEM`, `:sort confidence`, `:batch-approve >0.9`
4. Add command history (↑↓ navigation)
5. Add tab completion
6. Test common command patterns

**Deliverable:** Command mode works for power users

#### Task 3.5.10: Advanced Features (Optional)
**Goal:** Nice-to-have productivity features

**Steps:**
1. (Maybe implement later) Implement mark system ('ma', "'a") for bookmarks
2. Add duplicate detection suggestions panel
3. Create side-by-side comparison view
4. Add context panel showing related entities
5. Implement auto-resume from last session
6. Test advanced navigation

**Deliverable:** Advanced features enhance productivity

---

## Critical Files to Create/Modify

### New Files (Create)
1. `src/curation/interactive/__init__.py`
2. `src/curation/interactive/app.py` - Main Textual app (with reactive attributes)
3. `src/curation/interactive/screens.py` - ReviewScreen
4. `src/curation/interactive/session_tracker.py` - SessionTracker
5. `src/curation/interactive/queue_manager.py` - QueueManager
6. `src/curation/interactive/keybindings.py` - Key definitions
7. `src/curation/interactive/widgets/__init__.py`
8. `src/curation/interactive/widgets/candidate_list.py` - CandidateListWidget
9. `src/curation/interactive/widgets/detail_panel.py` - DetailPanel
10. `src/curation/interactive/widgets/status_bar.py` - StatusBar
11. `src/curation/interactive/widgets/edit_modal.py` - EditModalScreen
12. `src/curation/interactive/widgets/search_modal.py` - SearchModal
13. `src/curation/interactive/widgets/help_modal.py` - HelpModal
14. `scripts/review_entities_interactive.py` - Entry point

### Files to Modify
1. `pyproject.toml` - Add dependencies and entry point
2. `plans/developer-tasks.md` - Update checkboxes as tasks complete

### Files to Reference (No Changes)
- `src/curation/entity_approval.py` - EntityCurationService
- `src/curation/batch_operations.py` - BatchCurationService
- `src/curation/review_interface.py` - Keep for backward compatibility
- `src/storage/schemas.py` - EntityCandidate model
- `src/storage/neo4j_manager.py` - Database operations
- `src/utils/config.py` - Configuration loading

## Technical Decisions

### Why Textual?
- Modern, actively maintained Python TUI framework
- Reactive component model (similar to React)
- Built on Rich for styling (already in project)
- Good documentation and examples
- Handles terminal complexity (resize, colors, input)
- **Worker threads solve async/sync integration** (`@work(thread=True)`)

### Async/Sync Integration (CRITICAL)
- Neo4j operations are SYNCHRONOUS (neo4j driver)
- Textual is ASYNC-FIRST
- **Solution:** Use `@work(thread=True)` to run sync code in worker threads
- No need to convert Neo4jManager to async
- Textual handles lifecycle and thread pool management

### State Management Pattern (VALIDATED)
- **Use Textual's reactive attributes** (not separate ApplicationState class)
- Reactive attributes in App class (auto-update UI on change)
- Services remain stateless (called on-demand)
- Session data persisted on exit for resume
- Data binding between widgets and app state

### Service Integration (VALIDATED)
- Reuse existing services WITHOUT modification
- **Use `@work(thread=True)` decorator** for sync Neo4j operations
- Create/close connections per operation (not persistent)
- Keep business logic in services, UI logic in widgets

**Example:**
```python
@work(thread=True)
async def approve_candidate(self, candidate: EntityCandidate):
    service = self.get_curation_service()
    try:
        entity_id = service.approve_candidate(candidate)
        self.approved_count += 1
    finally:
        service.manager.close()
```

### Backward Compatibility
- Keep `ragagent-review` (old CLI) for scripting
- Add new `ragagent-review-interactive` for TUI
- Both use same underlying services
- Users can choose based on preference

## Testing Strategy

### Unit Tests
- Test reactive state updates
- Test SessionTracker calculations
- Test QueueManager filtering/sorting
- Use Textual's `run_test()` for headless testing

**Example:**
```python
@pytest.mark.asyncio
async def test_approve_workflow():
    app = ReviewApp()
    async with app.run_test() as pilot:
        await pilot.press("a")  # Approve
        assert app.approved_count == 1
```

### Integration Tests
- Test full approval workflow (select → approve → update)
- Test undo operation
- Test batch operations
- Mock Neo4jCandidateStore for isolation

### Manual Testing Checklist
- [ ] App launches without errors
- [ ] Navigate with arrow keys
- [ ] Approve candidate with 'a'
- [ ] Reject candidate with 'r'
- [ ] Edit candidate with 'e'
- [ ] Undo last action with 'u'
- [ ] Search with '/'
- [ ] Batch select with 'v'
- [ ] Batch approve selected
- [ ] Progress updates correctly
- [ ] Session stats accurate
- [ ] Quit with 'q' saves session

## Success Criteria

✅ **Critical Path (Must Have):**
1. App launches and connects to Neo4j
2. Displays list of candidates with filters
3. Navigate with keyboard (↑↓, PgUp/PgDn)
4. Single-key approve ('a') works
5. Single-key reject ('r') works
6. Single-key undo ('u') works
7. Edit modal works ('e')
8. Progress bar shows accurate stats
9. Session persists and can resume
10. Performance: <50ms navigation, <200ms actions

✅ **High Priority (Should Have):**
11. Search modal works ('/')
12. Batch operations work (multi-select + batch approve)
13. Detail panel shows full info
14. Help overlay shows keybindings ('?')

✅ **Nice to Have (Could Have):**
15. Command mode (':')
16. Duplicate suggestions
17. Comparison view
18. Bookmark marks

## Risks and Mitigations

### Risk 1: Textual Learning Curve
**Mitigation:** Start simple (Tasks 3.5.1, 3.5.2), reference Textual examples, use Context7 for docs

### Risk 2: Performance with Large Datasets
**Mitigation:** Implement pagination (50/page), lazy loading, test with 1000+ candidates early

### Risk 3: Terminal Compatibility Issues
**Mitigation:** Test on multiple terminals (iTerm, Terminal.app, tmux), use Textual's cross-platform support

### Risk 4: State Sync Complexity
**Mitigation:** Use Textual's reactive system (automatic synchronization)

## Architecture Validation Results ✅

**Status: APPROVED WITH RECOMMENDATIONS**

A comprehensive architectural review validated the approach with these key findings:

### ✅ Approved Decisions
1. Textual framework is appropriate for this use case
2. Component structure follows Textual best practices
3. Service integration strategy is sound (reuse without modification)
4. Implementation order is logical and achievable in 7 days
5. Testing strategy using `run_test()` is correct

### ⚠️ Critical Recommendations Incorporated
1. **State Management:** Use Textual's reactive attributes (not separate class)
   - Automatic UI updates via reactive system
   - Data binding between widgets and app
   - Watch methods for side effects

2. **Async/Sync Integration:** Use `@work(thread=True)` decorator
   - Solves sync Neo4j + async Textual integration
   - No need to convert services to async
   - Textual manages thread pool and lifecycle

3. **Connection Lifecycle:** Create/close per operation (not persistent)
   - Neo4j driver has connection pooling
   - Avoids timeout issues
   - Explicit cleanup prevents leaks

4. **Pagination:** 50 items/page for large datasets
   - Essential for 1000+ candidates
   - Lazy loading on scroll
   - Performance target: <200ms per action

5. **Testing:** Use Textual's headless testing with `run_test()`
   - pytest-asyncio for async tests
   - Mock Neo4jCandidateStore for isolation
   - E2E tests optional

### Key Architectural Insights
- **Reactive State:** Textual's reactive attributes eliminate manual UI synchronization
- **Worker Threads:** `@work(thread=True)` bridges sync services and async UI seamlessly
- **Service Pattern:** Stateless services with per-operation connections is correct approach
- **No Caching:** Neo4j is source of truth, query performance is acceptable with indexes

## Next Steps

1. ✅ Plan validated and approved
2. Add textual and pytest-asyncio dependencies
3. Create directory structure under `src/curation/interactive/`
4. Implement Task 3.5.1 (Foundation with reactive state)
5. Implement Task 3.5.2 (List Widget with worker threads)
6. Continue through 7-day implementation plan

---

**Implementation Status:** Completed
*Completion Date: 2025-12-22*

**Deferred Scope (Moved to Phase 6):**
- Flagging system (Schema update required)
- Sort command backend
- Export command logic
- Help screen widget
- Advanced comparison actions

**Next Steps:** Proceed to Phase 4.

*Plan created: 2025-12-21*
*Architecture validated by planning agent*
