# Entity Merge Workflow Specification

**Date:** 2025-12-22
**Status:** Planning Complete - Ready for Implementation
**Related Tasks:** Phase 3.5 Task 3.5.8, Phase 5 Task 5.6

---

## Overview

This document specifies two types of entity merge workflows for the Graph RAG system:

1. **Candidate-to-Candidate Merge** (Phase 3.5 Task 3.5.8) - Immediate implementation
2. **Candidate-to-Entity Merge** (Phase 5 Task 5.6) - Future enhancement

---

## Part 1: Candidate-to-Candidate Merge (Immediate)

### Purpose
Merge multiple entity candidates together to create a single approved entity. Used during initial curation when reviewing candidates before approval.

### When to Use
- Deduplicating candidates during initial review
- Consolidating similar entities before approval
- Example: "GPS", "Global Positioning System", "Gps" are the same system

### What Gets Merged

| Field | Merge Strategy |
|-------|---------------|
| **Canonical Name** | From primary candidate |
| **Entity Type** | From primary candidate |
| **Aliases** | Union of all unique aliases from all candidates |
| **Description** | Primary's description (fallback to first duplicate with description) |
| **Confidence** | Maximum confidence score across all candidates |
| **Mention Count** | Sum of mention counts |
| **Source Documents** | Union of all source documents |
| **Chunk IDs** | Union of all chunk IDs |
| **Provenance** | Stores all merged candidate_keys in entity properties |

### Candidate Status Updates
- **Primary:** `PENDING` → `APPROVED`
- **Duplicates:** `PENDING` → `REJECTED`
- **Result:** New Entity created with `EntityStatus.APPROVED`

### Backend Implementation

**Service Method:**
```python
# src/curation/entity_approval.py
def merge_candidates(
    self,
    primary: EntityCandidate,
    duplicates: Sequence[EntityCandidate]
) -> str:
    """Merge multiple candidates into a single approved entity."""
```

**Batch Service Method:**
```python
# src/curation/batch_operations.py
def batch_merge_clusters(
    self,
    clusters: Sequence[Sequence[EntityCandidate]],
    *,
    dry_run: bool = False,
) -> BatchOperationResult:
    """Merge multiple clusters of candidates."""
```

### UI Workflow

#### Step 1: Selection
1. User enters selection mode: `v` key
2. User selects 2+ candidates: `Space` key on each
3. Checkboxes show selected state: `[✓]`
4. Subtitle shows count: "Status: pending | Total: 50 | [SELECTION MODE] 3 selected"

#### Step 2: Initiate Merge
1. User presses `M` (capital M)
2. Validation:
   - At least 2 candidates selected
   - Filter out non-PENDING candidates (show warning)
   - Check for type conflicts (show warning, allow to continue)

#### Step 3: Primary Selection Modal
```
┌─ Select Primary Candidate ──────────────────────┐
│                                                  │
│ Choose which candidate should be the primary:   │
│                                                  │
│ ( ) GPS - SYSTEM - 0.95 conf - 15 mentions     │
│     Description: Navigation system using...      │
│                                                  │
│ (•) Global Positioning System - SYSTEM - 0.87  │ ← Default (highest conf)
│     Description: Satellite-based navigation...   │
│                                                  │
│ ( ) Gps - SYSTEM - 0.82 - 8 mentions           │
│     Description: [none]                          │
│                                                  │
│         [Select Primary]  [Cancel]               │
└──────────────────────────────────────────────────┘
```

#### Step 4: Merge Preview Modal
```
┌─ Merge Preview ─────────────────────────────────┐
│                                                  │
│ Primary Candidate (will become entity):         │
│ ► [•] Global Positioning System (SYSTEM) - 0.87│
│                                                  │
│ Candidates to merge (will be rejected):         │
│   [ ] GPS (SYSTEM) - 0.95                       │
│   [ ] Gps (SYSTEM) - 0.82                       │
│                                                  │
│ ─────────────────────────────────────────────── │
│ Merged Result Preview:                           │
│                                                  │
│ Name: Global Positioning System                  │
│ Type: SYSTEM                                     │
│ Aliases:                                         │
│   • GPS                                          │
│   • Global Positioning System                    │
│   • Gps                                          │
│ Description: Satellite-based navigation...       │
│ Confidence: 0.95 (max)                          │
│ Mentions: 38 (sum)                               │
│ Documents: 5 (union)                             │
│                                                  │
│     [Change Primary]  [Confirm Merge]  [Cancel] │
└──────────────────────────────────────────────────┘
```

#### Step 5: Execution
1. On confirm: execute merge
2. Call `batch_merge_clusters([[primary, ...duplicates]])`
3. Update session statistics
4. Show notification: "✓ Merged 3 candidates into: Global Positioning System"
5. Clear selection
6. Exit selection mode
7. Reload candidates

### Undo Support
- Press `u` key to undo merge
- Restores all candidate statuses to PENDING
- Deletes created entity
- Restores normalization table entries
- Reverts relationship candidate promotions

### Keybindings
- `v` - Toggle selection mode
- `Space` - Toggle current candidate selection
- `Ctrl+A` - Select all visible
- `Ctrl+D` - Deselect all
- `M` - Merge selected candidates (capital M)
- `u` - Undo last operation

### UI Components to Implement

**1. PrimarySelectionModal**
- File: `src/curation/interactive/widgets/primary_selection_modal.py`
- Shows candidate list with radio buttons
- Displays key info: name, type, confidence, mentions, description
- Default: highest confidence
- Returns: selected EntityCandidate

**2. MergePreviewModal**
- File: `src/curation/interactive/widgets/merge_preview_modal.py`
- Shows primary (highlighted)
- Shows duplicates list
- Shows merged result with all fields
- [Change Primary] returns to PrimarySelectionModal
- Returns: boolean (confirmed/cancelled)

**3. App Integration**
- File: `src/curation/interactive/app.py`
- Add `action_merge_candidates()` method
- Add `batch_merge_candidates(candidates)` worker method
- Handle modal flow and execution

### Error Handling

| Error Condition | Handling |
|----------------|----------|
| Less than 2 candidates | Show error: "Select at least 2 candidates to merge" |
| No PENDING candidates | Filter and warn: "X candidates filtered (not PENDING)" |
| Type conflicts | Warn but allow: "⚠️ Candidates have different types: SYSTEM, SUBSYSTEM" |
| Backend error | Rollback and show: "✗ Merge failed: {error}. Changes rolled back." |

### Testing Checklist
- [ ] Select 2 candidates, confirm checkboxes appear
- [ ] Press M, confirm primary selection modal appears
- [ ] Change primary, confirm preview updates
- [ ] Confirm merge, verify entity created with all aliases
- [ ] Verify normalization table updated
- [ ] Verify relationship candidates promoted
- [ ] Press u, verify undo restores candidates
- [ ] Test with type conflicts, confirm warning
- [ ] Test with <2 candidates, confirm error
- [ ] Test with non-PENDING, confirm filtered

---

## Part 2: Candidate-to-Entity Merge (Future - Phase 5)

### Purpose
Merge entity candidates into existing approved entities. Used for post-approval duplicate resolution and entity enrichment.

### When to Use
- Fixing duplicates discovered after approval
- Enriching existing entities with new aliases
- Preventing duplicates during incremental ingestion
- Example: New candidate "EPS" should be merged into existing entity "Electrical Power System"

### Differences from Candidate-to-Candidate

| Aspect | Candidate→Candidate | Candidate→Entity |
|--------|---------------------|------------------|
| **Result** | Creates new entity | Updates existing entity |
| **Target** | Selected candidates | Search for entity |
| **Status** | Primary→APPROVED, others→REJECTED | All→MERGED_INTO_ENTITY |
| **Keybinding** | `M` | `Shift+M` |
| **Undo** | Delete entity | Restore entity state |

### What Gets Merged

| Field | Merge Strategy |
|-------|---------------|
| **Aliases** | Add candidate's aliases to entity (deduplicated) |
| **Description** | Keep entity's (update only if entity has none) |
| **Mention Count** | Add candidate's count to entity's |
| **Source Documents** | Union with entity's documents |
| **Chunk IDs** | Union with entity's chunks |
| **Confidence** | Max(entity, candidate) |
| **Provenance** | Add candidate_key to entity's merged list |

### New Candidate Status
- Add `MERGED_INTO_ENTITY` to `CandidateStatus` enum
- Distinguishes from `REJECTED` (indicates successful merge, not rejection)

### Backend Implementation

**Entity Search:**
```python
# src/storage/neo4j_manager.py
def search_entities(
    self,
    query: str,
    entity_type: str | None = None,
    status: str | None = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Search for entities with fuzzy name matching."""
```

**Merge Operation:**
```python
# src/curation/entity_approval.py
def merge_candidate_into_entity(
    self,
    entity_id: str,
    candidate: EntityCandidate
) -> str:
    """Merge a candidate into an existing entity."""
```

**Batch Operation:**
```python
# src/curation/batch_operations.py
def batch_merge_into_entity(
    self,
    entity_id: str,
    candidates: Sequence[EntityCandidate]
) -> BatchOperationResult:
    """Merge multiple candidates into same entity."""
```

### UI Workflow

#### Step 1: Selection
1. User selects 1+ candidates in selection mode
2. User presses `Shift+M` (distinct from `M`)

#### Step 2: Entity Search Modal
```
┌─ Search for Entity ─────────────────────────────┐
│                                                  │
│ Search: [power system____________]   [Filter ▾] │
│                                                  │
│ Results:                                         │
│ ► Electrical Power System - SYSTEM              │
│   Aliases: EPS, Power System                     │
│   45 mentions | 0.95 confidence                  │
│                                                  │
│   Power Distribution Unit - SUBSYSTEM            │
│   Aliases: PDU                                   │
│   23 mentions | 0.89 confidence                  │
│                                                  │
│   Solar Power System - SUBSYSTEM                 │
│   Aliases: Solar Array                           │
│   18 mentions | 0.92 confidence                  │
│                                                  │
│         [Select Entity]  [Cancel]                │
└──────────────────────────────────────────────────┘
```

#### Step 3: Merge Preview Modal
```
┌─ Merge into Entity Preview ─────────────────────┐
│                                                  │
│ Target Entity (will be updated):                │
│ ► Electrical Power System (SYSTEM) - 0.95       │
│   Current aliases: EPS, Power System             │
│   45 mentions | 12 documents                     │
│                                                  │
│ Candidates to merge:                             │
│   • Electrical Power Subsystem (SYSTEM) - 0.87  │
│     Aliases: EPSS                                │
│   • Power Subsystem (SYSTEM) - 0.83             │
│                                                  │
│ ─────────────────────────────────────────────── │
│ Merged Result:                                   │
│                                                  │
│ Name: Electrical Power System (unchanged)        │
│ Type: SYSTEM (unchanged)                         │
│ Aliases:                                         │
│   • EPS (existing)                               │
│   • Power System (existing)                      │
│   • EPSS (new) ← highlighted                     │
│ Mentions: 45 + 23 = 68                          │
│ Documents: 12 ∪ 3 = 14                          │
│                                                  │
│            [Confirm Merge]  [Cancel]             │
└──────────────────────────────────────────────────┘
```

### Implementation Priority
**Phase 5 (Week 9)** - After incremental update system is in place

---

## Design Decisions Summary

### Confirmed Decisions
1. ✅ Start with candidate-to-candidate merge (Phase 3.5)
2. ✅ Add candidate-to-entity merge in Phase 5
3. ✅ Auto-select highest confidence as default primary
4. ✅ Allow type conflicts with warning
5. ✅ Use capital M for candidate merge, Shift+M for entity merge
6. ✅ Require minimum 2 candidates for merge
7. ✅ Show detailed preview before all merge operations
8. ✅ Full undo support via checkpoint system

### Implementation Sequence
1. **Now (Phase 3.5):** Candidate-to-candidate merge
   - Primary selection modal
   - Merge preview modal
   - Batch merge execution
   - Full undo support

2. **Later (Phase 5):** Candidate-to-entity merge
   - Entity search functionality
   - Entity search modal
   - Entity-candidate merge preview
   - Extended undo support
   - New MERGED_INTO_ENTITY status

---

## Success Criteria

### Phase 3.5 (Immediate)
- [ ] Can select and merge 2+ candidates
- [ ] Primary selection modal works correctly
- [ ] Merge preview shows accurate merged result
- [ ] Merge creates entity with all aliases
- [ ] Normalization table updated correctly
- [ ] Relationship candidates promoted
- [ ] Undo fully restores pre-merge state
- [ ] Type conflicts handled gracefully
- [ ] User experience is smooth and intuitive

### Phase 5 (Future)
- [ ] Can search and find existing entities
- [ ] Can merge candidates into entities
- [ ] Entity aliases updated correctly
- [ ] New status MERGED_INTO_ENTITY works
- [ ] Undo restores entity and candidates
- [ ] Integration with incremental updates

---

**Document Status:** ✅ Complete and ready for implementation
**Next Step:** Implement candidate-to-candidate merge (Phase 3.5 Task 3.5.8)
