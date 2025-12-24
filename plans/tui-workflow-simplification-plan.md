# TUI + Curation Workflow Simplification Plan

## Why

The current curation workflow is powerful but complicated:

- Multiple overlapping flows (entities vs relationships, neighborhood resolution, merge workflows)
- Many modal screens with non-obvious transitions (search/compare/merge/command)
- A “mode switch” concept (`review_mode`, selection mode, command mode) that increases mental load
- Undo/rollback semantics that are not consistently expressed in the UI

This plan fully implements a simplified, “single mental model” curation experience while keeping the
existing backend (`EntityCurationService`, `BatchCurationService`) as the authoritative source of
operations and safety.

## Goals (User-facing)

1. **One primary flow:** review entity candidates; relationships appear as contextual “neighborhood”
   tasks, not as a separate review mode.
2. **Fewer screens:** collapse multiple modals into a small set of consistent panels:
   - Candidate list (left)
   - Candidate detail (center/right)
   - Context / neighborhood panel (right or bottom)
   - One command palette (instead of separate command + search + entity-search modals)
3. **Safer batch work:** batch operations always preview, can be rolled back, and communicate what
   changed (and what can be undone).
4. **Predictable keys:** candidate-key normalization is consistent between ingestion and curation.

## Non-goals

- Multi-user collaborative curation
- A web UI replacement (keep TUI; web can remain future work)
- Rewriting the storage layer or changing Neo4j/Qdrant schemas

## Current Architecture (Baseline)

- Curation actions: `src/curation/entity_approval.py` (`EntityCurationService`)
- Batch actions: `src/curation/batch_operations.py` (`BatchCurationService`)
- TUI: `src/curation/interactive/app.py` + widgets/screens/modals
- CLI: `src/curation/review_interface.py`

## Target UX/IA (Information Architecture)

### Main Screen (Single Review Mode)

- Left: queue list (pending by default)
- Center: candidate detail (aliases/description/provenance/mentions)
- Right (context): duplicates + neighborhood (pending relationships and blocked edges)

### Command Palette (One Entry Point)

Replace distinct “search modal”, “entity search modal”, and “vim command mode” with one palette that:

- Supports fuzzy search for candidates in the current list
- Supports filters as structured “chips” (status/type/confidence)
- Supports actions (approve/reject/edit/merge/merge-into-existing/batch ops)
- Supports “jump to candidate” and “jump to entity”

### Neighborhood as First-class Context

When approving an entity candidate:

- Automatically compute neighborhood issues
- Show issues in the context panel (non-blocking) OR prompt with a single “Resolve now?” action
- Provide the minimal set of actions per issue:
  - Promote relationship (if peer is already approved)
  - Approve peer candidate (if peer pending)
  - Create missing peer entity (default type `CONCEPT`, with override)

Relationships are thus curated “on-demand” during entity approval rather than via a separate
relationship-review queue.

## Implementation Milestones

### Milestone 1 — Consolidate and Stabilize Backend Semantics

**Outcome:** the backend exposes the minimum set of primitives needed for the simplified UI.

Tasks:

- Ensure undo works uniformly for:
  - entity approve/merge/merge-into-existing
  - relationship candidate approve/reject
  - create missing peer entities
- Standardize candidate-key normalization (single helper shared across ingestion and curation)
- Ensure neighborhood lookups use candidate-key fragments instead of brittle raw string comparisons

Acceptance:

- Unit tests cover relationship-candidate undo and neighborhood behavior.

### Milestone 2 — Remove Relationship “Mode” From TUI

**Outcome:** the TUI always reviews entity candidates; relationships appear only as context.

Tasks:

- Remove `review_mode` toggle and any list loading logic that fetches relationship candidates as the
  primary queue.
- Update the status bar and subtitle logic to reflect the single mode.
- Update keybindings to remove mode-specific actions.

Acceptance:

- `scripts/review_entities_interactive.py` launches and can approve/reject/edit/merge entity
  candidates without any mode switches.

### Milestone 3 — Replace Multi-Modal UX With a Single Command Palette

**Outcome:** reduce screens and reduce navigation state.

Tasks:

- Implement `CommandPalette` widget/screen that unifies:
  - search
  - filter changes
  - entity lookup for merge-into-existing
  - action invocation (approve/reject/edit/merge/batch)
- Deprecate/remove `SearchModalScreen`, `EntitySearchModal`, and “vim-style command mode” UI.

Acceptance:

- All actions previously available via `:command`, `/search`, or merge-entity modal can be done from
  the palette.

### Milestone 4 — Context-First Neighborhood Panel

**Outcome:** neighborhood resolution becomes the main way users curate relationships.

Tasks:

- Implement a persistent `NeighborhoodPanel` showing computed issues for the current candidate.
- Provide inline actions per issue (promote/approve peer/create entity).
- Optional: add “Resolve selected issues” batch action that uses the same checkpoint/rollback
  semantics as other batch ops.

Acceptance:

- Approving an entity shows neighborhood issues without dismissing the main screen.
- Resolving issues updates UI and is undoable.

### Milestone 5 — Simplify Merge UX

**Outcome:** one consistent merge workflow (duplicate merge + merge into existing).

Tasks:

- Consolidate merge flows so there’s one entry point:
  - Merge selected candidates into a primary
  - Merge candidate into existing entity (via palette entity search)
- Ensure preview always renders the “resulting entity” and the candidate statuses that will change.
- Make “undo last” clearly communicated as “undo last action (and relationship promotions)”.

Acceptance:

- Merge flow requires fewer screens and fewer decisions; preview is always shown.

## Risk & Mitigation

- **Risk:** removing relationship review mode may hide relationship candidates.
  - **Mitigation:** ensure neighborhood panel surfaces pending relationships for the current entity
    and provide “open relationship queue” as an advanced action (optional, not default).
- **Risk:** command palette scope creep.
  - **Mitigation:** ship palette incrementally; keep existing modals until feature parity.

## Rollout Strategy

1. Land backend fixes (Milestone 1) with tests.
2. Implement UI changes behind a config flag (e.g. `curation.tui_simplified=true`).
3. Run internal dogfooding; remove old paths once stable.

## Acceptance Test Script (Manual)

1. Launch TUI: `uv run python scripts/review_entities_interactive.py`
2. Filter to pending, confidence ≥ 0.8
3. Approve an entity that has pending relationships
4. Resolve 1–2 neighborhood issues
5. Undo last action, confirm:
   - candidate status is restored
   - any created relationships are removed
6. Merge 2 candidates (preview → confirm)
7. Merge a candidate into an existing entity (preview → confirm)

