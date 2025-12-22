# Phase 5: Incremental Updates (Week 9)

## Overview

This phase implements the incremental update system that allows efficient reprocessing of modified documents without full reingestion. It includes change detection, differential chunk updates, and smart entity/relationship updates.

**Timeline:** Week 9  
**Architecture Reference:** [`graph-rag-architecture.md`](graph-rag-architecture.md)  
**Previous Phase:** [Phase 4 - Retrieval System](phase-4-retrieval-system.md)

---

## Task 5.1: Document Change Detection
**Priority:** High  
**Dependencies:** Task 1.10

**Description:**
Implement system to detect changes in documents and identify what needs reprocessing.

**Steps:**
1. Create [`src/pipeline/update_pipeline.py`] with update logic
2. Implement document checksum calculation
3. Store document metadata with checksums
4. Compare new files with stored metadata
5. Classify changes (new, modified, deleted, unchanged)
6. Identify affected chunks for modified documents
7. Track document versions

**Deliverables:**
- [`src/pipeline/update_pipeline.py`] with change detection
- Document metadata tracking
- Change classification logic
- Version tracking

**Acceptance Criteria:**
- Detects new documents
- Identifies modified documents via checksum
- Tracks deleted documents
- Compares document versions
- Fast change detection (<1s per document)
- Minimal false positives

---

## Task 5.2: Differential Chunk Update
**Priority:** High  
**Dependencies:** Task 5.1, Task 1.8

**Description:**
Implement efficient reprocessing of only changed document sections.

**Steps:**
1. Extend [`src/pipeline/update_pipeline.py`] with chunk diffing
2. Re-parse modified documents
3. Compare new chunks with existing chunks
4. Identify added, modified, deleted chunks
5. Update chunk embeddings only for changes
6. Update vector store with changes
7. Maintain chunk IDs for consistency

**Deliverables:**
- Chunk-level diff implementation
- Selective reprocessing logic
- Vector store update operations
- Chunk ID consistency

**Acceptance Criteria:**
- Identifies exact chunks that changed
- Only regenerates embeddings for changed chunks
- Updates vector store efficiently
- Preserves chunk IDs where possible
- Handles structural changes (new sections)
- Processes updates faster than full reingestion

---

## Task 5.3: Entity and Relationship Update Strategy
**Priority:** High  
**Dependencies:** Task 5.2, Task 2.6

**Description:**
Implement smart update strategy for entities and relationships when documents change.

**Steps:**
1. Re-extract entities from changed chunks
2. Compare with existing entities from that document
3. Update entity mention counts
4. Identify entities unique to deleted chunks
5. Re-extract relationships from changed chunks
6. Mark outdated relationships as inactive
7. Add new entities/relationships as candidates
8. Flag significant changes for review

**Deliverables:**
- Entity update logic in [`src/pipeline/update_pipeline.py`]
- Relationship update handling
- Change flagging for review
- Soft-delete for outdated data

**Acceptance Criteria:**
- Updates entity mention counts accurately
- Identifies entities that may no longer be valid
- Marks outdated relationships as inactive
- Adds new entities as candidates for review
- Preserves approved entities
- Flags major changes for human review

---

## Task 5.4: Graph Update Operations
**Priority:** Medium  
**Dependencies:** Task 5.3

**Description:**
Implement Neo4j operations for safely updating the graph without data loss.

**Steps:**
1. Extend [`src/storage/neo4j_manager.py`] with update operations
2. Implement soft-delete for relationships (mark inactive)
3. Implement entity merge during updates
4. Add versioning for entities and relationships
5. Create rollback functionality
6. Implement atomic update transactions
7. Add update validation

**Deliverables:**
- Update operations in [`src/storage/neo4j_manager.py`]
- Soft-delete implementation
- Versioning system
- Rollback capability

**Acceptance Criteria:**
- Safe updates without data loss
- Soft-deletes preserve history
- Version tracking for entities
- Atomic transactions prevent partial updates
- Rollback works correctly
- Update validation prevents corruption

---

## Task 5.5: Update Pipeline Orchestration
**Priority:** High  
**Dependencies:** Task 5.1, Task 5.2, Task 5.3, Task 5.4

**Description:**
Orchestrate complete update pipeline from change detection through graph updates.

**Steps:**
1. Complete [`src/pipeline/update_pipeline.py`] with full orchestration
2. Implement update workflow: detect → diff → reprocess → update
3. Add progress tracking for updates
4. Create update summary report
5. Implement dry-run mode
6. Add update scheduling (optional)
7. Create update CLI script

**Deliverables:**
- Complete update pipeline
- [`scripts/update_documents.py`] CLI script
- Update reports
- Dry-run capability

**Acceptance Criteria:**
- Processes document updates end-to-end
- Reports changes clearly
- Dry-run shows what would be updated
- Progress tracking for large updates
- Handles errors gracefully
- Significantly faster than full reingestion
- Preserves approved entity curation

---

## Task 5.6: Merge Candidates into Existing Entities
**Priority:** Medium
**Dependencies:** Task 5.3, Task 3.7
**Extension of:** Phase 3.5 Task 3.5.8 (Batch Operations UI)

**Description:**
Extend the merge workflow from Phase 3.5 to support merging entity candidates into existing approved entities. This enables fixing post-approval duplicates, enriching entities with newly discovered aliases, and handling ongoing ingestion without creating duplicate entities.

**Context:**
Phase 3.5 Task 3.5.8 implements candidate-to-candidate merging, which creates a new approved entity from multiple candidates. This task extends that capability to merge candidates into entities that already exist in the production graph. This is essential for:
- Fixing duplicate entities discovered after approval
- Enriching existing entities with new aliases found in new documents
- Preventing duplicate creation during incremental ingestion
- Consolidating similar entities post-approval

**Steps:**

1. **Backend: Entity Search Functionality**
   - Extend [`src/storage/neo4j_manager.py`] with entity search methods
   - Implement `search_entities(query, entity_type=None, status=None, limit=50)`
   - Support fuzzy name matching using normalization table
   - Return entities with: id, canonical_name, type, aliases, status, mention_count, confidence
   - Add filtering by entity type and status

2. **Backend: Merge Candidate into Entity Operation**
   - Add `merge_candidate_into_entity()` to [`src/curation/entity_approval.py`]
   - Signature: `merge_candidate_into_entity(entity_id: str, candidate: EntityCandidate) -> str`
   - Merge logic:
     - **Aliases**: Add candidate's aliases to entity's alias list (deduplicated)
     - **Description**: Keep entity's description (or update if entity has no description and candidate does)
     - **Mention Count**: Add candidate's mention count to entity's
     - **Source Documents**: Union candidate's documents with entity's
     - **Chunk IDs**: Union candidate's chunk_ids with entity's
     - **Confidence**: Take max of entity and candidate confidence
     - **Provenance**: Add candidate_key to entity's merged_candidate_keys
   - Update candidate status to `MERGED_INTO_ENTITY` (new status)
   - Update normalization table: map candidate's aliases to entity's canonical_name
   - Promote relationship candidates involving merged aliases
   - Create checkpoint for undo support
   - Return entity_id

3. **Backend: Batch Merge into Entities**
   - Add `batch_merge_into_entity()` to [`src/curation/batch_operations.py`]
   - Signature: `batch_merge_into_entity(entity_id: str, candidates: Sequence[EntityCandidate]) -> BatchOperationResult`
   - Iterate through candidates, calling `merge_candidate_into_entity()` for each
   - Use checkpoint/rollback for transaction safety
   - Return result with merged_count, failed list

4. **UI: Entity Search Modal** (`EntitySearchModal`)
   - New modal widget in [`src/curation/interactive/widgets/entity_search_modal.py`]
   - Search input with real-time filtering
   - Display search results in scrollable list
   - Show entity info: name, type, aliases, mention count, confidence
   - Allow filtering by entity type
   - Keyboard navigation (↑↓, Enter to select)
   - [Select Entity] and [Cancel] buttons
   - Bind to 'Shift+M' key (distinct from 'M' for candidate merge)

5. **UI: Entity-Candidate Merge Preview Modal** (`EntityCandidateMergePreviewModal`)
   - Shows existing entity details (current state)
   - Shows candidate(s) to be merged (what will be added)
   - Shows merged result preview:
     - **Aliases**: existing + new (highlighted in different color)
     - **Mention Count**: sum (show breakdown: "45 + 12 = 57")
     - **Source Documents**: union with count
     - **Description**: shows which description will be used
   - Warning if entity type doesn't match candidate type
   - [Confirm Merge] and [Cancel] buttons

6. **UI: Merge into Entity Workflow**
   - Extend [`src/curation/interactive/app.py`] with entity merge actions
   - Add `action_merge_into_entity()` bound to 'Shift+M'
   - Workflow:
     1. User selects 1+ PENDING candidates in selection mode
     2. User presses 'Shift+M' (distinct from 'M' for candidate merge)
     3. Show EntitySearchModal for entity selection
     4. User searches and selects target entity
     5. Show EntityCandidateMergePreviewModal
     6. On confirm: execute merge via `batch_merge_into_entity()`
     7. Update statistics, clear selection, reload candidates
   - Add worker method `merge_candidates_into_entity(entity_id, candidates)`

6a. **UI: Enhanced Duplicate Detection with Existing Entities**
   - Extend [`src/curation/interactive/widgets/duplicate_suggestions.py`]
   - Currently only checks against other EntityCandidates
   - **Enhancement**: Also query and compare against existing Entity nodes (status=APPROVED)
   - Display both types of matches:
     - **Candidate Duplicates**: Similar pending candidates (current behavior)
     - **Existing Entity Matches**: Similar approved entities (new)
   - Visual distinction:
     - Candidate duplicates: Blue/Cyan border, "Candidate" label
     - Entity matches: Green border, "Existing Entity" label, entity ID
   - Different actions available:
     - Candidate duplicates → Merge candidates (M key)
     - Entity matches → Quick link to merge into entity workflow (Shift+M)
   - Implementation:
     - Add `_find_existing_entity_matches()` method
     - Query Neo4j for entities with similar names/aliases
     - Combine results from both candidate and entity searches
     - Sort by similarity, show top 5 total (mix of both types)
   - Performance: Cache entity queries, use indexes for fast name lookup

7. **Schema: New Candidate Status**
   - Add `MERGED_INTO_ENTITY` to CandidateStatus enum in [`src/storage/schemas.py`]
   - Distinguishes from `REJECTED` (indicates successful merge rather than rejection)
   - Update Neo4j constraints to allow new status
   - Update candidate filtering to handle new status

8. **Undo Support**
   - Extend undo system to handle entity-candidate merges
   - On undo:
     - Remove merged aliases from entity
     - Restore entity's previous mention count, documents, etc.
     - Restore candidate status to PENDING
     - Revert normalization table entries
   - Store pre-merge entity snapshot in UndoAction

9. **Validation and Error Handling**
   - Validate entity exists before merge
   - Check entity is in APPROVED status
   - Warn if candidate type differs from entity type (allow but warn)
   - Handle case where entity was deleted after selection
   - Graceful error handling with rollback
   - User-friendly error messages

10. **Documentation and Testing**
    - Document merge workflows in user guide
    - Add examples of when to use each merge type
    - Create integration tests for entity merging
    - Test undo functionality
    - Test error cases (deleted entity, type mismatches)

**Deliverables:**
- Entity search functionality in Neo4jManager
- `merge_candidate_into_entity()` operation
- `batch_merge_into_entity()` for multiple candidates
- EntitySearchModal widget
- EntityCandidateMergePreviewModal widget
- Merge into entity workflow in interactive UI
- New MERGED_INTO_ENTITY status
- Extended undo support for entity merges
- Comprehensive validation and error handling

**Acceptance Criteria:**
- ✓ Can search for existing entities by name
- ✓ Search supports fuzzy matching via normalization table
- ✓ Can filter search results by entity type
- ✓ Entity search modal shows relevant entity details
- ✓ Merge preview shows current entity, candidates, and merged result
- ✓ New aliases are clearly distinguished in preview
- ✓ Merge adds all candidate aliases to entity
- ✓ Merge updates entity mention counts correctly
- ✓ Merge unions source documents and chunk IDs
- ✓ Candidate status updated to MERGED_INTO_ENTITY
- ✓ Normalization table updated with merged aliases
- ✓ Relationship candidates promoted correctly
- ✓ Undo restores entity and candidate to pre-merge state
- ✓ Type mismatch warnings shown but allowed
- ✓ Graceful handling of deleted entities
- ✓ Can merge single candidate into entity
- ✓ Can merge multiple candidates into same entity (batch)
- ✓ Session statistics track entity merges separately

**Comparison: Candidate-to-Candidate vs Candidate-to-Entity Merge**

| Aspect | Candidate→Candidate (Task 3.5.8) | Candidate→Entity (Task 5.6) |
|--------|-----------------------------------|------------------------------|
| **When to Use** | Initial curation, before approval | Post-approval duplicates, enrichment |
| **Result** | Creates new approved entity | Updates existing entity |
| **Primary Selection** | User chooses primary candidate | Target entity is existing entity |
| **Candidate Statuses** | Primary→APPROVED, others→REJECTED | All→MERGED_INTO_ENTITY |
| **Entity Creation** | Yes (new entity created) | No (entity updated in place) |
| **Keybinding** | 'M' (capital M) | 'Shift+M' |
| **Search Required** | No (select from visible candidates) | Yes (search for target entity) |
| **Undo** | Deletes created entity, restores candidates | Restores entity state, restores candidates |
| **Use Case** | "These 3 candidates are the same thing" | "This candidate is another alias for that entity" |

**Testing Checklist:**
- [ ] Search for entity by partial name, confirm results appear
- [ ] Select entity from search, confirm details shown
- [ ] Select candidate, press Shift+M, search for entity, confirm merge
- [ ] Verify candidate aliases added to entity
- [ ] Verify entity mention count increased
- [ ] Verify candidate status is MERGED_INTO_ENTITY
- [ ] Press 'u', verify entity restored and candidate status reset
- [ ] Try merging with non-existent entity ID, confirm error handling
- [ ] Merge candidate with different type, confirm warning shown
- [ ] Merge multiple candidates into same entity, verify batch works

**Future Enhancements:**
- Auto-suggest likely target entities based on similarity
- Bulk entity search and merge suggestions
- Entity-to-entity merging (not just candidate-to-entity)
- Visual graph view showing merge impact

---

## Phase 5 Summary

**Key Deliverables:**
- Document change detection system
- Differential chunk update mechanism
- Smart entity and relationship updates
- Safe graph update operations
- Complete update pipeline orchestration
- CLI tools for incremental updates

**Success Metrics:**
- Update pipeline significantly faster than full reingestion
- Change detection accurate and efficient
- Entity curation preserved through updates
- Graph updates maintain data integrity
- System handles document modifications gracefully

**Next Phase:** [Phase 6 - Discovery & Polish](phase-6-discovery-polish.md)