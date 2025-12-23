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
1. [x] Create [`src/pipeline/update_pipeline.py`] with update logic
2. [x] Implement document checksum calculation
3. [x] Store document metadata with checksums
4. [x] Compare new files with stored metadata
5. [x] Classify changes (new, modified, deleted, unchanged)
6. [x] Identify affected chunks for modified documents
7. [x] Track document versions

**Deliverables:**
- [`src/pipeline/update_pipeline.py`] with change detection
- Document metadata tracking
- Change classification logic
- Version tracking

**Acceptance Criteria:**
- ✓ Detects new documents
- ✓ Identifies modified documents via checksum
- ✓ Tracks deleted documents
- ✓ Compares document versions
- ✓ Fast change detection (<1s per document)
- ✓ Minimal false positives

---

## Task 5.2: Differential Chunk Update
**Priority:** High  
**Dependencies:** Task 5.1, Task 1.8

**Description:**
Implement efficient reprocessing of only changed document sections.

**Steps:**
1. [x] Extend [`src/pipeline/update_pipeline.py`] with chunk diffing
2. [x] Re-parse modified documents
3. [x] Compare new chunks with existing chunks
4. [x] Identify added, modified, deleted chunks
5. [x] Update chunk embeddings only for changes
6. [x] Update vector store with changes
7. [x] Maintain chunk IDs for consistency

**Deliverables:**
- Chunk-level diff implementation
- Selective reprocessing logic
- Vector store update operations
- Chunk ID consistency

**Acceptance Criteria:**
- ✓ Identifies exact chunks that changed
- ✓ Only regenerates embeddings for changed chunks
- ✓ Updates vector store efficiently
- ✓ Preserves chunk IDs where possible
- ✓ Handles structural changes (new sections)
- ✓ Processes updates faster than full reingestion

---

## Task 5.3: Entity and Relationship Update Strategy
**Priority:** High  
**Dependencies:** Task 5.2, Task 2.6

**Description:**
Implement smart update strategy for entities and relationships when documents change.

**Steps:**
1. [x] Re-extract entities from changed chunks
2. [x] Compare with existing entities from that document
3. [x] Update entity mention counts
4. [x] Identify entities unique to deleted chunks
5. [x] Re-extract relationships from changed chunks
6. [x] Mark outdated relationships as inactive
7. [x] Add new entities/relationships as candidates
8. [x] Flag significant changes for review

**Deliverables:**
- Entity update logic in [`src/pipeline/update_pipeline.py`]
- Relationship update handling
- Change flagging for review
- Soft-delete for outdated data

**Acceptance Criteria:**
- ✓ Updates entity mention counts accurately
- ✓ Identifies entities that may no longer be valid
- ✓ Marks outdated relationships as inactive
- ✓ Adds new entities as candidates for review
- ✓ Preserves approved entities
- ✓ Flags major changes for human review

---

## Task 5.4: Graph Update Operations
**Priority:** Medium  
**Dependencies:** Task 5.3

**Description:**
Implement Neo4j operations for safely updating the graph without data loss.

**Steps:**
1. [x] Extend [`src/storage/neo4j_manager.py`] with update operations
2. [x] Implement soft-delete for relationships (mark inactive)
3. [x] Implement entity merge during updates
4. [x] Add versioning for entities and relationships
5. [x] Create rollback functionality
6. [x] Implement atomic update transactions
7. [x] Add update validation

**Deliverables:**
- Update operations in [`src/storage/neo4j_manager.py`]
- Soft-delete implementation
- Versioning system
- Rollback capability

**Acceptance Criteria:**
- ✓ Safe updates without data loss
- ✓ Soft-deletes preserve history
- ✓ Version tracking for entities
- ✓ Atomic transactions prevent partial updates
- ✓ Rollback works correctly
- ✓ Update validation prevents corruption

---

## Task 5.5: Update Pipeline Orchestration
**Priority:** High  
**Dependencies:** Task 5.1, Task 5.2, Task 5.3, Task 5.4

**Description:**
Orchestrate complete update pipeline from change detection through graph updates.

**Steps:**
1. [x] Complete [`src/pipeline/update_pipeline.py`] with full orchestration
2. [x] Implement update workflow: detect → diff → reprocess → update
3. [x] Add progress tracking for updates
4. [x] Create update summary report
5. [x] Implement dry-run mode
6. [ ] Add update scheduling (optional)
7. [x] Create update CLI script

**Deliverables:**
- Complete update pipeline
- [`scripts/update_documents.py`] CLI script
- Update reports
- Dry-run capability

**Acceptance Criteria:**
- ✓ Processes document updates end-to-end
- ✓ Reports changes clearly
- ✓ Dry-run shows what would be updated
- ✓ Progress tracking for large updates
- ✓ Handles errors gracefully
- ✓ Significantly faster than full reingestion
- ✓ Preserves approved entity curation

---

## Task 5.6: Merge Candidates into Existing Entities
**Priority:** Medium
**Dependencies:** Task 5.3, Task 3.7
**Extension of:** Phase 3.5 Task 3.5.8 (Batch Operations UI)

**Description:**
Extend the merge workflow from Phase 3.5 to support merging entity candidates into existing approved entities. This enables fixing post-approval duplicates, enriching entities with newly discovered aliases, and handling ongoing ingestion without creating duplicate entities.

**Steps:**

1. [x] **Backend: Entity Search Functionality**
2. [x] **Backend: Merge Candidate into Entity Operation**
3. [x] **Backend: Batch Merge into Entities**
4. [ ] **UI: Entity Search Modal** (`EntitySearchModal`)
5. [ ] **UI: Entity-Candidate Merge Preview Modal** (`EntityCandidateMergePreviewModal`)
6. [ ] **UI: Merge into Entity Workflow**
6a. [ ] **UI: Enhanced Duplicate Detection with Existing Entities**
7. [x] **Schema: New Candidate Status**
8. [x] **Undo Support**
9. [x] **Validation and Error Handling**
10. [x] **Documentation and Testing**

**Deliverables:**
- Entity search functionality in Neo4jManager
- `merge_candidate_into_entity()` operation
- `batch_merge_into_entity()` for multiple candidates
- New MERGED_INTO_ENTITY status
- Extended undo support for entity merges
- Comprehensive validation and error handling

**Acceptance Criteria:**
- ✓ Can search for existing entities by name
- ✓ Search supports fuzzy matching via normalization table
- ✓ Can filter search results by entity type
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