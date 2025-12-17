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