# Relationship Extraction Improvement Plan

## Summary

Improve relationship extraction quality by:
1. Removing CooccurrenceRelationshipExtractor (creates N² generic RELATED_TO relationships)
2. Removing RELATED_TO from schema (too generic, not used in retrieval)
3. Adding relationship validation (confidence threshold + entity validation)
4. Fixing IS_A over-generation from dependency parser
5. Extending curation workflow to relationships

**Expected Impact**: 60-85% reduction in relationship candidates with significantly higher quality.

## Root Cause Analysis

### Problem 1: CooccurrenceRelationshipExtractor
- **Location**: `src/extraction/cooccurrence_extractor.py`
- **Issue**: Creates combinatorial RELATED_TO relationships for all entity pairs in a chunk (8 entities = 28 relationships)
- **Impact**: 50-80% of all relationships are generic RELATED_TO with confidence=0.4
- **Always enabled** at `src/pipeline/ingestion_pipeline.py:263`

### Problem 2: RELATED_TO Type
- **Defined** at `src/storage/schemas.py:374`
- **Referenced** in `config/extraction_prompts.yaml:121`
- **Fallback** in `src/extraction/pattern_extractor.py:45`
- **Not in active config** - already excluded from `config/config.yaml` relationship_types list
- **Never used in retrieval** - graph_retriever.py filters by specific types only

### Problem 3: No Relationship Validation
- `src/pipeline/ingestion_pipeline.py:1319-1389` (_store_relationship_candidates) accepts ALL relationships
- No confidence threshold filtering
- No entity existence validation (source/target can be arbitrary text)
- No post-extraction quality checks

### Problem 4: IS_A Over-Generation
- **Location**: `src/extraction/dependency_extractor.py:169`
- Maps ALL "be" verbs to IS_A relationship
- Creates noise like "Battery is critical" → Battery IS_A critical
- Comment in code acknowledges: "Very broad, might need filtering"

---

## Implementation Plan

### Phase 1: Remove CooccurrenceRelationshipExtractor

**Files to Delete**:
- `src/extraction/cooccurrence_extractor.py` (73 lines)
- `tests/test_extraction/test_cooccurrence_extractor.py`

**Files to Modify**:
- `src/pipeline/ingestion_pipeline.py`:
  - Remove import at ~line 60
  - Remove initialization at line 263
  - Remove execution in `_extract_rule_based_relationships()` at lines 1418-1433

**Verification**: No RELATED_TO relationships created after this change.

---

### Phase 2: Remove RELATED_TO from Schema

**Files to Modify**:

1. **`src/storage/schemas.py:374`**
   - Delete `RELATED_TO = "RELATED_TO"` from RelationshipType enum

2. **`config/extraction_prompts.yaml:121`**
   - Remove `- RELATED_TO: source has a generic relation to target` from list

3. **`src/extraction/pattern_extractor.py:45`**
   - Change fallback from `"RELATED_TO"` to `None`
   - Add validation to skip patterns without explicit relationship_type:
   ```python
   rel_type = group.get("relationship_type")
   if not rel_type:
       logger.warning(f"Skipping pattern without relationship_type: {pattern_str}")
       continue
   ```

4. **`scripts/cleanup_relationships.py:129`**
   - Change default `type_filter` from `"RELATED_TO"` to `"ALL"`

**Files to Update** (test fixtures):
- `tests/test_curation/test_relationship_candidate_undo.py` - replace RELATED_TO with DEPENDS_ON
- `tests/test_curation/test_relationship_promotion.py` - replace RELATED_TO with DEPENDS_ON
- `tests/test_curation/test_relationship_promotion_invalid_mentions.py` - replace RELATED_TO with DEPENDS_ON

**Migration Script** (create `scripts/migrate_related_to.py`):
- Report on existing RELATED_TO relationships in database
- Bulk reject pending RELATED_TO candidates
- Validation command to ensure clean removal

---

### Phase 3: Add Relationship Validation

#### Step 3.1: Configuration Schema

**File**: `src/utils/config.py` (add new class)
```python
class RelationshipValidationConfig(BaseModel):
    """Configuration for relationship validation."""
    min_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    validate_entity_existence: bool = Field(default=True)
    fuzzy_entity_match_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    max_entity_name_length: int = Field(default=100)
```

Add to `ExtractionConfig`:
```python
relationship_validation: RelationshipValidationConfig = Field(default_factory=RelationshipValidationConfig)
```

**File**: `config/config.yaml` (add after line 98)
```yaml
  relationship_validation:
    min_confidence_threshold: 0.5
    validate_entity_existence: true
    fuzzy_entity_match_threshold: 0.85
    max_entity_name_length: 100
```

#### Step 3.2: Create Validator Module

**New File**: `src/extraction/relationship_validator.py`

**Key Methods**:
- `validate_relationship(relationship, known_entities)` → ValidationResult
  - Check confidence threshold
  - Check entity name length
  - Validate source/target exist in known_entities (with fuzzy matching)
  - Check IS_A relationship quality (reject if target is adjective)
- `filter_relationships(relationships, known_entities)` → (valid, rejected)

**Implementation Details**:
- Use `StringNormalizer` for entity name normalization
- Use `FuzzyMatcher` for entity existence validation (threshold=0.85)
- Return both valid relationships and rejected with reasons for logging
- ~150 lines of code

#### Step 3.3: Integrate in Pipeline

**File**: `src/pipeline/ingestion_pipeline.py`

**Changes**:
1. Add import: `from src.extraction.relationship_validator import RelationshipValidator`
2. Initialize validator in `initialize()` method (~line 285)
3. Modify `_extract_rule_based_relationships()` (lines 1391-1435):
   - Gather known entities from `llm_entities` + `spacy_entities`
   - Call `validator.filter_relationships()` before storing
   - Log filtered count
4. Modify `_extract_llm_relationships()` (lines 1437-1495):
   - Same pattern: validate before storing
   - Log filtered count
5. Update `IngestionResult` to include `relationships_filtered` count

**Validation Points**:
- Confidence threshold (default 0.5)
- Entity existence (fuzzy match against extracted entities)
- Entity name length (max 100 chars)
- IS_A semantic check (reject adjective targets)

---

### Phase 4: Fix IS_A and Review SIMILAR_TO

#### IS_A Fix

**File**: `src/extraction/dependency_extractor.py:167-170`

**Change**:
```python
# Remove automatic IS_A for "be" verb
# This was too noisy and created invalid relationships
# IS_A relationships should come from:
# 1. Pattern extractor with explicit "is a type of" patterns
# 2. LLM extractor with semantic understanding

# Delete lines 167-169:
# if lemma == "be":
#     return "IS_A"

return None
```

**Rationale**:
- Pattern extractor already has proper IS_A patterns ("is a type of", "is a kind of")
- LLM extractor can create IS_A when semantically appropriate
- Dependency parser creating IS_A for all "be" verbs is too noisy

#### IS_A Validation Enhancement

**File**: `src/extraction/relationship_validator.py`

Add semantic check in `validate_relationship()`:
```python
# IS_A relationship semantic check
if relationship.type == "IS_A":
    target_lower = relationship.target.lower()

    # Reject IS_A if target looks like adjective/attribute
    adjective_indicators = [
        "critical", "important", "essential", "primary", "secondary",
        "active", "passive", "normal", "abnormal", "high", "low",
    ]
    if any(word in target_lower for word in adjective_indicators):
        return ValidationResult(
            valid=False,
            reason=f"IS_A target '{relationship.target}' appears to be adjective, not type"
        )
```

#### SIMILAR_TO Assessment

**Finding**: SIMILAR_TO is acceptable as-is
- Only created by LLM extractor with semantic understanding
- Has clear meaning in prompts
- Useful for retrieval (finding analogous entities)
- Already subject to confidence scoring

**Action**: No changes needed. Monitor usage after validation is in place.

#### Prompt Clarifications

**File**: `config/extraction_prompts.yaml`

Update guidance (after line 137):
```yaml
    - For IS_A: only use when target is a category/type, not an attribute or adjective
    - For SIMILAR_TO: only use when entities serve similar purposes, not just mentioned together
```

---

### Phase 5: Extend Curation to Relationships

#### Step 5.1: CLI Commands

**File**: `src/curation/review_interface.py`

**Add relationship command group** (after line 30):
```python
relationship_app = typer.Typer(help="Review and curate relationship candidates.")
app.add_typer(relationship_app, name="relationship")
```

**Commands to Add**:
- `relationship queue` - List relationship candidates with filters (status, type, confidence)
- `relationship show <key>` - Show details of a relationship candidate
- `relationship approve <key>` - Approve a relationship candidate
- `relationship reject <key>` - Reject a relationship candidate
- `relationship batch-approve` - Batch approve high-confidence relationships

**Usage Examples**:
```bash
uv run ragagent-review relationship queue --status pending --min-confidence 0.7
uv run ragagent-review relationship show "<key>"
uv run ragagent-review relationship approve "<key>"
uv run ragagent-review relationship batch-approve --min-confidence 0.8 --dry-run
```

#### Step 5.2: Neo4j Query Methods

**File**: `src/storage/neo4j_manager.py`

**Add methods** (around line 1500):
- `get_relationship_candidates(status, rel_type, min_confidence, limit, offset)` → List[Dict]
- `get_relationship_candidate(identifier)` → Optional[Dict]

**Query Logic**:
```cypher
MATCH (r:RelationshipCandidate)
WHERE ($status IS NULL OR r.status = $status)
  AND ($rel_type IS NULL OR r.type = $rel_type)
  AND ($min_confidence IS NULL OR r.confidence_score >= $min_confidence)
RETURN r
ORDER BY r.confidence_score DESC, r.mention_count DESC
```

#### Step 5.3: Interactive TUI Updates

**File**: `src/curation/interactive/app.py`

**Add mode toggle**:
- Create `ReviewMode` enum (ENTITIES, RELATIONSHIPS)
- Add `action_toggle_mode()` to switch between entity and relationship review
- Add keybinding `m` for mode toggle

**New Widget**: `src/curation/interactive/widgets/relationship_list.py`
- DataTable showing: Source, Type, Target, Confidence, Status
- Reuses existing session tracker and undo system

**Integration**:
- Reuse existing command palette (add relationship commands)
- Reuse existing batch operations modal
- Update context panel to show relationship details in relationship mode

#### Step 5.4: Documentation

**File**: `CLAUDE.md`

Add section after "Entity Discovery and Curation":
```markdown
### Relationship Curation
```bash
# Review relationship candidates
uv run ragagent-review relationship queue --status pending --min-confidence 0.7
uv run ragagent-review relationship approve "<key>"
uv run ragagent-review relationship reject "<key>"

# Batch operations
uv run ragagent-review relationship batch-approve --min-confidence 0.8 --dry-run

# Interactive TUI (press 'm' to toggle between entities and relationships)
uv run ragagent-review-interactive
```
```

---

### Phase 6: Testing

#### Tests to Update
1. Delete: `tests/test_extraction/test_cooccurrence_extractor.py`
2. Update test fixtures: Replace RELATED_TO with DEPENDS_ON in:
   - `tests/test_curation/test_relationship_candidate_undo.py`
   - `tests/test_curation/test_relationship_promotion.py`
   - `tests/test_curation/test_relationship_promotion_invalid_mentions.py`

#### New Tests to Create

**File**: `tests/test_extraction/test_relationship_validator.py`
- Test confidence threshold filtering
- Test entity existence validation (with fuzzy matching)
- Test IS_A adjective rejection
- Test entity name length limits
- Test batch filtering

**File**: `tests/test_extraction/test_dependency_extractor_is_a.py`
- Test that "be" verb + adjective doesn't create IS_A
- Test that dependency extractor never returns IS_A
- Test that IS_A only comes from patterns or LLM

**File**: `tests/test_pipeline/test_relationship_extraction_integration.py`
- Test validation integration in pipeline
- Test filtered count statistics
- Test no RELATED_TO relationships created

---

### Phase 7: Migration and Deployment

#### Migration Script

**Create**: `scripts/migrate_related_to.py`

**Commands**:
- `report` - Show count and examples of existing RELATED_TO relationships
- `reject-related-to` - Bulk reject all pending RELATED_TO candidates
- `validate-schema` - Verify no active RELATED_TO relationships remain

**Usage**:
```bash
# Pre-deployment report
uv run python scripts/migrate_related_to.py report

# Execute migration
uv run python scripts/migrate_related_to.py reject-related-to --no-dry-run

# Post-deployment validation
uv run python scripts/migrate_related_to.py validate-schema
```

#### Deployment Checklist

**Pre-Deployment**:
- [ ] Run full test suite: `uv run pytest`
- [ ] Backup Neo4j database
- [ ] Run migration report to understand impact

**Deployment**:
- [ ] Deploy code changes
- [ ] Run migration script to reject RELATED_TO candidates
- [ ] Validate schema cleanup
- [ ] Reingest sample documents to verify behavior

**Post-Deployment**:
- [ ] Review relationship quality with CLI
- [ ] Batch approve high-confidence relationships (0.8+)
- [ ] Monitor filtered relationship statistics in logs
- [ ] Tune confidence threshold if needed (config parameter)

#### Configuration Tuning

**Initial Settings** (conservative):
```yaml
extraction:
  relationship_validation:
    min_confidence_threshold: 0.5  # Start conservative
    validate_entity_existence: true
    fuzzy_entity_match_threshold: 0.85
    max_entity_name_length: 100
```

**Tuning Guidance**:
- Monitor `relationships_filtered` count in logs
- If too many false negatives: lower threshold to 0.4
- If too many false positives: raise threshold to 0.6-0.7
- Review rejected relationships to identify patterns

---

## Critical Files Summary

### Files to Delete (2)
1. `src/extraction/cooccurrence_extractor.py`
2. `tests/test_extraction/test_cooccurrence_extractor.py`

### Files to Modify (12)
1. `src/pipeline/ingestion_pipeline.py` - Remove cooccurrence, add validation
2. `src/storage/schemas.py` - Remove RELATED_TO enum (line 374)
3. `config/extraction_prompts.yaml` - Remove RELATED_TO (line 121), clarify IS_A/SIMILAR_TO
4. `src/extraction/pattern_extractor.py` - Remove RELATED_TO fallback (line 45)
5. `src/extraction/dependency_extractor.py` - Remove IS_A for "be" verbs (lines 167-169)
6. `scripts/cleanup_relationships.py` - Update default filter (line 129)
7. `src/utils/config.py` - Add RelationshipValidationConfig
8. `config/config.yaml` - Add validation configuration
9. `src/curation/review_interface.py` - Add relationship commands
10. `src/storage/neo4j_manager.py` - Add relationship candidate queries
11. `src/curation/interactive/app.py` - Add relationship mode toggle
12. `CLAUDE.md` - Document new features

### Files to Create (6)
1. `src/extraction/relationship_validator.py` - New validation module (~150 lines)
2. `src/curation/interactive/widgets/relationship_list.py` - TUI widget (~50 lines)
3. `tests/test_extraction/test_relationship_validator.py` - Validation tests
4. `tests/test_pipeline/test_relationship_extraction_integration.py` - Integration tests
5. `tests/test_extraction/test_dependency_extractor_is_a.py` - IS_A fix tests
6. `scripts/migrate_related_to.py` - Migration script (~150 lines)

---

## Expected Outcomes

### Quantitative Impact
- **60-85% reduction** in relationship candidates
- **Higher average confidence** (no more 0.4 cooccurrence relationships)
- **Meaningful relationships only** (26 specific types, no generic RELATED_TO)

### Qualitative Impact
- **Higher precision** - validation filters low-quality extractions
- **Clearer semantics** - every relationship has specific meaning
- **Better retrieval** - graph traversal uses meaningful relationships only
- **Quality control** - manual curation enables continuous improvement

### Risk Mitigation
- Conservative validation thresholds (0.5 confidence)
- Optional validation (can be disabled in config)
- Migration script handles existing data
- Comprehensive testing before deployment
- Tuning recommendations for post-deployment

---

## Implementation Priority

### High Priority (Do First)
1. **Phase 1-2**: Remove cooccurrence extractor and RELATED_TO (cleanup, high impact)
2. **Phase 3**: Add validation (foundation for quality)
3. **Phase 4**: Fix IS_A (reduce noise)

### Medium Priority (Do Second)
4. **Phase 5**: Extend curation (enables quality control)
5. **Phase 6**: Testing (ensure correctness)

### Lower Priority (Do Last)
6. **Phase 7**: Migration and deployment (operational)

### Estimated Effort
- **High Priority**: 4 days
- **Medium Priority**: 5 days
- **Lower Priority**: 1 day
- **Total**: ~10 days (2 weeks)
