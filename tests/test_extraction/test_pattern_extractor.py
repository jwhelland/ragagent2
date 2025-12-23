from __future__ import annotations

import pytest
from src.extraction.pattern_extractor import PatternRelationshipExtractor
from src.extraction.models import ExtractedRelationship

class _Chunk:
    def __init__(self, content: str, chunk_id: str = "c1", document_id: str = "d1"):
        self.content = content
        self.chunk_id = chunk_id
        self.document_id = document_id

@pytest.fixture
def extractor():
    return PatternRelationshipExtractor()

def test_extract_is_a(extractor):
    chunk = _Chunk("A solar panel is a type of power source.")
    rels = extractor.extract_relationships(chunk)
    assert len(rels) == 1
    # Expect "a solar panel" as regex captures it
    assert rels[0].source.lower() == "a solar panel"
    assert rels[0].target.lower() == "power source"
    assert rels[0].type == "IS_A"

def test_extract_part_of(extractor):
    chunk = _Chunk("The EPS consists of a battery and a solar array.")
    rels = extractor.extract_relationships(chunk)
    assert len(rels) >= 1
    
    battery_rel = next((r for r in rels if "battery" in r.source), None)
    assert battery_rel
    assert battery_rel.target == "The EPS" or battery_rel.target == "EPS"
    assert battery_rel.type == "PART_OF"

def test_extract_controls(extractor):
    chunk = _Chunk("The OBC controls the power unit.")
    rels = extractor.extract_relationships(chunk)
    assert len(rels) == 1
    assert "OBC" in rels[0].source
    assert "power unit" in rels[0].target
    assert rels[0].type == "CONTROLS"

def test_no_match(extractor):
    chunk = _Chunk("The sun is bright today.")
    rels = extractor.extract_relationships(chunk)
    assert len(rels) == 0

def test_handles_empty_chunk(extractor):
    chunk = _Chunk("")
    rels = extractor.extract_relationships(chunk)
    assert len(rels) == 0
