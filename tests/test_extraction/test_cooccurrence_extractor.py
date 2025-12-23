from __future__ import annotations

import pytest
from src.extraction.cooccurrence_extractor import CooccurrenceRelationshipExtractor

class _Chunk:
    def __init__(self, content: str, chunk_id: str = "c1", document_id: str = "d1"):
        self.content = content
        self.chunk_id = chunk_id
        self.document_id = document_id

@pytest.fixture
def extractor():
    return CooccurrenceRelationshipExtractor(confidence=0.4)

def test_extract_cooccurrence(extractor):
    chunk = _Chunk("The battery and the OBC are in the same box.")
    known_entities = [
        {"name": "battery", "type": "COMPONENT"},
        {"name": "OBC", "type": "COMPONENT"}
    ]
    
    rels = extractor.extract_relationships(chunk, known_entities=known_entities)
    
    assert len(rels) == 1
    rel = rels[0]
    assert rel.type == "RELATED_TO"
    assert rel.confidence == 0.4
    assert set([rel.source, rel.target]) == {"battery", "OBC"}
    assert rel.source_extractor == "cooccurrence"

def test_extract_cooccurrence_multiple(extractor):
    chunk = _Chunk("System components: battery, solar panel, sensor.")
    known_entities = [
        {"name": "battery", "type": "COMPONENT"},
        {"name": "solar panel", "type": "COMPONENT"},
        {"name": "sensor", "type": "COMPONENT"}
    ]
    
    rels = extractor.extract_relationships(chunk, known_entities=known_entities)
    
    # 3 entities -> 3 combinations (C(3,2) = 3)
    assert len(rels) == 3
    
    pairs = set([tuple(sorted([r.source, r.target])) for r in rels])
    expected = {
        tuple(sorted(["battery", "solar panel"])),
        tuple(sorted(["battery", "sensor"])),
        tuple(sorted(["solar panel", "sensor"]))
    }
    assert pairs == expected

def test_extract_cooccurrence_empty(extractor):
    chunk = _Chunk("No entities here.")
    rels = extractor.extract_relationships(chunk, known_entities=[])
    assert len(rels) == 0

def test_extract_cooccurrence_single(extractor):
    chunk = _Chunk("Only one entity: battery.")
    rels = extractor.extract_relationships(chunk, known_entities=[{"name": "battery"}])
    assert len(rels) == 0
