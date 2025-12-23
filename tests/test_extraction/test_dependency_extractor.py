from __future__ import annotations

import pytest
import spacy
from src.extraction.dependency_extractor import DependencyRelationshipExtractor
from src.utils.config import SpacyConfig

class _Chunk:
    def __init__(self, content: str, chunk_id: str = "c1", document_id: str = "d1"):
        self.content = content
        self.chunk_id = chunk_id
        self.document_id = document_id

@pytest.fixture
def extractor():
    # Model should now be available
    config = SpacyConfig(model="en_core_web_sm")
    return DependencyRelationshipExtractor(config=config)

def test_extract_controls_svo(extractor):
    chunk = _Chunk("The main computer controls the thrusters.")
    rels = extractor.extract_relationships(chunk)
    
    assert len(rels) >= 1
    # Filter to find the relevant relationship in case of noise
    rel = next((r for r in rels if r.type == "CONTROLS"), None)
    assert rel is not None
    assert "computer" in rel.source
    assert "thrusters" in rel.target

def test_extract_monitors_svo(extractor):
    chunk = _Chunk("Sensors monitor the temperature.")
    rels = extractor.extract_relationships(chunk)
    
    assert len(rels) >= 1
    rel = next((r for r in rels if r.type == "MONITORS"), None)
    assert rel is not None
    assert "Sensors" in rel.source
    assert "temperature" in rel.target

def test_extract_contains_svo(extractor):
    chunk = _Chunk("The module includes a battery.")
    rels = extractor.extract_relationships(chunk)
    
    assert len(rels) >= 1
    rel = next((r for r in rels if r.type == "CONTAINS"), None)
    assert rel is not None
    assert "module" in rel.source
    assert "battery" in rel.target

def test_no_relationship(extractor):
    chunk = _Chunk("The system is running.")
    rels = extractor.extract_relationships(chunk)
    # "running" is not in our verb map
    # We might match "is" -> IS_A depending on config, but "running" isn't a mapped entity usually unless configured
    # In dependency extractor, we map "be" -> IS_A. 
    # "system is running" -> nsubj(running, system), aux(running, is).
    # The verb head is "running". We don't map "run".
    
    matched_rels = [r for r in rels if r.source_extractor == "spacy_dependency"]
    assert len(matched_rels) == 0