from __future__ import annotations

from typing import List

from pytest import approx

from src.extraction.entity_merger import EntityMerger
from src.extraction.llm_extractor import LLMExtractedEntity
from src.extraction.spacy_extractor import ExtractedEntity


def _spacy_entity(
    text: str,
    label: str,
    confidence: float,
    chunk_id: str = "c1",
    document_id: str = "doc1",
) -> ExtractedEntity:
    return ExtractedEntity(
        text=text,
        label=label,
        confidence=confidence,
        start_char=0,
        end_char=len(text),
        sentence=text,
        context=text,
        chunk_id=chunk_id,
        document_id=document_id,
    )


def _llm_entity(
    name: str,
    ent_type: str,
    confidence: float,
    aliases: List[str] | None = None,
    chunk_id: str = "c1",
    document_id: str = "doc1",
    description: str = "",
) -> LLMExtractedEntity:
    return LLMExtractedEntity(
        name=name,
        type=ent_type,
        confidence=confidence,
        aliases=aliases or [],
        chunk_id=chunk_id,
        document_id=document_id,
        description=description,
    )


def test_merger_combines_spacy_and_llm_entities() -> None:
    spacy_entities = [_spacy_entity("solar array", "COMPONENT", 0.6)]
    llm_entities = [_llm_entity("solar array", "COMPONENT", 0.82, aliases=["solar panels"])]

    merger = EntityMerger()
    merged = merger.merge(spacy_entities, llm_entities)

    assert len(merged) == 1
    candidate = merged[0]
    assert candidate.canonical_name == "solar array"
    assert candidate.resolved_type == "COMPONENT"
    assert candidate.mention_count == 2
    assert candidate.combined_confidence > 0.82
    assert {p.extractor for p in candidate.provenance} == {"spacy", "llm"}
    assert set(candidate.aliases) >= {"solar array", "solar panels"}


def test_merger_resolves_type_conflicts() -> None:
    spacy_entities = [_spacy_entity("platform", "SYSTEM", 0.7)]
    llm_entities = [_llm_entity("platform", "COMPONENT", 0.55)]

    merger = EntityMerger()
    merged = merger.merge(spacy_entities, llm_entities)

    assert len(merged) == 1
    candidate = merged[0]
    assert candidate.resolved_type == "SYSTEM"
    assert "COMPONENT" in candidate.conflicting_types
    assert candidate.combined_confidence == approx(1 - (1 - 0.63) * (1 - 0.55) + 0.05, rel=0.05)
    assert candidate.chunk_id == "c1"


def test_merger_deduplicates_with_aliases_and_multiple_mentions() -> None:
    spacy_entities = [
        _spacy_entity("electrical power system", "SUBSYSTEM", 0.65),
        _spacy_entity("electrical power system", "SUBSYSTEM", 0.5),
    ]
    llm_entities = [
        _llm_entity(
            "EPS",
            "SUBSYSTEM",
            0.6,
            aliases=["electrical power system"],
            description="Electrical power subsystem",
        )
    ]

    merger = EntityMerger()
    merged = merger.merge(spacy_entities, llm_entities)

    assert len(merged) == 1
    candidate = merged[0]
    assert candidate.canonical_name == "EPS"
    assert "electrical power system" in candidate.aliases
    assert candidate.mention_count == 3  # two spaCy hits + one LLM hit
    assert candidate.resolved_type == "SUBSYSTEM"


def test_merger_does_not_merge_across_chunks() -> None:
    spacy_entities = [_spacy_entity("battery", "COMPONENT", 0.7, chunk_id="c1")]
    llm_entities = [_llm_entity("battery", "COMPONENT", 0.9, chunk_id="c2")]

    merger = EntityMerger()
    merged = merger.merge(spacy_entities, llm_entities)

    assert len(merged) == 2
    assert {candidate.chunk_id for candidate in merged} == {"c1", "c2"}


def test_merger_filters_disallowed_types() -> None:
    spacy_entities = [_spacy_entity("battery", "COMPONENT", 0.7)]
    llm_entities = [_llm_entity("platform", "SYSTEM", 0.9)]

    merger = EntityMerger(allowed_types=["SYSTEM"])
    merged = merger.merge(spacy_entities, llm_entities)

    assert len(merged) == 1
    assert merged[0].canonical_name == "platform"
    assert merged[0].resolved_type == "SYSTEM"


def test_merger_preserves_spacy_provenance_fields() -> None:
    spacy_entities = [
        ExtractedEntity(
            text="solar array",
            label="COMPONENT",
            confidence=0.6,
            start_char=10,
            end_char=20,
            sentence="The solar array is deployed.",
            context="... The solar array is deployed. ...",
            chunk_id="c1",
            document_id="doc1",
            metadata={"source": "ner"},
        )
    ]

    merger = EntityMerger()
    merged = merger.merge(spacy_entities, [])

    assert len(merged) == 1
    prov = merged[0].provenance[0]
    assert prov.extractor == "spacy"
    assert prov.start_char == 10
    assert prov.end_char == 20
    assert prov.sentence == "The solar array is deployed."
    assert prov.context == "... The solar array is deployed. ..."
    assert prov.metadata["source"] == "ner"


def test_merger_handles_empty_inputs() -> None:
    merger = EntityMerger()
    assert merger.merge([], []) == []


def test_merger_uses_string_normalizer_for_matching() -> None:
    spacy_entities = [_spacy_entity("C++ Controller", "COMPONENT", 0.6)]
    llm_entities = [_llm_entity("c++ controller", "COMPONENT", 0.7, aliases=["controller (C++)"])]

    merger = EntityMerger()
    merged = merger.merge(spacy_entities, llm_entities)

    assert len(merged) == 1
    candidate = merged[0]
    assert "c++" in candidate.canonical_name.lower()
    assert candidate.canonical_normalized == "c++ controller"
    assert candidate.mention_count == 2
