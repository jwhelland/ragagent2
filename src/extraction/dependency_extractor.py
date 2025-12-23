"""Syntactic dependency-based relationship extractor."""

from typing import Any, Dict, List, Optional, Set

import spacy
from loguru import logger
from spacy.language import Language
from spacy.matcher import DependencyMatcher
from spacy.tokens import Doc

from src.extraction.models import ExtractedRelationship
from src.utils.config import SpacyConfig


class DependencyRelationshipExtractor:
    """Extracts relationships using spaCy dependency parsing."""

    def __init__(
        self,
        config: Optional[SpacyConfig] = None,
        nlp: Optional[Language] = None,
    ) -> None:
        self.config = config or SpacyConfig()
        # Use provided nlp or load a new one. Ideally reuse the one from SpacyExtractor to save memory.
        self.nlp: Language = nlp or self._load_model(self.config.model)
        
        self.matcher = DependencyMatcher(self.nlp.vocab)
        self._register_patterns()
        
        logger.info("Initialized DependencyRelationshipExtractor")

    def extract_relationships(
        self,
        chunk: Any,
        *,
        known_entities: Optional[List[Dict[str, Any]]] = None, # Not strictly used yet, but good for filtering
        document_context: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractedRelationship]:
        """Extract relationships from a chunk using dependency matching."""
        text = getattr(chunk, "content", None)
        if text is None and isinstance(chunk, dict):
            text = chunk.get("content")
        
        if not text:
            return []

        chunk_id = getattr(chunk, "chunk_id", None) or getattr(chunk, "id", None) or (chunk.get("chunk_id") if isinstance(chunk, dict) else None)
        document_id = getattr(chunk, "document_id", None) or (chunk.get("document_id") if isinstance(chunk, dict) else None)

        # Process text
        doc = self.nlp(text)
        matches = self.matcher(doc)

        relationships: List[ExtractedRelationship] = []
        
        for match_id, token_ids in matches:
            pattern_name = self.nlp.vocab.strings[match_id]
            
            # Pattern format assumes: [subject, verb, object]
            if len(token_ids) != 3:
                continue
                
            verb_idx, subj_idx, obj_idx = token_ids
            subject_text = self._expand_noun_phrase(doc[subj_idx])
            object_text = self._expand_noun_phrase(doc[obj_idx])
            verb_token = doc[verb_idx]
            
            # Map verb to relationship type
            rel_type = self._map_verb_to_type(verb_token.lemma_, pattern_name)
            
            if not rel_type:
                continue

            relationships.append(
                ExtractedRelationship(
                    source=subject_text,
                    target=object_text,
                    type=rel_type,
                    description=f"Extracted via dependency: {verb_token.text}",
                    confidence=0.7, # Moderate confidence for syntactic extraction
                    bidirectional=False,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    source_extractor="spacy_dependency",
                    raw={
                        "pattern": pattern_name,
                        "verb": verb_token.text,
                        "subject": subject_text,
                        "object": object_text
                    }
                )
            )

        return relationships

    def _load_model(self, model_name: str) -> Language:
        try:
            return spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed. "
            ) from exc

    def _register_patterns(self) -> None:
        """Register dependency patterns for SVO triples."""
        
        # Simple SVO: Subject -> Verb -> Object
        svo_pattern = [
            {
                "RIGHT_ID": "verb",
                "RIGHT_ATTRS": {"POS": "VERB"}
            },
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "subject",
                "RIGHT_ATTRS": {"DEP": {"IN": ["nsubj", "nsubjpass"]}}
            },
            {
                "LEFT_ID": "verb",
                "REL_OP": ">",
                "RIGHT_ID": "object",
                "RIGHT_ATTRS": {"DEP": {"IN": ["dobj", "pobj", "attr"]}} # attr handles "X is Y"
            }
        ]
        
        self.matcher.add("SVO", [svo_pattern])

    def _expand_noun_phrase(self, token) -> str:
        """Expand a token to its full noun phrase."""
        # This is a simple heuristic. A more robust way uses doc.noun_chunks
        # but matching tokens to chunks can be tricky.
        # We'll use the subtree but filter for relevant parts.
        
        # Prefer using the noun chunk if the token is inside one
        for chunk in token.doc.noun_chunks:
            if token in chunk:
                return chunk.text
        
        # Fallback: simple subtree expansion (might be too greedy)
        return "".join([t.text_with_ws for t in token.subtree]).strip()

    def _map_verb_to_type(self, lemma: str, pattern_name: str) -> Optional[str]:
        """Map verb lemmas to relationship types."""
        verb_map = {
            "control": "CONTROLS",
            "monitor": "MONITORS",
            "measure": "MONITORS",
            "contain": "CONTAINS",
            "include": "CONTAINS",
            "comprise": "CONTAINS",
            "provide": "PROVIDES", # ambiguous
            "power": "PROVIDES_POWER_TO",
            "feed": "PROVIDES_POWER_TO",
            "trigger": "TRIGGERS",
            "cause": "CAUSES",
            "connect": "CONNECTS_TO",
            "support": "SUPPORTS",
            "require": "REQUIRES",
        }
        
        if lemma in verb_map:
            return verb_map[lemma]
            
        # "Is" handling for hierarchies
        if lemma == "be":
            return "IS_A" # Very broad, might need filtering
            
        return None
