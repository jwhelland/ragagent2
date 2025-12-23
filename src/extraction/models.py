"""Shared data models for extraction modules."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ExtractedEntity(BaseModel):
    """Structured representation of an extracted entity."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str
    description: str = ""
    aliases: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    source: str = "unknown"
    raw: Dict[str, Any] | None = None
    
    # Optional fields for provenance (e.g. from spaCy)
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    sentence: Optional[str] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExtractedRelationship(BaseModel):
    """Structured representation of an extracted relationship."""

    model_config = ConfigDict(extra="forbid")

    source: str
    type: str
    target: str
    description: str = ""
    confidence: float = 0.0
    bidirectional: bool = False
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    source_extractor: str = "unknown"
    raw: Dict[str, Any] | None = None
