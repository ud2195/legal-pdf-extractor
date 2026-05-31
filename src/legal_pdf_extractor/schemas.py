from typing import Any

from pydantic import BaseModel, Field


class Source(BaseModel):
    page: int = Field(ge=1)
    snippet: str = Field(min_length=1)


class ExtractionResponse(BaseModel):
    value: Any = None
    found: bool
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    sources: list[Source] = Field(default_factory=list)
    error: dict[str, str] | None = None


class TextChunk(BaseModel):
    chunk_id: str
    doc_hash: str
    page: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(TextChunk):
    score: float = 0.0
