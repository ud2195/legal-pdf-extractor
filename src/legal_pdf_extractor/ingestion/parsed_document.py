from pydantic import BaseModel, Field

from legal_pdf_extractor.schemas import TextChunk


class ParsedPage(BaseModel):
    page: int = Field(ge=1)
    text: str
    metadata: dict[str, str] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    doc_hash: str
    pages: list[ParsedPage]
    chunks: list[TextChunk] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
