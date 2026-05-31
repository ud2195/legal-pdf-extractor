from typing import Any

from legal_pdf_extractor.schemas import ExtractionResponse, Source


def format_response(
    value: Any,
    found: bool,
    sources: list[Source],
    confidence: float | None = None,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    return ExtractionResponse(
        value=value,
        found=found,
        confidence=confidence,
        sources=sources,
        error=error,
    ).model_dump(exclude_none=True)
