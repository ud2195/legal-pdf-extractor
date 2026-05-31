from typing import Any

from legal_pdf_extractor.extraction.examples import normalize_examples
from legal_pdf_extractor.extraction.prompts import ExtractionPromptBuilder
from legal_pdf_extractor.llm.base import LLMClient
from legal_pdf_extractor.schemas import RetrievedChunk


class LegalTermExtractor:
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_builder: ExtractionPromptBuilder | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or ExtractionPromptBuilder()

    def extract(
        self,
        query: str,
        output_type: str,
        chunks: list[RetrievedChunk],
        examples: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        prompt = self.prompt_builder.build(
            query=query,
            output_type=output_type,
            chunks=chunks,
            examples=normalize_examples(examples),
        )
        return self.llm_client.extract_json(prompt=prompt, output_type=output_type)
