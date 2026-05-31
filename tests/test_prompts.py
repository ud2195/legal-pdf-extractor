from legal_pdf_extractor.extraction.prompts import ExtractionPromptBuilder
from legal_pdf_extractor.schemas import RetrievedChunk


def test_prompt_builder_omits_examples_when_not_provided() -> None:
    prompt = ExtractionPromptBuilder().build(
        query="Who is the tenant?",
        output_type="string",
        chunks=[
            RetrievedChunk(
                chunk_id="abc:p1:c0",
                doc_hash="abc",
                page=1,
                text="Tenant: Greenfield Properties LLC.",
            )
        ],
    )

    assert "Few-shot examples" not in prompt
    assert "chunk_id=" not in prompt
    assert "[page=1]" in prompt
    assert "Tenant: Greenfield Properties LLC." in prompt


def test_prompt_builder_appends_examples_when_provided() -> None:
    prompt = ExtractionPromptBuilder().build(
        query="Who is the tenant?",
        output_type="string",
        chunks=[],
        examples=[
            {
                "input": "Tenant is Acme LLC.",
                "output": {"value": "Acme LLC", "found": True, "sources": []},
            }
        ],
    )

    assert "Few-shot examples" in prompt
    assert "Acme LLC" in prompt
