from pathlib import Path
from typing import Any

from legal_pdf_extractor.api import extract
from legal_pdf_extractor.indexing.base import EmbeddingModel
from legal_pdf_extractor.ingestion.base import PdfParser
from legal_pdf_extractor.ingestion.parsed_document import ParsedDocument, ParsedPage
from legal_pdf_extractor.schemas import TextChunk


class DummyLLM:
    def extract_json(self, prompt: str, output_type: str) -> dict[str, Any]:
        return {
            "value": "Greenfield Properties LLC",
            "found": True,
            "sources": [
                {
                    "page": 1,
                    "snippet": "Tenant: Greenfield Properties LLC.",
                }
            ],
        }


class FakePdfParser(PdfParser):
    def parse(self, pdf_path: str | Path, doc_hash: str) -> ParsedDocument:
        text = (
            "Tenant: Greenfield Properties LLC. "
            "The Tenant shall provide 60 days written notice."
        )
        return ParsedDocument(
            doc_hash=doc_hash,
            pages=[ParsedPage(page=1, text=text)],
            chunks=[
                TextChunk(
                    chunk_id=f"{doc_hash}:docling:c0",
                    doc_hash=doc_hash,
                    page=1,
                    text=text,
                    metadata={"chunker": "test", "pages": [1], "raw_text": text},
                )
            ],
        )


class TrackingPdfParser(FakePdfParser):
    def __init__(self) -> None:
        self.seen_pdf_path: Path | None = None

    def parse(self, pdf_path: str | Path, doc_hash: str) -> ParsedDocument:
        self.seen_pdf_path = Path(pdf_path)
        return super().parse(pdf_path, doc_hash)


class FakeEmbeddingModel(EmbeddingModel):
    def embed(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


def test_extract_orchestrates_hash_cached_pipeline(tmp_path: Path) -> None:
    result = extract(
        pdf=Path("tests/fixtures/sample_contract.pdf"),
        query="Who is the tenant?",
        output_type="string",
        cache_dir=tmp_path,
        llm_client=DummyLLM(),
        parser=FakePdfParser(),
        embeddings=FakeEmbeddingModel(),
    )

    assert result["found"] is True
    assert result["value"] == "Greenfield Properties LLC"
    assert result["confidence"] > 0.0
    assert result["sources"][0]["page"] == 1


def test_extract_cleans_up_temp_pdf_for_bytes_input(tmp_path: Path) -> None:
    parser = TrackingPdfParser()

    result = extract(
        pdf=Path("tests/fixtures/sample_contract.pdf").read_bytes(),
        query="Who is the tenant?",
        output_type="string",
        cache_dir=tmp_path,
        llm_client=DummyLLM(),
        parser=parser,
        embeddings=FakeEmbeddingModel(),
    )

    assert result["found"] is True
    assert parser.seen_pdf_path is not None
    assert parser.seen_pdf_path.suffix == ".pdf"
    assert not parser.seen_pdf_path.exists()
