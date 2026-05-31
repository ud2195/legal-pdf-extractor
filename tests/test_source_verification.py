import pytest

from legal_pdf_extractor.errors import SourceVerificationError
from legal_pdf_extractor.ingestion.parsed_document import ParsedDocument, ParsedPage
from legal_pdf_extractor.schemas import RetrievedChunk
from legal_pdf_extractor.validation.source_verifier import SourceVerifier


def test_source_verifier_accepts_verbatim_snippet_from_retrieved_chunks() -> None:
    document = ParsedDocument(
        doc_hash="abc",
        pages=[ParsedPage(page=1, text="The Tenant shall provide 60 days written notice.")],
    )
    chunks = [
        RetrievedChunk(
            chunk_id="abc:p1:c0",
            doc_hash="abc",
            page=1,
            text="The Tenant shall provide 60 days written notice.",
        )
    ]

    sources = SourceVerifier().verify(
        [{"page": 1, "snippet": "60 days written notice"}],
        document=document,
        found=True,
        chunks=chunks,
    )

    assert sources[0].page == 1
    assert sources[0].snippet == "60 days written notice"


def test_source_verifier_replaces_modified_snippet_with_best_chunk_match() -> None:
    document = ParsedDocument(
        doc_hash="abc",
        pages=[ParsedPage(page=1, text="The Tenant shall provide 60 days written notice.")],
    )
    chunks = [
        RetrievedChunk(
            chunk_id="abc:p1:c0",
            doc_hash="abc",
            page=1,
            text="The Tenant shall provide 60 days written notice.",
        )
    ]

    sources = SourceVerifier().verify(
        [{"page": 1, "snippet": "Tenant must give sixty days notice"}],
        document=document,
        found=True,
        chunks=chunks,
    )

    assert sources[0].page == 1
    assert sources[0].snippet == "Tenant shall provide 60 days written"


def test_source_verifier_rejects_found_answer_without_sources() -> None:
    document = ParsedDocument(doc_hash="abc", pages=[ParsedPage(page=1, text="Actual text.")])
    chunks = [
        RetrievedChunk(
            chunk_id="abc:p1:c0",
            doc_hash="abc",
            page=1,
            text="Actual text.",
        )
    ]

    with pytest.raises(SourceVerificationError):
        SourceVerifier().verify([], document=document, found=True, chunks=chunks)


def test_source_verifier_requires_retrieved_chunks_for_found_answer() -> None:
    document = ParsedDocument(doc_hash="abc", pages=[ParsedPage(page=1, text="Actual text.")])

    with pytest.raises(SourceVerificationError, match="No source evidence"):
        SourceVerifier().verify(
            [{"page": 1, "snippet": "Actual text."}],
            document=document,
            found=True,
            chunks=None,
        )
