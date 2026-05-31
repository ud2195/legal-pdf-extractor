from legal_pdf_extractor.schemas import RetrievedChunk, Source
from legal_pdf_extractor.validation.confidence import ConfidenceScorer


def test_confidence_uses_verified_source_and_candidate_rank() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="before",
            doc_hash="abc",
            page=1,
            text="Nearby context with Tenant: Greenfield Properties LLC.",
            metadata={"retrieval_role": "context", "confidence_rank": 0},
        ),
        RetrievedChunk(
            chunk_id="hit",
            doc_hash="abc",
            page=1,
            text="The lease identifies the tenant.",
            metadata={"retrieval_role": "candidate", "confidence_rank": 0},
            score=0.9,
        ),
    ]

    confidence = ConfidenceScorer().score(
        found=True,
        sources=[Source(page=1, snippet="Tenant: Greenfield Properties LLC")],
        chunks=chunks,
    )

    assert confidence == 1.0


def test_confidence_is_zero_without_verified_source() -> None:
    confidence = ConfidenceScorer().score(
        found=True,
        sources=[Source(page=1, snippet="Missing source")],
        chunks=[
            RetrievedChunk(
                chunk_id="hit",
                doc_hash="abc",
                page=1,
                text="Actual source text.",
                metadata={"retrieval_role": "candidate", "confidence_rank": 0},
            )
        ],
    )

    assert confidence == 0.0
