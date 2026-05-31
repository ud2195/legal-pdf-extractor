from pathlib import Path
from typing import Any

from legal_pdf_extractor.config import load_settings
from legal_pdf_extractor.documents.cache import DocumentCache
from legal_pdf_extractor.documents.hashing import compute_document_hash
from legal_pdf_extractor.documents.models import NormalizedPdf
from legal_pdf_extractor.documents.normalization import (
    PdfInput,
    cleanup_normalized_pdf,
    normalize_pdf_input,
)
from legal_pdf_extractor.errors import LegalPdfExtractorError
from legal_pdf_extractor.extraction.extractor import LegalTermExtractor
from legal_pdf_extractor.indexing.base import EmbeddingModel
from legal_pdf_extractor.indexing.embeddings import OpenAIEmbeddingModel
from legal_pdf_extractor.indexing.index_manager import IndexManager
from legal_pdf_extractor.ingestion.base import PdfParser
from legal_pdf_extractor.llm.base import LLMClient
from legal_pdf_extractor.llm.openai_client import OpenAILLMClient
from legal_pdf_extractor.output.formatter import format_response
from legal_pdf_extractor.retrieval.retriever import Retriever
from legal_pdf_extractor.validation.confidence import ConfidenceScorer
from legal_pdf_extractor.validation.source_verifier import SourceVerifier
from legal_pdf_extractor.validation.type_validator import TypeValidator


def extract(
    pdf: PdfInput,
    query: str,
    output_type: str,
    examples: list[dict[str, Any]] | None = None,
    *,
    cache_dir: str | Path | None = None,
    llm_client: LLMClient | None = None,
    parser: PdfParser | None = None,
    embeddings: EmbeddingModel | None = None,
) -> dict[str, Any]:
    normalized: NormalizedPdf | None = None
    try:
        settings = load_settings(cache_dir=cache_dir)
        normalized = normalize_pdf_input(pdf)
        doc_hash = compute_document_hash(normalized.path)

        cache = DocumentCache(settings.cache_dir)
        embedding_model = embeddings or OpenAIEmbeddingModel(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            batch_size=settings.embedding_batch_size,
            max_workers=settings.embedding_max_workers,
        )
        parsed_document, vector_store = IndexManager(
            cache,
            parser=parser,
            embeddings=embedding_model,
        ).ensure_index(
            pdf_path=normalized.path,
            doc_hash=doc_hash,
        )
        chunks = Retriever(embeddings=embedding_model).retrieve(query=query, store=vector_store)
        client = llm_client or OpenAILLMClient(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
        raw_result = LegalTermExtractor(client).extract(
            query=query,
            output_type=output_type,
            chunks=chunks,
            examples=examples,
        )

        found = bool(raw_result.get("found"))
        value = TypeValidator().validate(raw_result.get("value"), output_type, found=found)
        sources = SourceVerifier().verify(
            sources=raw_result.get("sources", []),
            document=parsed_document,
            found=found,
            chunks=chunks,
        )
        confidence = ConfidenceScorer().score(found=found, sources=sources, chunks=chunks)
        return format_response(
            value=value,
            found=found,
            confidence=confidence,
            sources=sources,
        )
    except LegalPdfExtractorError as exc:
        return format_response(
            value=None,
            found=False,
            confidence=0.0,
            sources=[],
            error=exc.to_error_payload(),
        )
    except Exception as exc:
        return format_response(
            value=None,
            found=False,
            confidence=0.0,
            sources=[],
            error={"code": "unexpected_error", "message": str(exc)},
        )
    finally:
        if normalized is not None:
            cleanup_normalized_pdf(normalized)
