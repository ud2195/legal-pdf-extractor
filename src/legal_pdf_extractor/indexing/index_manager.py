from pathlib import Path
from typing import TYPE_CHECKING

from legal_pdf_extractor.documents.cache import DocumentCache
from legal_pdf_extractor.errors import PdfParseError
from legal_pdf_extractor.indexing.base import EmbeddingModel
from legal_pdf_extractor.ingestion.base import PdfParser
from legal_pdf_extractor.ingestion.docling_parser import DoclingPdfParser
from legal_pdf_extractor.ingestion.parsed_document import ParsedDocument

if TYPE_CHECKING:
    from legal_pdf_extractor.indexing.vector_store import FaissVectorStore


class IndexManager:
    def __init__(
        self,
        cache: DocumentCache,
        embeddings: EmbeddingModel,
        parser: PdfParser | None = None,
    ) -> None:
        self.cache = cache
        self.parser = parser or DoclingPdfParser()
        self.embeddings = embeddings

    def ensure_index(
        self,
        pdf_path: str | Path,
        doc_hash: str,
    ) -> tuple[ParsedDocument, "FaissVectorStore"]:
        parsed = self.cache.load_parsed(doc_hash)
        if parsed and self.cache.has_index(doc_hash):
            return parsed, self._build_store(doc_hash)

        parsed = parsed or self.parser.parse(pdf_path, doc_hash)
        self.cache.save_parsed(parsed)
        chunks = parsed.chunks
        if not chunks:
            raise PdfParseError("No chunks could be produced from the PDF.")

        store = self._build_store(doc_hash)
        embeddings = self.embeddings.embed_many([chunk.text for chunk in chunks])

        from legal_pdf_extractor.indexing.vector_store import VectorRecord

        records = [
            VectorRecord(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        store.save(doc_hash=doc_hash, records=records)
        return parsed, store

    def _build_store(self, doc_hash: str) -> "FaissVectorStore":
        from legal_pdf_extractor.indexing.vector_store import FaissVectorStore

        return FaissVectorStore(
            index_path=self.cache.index_path(doc_hash),
            metadata_path=self.cache.index_metadata_path(doc_hash),
        )
