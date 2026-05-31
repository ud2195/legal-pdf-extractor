from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from legal_pdf_extractor.errors import PdfParseError
from legal_pdf_extractor.indexing.base import DocumentChunker
from legal_pdf_extractor.indexing.docling_hybrid_chunker import DoclingHybridChunker
from legal_pdf_extractor.ingestion.base import PdfParser
from legal_pdf_extractor.ingestion.parsed_document import ParsedDocument, ParsedPage
from legal_pdf_extractor.schemas import TextChunk


class DoclingPdfParser(PdfParser):
    def __init__(
        self,
        converter: DocumentConverter | None = None,
        chunker: DocumentChunker | None = None,
    ) -> None:
        self.converter = converter or _build_converter()
        self.chunker = chunker

    def parse(self, pdf_path: str | Path, doc_hash: str) -> ParsedDocument:
        document = self.converter.convert(Path(pdf_path)).document
        chunker = self.chunker or _build_chunker()
        chunks = [
            chunk
            for chunk in chunker.chunk(document, doc_hash=doc_hash)
            if chunk.text.strip()
        ]
        if not chunks:
            raise PdfParseError("No chunks could be extracted from the PDF.")

        return ParsedDocument(
            doc_hash=doc_hash,
            pages=_pages_from_chunks(chunks),
            chunks=chunks,
            metadata={"parser": "docling_hybrid"},
        )


def _build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )


def _build_chunker() -> DocumentChunker:
    return DoclingHybridChunker()


def _pages_from_chunks(chunks: list[TextChunk]) -> list[ParsedPage]:
    page_text: dict[int, list[str]] = {}
    for chunk in chunks:
        pages = chunk.metadata.get("pages") or [chunk.page]
        for page in pages:
            page_text.setdefault(page, []).append(chunk.text)
    return [
        ParsedPage(
            page=page,
            text="\n\n".join(texts),
            metadata={"source": "docling_hybrid_chunks"},
        )
        for page, texts in sorted(page_text.items())
    ]
