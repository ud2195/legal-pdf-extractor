from abc import ABC, abstractmethod
from pathlib import Path

from legal_pdf_extractor.ingestion.parsed_document import ParsedDocument


class PdfParser(ABC):
    @abstractmethod
    def parse(self, pdf_path: str | Path, doc_hash: str) -> ParsedDocument:
        raise NotImplementedError
