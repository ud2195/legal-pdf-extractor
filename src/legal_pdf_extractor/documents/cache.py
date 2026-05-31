from pathlib import Path

from legal_pdf_extractor.ingestion.parsed_document import ParsedDocument


class DocumentCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def document_dir(self, doc_hash: str) -> Path:
        return self.root / doc_hash

    def parsed_path(self, doc_hash: str) -> Path:
        return self.document_dir(doc_hash) / "parsed_document.json"

    def index_path(self, doc_hash: str) -> Path:
        return self.document_dir(doc_hash) / "vector_index.faiss"

    def index_metadata_path(self, doc_hash: str) -> Path:
        return self.document_dir(doc_hash) / "vector_metadata.json"

    def has_index(self, doc_hash: str) -> bool:
        return self.index_path(doc_hash).exists() and self.index_metadata_path(doc_hash).exists()

    def load_parsed(self, doc_hash: str) -> ParsedDocument | None:
        path = self.parsed_path(doc_hash)
        if not path.exists():
            return None
        return ParsedDocument.model_validate_json(path.read_text())

    def save_parsed(self, document: ParsedDocument) -> None:
        path = self.parsed_path(document.doc_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document.model_dump_json(indent=2))
