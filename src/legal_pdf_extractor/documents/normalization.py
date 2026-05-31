import tempfile
from pathlib import Path

from legal_pdf_extractor.documents.models import NormalizedPdf
from legal_pdf_extractor.errors import PdfNormalizationError

PdfInput = bytes | bytearray | str | Path


def normalize_pdf_input(pdf: PdfInput) -> NormalizedPdf:
    if isinstance(pdf, (bytes, bytearray)):
        content = bytes(pdf)
        if not content:
            raise PdfNormalizationError("PDF input is empty.")
        if not content.lstrip().startswith(b"%PDF"):
            raise PdfNormalizationError("Input does not look like a PDF document.")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            return NormalizedPdf(path=Path(tmp.name), is_temp=True)
    else:
        path = Path(pdf).expanduser()
        if not path.exists():
            raise PdfNormalizationError(f"PDF path does not exist: {path}")
        if not path.is_file():
            raise PdfNormalizationError(f"PDF path is not a file: {path}")
        if path.suffix.lower() != ".pdf":
            raise PdfNormalizationError(f"PDF path must have a .pdf suffix: {path}")
        if not _has_pdf_header(path):
            raise PdfNormalizationError("Input does not look like a PDF document.")
        return NormalizedPdf(path=path, is_temp=False)


def cleanup_normalized_pdf(normalized: NormalizedPdf) -> None:
    if normalized.is_temp:
        normalized.path.unlink(missing_ok=True)


def _has_pdf_header(path: Path) -> bool:
    with path.open("rb") as handle:
        prefix = handle.read(1024)
    return bool(prefix) and prefix.lstrip().startswith(b"%PDF")
