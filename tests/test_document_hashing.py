import pytest

from legal_pdf_extractor.documents.hashing import compute_document_hash
from legal_pdf_extractor.documents.normalization import (
    cleanup_normalized_pdf,
    normalize_pdf_input,
)
from legal_pdf_extractor.errors import PdfNormalizationError


def test_document_hash_is_stable() -> None:
    path = "tests/fixtures/sample_contract.pdf"

    assert compute_document_hash(path) == compute_document_hash(path)


def test_normalize_pdf_accepts_path() -> None:
    normalized = normalize_pdf_input("tests/fixtures/sample_contract.pdf")

    assert normalized.path.name == "sample_contract.pdf"
    assert normalized.is_temp is False


def test_normalize_pdf_rejects_non_pdf_suffix(tmp_path) -> None:
    path = tmp_path / "sample.txt"
    path.write_bytes(b"%PDF-1.4\n")

    with pytest.raises(PdfNormalizationError):
        normalize_pdf_input(path)


def test_normalize_pdf_bytes_creates_temp_pdf_and_cleanup() -> None:
    normalized = normalize_pdf_input(b"%PDF-1.4\nsame content")

    assert normalized.path.exists()
    assert normalized.path.suffix == ".pdf"
    assert normalized.is_temp is True

    cleanup_normalized_pdf(normalized)

    assert not normalized.path.exists()
