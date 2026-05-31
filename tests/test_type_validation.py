import pytest

from legal_pdf_extractor.errors import TypeValidationError
from legal_pdf_extractor.validation.type_validator import TypeValidator


def test_type_validator_accepts_supported_values() -> None:
    validator = TypeValidator()

    assert validator.validate("Acme", "string") == "Acme"
    assert validator.validate("2024-03-15", "date") == "2024-03-15"
    assert validator.validate(5.25, "number") == 5.25
    assert validator.validate(["A", "B"], "array[string]") == ["A", "B"]


def test_type_validator_rejects_wrong_type() -> None:
    with pytest.raises(TypeValidationError):
        TypeValidator().validate("5.25", "number")
