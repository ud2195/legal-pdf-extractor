from datetime import date
from typing import Any

from legal_pdf_extractor.errors import TypeValidationError
from legal_pdf_extractor.validation.output_types import OutputType, parse_output_type


class TypeValidator:
    def validate(self, value: Any, output_type: str | OutputType, found: bool = True) -> Any:
        spec = parse_output_type(output_type) if isinstance(output_type, str) else output_type
        if not found:
            return None
        if spec.is_array:
            if not isinstance(value, list):
                raise TypeValidationError("Expected an array value.")
            return [self._validate_scalar(item, spec.item_type) for item in value]
        return self._validate_scalar(value, spec.item_type)

    def _validate_scalar(self, value: Any, item_type: str) -> Any:
        if item_type == "string":
            if not isinstance(value, str):
                raise TypeValidationError("Expected a string value.")
            return value
        if item_type == "date":
            if not isinstance(value, str):
                raise TypeValidationError("Expected an ISO-8601 date string.")
            try:
                date.fromisoformat(value)
            except ValueError as exc:
                raise TypeValidationError("Expected an ISO-8601 date string.") from exc
            return value
        if item_type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeValidationError("Expected a numeric value.")
            return value
        raise TypeValidationError(f"Unsupported scalar type: {item_type}")
