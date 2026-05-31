from dataclasses import dataclass

from legal_pdf_extractor.errors import TypeValidationError


@dataclass(frozen=True)
class OutputType:
    item_type: str
    is_array: bool = False


SUPPORTED_SCALARS = {"string", "date", "number"}


def parse_output_type(value: str) -> OutputType:
    normalized = value.strip().lower()
    if normalized in SUPPORTED_SCALARS:
        return OutputType(item_type=normalized)
    if normalized.startswith("array[") and normalized.endswith("]"):
        item_type = normalized[len("array[") : -1].strip()
        if item_type in SUPPORTED_SCALARS:
            return OutputType(item_type=item_type, is_array=True)
    raise TypeValidationError(
        "Unsupported output_type. Use string, date, number, array[string], "
        "array[date], or array[number]."
    )
