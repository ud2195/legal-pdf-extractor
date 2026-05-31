import json
from importlib import resources
from pathlib import Path
from string import Template
from typing import Any

from legal_pdf_extractor.schemas import RetrievedChunk


class ExtractionPromptBuilder:
    def __init__(self, template_path: str | Path | None = None) -> None:
        self.template_path = Path(template_path) if template_path is not None else None

    def build(
        self,
        query: str,
        output_type: str,
        chunks: list[RetrievedChunk],
        examples: list[dict[str, Any]] | None = None,
    ) -> str:
        context = "\n\n".join(
            f"[page={chunk.page}]\n{chunk.text}" for chunk in chunks
        )
        examples_section = self._format_examples(examples)
        return Template(self._load_template()).substitute(
            query=query,
            output_type=output_type,
            examples_section=examples_section,
            context=context,
        ).strip()

    def _load_template(self) -> str:
        if self.template_path is not None:
            return self.template_path.read_text()
        return resources.files(__package__).joinpath("extraction_prompt.txt").read_text()

    def _format_examples(self, examples: list[dict[str, Any]] | None) -> str:
        if not examples:
            return ""
        examples_text = json.dumps(examples, ensure_ascii=False, indent=2)
        return f"\nFew-shot examples:\n{examples_text}"
