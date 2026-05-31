import json
from pathlib import Path

import typer
from rich.console import Console

from legal_pdf_extractor.api import extract as extract_term

app = typer.Typer(help="Extract typed values from legal PDFs with source attribution.")
console = Console()


@app.callback()
def main() -> None:
    """Extract typed values from legal PDFs with source attribution."""


@app.command("extract")
def extract_command(
    pdf: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False, help="PDF path."),
    query: str = typer.Option(..., help="Single legal term to extract."),
    output_type: str = typer.Option(
        ...,
        help=r"string, date, number, array\[string], array\[date], or array\[number].",
    ),
    examples_json: str | None = typer.Option(None, help="Few-shot examples as a JSON array."),
    cache_dir: Path | None = typer.Option(None, help="Override cache directory."),
    pretty: bool = typer.Option(True, help="Pretty-print JSON output."),
) -> None:
    examples = json.loads(examples_json) if examples_json else None
    result = extract_term(
        pdf=pdf,
        query=query,
        output_type=output_type,
        examples=examples,
        cache_dir=cache_dir,
    )
    if pretty:
        console.print_json(data=result)
    else:
        console.print(json.dumps(result, separators=(",", ":")))
