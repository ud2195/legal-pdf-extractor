from typer.testing import CliRunner

from legal_pdf_extractor.cli import app


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["extract", "--help"])

    assert result.exit_code == 0
    assert "Single legal term" in result.output
