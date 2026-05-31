from pathlib import Path

from pydantic import BaseModel


class NormalizedPdf(BaseModel):
    path: Path
    is_temp: bool = False
