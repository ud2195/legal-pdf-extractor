import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CACHE_DIR = Path(".cache/legal_pdf_extractor")
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_EMBEDDING_MAX_WORKERS = 4


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str
    openai_embedding_model: str
    cache_dir: Path
    embedding_batch_size: int
    embedding_max_workers: int


@lru_cache(maxsize=1)
def load_environment() -> None:
    load_dotenv()


def load_settings(cache_dir: str | Path | None = None) -> Settings:
    load_environment()
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=DEFAULT_OPENAI_MODEL,
        openai_embedding_model=DEFAULT_OPENAI_EMBEDDING_MODEL,
        cache_dir=Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR,
        embedding_batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
        embedding_max_workers=DEFAULT_EMBEDDING_MAX_WORKERS,
    )
