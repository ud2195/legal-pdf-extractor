import concurrent.futures
from collections.abc import Iterator
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from legal_pdf_extractor.errors import EmbeddingModelError
from legal_pdf_extractor.indexing.base import EmbeddingModel


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        api_key: str | None,
        model: str,
        *,
        client: Any | None = None,
        batch_size: int = 64,
        max_workers: int = 4,
        max_retries: int = 5,
    ) -> None:
        if not api_key and client is None:
            raise EmbeddingModelError("OPENAI_API_KEY is required for OpenAI embeddings.")
        self.client = client if client is not None else OpenAI(api_key=api_key, max_retries=0)
        self.model = model
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_retries = max_retries

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        batches = _batched(texts, self.batch_size)
        embeddings: list[list[float]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._embed_batch_with_retry, batch)
                for batch in batches
            ]
            for future in futures:
                embeddings.extend(future.result())
        return embeddings

    def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        try:
            response = Retrying(
                retry=retry_if_exception_type(
                    (RateLimitError, APIConnectionError, APITimeoutError, APIError)
                ),
                stop=stop_after_attempt(self.max_retries + 1),
                wait=wait_exponential(multiplier=1, max=30),
                reraise=True,
            )(
                self.client.embeddings.create,
                model=self.model,
                input=texts,
                encoding_format="float",
            )
        except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as exc:
            raise EmbeddingModelError(f"OpenAI embeddings failed after retries: {exc}") from exc

        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]


def _batched(texts: list[str], batch_size: int) -> Iterator[list[str]]:
    for offset in range(0, len(texts), batch_size):
        yield texts[offset : offset + batch_size]
