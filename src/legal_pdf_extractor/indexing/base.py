from abc import ABC, abstractmethod
from typing import Any

from legal_pdf_extractor.schemas import RetrievedChunk, TextChunk


class DocumentChunker(ABC):
    @abstractmethod
    def chunk(self, document: Any, doc_hash: str) -> list[TextChunk]:
        raise NotImplementedError


class EmbeddingModel(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class VectorStore(ABC):
    @abstractmethod
    def exists(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def chunks(self) -> list[TextChunk]:
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 6) -> list[RetrievedChunk]:
        raise NotImplementedError
