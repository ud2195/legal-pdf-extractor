from abc import ABC, abstractmethod

from legal_pdf_extractor.indexing.base import VectorStore
from legal_pdf_extractor.schemas import RetrievedChunk


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, store: VectorStore) -> list[RetrievedChunk]:
        raise NotImplementedError
