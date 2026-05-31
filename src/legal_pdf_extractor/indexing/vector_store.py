import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np
from pydantic import BaseModel, Field

from legal_pdf_extractor.indexing.base import VectorStore
from legal_pdf_extractor.schemas import RetrievedChunk, TextChunk


class VectorRecord(BaseModel):
    chunk: TextChunk
    embedding: list[float]


class VectorMetadata(BaseModel):
    doc_hash: str
    dimensions: int
    chunks: list[TextChunk] = Field(default_factory=list)


class FaissVectorStore(VectorStore):
    def __init__(self, index_path: str | Path, metadata_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

    def exists(self) -> bool:
        return self.index_path.exists() and self.metadata_path.exists()

    def save(self, doc_hash: str, records: list[VectorRecord]) -> None:
        if not records:
            raise ValueError("Cannot save an empty FAISS index.")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        dimensions = len(records[0].embedding)
        if dimensions == 0:
            raise ValueError("Cannot save zero-dimensional embeddings.")
        index = faiss.IndexFlatIP(dimensions)

        embeddings = _to_float32_matrix([record.embedding for record in records])
        index.add(embeddings)

        faiss.write_index(index, str(self.index_path))
        metadata = VectorMetadata(
            doc_hash=doc_hash,
            dimensions=dimensions,
            chunks=[record.chunk for record in records],
        )
        self.metadata_path.write_text(metadata.model_dump_json(indent=2))

    def load_metadata(self) -> VectorMetadata:
        return VectorMetadata.model_validate_json(self.metadata_path.read_text())

    def chunks(self) -> list[TextChunk]:
        if not self.exists():
            return []
        return self.load_metadata().chunks

    def search(self, query_embedding: list[float], top_k: int = 6) -> list[RetrievedChunk]:
        if not self.exists():
            return []

        metadata = self.load_metadata()
        if not metadata.chunks:
            return []
        if len(query_embedding) != metadata.dimensions:
            return []

        index = faiss.read_index(str(self.index_path))
        query = _to_float32_matrix([query_embedding])
        scores, indices = index.search(query, min(top_k, len(metadata.chunks)))

        results: list[RetrievedChunk] = []
        for score, index_position in zip(scores[0], indices[0], strict=False):
            if index_position < 0:
                continue
            chunk = metadata.chunks[int(index_position)]
            results.append(
                RetrievedChunk(
                    **chunk.model_dump(),
                    score=float(score),
                )
            )
        return results


def _to_float32_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D embedding matrix.")
    return np.ascontiguousarray(matrix)
