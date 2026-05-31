from legal_pdf_extractor.indexing.base import EmbeddingModel, VectorStore
from legal_pdf_extractor.retrieval.retriever import Retriever
from legal_pdf_extractor.schemas import RetrievedChunk, TextChunk


class FakeEmbeddingModel(EmbeddingModel):
    def embed(self, text: str) -> list[float]:
        return [1.0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self._chunks = [
            TextChunk(chunk_id="dense", doc_hash="abc", page=1, text="General lease terms."),
            TextChunk(
                chunk_id="bm25",
                doc_hash="abc",
                page=2,
                text="Tenant must provide renewal notice before expiration.",
            ),
            TextChunk(
                chunk_id="after",
                doc_hash="abc",
                page=3,
                text="Additional surrounding context.",
            ),
        ]
        self.top_k_values: list[int] = []

    def exists(self) -> bool:
        return True

    def chunks(self) -> list[TextChunk]:
        return self._chunks

    def search(self, query_embedding: list[float], top_k: int = 6) -> list[RetrievedChunk]:
        self.top_k_values.append(top_k)
        return [RetrievedChunk(**self._chunks[0].model_dump(), score=1.0)]


def test_retriever_merges_vector_and_bm25_candidates() -> None:
    results = Retriever(embeddings=FakeEmbeddingModel(), top_k=2).retrieve(
        query="renewal notice",
        store=FakeVectorStore(),
    )

    assert [result.chunk_id for result in results] == ["dense", "bm25", "after"]


def test_retriever_defaults_to_top_five() -> None:
    store = FakeVectorStore()

    Retriever(embeddings=FakeEmbeddingModel()).retrieve(
        query="renewal notice",
        store=store,
    )

    assert store.top_k_values == [5]


def test_retriever_expands_vector_hit_with_neighboring_chunks() -> None:
    class NeighborVectorStore(FakeVectorStore):
        def __init__(self) -> None:
            super().__init__()
            self._chunks = [
                TextChunk(chunk_id="before", doc_hash="abc", page=1, text="Execution date."),
                TextChunk(chunk_id="hit", doc_hash="abc", page=2, text="Effective term."),
                TextChunk(chunk_id="after", doc_hash="abc", page=3, text="Renewal details."),
            ]

        def search(
            self, query_embedding: list[float], top_k: int = 6
        ) -> list[RetrievedChunk]:
            self.top_k_values.append(top_k)
            return [RetrievedChunk(**self._chunks[1].model_dump(), score=1.0)]

    results = Retriever(embeddings=FakeEmbeddingModel(), top_k=1).retrieve(
        query="semantic-only",
        store=NeighborVectorStore(),
    )

    assert [result.chunk_id for result in results] == ["before", "hit", "after"]
