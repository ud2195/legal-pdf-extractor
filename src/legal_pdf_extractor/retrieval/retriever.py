from legal_pdf_extractor.indexing.base import EmbeddingModel, VectorStore
from legal_pdf_extractor.retrieval.base import BaseRetriever
from legal_pdf_extractor.retrieval.ranking import top_bm25_chunks
from legal_pdf_extractor.schemas import RetrievedChunk, TextChunk


class Retriever(BaseRetriever):
    def __init__(
        self,
        embeddings: EmbeddingModel,
        top_k: int = 5,
        context_window: int = 1,
    ) -> None:
        self.embeddings = embeddings
        self.top_k = top_k
        self.context_window = context_window

    def retrieve(self, query: str, store: VectorStore) -> list[RetrievedChunk]:
        query_embedding = self.embeddings.embed(query)
        vector_candidates = store.search(query_embedding, top_k=self.top_k)
        all_chunks = store.chunks()
        bm25_candidates = top_bm25_chunks(query, all_chunks, limit=self.top_k)

        candidates: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()
        for chunk in [*vector_candidates, *bm25_candidates]:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            candidates.append(_with_retrieval_metadata(chunk, len(candidates), "candidate"))
            seen_chunk_ids.add(chunk.chunk_id)
        return _expand_with_context(candidates, all_chunks, self.context_window)


def _expand_with_context(
    candidates: list[RetrievedChunk],
    all_chunks: list[TextChunk],
    context_window: int,
) -> list[RetrievedChunk]:
    if context_window <= 0 or not candidates:
        return candidates

    positions = {chunk.chunk_id: index for index, chunk in enumerate(all_chunks)}
    expanded: list[RetrievedChunk] = []
    seen_chunk_ids: set[str] = set()
    candidate_by_id = {chunk.chunk_id: chunk for chunk in candidates}

    for candidate in candidates:
        position = positions.get(candidate.chunk_id)
        if position is None:
            window_chunks: list[RetrievedChunk | TextChunk] = [candidate]
        else:
            start = max(0, position - context_window)
            end = min(len(all_chunks), position + context_window + 1)
            window_chunks = all_chunks[start:end]

        for chunk in window_chunks:
            if chunk.chunk_id in seen_chunk_ids:
                continue
            expanded_chunk = candidate_by_id.get(chunk.chunk_id)
            if expanded_chunk is None:
                expanded_chunk = _with_retrieval_metadata(
                    chunk,
                    candidate.metadata["confidence_rank"],
                    "context",
                )
            expanded.append(
                expanded_chunk
            )
            seen_chunk_ids.add(chunk.chunk_id)

    return expanded


def _with_retrieval_metadata(
    chunk: RetrievedChunk | TextChunk,
    confidence_rank: int,
    retrieval_role: str,
) -> RetrievedChunk:
    metadata = {
        **chunk.metadata,
        "retrieval_role": retrieval_role,
        "confidence_rank": confidence_rank,
    }
    return RetrievedChunk(
        **{
            **chunk.model_dump(),
            "metadata": metadata,
        }
    )
