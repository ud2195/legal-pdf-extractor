from legal_pdf_extractor.schemas import RetrievedChunk, Source


class ConfidenceScorer:
    def score(
        self,
        found: bool,
        sources: list[Source],
        chunks: list[RetrievedChunk],
    ) -> float:
        if not found or not sources or not chunks:
            return 0.0

        matched_ranks: list[int] = []
        matched_sources = 0
        for source in sources:
            ranks = [
                chunk.metadata["confidence_rank"]
                for chunk in chunks
                if source.snippet in chunk.text and "confidence_rank" in chunk.metadata
            ]
            if ranks:
                matched_sources += 1
                matched_ranks.append(min(ranks))

        if not matched_sources:
            return 0.0

        source_score = 0.85 * (matched_sources / len(sources))
        rank_bonus = _rank_bonus(min(matched_ranks), chunks) if matched_ranks else 0.0
        return round(min(1.0, source_score + rank_bonus), 2)


def _rank_bonus(rank: int, chunks: list[RetrievedChunk]) -> float:
    ranked_chunks = [
        chunk
        for chunk in chunks
        if isinstance(chunk.metadata.get("confidence_rank"), int)
        and chunk.metadata.get("retrieval_role") == "candidate"
    ]
    rank_count = max(1, len(ranked_chunks))
    return max(0.0, 0.15 * (1 - rank / rank_count))
