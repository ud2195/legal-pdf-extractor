import re

from rank_bm25 import BM25Okapi

from legal_pdf_extractor.schemas import RetrievedChunk, TextChunk

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def top_bm25_chunks(query: str, chunks: list[TextChunk], limit: int) -> list[RetrievedChunk]:
    query_tokens = _tokenize(query)
    if not query_tokens or not chunks:
        return []

    tokenized_chunks = [_tokenize(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(query_tokens)
    query_terms = set(query_tokens)
    ranked = sorted(
        (
            (chunk, score)
            for chunk, tokens, score in zip(chunks, tokenized_chunks, scores, strict=True)
            if query_terms & set(tokens)
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        RetrievedChunk(**chunk.model_dump(), score=float(score))
        for chunk, score in ranked[:limit]
    ]


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())
