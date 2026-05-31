import re

from rapidfuzz import fuzz, process

from legal_pdf_extractor.errors import SourceVerificationError
from legal_pdf_extractor.ingestion.parsed_document import ParsedDocument
from legal_pdf_extractor.schemas import RetrievedChunk, Source


class SourceVerifier:
    def verify(
        self,
        sources: list[dict] | list[Source],
        document: ParsedDocument,
        found: bool,
        chunks: list[RetrievedChunk] | None = None,
    ) -> list[Source]:
        if not found:
            return []

        evidence = chunks or []
        if not evidence:
            raise SourceVerificationError("No source evidence is available for verification.")

        verified: list[Source] = []
        for raw_source in sources:
            source = (
                raw_source if isinstance(raw_source, Source) else Source.model_validate(raw_source)
            )
            exact_match = _exact_match(source.snippet, evidence)
            if exact_match is not None:
                verified.append(exact_match)
                continue
            verified.append(_best_fuzzy_match(source.snippet, evidence))

        if not verified:
            raise SourceVerificationError("A found answer must include at least one source.")
        return verified


def _exact_match(snippet: str, chunks: list[RetrievedChunk]) -> Source | None:
    for chunk in chunks:
        if snippet in chunk.text:
            return Source(page=chunk.page, snippet=snippet)
    return None


def _best_fuzzy_match(snippet: str, chunks: list[RetrievedChunk]) -> Source:
    choices: dict[int, str] = {}
    pages: dict[int, int] = {}
    index = 0
    for chunk in chunks:
        for candidate in _candidate_snippets(chunk.text, snippet):
            choices[index] = candidate
            pages[index] = chunk.page
            index += 1

    match = process.extractOne(
        snippet,
        choices,
        scorer=fuzz.WRatio,
    )
    if match is None:
        return Source(page=chunks[0].page, snippet=chunks[0].text)
    matched_snippet, _, matched_index = match
    return Source(page=pages[matched_index], snippet=matched_snippet)


def _candidate_snippets(text: str, snippet: str) -> list[str]:
    words = list(re.finditer(r"\S+", text))
    if not words:
        return [text]

    window_size = max(1, len(re.findall(r"\S+", snippet)))
    if len(words) <= window_size:
        return [text]

    candidates: list[str] = []
    for start in range(0, len(words) - window_size + 1):
        end = start + window_size - 1
        candidates.append(text[words[start].start() : words[end].end()])
    return candidates
