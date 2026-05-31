from typing import Any, cast

import tiktoken
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

from legal_pdf_extractor.config import DEFAULT_OPENAI_EMBEDDING_MODEL
from legal_pdf_extractor.indexing.base import DocumentChunker
from legal_pdf_extractor.schemas import TextChunk


class DoclingHybridChunker(DocumentChunker):
    def __init__(
        self,
        merge_peers: bool = True,
        model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
        max_tokens: int = 800,
    ) -> None:
        tokenizer = OpenAITokenizer(
            tokenizer=tiktoken.encoding_for_model(model),
            max_tokens=max_tokens,
        )
        self.chunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=merge_peers,
        )

    def chunk(self, document: Any, doc_hash: str) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for index, docling_chunk in enumerate(self.chunker.chunk(dl_doc=document)):
            docling_chunk = cast(Any, docling_chunk)
            contextualized_text = self.chunker.contextualize(docling_chunk)
            if not contextualized_text.strip():
                continue
            pages = _chunk_pages(docling_chunk)
            primary_page = pages[0] if pages else 1
            chunks.append(
                TextChunk(
                    chunk_id=f"{doc_hash}:docling:c{index}",
                    doc_hash=doc_hash,
                    page=primary_page,
                    text=contextualized_text,
                    metadata={
                        "chunker": "docling_hybrid",
                        "pages": pages,
                        "raw_text": docling_chunk.text,
                    },
                )
            )
        return chunks


def _chunk_pages(chunk: Any) -> list[int]:
    pages: set[int] = set()
    for item in chunk.meta.doc_items:
        for prov in item.prov or []:
            pages.add(prov.page_no)
    return sorted(pages)
