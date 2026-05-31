from types import SimpleNamespace

from legal_pdf_extractor.indexing.docling_hybrid_chunker import DoclingHybridChunker


class FakeHybridChunker:
    def chunk(self, dl_doc):
        return iter(dl_doc.chunks)

    def contextualize(self, chunk):
        return f"{' > '.join(chunk.meta.headings)}\n{chunk.text}"


def test_docling_hybrid_chunker_uses_contextualized_text() -> None:
    docling_chunker = DoclingHybridChunker()
    docling_chunker.chunker = FakeHybridChunker()
    fake_chunk = SimpleNamespace(
        text="The agreement starts on May 8, 2014.",
        meta=SimpleNamespace(
            headings=["1. Effective Date and Term."],
            doc_items=[
                SimpleNamespace(
                    prov=[SimpleNamespace(page_no=2)],
                )
            ],
        ),
    )
    fake_document = SimpleNamespace(chunks=[fake_chunk])

    chunks = docling_chunker.chunk(fake_document, doc_hash="abc")

    assert chunks[0].text.startswith("1. Effective Date and Term.")
    assert chunks[0].page == 2
    assert chunks[0].metadata["pages"] == [2]
    assert chunks[0].metadata["chunker"] == "docling_hybrid"
    assert chunks[0].metadata["raw_text"] == "The agreement starts on May 8, 2014."
    assert set(chunks[0].metadata) == {"chunker", "pages", "raw_text"}
