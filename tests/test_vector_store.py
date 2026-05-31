from legal_pdf_extractor.indexing.vector_store import FaissVectorStore, VectorRecord
from legal_pdf_extractor.schemas import TextChunk


def test_faiss_vector_store_persists_and_searches(tmp_path) -> None:
    store = FaissVectorStore(
        index_path=tmp_path / "vector_index.faiss",
        metadata_path=tmp_path / "vector_metadata.json",
    )
    record = VectorRecord(
        chunk=TextChunk(
            chunk_id="abc:p1:c0",
            doc_hash="abc",
            page=1,
            text="Tenant: Greenfield Properties LLC.",
        ),
        embedding=[1.0, 0.0, 0.0],
    )

    store.save(doc_hash="abc", records=[record])
    results = store.search([1.0, 0.0, 0.0], top_k=1)

    assert store.exists()
    assert results[0].chunk_id == "abc:p1:c0"
    assert results[0].score > 0.99
