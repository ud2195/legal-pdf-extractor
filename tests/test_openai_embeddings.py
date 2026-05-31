from types import SimpleNamespace

from legal_pdf_extractor.indexing.embeddings import OpenAIEmbeddingModel


class FakeEmbeddingsClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def create(self, model: str, input: list[str], encoding_format: str):
        self.calls.append(input)
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(len(text)), float(index)])
                for index, text in enumerate(input)
            ]
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsClient()


def test_openai_embedding_model_batches_and_preserves_order() -> None:
    client = FakeOpenAIClient()
    model = OpenAIEmbeddingModel(
        api_key=None,
        model="text-embedding-3-small",
        client=client,
        batch_size=2,
        max_workers=2,
    )

    embeddings = model.embed_many(["a", "bbbb", "cc"])

    assert embeddings == [[1.0, 0.0], [4.0, 1.0], [2.0, 0.0]]
    assert sorted(client.embeddings.calls, key=len, reverse=True) == [["a", "bbbb"], ["cc"]]
