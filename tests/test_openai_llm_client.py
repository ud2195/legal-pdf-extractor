from types import SimpleNamespace
from typing import Any

import httpx
from openai import APIConnectionError

from legal_pdf_extractor.llm.openai_client import OpenAILLMClient


class FakeChatCompletions:
    def __init__(self, failures_before_success: int = 0) -> None:
        self.kwargs: dict[str, Any] | None = None
        self.call_count = 0
        self.failures_before_success = failures_before_success

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.call_count += 1
        self.kwargs = kwargs
        if self.call_count <= self.failures_before_success:
            raise APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"value": "Acme LLC", "found": true, "sources": []}'
                    )
                )
            ]
        )


def test_openai_llm_client_uses_structured_outputs() -> None:
    completions = FakeChatCompletions()
    client = OpenAILLMClient(
        api_key=None,
        model="gpt-test",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    result = client.extract_json("Extract the tenant.", output_type="array[number]")

    assert result["value"] == "Acme LLC"
    assert completions.call_count == 1
    assert completions.kwargs is not None
    response_format = completions.kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    value_schema = response_format["json_schema"]["schema"]["properties"]["value"]
    assert value_schema == {
        "anyOf": [
            {"type": "array", "items": {"type": "number"}},
            {"type": "null"},
        ]
    }


def test_openai_llm_client_retries_transient_errors() -> None:
    completions = FakeChatCompletions(failures_before_success=2)
    client = OpenAILLMClient(
        api_key=None,
        model="gpt-test",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        max_retries=2,
    )

    result = client.extract_json("Extract the tenant.", output_type="string")

    assert result["value"] == "Acme LLC"
    assert completions.call_count == 3
