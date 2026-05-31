import json
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from legal_pdf_extractor.errors import LLMError
from legal_pdf_extractor.llm.base import LLMClient
from legal_pdf_extractor.validation.output_types import OutputType, parse_output_type


class OpenAILLMClient(LLMClient):
    def __init__(
        self,
        api_key: str | None,
        model: str,
        *,
        client: Any | None = None,
        max_retries: int = 3,
    ) -> None:
        if not api_key and client is None:
            raise LLMError("OPENAI_API_KEY is required for OpenAI extraction.")
        self.client = client if client is not None else OpenAI(api_key=api_key, max_retries=0)
        self.model = model
        self.max_retries = max_retries

    def extract_json(self, prompt: str, output_type: str) -> dict[str, Any]:
        try:
            response = Retrying(
                retry=retry_if_exception_type(
                    (RateLimitError, APIConnectionError, APITimeoutError, APIError)
                ),
                stop=stop_after_attempt(self.max_retries + 1),
                wait=wait_exponential(multiplier=1, max=20),
                reraise=True,
            )(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Return only valid JSON for legal extraction tasks.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "legal_extraction_response",
                        "strict": True,
                        "schema": _extraction_response_schema(output_type),
                    },
                },
                temperature=0,
            )
            content = response.choices[0].message.content
            if not content:
                raise LLMError("OpenAI returned an empty response.")
            return json.loads(content)
        except LLMError:
            raise
        except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as exc:
            raise LLMError(f"OpenAI extraction failed after retries: {exc}") from exc
        except Exception as exc:
            raise LLMError(f"OpenAI extraction failed: {exc}") from exc


def _extraction_response_schema(output_type: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "value": _value_schema(parse_output_type(output_type)),
            "found": {"type": "boolean"},
            "sources": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer"},
                        "snippet": {"type": "string"},
                    },
                    "required": ["page", "snippet"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["value", "found", "sources"],
        "additionalProperties": False,
    }


def _value_schema(output_type: OutputType) -> dict[str, Any]:
    value_schema = _scalar_value_schema(output_type.item_type)
    if output_type.is_array:
        value_schema = {
            "type": "array",
            "items": value_schema,
        }
    return {
        "anyOf": [
            value_schema,
            {"type": "null"},
        ]
    }


def _scalar_value_schema(item_type: str) -> dict[str, Any]:
    if item_type == "number":
        return {"type": "number"}
    if item_type == "date":
        return {
            "type": "string",
            "description": "An ISO-8601 date string.",
        }
    return {"type": "string"}
