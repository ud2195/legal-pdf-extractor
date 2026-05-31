from abc import ABC, abstractmethod
from typing import Any


class LLMClient(ABC):
    @abstractmethod
    def extract_json(self, prompt: str, output_type: str) -> dict[str, Any]:
        raise NotImplementedError
