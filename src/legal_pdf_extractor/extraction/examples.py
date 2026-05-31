from typing import Any


def normalize_examples(examples: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not examples:
        return []
    normalized = []
    for example in examples:
        if "input" in example and "output" in example:
            normalized.append({"input": example["input"], "output": example["output"]})
    return normalized
