"""Loader for the ProofWriter data distributed with Logic-LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast


GoldLabel = Literal["TRUE", "FALSE", "UNKNOWN"]

_ANSWER_TO_LABEL: dict[str, GoldLabel] = {
    "A": "TRUE",
    "B": "FALSE",
    "C": "UNKNOWN",
}
_REQUIRED_STRING_FIELDS = ("id", "context", "question", "answer")


class ProofWriterValidationError(ValueError):
    """Raised when a ProofWriter file or example does not match its schema."""


@dataclass(frozen=True)
class ProofWriterExample:
    id: str
    context: str
    question: str
    gold_label: GoldLabel


def _parse_example(raw: object, index: int) -> ProofWriterExample:
    location = f"example at index {index}"
    if not isinstance(raw, dict):
        raise ProofWriterValidationError(f"{location} must be a JSON object")

    for field in _REQUIRED_STRING_FIELDS:
        if field not in raw:
            raise ProofWriterValidationError(
                f"{location} is missing required field {field!r}"
            )
        if not isinstance(raw[field], str):
            raise ProofWriterValidationError(
                f"{location} field {field!r} must be a string"
            )

    answer = cast(str, raw["answer"])
    try:
        gold_label = _ANSWER_TO_LABEL[answer]
    except KeyError as exc:
        raise ProofWriterValidationError(
            f"{location} has invalid answer {answer!r}; expected one of A, B, C"
        ) from exc

    return ProofWriterExample(
        id=cast(str, raw["id"]),
        context=cast(str, raw["context"]),
        question=cast(str, raw["question"]),
        gold_label=gold_label,
    )


def load_proofwriter_file(path: Path) -> list[ProofWriterExample]:
    """Load Logic-LLM's JSON-array ProofWriter format and normalize its labels."""

    try:
        with path.open("r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ProofWriterValidationError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(raw_data, list):
        raise ProofWriterValidationError(
            f"ProofWriter file {path} must contain a JSON array"
        )

    return [_parse_example(raw, index) for index, raw in enumerate(raw_data)]
