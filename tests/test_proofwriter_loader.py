import json

import pytest

from string_agent.datasets.proofwriter import (
    ProofWriterValidationError,
    load_proofwriter_file,
)


def _write_fixture(tmp_path, examples):
    path = tmp_path / "proofwriter.json"
    path.write_text(json.dumps(examples), encoding="utf-8")
    return path


def _example(answer="A"):
    return {
        "id": "ProofWriter_AttNoneg-OWA-D5-1041_Q1",
        "context": "Bob is cold. Charlie is kind.",
        "question": (
            "Based on the above information, is the following statement true, "
            "false, or unknown? Charlie is kind."
        ),
        "options": ["A) True", "B) False", "C) Unknown"],
        "answer": answer,
    }


@pytest.mark.parametrize(
    ("answer", "expected_label"),
    [("A", "TRUE"), ("B", "FALSE"), ("C", "UNKNOWN")],
)
def test_loads_and_normalizes_actual_schema(tmp_path, answer, expected_label):
    raw = _example(answer)
    loaded = load_proofwriter_file(_write_fixture(tmp_path, [raw]))

    assert len(loaded) == 1
    assert loaded[0].id == raw["id"]
    assert loaded[0].context == raw["context"]
    assert loaded[0].question == raw["question"]
    assert loaded[0].gold_label == expected_label


def test_invalid_answer_fails_explicitly(tmp_path):
    with pytest.raises(ProofWriterValidationError, match="invalid answer"):
        load_proofwriter_file(_write_fixture(tmp_path, [_example("D")]))


@pytest.mark.parametrize("missing_field", ["id", "context", "question", "answer"])
def test_missing_required_field_fails_explicitly(tmp_path, missing_field):
    raw = _example()
    del raw[missing_field]

    with pytest.raises(ProofWriterValidationError, match=missing_field):
        load_proofwriter_file(_write_fixture(tmp_path, [raw]))
