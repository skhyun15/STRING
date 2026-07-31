from dataclasses import fields
from pathlib import Path
from typing import get_args

import pytest

from string_agent.datasets import ProofWriterExample
from string_agent.integration import explanation_refiner
from string_agent.integration.explanation_refiner import (
    QuestionFormatError,
    VerificationResult,
    VerificationStatus,
    extract_query_statement,
    verify_positive,
)


QUESTION = (
    "Based on the above information, is the following statement true, false, "
    "or unknown? Charlie is kind."
)


def test_extracts_query_statement_deterministically():
    assert extract_query_statement(QUESTION) == "Charlie is kind."


@pytest.mark.parametrize(
    "question",
    [
        "Is Charlie kind?",
        "Based on the above information, Charlie is kind.",
        (
            "Based on the above information, is the following statement true, "
            "false, or unknown?   "
        ),
    ],
)
def test_rejects_malformed_question_prefix(question):
    with pytest.raises(QuestionFormatError):
        extract_query_statement(question)


def test_gold_label_is_not_in_verifier_input(monkeypatch):
    captured = {}

    def fake_verify(natural_input, example_id, *, max_iterations):
        captured["natural_input"] = natural_input
        captured["example_id"] = example_id
        captured["max_iterations"] = max_iterations
        return VerificationResult("NOT_PROVED", Path("query.thy"), 1, "not proved")

    monkeypatch.setattr(explanation_refiner, "_verify_natural_language", fake_verify)
    example = ProofWriterExample(
        id="ProofWriter_test_Q1",
        context="Charlie is kind.",
        question=QUESTION,
        gold_label="FALSE",
    )

    result = verify_positive(example)

    assert result.status == "NOT_PROVED"
    assert captured == {
        "natural_input": explanation_refiner.NaturalLanguageInput(
            premise="none",
            explanation="Charlie is kind.",
            hypothesis="Charlie is kind.",
        ),
        "example_id": "ProofWriter_test_Q1",
        "max_iterations": 1,
    }
    assert "gold_label" not in {field.name for field in fields(captured["natural_input"])}


def test_verification_result_status_structure():
    assert set(get_args(VerificationStatus)) == {
        "PROVED",
        "NOT_PROVED",
        "FORMALISATION_ERROR",
        "VERIFIER_ERROR",
    }
    result = VerificationResult(
        status="PROVED",
        theory_path=Path("proof.thy"),
        iterations=1,
        error=None,
        syntax_valid=True,
        proof_valid=True,
    )
    assert result.status == "PROVED"
    assert result.theory_path == Path("proof.thy")
    assert result.iterations == 1
