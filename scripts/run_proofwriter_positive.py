"""Run the Phase 3 positive-query integration on one ProofWriter dev example."""

from pathlib import Path

from string_agent.datasets import load_proofwriter_file
from string_agent.integration.explanation_refiner import (
    _build_natural_language_input,
    extract_query_statement,
    verify_positive,
)


EXAMPLE_ID = "ProofWriter_AttNoneg-OWA-D5-1041_Q1"


def main() -> None:
    examples = load_proofwriter_file(
        Path("external/Logic-LLM/data/ProofWriter/dev.json")
    )
    example = next(item for item in examples if item.id == EXAMPLE_ID)
    natural_input = _build_natural_language_input(example)

    print(f"id: {example.id}")
    print(f"context: {example.context}")
    print(f"question: {example.question}")
    print(f"gold_label: {example.gold_label}")
    print(f"positive_query: {extract_query_statement(example.question)}")
    print(f"natural_language_input: {natural_input!r}")
    print(f"verification_result: {verify_positive(example)!r}")


if __name__ == "__main__":
    main()
