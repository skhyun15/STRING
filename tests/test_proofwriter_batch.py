import importlib.util
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "run_proofwriter_batch.py"
SPEC = importlib.util.spec_from_file_location("proofwriter_batch", MODULE)
batch = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(batch)


CONTEXT = "Charlie is kind. All kind things are quiet."


def artifact(explanation):
    return {"status": "VALID", "final_validity": True, "initial_candidate": {"explanation": explanation}, "iterations": []}


def test_query_identical_to_source_fact_is_directly_grounded():
    grounding = batch._grounding(artifact("Charlie is kind."), CONTEXT, "Charlie is kind.", "query")
    assert grounding[0]["grounding"] == "GROUNDED_DIRECT_FACT"


def test_ungrounded_query_repetition_is_circular():
    grounding = batch._grounding(artifact("Charlie is nice."), CONTEXT, "Charlie is nice.", "query")
    assert grounding[0]["grounding"] == "CIRCULAR_QUERY"


def test_complement_identical_to_source_fact_is_directly_grounded():
    grounding = batch._grounding(artifact("Charlie is not kind."), "Charlie is not kind.", "Charlie is not kind.", "complement")
    assert grounding[0]["grounding"] == "GROUNDED_DIRECT_FACT"


def test_ungrounded_complement_repetition_is_circular():
    grounding = batch._grounding(artifact("Charlie is not kind."), CONTEXT, "Charlie is not kind.", "complement")
    assert grounding[0]["grounding"] == "CIRCULAR_COMPLEMENT"


def test_direct_fact_fast_path_and_supported_verdict():
    source = batch._source_items(CONTEXT)[0][0]
    direct = batch._direct_fact_artifact("Charlie is kind.", source)
    assert direct["total_openai_api_calls"] == 0
    assert batch._verdict("ACCEPTED", "REJECTED") == "SUPPORTED"


def test_runtime_failure_is_unresolved_not_unknown():
    assert batch._verdict("TIMEOUT", "REJECTED") == "UNRESOLVED"
    assert batch._verdict("REJECTED", "ERROR") == "UNRESOLVED"


def test_ungrounded_refinement_categories_are_specific():
    value = artifact("A new biomedical fact.\nAll novel things are useful.\nTherefore this bridges the gap.")
    statuses = [x["grounding"] for x in batch._grounding(value, CONTEXT, "Other claim.", "query")]
    assert statuses == ["UNGROUNDED_NEW_FACT", "UNGROUNDED_NEW_RULE", "UNGROUNDED_BRIDGE"]


def test_compressed_universal_rules_are_rules_not_facts():
    facts, rules = batch._source_items(
        "Charlie is kind. Red, cold things are round. Cold, smart things are red."
    )
    assert [x["text"] for x in facts] == ["Charlie is kind."]
    assert [x["text"] for x in rules] == [
        "Red, cold things are round.",
        "Cold, smart things are red.",
    ]
