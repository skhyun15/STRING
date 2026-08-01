import importlib.util, json
from pathlib import Path
import pytest
P=Path(__file__).resolve().parents[1]/"scripts/run_bioasq_audit.py"; S=importlib.util.spec_from_file_location("bioaudit",P); m=importlib.util.module_from_spec(S); S.loader.exec_module(m)
def test_load_prompt_and_graph(tmp_path):
    data={"questions":[{"id":"q1","body":"Question?","type":"factoid","documents":["d"],"snippets":[{"text":"Evidence.","document":"d"}],"ideal_answer":["gold"],"exact_answer":"gold"}]}
    p=tmp_path/"d.json"; p.write_text(json.dumps(data)); q=m.load_question(p,"q1")
    assert q["snippets"][0]["id"]=="snippet-1"
    prompt=m.generation_prompt(q); assert "gold" not in prompt and "ideal_answer" not in prompt and "exact_answer" not in prompt
    generated={"explanation_steps":[],"atomic_claims":[{"claim_id":"claim-1","text":"Evidence.","evidence_ids":["snippet-1"]}]}
    audited={"claims":[{"claim_id":"claim-1","text":"Evidence.","evidence_ids":["snippet-1"],"grounding_status":"DIRECTLY_SUPPORTED","acceptance_status":"ACCEPTED"}],"refined_claims":[],"refined_answer":"Evidence."}; m.normalize_audit(audited)
    graph=m.build_graph(q,generated,audited); assert any(e["source"]=="snippet-1" for e in graph["edges"])
def test_normalize_audit_enums():
    value={"claims":[{"grounding_status":"directly_supported","acceptance_status":"accepted"}],"refined_claims":[]}
    m.normalize_audit(value); assert value["claims"][0]["grounding_status"]=="DIRECTLY_SUPPORTED"; assert value["claims"][0]["acceptance_status"]=="ACCEPTED"
    alias={"claims":[{"grounding_status":"DIRECT","acceptance_status":"accepted"}],"refined_claims":[]}; m.normalize_audit(alias); assert alias["claims"][0]["grounding_status"]=="DIRECTLY_SUPPORTED"
    supported={"claims":[{"grounding_status":"SUPPORTED","acceptance_status":"accepted","evidence_ids":["snippet-1","snippet-2"]}],"refined_claims":[]}; m.normalize_audit(supported); assert supported["claims"][0]["grounding_status"]=="COMPOSITIONALLY_SUPPORTED"
    indirect={"claims":[{"grounding_status":"INDIRECTLY_SUPPORTED","acceptance_status":"accepted","evidence_ids":["snippet-1"]}],"refined_claims":[]}; m.normalize_audit(indirect); assert indirect["claims"][0]["grounding_status"]=="COMPOSITIONALLY_SUPPORTED"
    indirect_short={"claims":[{"grounding_status":"INDIRECT","acceptance_status":"accepted","evidence_ids":["snippet-1"]}],"refined_claims":[]}; m.normalize_audit(indirect_short); assert indirect_short["claims"][0]["grounding_status"]=="COMPOSITIONALLY_SUPPORTED"
def test_direct_claim_stays_accepted_while_bridge_dependent_layers_are_conditional():
    q={"body":"Q?","snippets":[{"id":"snippet-1","text":"A","document":"d"},{"id":"snippet-2","text":"B","document":"d"}]}
    generated={"explanation_steps":[]}
    claim={"claim_id":"c1","text":"A","evidence_ids":["snippet-1"],"grounding_status":"DIRECTLY_SUPPORTED","acceptance_status":"ACCEPTED_WITH_ASSUMPTION","intermediate_inference":"A and B imply C","bridge_assumption":{"text":"A biomedical bridge","status":"REASONABLE_BUT_UNSTATED"}}
    refined={"claim_id":"c1","text":"A","evidence_ids":["snippet-1"],"grounding_status":"DIRECTLY_SUPPORTED","acceptance_status":"ACCEPTED","previous_claim_id":"c1"}
    audited={"claims":[claim],"refined_claims":[refined],"refined_answer":"C"}; m.normalize_audit(audited)
    assert claim["acceptance_status"]=="ACCEPTED"
    assert claim["intermediate_acceptance_status"]=="ACCEPTED_WITH_ASSUMPTION"
    assert audited["conclusion_status"]["acceptance_status"]=="ACCEPTED_WITH_ASSUMPTION"
    graph=m.build_graph(q,generated,audited)
    assert sum(n["type"]=="EVIDENCE_SNIPPET" for n in graph["nodes"])==2
    assert any(n["type"]=="INTERMEDIATE_INFERENCE" for n in graph["nodes"])
    assert any(n["type"]=="ASSUMPTION" for n in graph["nodes"])
    by_id={n["id"]:n for n in graph["nodes"]}
    assert by_id["c1"]["acceptance_status"]=="ACCEPTED"
    assert by_id["inference-c1"]["acceptance_status"]=="ACCEPTED_WITH_ASSUMPTION"
    assert by_id["conclusion"]["acceptance_status"]=="ACCEPTED_WITH_ASSUMPTION"


def _question():
    return {"body":"Q?","snippets":[{"id":"snippet-1","text":"A","document":"d"},{"id":"snippet-2","text":"B","document":"d"}]}


def _claim(**updates):
    value={"claim_id":"c1","text":"A","evidence_ids":["snippet-1"],"grounding_status":"DIRECTLY_SUPPORTED","acceptance_status":"ACCEPTED"}
    value.update(updates)
    return value


def test_direct_claim_without_refinement_is_accepted_and_supports_conclusion():
    audited={"claims":[_claim()],"refined_claims":[],"refined_answer":"A"}
    m.normalize_audit(audited,{"snippet-1","snippet-2"})
    m.validate_audit_output(audited,{"snippet-1","snippet-2"})
    graph=m.build_graph(_question(),{"explanation_steps":[]},audited)
    assert audited["claims"][0]["acceptance_status"]=="ACCEPTED"
    assert audited["conclusion_status"]["acceptance_status"]=="ACCEPTED"
    assert {"source":"c1","target":"conclusion","type":"supports"} in graph["edges"]


def test_direct_claim_with_unchanged_refinement_is_accepted():
    refined=_claim(previous_claim_id="c1")
    audited={"claims":[_claim()],"refined_claims":[refined],"refined_answer":"A"}
    m.normalize_audit(audited,{"snippet-1"})
    m.validate_audit_output(audited,{"snippet-1"})
    graph=m.build_graph(_question(),{"explanation_steps":[]},audited)
    assert refined["acceptance_status"]=="ACCEPTED"
    assert {"source":"refined-c1","target":"conclusion","type":"supports"} in graph["edges"]


def test_bridge_affects_only_intermediate_and_conclusion():
    claim=_claim(intermediate_inference="A plus a bridge implies B",bridge_assumption={"text":"bridge","status":"REASONABLE_BUT_UNSTATED"})
    audited={"claims":[claim],"refined_claims":[],"refined_answer":"B"}
    m.normalize_audit(audited,{"snippet-1"})
    assert claim["acceptance_status"]=="ACCEPTED"
    assert claim["intermediate_acceptance_status"]=="ACCEPTED_WITH_ASSUMPTION"
    assert audited["conclusion_status"]["acceptance_status"]=="ACCEPTED_WITH_ASSUMPTION"


def test_overstated_original_does_not_support_conclusion_but_refinement_can():
    original=_claim(grounding_status="OVERSTATED",acceptance_status="ACCEPTED")
    refined=_claim(claim_id="c2",previous_claim_id="c1")
    audited={"claims":[original],"refined_claims":[refined],"refined_answer":"A"}
    m.normalize_audit(audited,{"snippet-1"})
    m.validate_audit_output(audited,{"snippet-1"})
    graph=m.build_graph(_question(),{"explanation_steps":[]},audited)
    conclusion_sources={e["source"] for e in graph["edges"] if e["target"]=="conclusion" and e["type"]=="supports"}
    assert original["acceptance_status"]=="REJECTED"
    assert conclusion_sources=={"refined-c2"}


def test_invalid_previous_claim_id_is_structured_validation_error():
    audited={"claims":[_claim()],"refined_claims":[_claim(claim_id="c2",previous_claim_id="missing")],"refined_answer":"A"}
    m.normalize_audit(audited,{"snippet-1"})
    with pytest.raises(m.ModelOutputValidationError) as error:
        m.validate_audit_output(audited,{"snippet-1"})
    assert error.value.as_dict()["code"]=="MODEL_OUTPUT_VALIDATION_ERROR"
    assert error.value.path=="$.refined_claims[0].previous_claim_id"


@pytest.mark.parametrize("payload", [{}, {"atomic_claims":[{"claim_id":"c1"}]}])
def test_malformed_generation_output_is_structured_validation_error(payload):
    with pytest.raises(m.ModelOutputValidationError) as error:
        m.validate_generation_output(payload,{"snippet-1"})
    assert error.value.as_dict()["code"]=="MODEL_OUTPUT_VALIDATION_ERROR"


def test_malformed_audit_output_is_structured_validation_error_not_key_error():
    malformed={"claims":[{"claim_id":"c1","text":"A","evidence_ids":["snippet-1"]}],"refined_claims":[],"refined_answer":"A"}
    with pytest.raises(m.ModelOutputValidationError) as error:
        m.validate_raw_audit_output(malformed)
    assert error.value.as_dict()["code"]=="MODEL_OUTPUT_VALIDATION_ERROR"
    assert error.value.path=="$.claims[0].grounding_status"


def test_artifact_and_graph_statuses_are_consistent():
    audited={"claims":[_claim()],"refined_claims":[],"refined_answer":"A"}
    m.normalize_audit(audited,{"snippet-1"})
    m.validate_audit_output(audited,{"snippet-1"})
    graph=m.build_graph(_question(),{"explanation_steps":[]},audited)
    nodes={node["id"]:node for node in graph["nodes"]}
    claim=audited["claims"][0]
    assert nodes[claim["claim_id"]]["grounding_status"]==claim["grounding_status"]
    assert nodes[claim["claim_id"]]["acceptance_status"]==claim["acceptance_status"]
    assert nodes["conclusion"]["grounding_status"]==audited["conclusion_status"]["grounding_status"]
    assert nodes["conclusion"]["acceptance_status"]==audited["conclusion_status"]["acceptance_status"]
