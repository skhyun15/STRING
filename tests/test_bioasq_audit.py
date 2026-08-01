import importlib.util, json
from pathlib import Path
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
