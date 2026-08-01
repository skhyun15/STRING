"""Run one BioASQ claim-evidence audit with gpt-4.1."""
from __future__ import annotations

import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from string_agent.integration.explanation_refiner import _load_openai_configuration
from openai import OpenAI

ALLOWED_GROUNDING = {"DIRECTLY_SUPPORTED", "COMPOSITIONALLY_SUPPORTED", "PARTIALLY_SUPPORTED", "CONTRADICTED", "UNGROUNDED", "OVERSTATED", "UNKNOWN"}
ALLOWED_ACCEPTANCE = {"ACCEPTED", "ACCEPTED_WITH_ASSUMPTION", "REJECTED", "NOT_EVALUATED"}
ALLOWED_BRIDGE = {"EXPLICIT", "REASONABLE_BUT_UNSTATED", "UNSUPPORTED"}


class ModelOutputValidationError(ValueError):
    code = "MODEL_OUTPUT_VALIDATION_ERROR"

    def __init__(self, message: str, *, path: str = "$") -> None:
        self.path = path
        super().__init__(f"{self.code} at {path}: {message}")

    def as_dict(self) -> dict:
        return {"code": self.code, "path": self.path, "message": str(self)}


def _require(value: object, expected: type, path: str) -> object:
    if not isinstance(value, expected):
        raise ModelOutputValidationError(
            f"expected {expected.__name__}, got {type(value).__name__}", path=path
        )
    return value


def validate_generation_output(generated: dict, valid_evidence_ids: set[str]) -> None:
    _require(generated, dict, "$")
    claims = generated.get("atomic_claims")
    _require(claims, list, "$.atomic_claims")
    if not claims:
        raise ModelOutputValidationError("must not be empty", path="$.atomic_claims")
    for index, claim in enumerate(claims):
        path = f"$.atomic_claims[{index}]"
        _require(claim, dict, path)
        for field in ("claim_id", "text"):
            value = claim.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ModelOutputValidationError("must be a non-empty string", path=f"{path}.{field}")
        evidence_ids = claim.get("evidence_ids")
        _require(evidence_ids, list, f"{path}.evidence_ids")
        if not all(isinstance(item, str) for item in evidence_ids):
            raise ModelOutputValidationError("must contain strings", path=f"{path}.evidence_ids")
        if not set(evidence_ids) <= valid_evidence_ids:
            raise ModelOutputValidationError("contains an unknown evidence ID", path=f"{path}.evidence_ids")


def validate_raw_audit_output(audited: dict) -> None:
    _require(audited, dict, "$")
    claims = audited.get("claims")
    _require(claims, list, "$.claims")
    refined = audited.get("refined_claims", [])
    _require(refined, list, "$.refined_claims")
    for group_name, group in (("claims", claims), ("refined_claims", refined)):
        for index, claim in enumerate(group):
            path = f"$.{group_name}[{index}]"
            _require(claim, dict, path)
            for field in ("claim_id", "text", "grounding_status", "acceptance_status"):
                value = claim.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ModelOutputValidationError("must be a non-empty string", path=f"{path}.{field}")
            evidence_ids = claim.get("evidence_ids")
            _require(evidence_ids, list, f"{path}.evidence_ids")
    if "refined_answer" not in audited or audited["refined_answer"] is not None and not isinstance(audited["refined_answer"], str):
        raise ModelOutputValidationError("must be a string or null", path="$.refined_answer")


def validate_audit_output(audited: dict, valid_evidence_ids: set[str]) -> None:
    _require(audited, dict, "$")
    claims = audited.get("claims")
    _require(claims, list, "$.claims")
    refined = audited.get("refined_claims", [])
    _require(refined, list, "$.refined_claims")
    original_ids: set[str] = set()
    for group_name, group in (("claims", claims), ("refined_claims", refined)):
        for index, claim in enumerate(group):
            path = f"$.{group_name}[{index}]"
            _require(claim, dict, path)
            for field in ("claim_id", "text"):
                value = claim.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ModelOutputValidationError("must be a non-empty string", path=f"{path}.{field}")
            evidence_ids = claim.get("evidence_ids")
            _require(evidence_ids, list, f"{path}.evidence_ids")
            if not all(isinstance(item, str) for item in evidence_ids):
                raise ModelOutputValidationError("must contain strings", path=f"{path}.evidence_ids")
            if not set(evidence_ids) <= valid_evidence_ids:
                raise ModelOutputValidationError("contains an unknown evidence ID", path=f"{path}.evidence_ids")
            grounding = claim.get("grounding_status")
            if grounding not in ALLOWED_GROUNDING:
                raise ModelOutputValidationError(f"invalid grounding status {grounding!r}", path=f"{path}.grounding_status")
            acceptance = claim.get("acceptance_status")
            if acceptance not in ALLOWED_ACCEPTANCE:
                raise ModelOutputValidationError(f"invalid acceptance status {acceptance!r}", path=f"{path}.acceptance_status")
            bridge = claim.get("bridge_assumption") or {}
            _require(bridge, dict, f"{path}.bridge_assumption")
            if bridge and bridge.get("status") not in ALLOWED_BRIDGE:
                raise ModelOutputValidationError("invalid bridge status", path=f"{path}.bridge_assumption.status")
            if group_name == "claims":
                if claim["claim_id"] in original_ids:
                    raise ModelOutputValidationError("duplicate claim ID", path=f"{path}.claim_id")
                original_ids.add(claim["claim_id"])
    for index, claim in enumerate(refined):
        previous = claim.get("previous_claim_id")
        if not isinstance(previous, str) or previous not in original_ids:
            raise ModelOutputValidationError(
                "must reference an existing original claim ID",
                path=f"$.refined_claims[{index}].previous_claim_id",
            )
    if "refined_answer" not in audited or audited["refined_answer"] is not None and not isinstance(audited["refined_answer"], str):
        raise ModelOutputValidationError("must be a string or null", path="$.refined_answer")
    conclusion = audited.get("conclusion_status")
    _require(conclusion, dict, "$.conclusion_status")
    if conclusion.get("grounding_status") not in ALLOWED_GROUNDING:
        raise ModelOutputValidationError("invalid grounding status", path="$.conclusion_status.grounding_status")
    if conclusion.get("acceptance_status") not in ALLOWED_ACCEPTANCE:
        raise ModelOutputValidationError("invalid acceptance status", path="$.conclusion_status.acceptance_status")

def load_question(path: Path, question_id: str) -> dict:
    questions = json.loads(path.read_text(encoding="utf-8"))["questions"]
    matches = [q for q in questions if q.get("id") == question_id]
    if len(matches) != 1: raise ValueError(f"expected exactly one question {question_id}, found {len(matches)}")
    q = matches[0]
    snippets = [{"id": f"snippet-{i}", "text": s["text"], "document": s.get("document")} for i, s in enumerate(q["snippets"], 1)]
    return {"id": q["id"], "body": q["body"], "type": q["type"], "documents": q["documents"], "snippets": snippets, "ideal_answer": q.get("ideal_answer"), "exact_answer": q.get("exact_answer")}

def generation_prompt(q: dict) -> str:
    evidence = "\n".join(f"{s['id']}: {s['text']}" for s in q["snippets"])
    return f"""Use only the evidence snippets. Do not add outside biomedical knowledge.
Question: {q['body']}
Evidence:\n{evidence}
Return JSON with concise_answer, atomic_claims (claim_id,text,evidence_ids), explanation_steps (step_id,text,evidence_ids), uncertainty."""

def audit_prompt(q: dict, generated: dict) -> str:
    evidence = "\n".join(f"{s['id']}: {s['text']}" for s in q["snippets"])
    return f"""Audit and refine using only these snippets. Remove unsupported claims, weaken wording, correct evidence links, or state uncertainty. Never add a biomedical fact.
Question: {q['body']}
Evidence:\n{evidence}
Generated candidate:\n{json.dumps(generated, ensure_ascii=False)}
For every compositionally supported claim include evidence_subclaims (evidence_id, subclaim), intermediate_inference, and bridge_assumption (text, status, explicit_in_snippets). Bridge status must be EXPLICIT, REASONABLE_BUT_UNSTATED, or UNSUPPORTED. Keep a directly supported atomic claim ACCEPTED. A necessary REASONABLE_BUT_UNSTATED bridge makes only its dependent intermediate inference and conclusion ACCEPTED_WITH_ASSUMPTION. Refinement may only split snippet-specific subclaims, decompose over-combined claims, weaken unstated causation, or expose assumptions.
Return JSON with claims (claim_id,text,evidence_ids,grounding_status,relation_strength,rationale,acceptance_status,evidence_subclaims,intermediate_inference,bridge_assumption), refined_answer, refined_claims (claim_id,text,evidence_ids,grounding_status,acceptance_status,previous_claim_id), refinement_summary, refinement_metadata."""

def _call(client, prompt: str) -> tuple[dict, str, dict]:
    response = client.chat.completions.create(model="gpt-4.1", temperature=0, response_format={"type":"json_object"}, messages=[{"role":"system","content":"Return valid JSON only."},{"role":"user","content":prompt}])
    raw = response.choices[0].message.content
    usage = response.usage.model_dump() if response.usage else {}
    return json.loads(raw), raw, usage

def build_graph(q: dict, generated: dict, audited: dict) -> dict:
    nodes=[{"id":"question","type":"QUESTION","label":q["body"],"grounding_status":"NOT_APPLICABLE","acceptance_status":"ACCEPTED","verifier_status":"NOT_APPLICABLE"}]
    edges=[]
    for s in q["snippets"]: nodes.append({"id":s["id"],"type":"EVIDENCE_SNIPPET","label":s["text"],"grounding_status":"SOURCE","acceptance_status":"ACCEPTED","verifier_status":"NOT_APPLICABLE"})
    for c in audited["claims"]:
        nodes.append({"id":c["claim_id"],"type":"ANSWER_CLAIM","label":c["text"],"grounding_status":c["grounding_status"],"acceptance_status":c["acceptance_status"],"verifier_status":"NOT_APPLICABLE"})
        relation="supports" if c["grounding_status"] in {"DIRECTLY_SUPPORTED","COMPOSITIONALLY_SUPPORTED"} else "partially_supports"
        for eid in c["evidence_ids"]: edges.append({"source":eid,"target":c["claim_id"],"type":relation})
        if c.get("intermediate_inference"):
            iid="inference-"+c["claim_id"]
            nodes.append({"id":iid,"type":"INTERMEDIATE_INFERENCE","label":c["intermediate_inference"],"grounding_status":c["grounding_status"],"acceptance_status":c["intermediate_acceptance_status"],"verifier_status":"NOT_APPLICABLE"})
            edges.append({"source":iid,"target":c["claim_id"],"type":"derived_from"})
        bridge=c.get("bridge_assumption") or {}
        if bridge.get("text"):
            aid="assumption-"+c["claim_id"]
            nodes.append({"id":aid,"type":"ASSUMPTION","label":bridge["text"],"assumption_status":bridge.get("status"),"grounding_status":"UNGROUNDED" if bridge.get("status")=="UNSUPPORTED" else "UNKNOWN","acceptance_status":c["intermediate_acceptance_status"],"verifier_status":"NOT_APPLICABLE"})
            edges.append({"source":aid,"target":c["claim_id"],"type":"depends_on"})
    for s in generated.get("explanation_steps",[]):
        nodes.append({"id":s["step_id"],"type":"EXPLANATION_STEP","label":s["text"],"grounding_status":"UNKNOWN","acceptance_status":"UNVERIFIED","verifier_status":"NOT_APPLICABLE"})
        for eid in s.get("evidence_ids",[]): edges.append({"source":eid,"target":s["step_id"],"type":"grounded_in"})
    for c in audited.get("refined_claims",[]):
        rid="refined-"+c["claim_id"]
        nodes.append({"id":rid,"type":"REFINED_CLAIM","label":c["text"],"grounding_status":c["grounding_status"],"acceptance_status":c["acceptance_status"],"verifier_status":"NOT_APPLICABLE"})
        if c.get("previous_claim_id"): edges.append({"source":c["previous_claim_id"],"target":rid,"type":"revised_to"})
        relation = "supports" if c["acceptance_status"] == "ACCEPTED" else "partially_supports"
        for eid in c.get("evidence_ids",[]): edges.append({"source":eid,"target":rid,"type":relation})
    conclusion=audited["conclusion_status"]
    nodes.append({"id":"conclusion","type":"CONCLUSION","label":audited["refined_answer"],"grounding_status":conclusion["grounding_status"],"acceptance_status":conclusion["acceptance_status"],"verifier_status":"NOT_APPLICABLE"})
    effective = audited.get("refined_claims") or audited["claims"]
    refined_mode = bool(audited.get("refined_claims"))
    for c in effective:
        if c["acceptance_status"] == "ACCEPTED":
            source = "refined-" + c["claim_id"] if refined_mode else c["claim_id"]
            edges.append({"source":source,"target":"conclusion","type":"supports"})
    graph = {"nodes":nodes,"edges":edges}
    node_ids = {node["id"] for node in nodes}
    for edge in edges:
        assert edge["source"] in node_ids, f"missing graph source node: {edge['source']}"
        assert edge["target"] in node_ids, f"missing graph target node: {edge['target']}"
    return graph

def normalize_audit(audited: dict, valid_evidence_ids: set[str] | None = None) -> dict:
    grounding_aliases = {
        "DIRECT": "DIRECTLY_SUPPORTED",
        "COMPOSITIONAL": "COMPOSITIONALLY_SUPPORTED",
        "INDIRECT": "COMPOSITIONALLY_SUPPORTED",
        "INDIRECTLY_SUPPORTED": "COMPOSITIONALLY_SUPPORTED",
        "PARTIAL": "PARTIALLY_SUPPORTED",
    }
    for claim in audited.get("claims", []) + audited.get("refined_claims", []):
        claim["grounding_status"] = str(claim.get("grounding_status", "UNKNOWN")).upper()
        claim["grounding_status"] = grounding_aliases.get(claim["grounding_status"], claim["grounding_status"])
        if claim["grounding_status"] == "SUPPORTED":
            claim["grounding_status"] = "COMPOSITIONALLY_SUPPORTED" if len(set(claim.get("evidence_ids", []))) > 1 else "DIRECTLY_SUPPORTED"
        claim["acceptance_status"] = str(claim.get("acceptance_status", "REJECTED")).upper()
        bridge=claim.get("bridge_assumption") or {}
        if bridge:
            bridge["status"]=str(bridge.get("status","UNSUPPORTED")).upper()
        evidence_ids = claim.get("evidence_ids", [])
        evidence_valid = bool(evidence_ids) and (
            valid_evidence_ids is None or set(evidence_ids) <= valid_evidence_ids
        )
        if claim["grounding_status"] == "DIRECTLY_SUPPORTED" and evidence_valid:
            claim["acceptance_status"] = "ACCEPTED"
        elif claim["grounding_status"] in {"CONTRADICTED", "UNGROUNDED", "OVERSTATED"}:
            claim["acceptance_status"] = "REJECTED"
        bridge_status=bridge.get("status")
        claim["intermediate_acceptance_status"] = (
            "ACCEPTED_WITH_ASSUMPTION" if bridge_status == "REASONABLE_BUT_UNSTATED"
            else "REJECTED" if bridge_status == "UNSUPPORTED"
            else claim["acceptance_status"]
        )
    source_claims=audited.get("claims", [])
    intermediate_statuses=[c.get("intermediate_acceptance_status") for c in source_claims if c.get("intermediate_inference")]
    refined=audited.get("refined_claims", [])
    effective_claims = refined or source_claims
    accepted_claims = [c for c in effective_claims if c.get("acceptance_status") == "ACCEPTED"]
    base_accepted = bool(accepted_claims)
    if "REJECTED" in intermediate_statuses or not base_accepted:
        conclusion_acceptance="REJECTED"
    elif "ACCEPTED_WITH_ASSUMPTION" in intermediate_statuses:
        conclusion_acceptance="ACCEPTED_WITH_ASSUMPTION"
    else:
        conclusion_acceptance="ACCEPTED"
    audited["conclusion_status"]={
        "grounding_status": "COMPOSITIONALLY_SUPPORTED" if len(accepted_claims) > 1 else (accepted_claims[0]["grounding_status"] if accepted_claims else "UNKNOWN"),
        "acceptance_status": conclusion_acceptance,
        "verifier_status": "NOT_APPLICABLE",
    }
    return audited

def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,required=True); p.add_argument("--question-id",required=True); p.add_argument("--output-root",type=Path,default=ROOT/"outputs/bioasq_audit"); p.add_argument("--run-id"); args=p.parse_args()
    q=load_question(args.dataset,args.question_id); out=args.output_root/q["id"]/(Path("runs")/args.run_id if args.run_id else Path()); out.mkdir(parents=True,exist_ok=True); (out/"debug").mkdir(exist_ok=True)
    events=out/"timing_events.jsonl"; debug=out/"debug/model_responses.jsonl"; started=time.perf_counter(); calls=0; usage=[]
    def event(phase,event):
        with events.open("a",encoding="utf-8") as f: f.write(json.dumps({"timestamp":datetime.now(timezone.utc).isoformat(),"phase":phase,"event":event})+"\n")
    _load_openai_configuration(ROOT/"explanation_refinement","gpt-4.1"); client=OpenAI()
    prompt1=generation_prompt(q); event("answer_generation","start"); generated,raw,u=_call(client,prompt1); calls+=1; usage.append(u); event("answer_generation","end")
    with debug.open("a",encoding="utf-8") as f: f.write(json.dumps({"phase":"answer_generation","raw_response":raw},ensure_ascii=False)+"\n")
    valid_ids={s["id"] for s in q["snippets"]}
    validate_generation_output(generated, valid_ids)
    prompt2=audit_prompt(q,generated); event("grounding_refinement","start"); audited,raw,u=_call(client,prompt2); calls+=1; usage.append(u); event("grounding_refinement","end"); validate_raw_audit_output(audited); audited=normalize_audit(audited, valid_ids)
    with debug.open("a",encoding="utf-8") as f: f.write(json.dumps({"phase":"grounding_refinement","raw_response":raw},ensure_ascii=False)+"\n")
    validate_audit_output(audited, valid_ids)
    graph=build_graph(q,generated,audited); runtime=time.perf_counter()-started
    artifact={"schema_version":1,"run_id":args.run_id,"input":q,"generation_prompt_excludes_gold":("ideal_answer" not in prompt1 and "exact_answer" not in prompt1),"generated":generated,"grounding":audited["claims"],"refinement":{"summary":audited.get("refinement_summary"),"metadata":audited.get("refinement_metadata"),"refined_answer":audited["refined_answer"],"refined_claims":audited.get("refined_claims",[])},"conclusion_status":audited["conclusion_status"],"final_accepted_answer":audited["refined_answer"],"verifier_status":"NOT_APPLICABLE","posthoc_gold_comparison":{"ideal_answer":q["ideal_answer"],"exact_answer":q["exact_answer"]},"api_calls":calls,"usage":usage,"runtime_seconds":runtime}
    (out/"artifact.json").write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding="utf-8"); (out/"graph.json").write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding="utf-8")
    # Keep the BioASQ presentation in the frontend template so generated runs
    # all receive the same auditable dashboard without changing artifact data.
    template = ROOT / "demo" / "bioasq.html"
    if not template.exists():
        raise FileNotFoundError(f"BioASQ renderer template missing: {template}")
    (out / "index.html").write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"answer":generated["concise_answer"],"claims":len(audited["claims"]),"accepted":sum(c["acceptance_status"]=="ACCEPTED" for c in audited["claims"]),"api_calls":calls,"runtime":runtime,"output":str(out)},ensure_ascii=False))
    return 0
if __name__=="__main__": raise SystemExit(main())
