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
    claims={c["claim_id"]:c for c in audited["claims"]}
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
        for eid in c.get("evidence_ids",[]): edges.append({"source":eid,"target":rid,"type":"supports"})
    conclusion=audited["conclusion_status"]
    nodes.append({"id":"conclusion","type":"CONCLUSION","label":audited["refined_answer"],"grounding_status":conclusion["grounding_status"],"acceptance_status":conclusion["acceptance_status"],"verifier_status":"NOT_APPLICABLE"})
    for c in audited.get("refined_claims",[]): edges.append({"source":"refined-"+c["claim_id"],"target":"conclusion","type":"derived_from"})
    return {"nodes":nodes,"edges":edges}

def normalize_audit(audited: dict) -> dict:
    grounding_aliases = {"DIRECT": "DIRECTLY_SUPPORTED", "COMPOSITIONAL": "COMPOSITIONALLY_SUPPORTED", "PARTIAL": "PARTIALLY_SUPPORTED"}
    for claim in audited.get("claims", []) + audited.get("refined_claims", []):
        claim["grounding_status"] = str(claim.get("grounding_status", "UNKNOWN")).upper()
        claim["grounding_status"] = grounding_aliases.get(claim["grounding_status"], claim["grounding_status"])
        if claim["grounding_status"] == "SUPPORTED":
            claim["grounding_status"] = "COMPOSITIONALLY_SUPPORTED" if len(set(claim.get("evidence_ids", []))) > 1 else "DIRECTLY_SUPPORTED"
        claim["acceptance_status"] = str(claim.get("acceptance_status", "REJECTED")).upper()
        bridge=claim.get("bridge_assumption") or {}
        if bridge:
            bridge["status"]=str(bridge.get("status","UNSUPPORTED")).upper()
        if claim["grounding_status"] == "DIRECTLY_SUPPORTED":
            claim["acceptance_status"] = "ACCEPTED"
        bridge_status=bridge.get("status")
        claim["intermediate_acceptance_status"] = (
            "ACCEPTED_WITH_ASSUMPTION" if bridge_status == "REASONABLE_BUT_UNSTATED"
            else "REJECTED" if bridge_status == "UNSUPPORTED"
            else claim["acceptance_status"]
        )
    source_claims=audited.get("claims", [])
    intermediate_statuses=[c.get("intermediate_acceptance_status") for c in source_claims if c.get("intermediate_inference")]
    refined=audited.get("refined_claims", [])
    base_accepted=bool(refined) and all(c.get("acceptance_status") == "ACCEPTED" for c in refined)
    if "REJECTED" in intermediate_statuses or not base_accepted:
        conclusion_acceptance="REJECTED"
    elif "ACCEPTED_WITH_ASSUMPTION" in intermediate_statuses:
        conclusion_acceptance="ACCEPTED_WITH_ASSUMPTION"
    else:
        conclusion_acceptance="ACCEPTED"
    audited["conclusion_status"]={
        "grounding_status": "COMPOSITIONALLY_SUPPORTED" if len(refined) > 1 else (refined[0]["grounding_status"] if refined else "UNKNOWN"),
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
    if not generated.get("atomic_claims"): raise RuntimeError("no atomic claims generated")
    prompt2=audit_prompt(q,generated); event("grounding_refinement","start"); audited,raw,u=_call(client,prompt2); calls+=1; usage.append(u); event("grounding_refinement","end"); audited=normalize_audit(audited)
    with debug.open("a",encoding="utf-8") as f: f.write(json.dumps({"phase":"grounding_refinement","raw_response":raw},ensure_ascii=False)+"\n")
    valid_ids={s["id"] for s in q["snippets"]}
    for c in audited["claims"]+audited.get("refined_claims",[]):
        if c["grounding_status"] not in ALLOWED_GROUNDING or not set(c.get("evidence_ids",[]))<=valid_ids: raise RuntimeError("invalid grounding output")
    graph=build_graph(q,generated,audited); runtime=time.perf_counter()-started
    artifact={"schema_version":1,"run_id":args.run_id,"input":q,"generation_prompt_excludes_gold":("ideal_answer" not in prompt1 and "exact_answer" not in prompt1),"generated":generated,"grounding":audited["claims"],"refinement":{"summary":audited.get("refinement_summary"),"metadata":audited.get("refinement_metadata"),"refined_answer":audited["refined_answer"],"refined_claims":audited.get("refined_claims",[])},"conclusion_status":audited["conclusion_status"],"final_accepted_answer":audited["refined_answer"],"verifier_status":"NOT_APPLICABLE","posthoc_gold_comparison":{"ideal_answer":q["ideal_answer"],"exact_answer":q["exact_answer"]},"api_calls":calls,"usage":usage,"runtime_seconds":runtime}
    (out/"artifact.json").write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding="utf-8"); (out/"graph.json").write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"index.html").write_text("""<!doctype html><meta charset=utf-8><script src='https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js'></script><h1>BioASQ Evidence Audit</h1><pre id=m></pre><div id=g style='height:600px'></div><script>Promise.all(['artifact.json','graph.json'].map(x=>fetch(x).then(r=>r.json()))).then(([a,x])=>{m.textContent=JSON.stringify({question:a.input.body,generated:a.generated.concise_answer,final:a.final_accepted_answer,gold:a.posthoc_gold_comparison},null,2);cytoscape({container:g,elements:[...x.nodes.map(n=>({data:n})),...x.edges.map((e,i)=>({data:{id:'e'+i,...e}}))],style:[{selector:'node',style:{label:'data(label)','text-wrap':'wrap','text-max-width':220}},{selector:'edge',style:{label:'data(type)','target-arrow-shape':'triangle','curve-style':'bezier'}}],layout:{name:'cose'}})})</script>""",encoding="utf-8")
    print(json.dumps({"answer":generated["concise_answer"],"claims":len(audited["claims"]),"accepted":sum(c["acceptance_status"]=="ACCEPTED" for c in audited["claims"]),"api_calls":calls,"runtime":runtime,"output":str(out)},ensure_ascii=False))
    return 0
if __name__=="__main__": raise SystemExit(main())
