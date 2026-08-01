"""Sequential ProofWriter batch runner backed only by the upstream adapter."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from string_agent.datasets.proofwriter import ProofWriterExample, load_proofwriter_file
from string_agent.integration import ExplanationRefiner, RunStatus, artifact_to_dict


def _candidate_worker(queue, premise: str, hypothesis: str, output_root: str, run_id: str) -> None:
    os.setsid()
    try:
        refiner = ExplanationRefiner(model="gpt-4.1", max_iterations=2, output_root=Path(output_root))
        artifact, path = refiner.run(premise=premise, hypothesis=hypothesis, run_id=run_id)
        queue.put({"ok": True, "artifact": artifact_to_dict(artifact), "path": str(path)})
    except Exception as exc:
        queue.put({"ok": False, "error": repr(exc)})


def _source_items(context: str) -> tuple[list[dict], list[dict]]:
    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", context) if x.strip()]
    facts, rules = [], []
    for index, text in enumerate(sentences, 1):
        item = {"id": f"source-{index}", "text": text}
        is_rule = re.search(r"\b(if|all|every|then)\b", text, re.I) or re.search(
            r"(?:[A-Za-z-]+(?:,\s*[A-Za-z-]+)*)\s+things\s+are\s+", text, re.I
        )
        (rules if is_rule else facts).append(item)
    return facts, rules


def _normal(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _direct_fact_artifact(hypothesis: str, source: dict) -> dict:
    return {"status": "DIRECT_FACT", "final_validity": None, "iteration_count": 0,
            "total_openai_api_calls": 0, "total_timing_seconds": 0.0,
            "initial_candidate": {"proposed_answer": "TRUE", "explanation": source["text"], "graph": {"steps": []}},
            "final_graph": {"steps": []}, "direct_fact_source": source}


def _grounding(artifact: dict, context: str, hypothesis: str, role: str) -> list[dict]:
    facts, rules = _source_items(context)
    normalized_facts = {_normal(x["text"]): x for x in facts}
    normalized_rules = {_normal(x["text"]): x for x in rules}
    steps = []
    initial = artifact.get("initial_candidate") or {}
    explanations = [initial.get("explanation", "")]
    explanations += [i.get("input_explanation", "") for i in artifact.get("iterations", [])]
    seen = set()
    for explanation in explanations:
        for line in explanation.splitlines():
            text = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            if re.match(r"^(refine strategy|updated explanatory sentences)\b", text, re.I):
                continue
            key = _normal(text)
            if key in normalized_facts:
                category, source = "GROUNDED_DIRECT_FACT", normalized_facts[key]
            elif key in normalized_rules:
                category, source = "GROUNDED", normalized_rules[key]
            elif key == _normal(hypothesis):
                category = "CIRCULAR_QUERY" if role == "query" else "CIRCULAR_COMPLEMENT"
                source = None
            elif re.search(r"\b(all|every|if|then|whenever)\b", text, re.I):
                category, source = "UNGROUNDED_NEW_RULE", None
            elif re.search(r"\b(therefore|thus|hence|because|implies|means)\b", text, re.I):
                category, source = "UNGROUNDED_BRIDGE", None
            else:
                category, source = "UNGROUNDED_NEW_FACT", None
            steps.append({"text": text, "cited_source_sentence_id": source["id"] if source else None, "source_text": source["text"] if source else None, "derived_claim": text, "grounding": category})
    return steps


def _statement(question: str) -> str:
    marker = "?"
    return question.split(marker, 1)[1].strip() if marker in question else question.strip()


def _complement(statement: str) -> str:
    match = re.match(r"^(.*?)(?: is| are) not (.+)$", statement, re.I)
    if match:
        return f"{match.group(1)}{statement[len(match.group(1)):].replace(' not ', ' ', 1)}"
    match = re.match(r"^(.*?)( is| are) (.+)$", statement, re.I)
    if match:
        return f"{match.group(1)}{match.group(2)} not {match.group(3)}"
    return f"It is not the case that {statement}"


def _candidate_status(artifact: dict) -> str:
    status = artifact["status"]
    if status == "ERROR":
        return "ERROR"
    if artifact.get("final_validity") is True:
        return "PROVED"
    return "NOT_PROVED"


def _acceptance(formal_status: str, grounding: list[dict]) -> str:
    grounded = bool(grounding) and all(x["grounding"].startswith("GROUNDED") for x in grounding)
    return "ACCEPTED" if formal_status in {"PROVED", "DIRECT_FACT"} and grounded else "REJECTED"


def _verdict(query: str, complement: str) -> str:
    if any(status in {"ERROR", "TIMEOUT", "FORMALISATION_ADAPTER_ERROR"} for status in (query, complement)):
        return "UNRESOLVED"
    if query == "ACCEPTED" and complement == "ACCEPTED":
        return "INCONSISTENT"
    if query == "ACCEPTED":
        return "SUPPORTED"
    if complement == "ACCEPTED":
        return "CONTRADICTED"
    return "UNKNOWN"


def _system_status(query: dict, complement: dict) -> str:
    failures = [x.get("system_status") for x in (query, complement) if x.get("system_status") != "OK"]
    return failures[0] if failures else "OK"


def _graph(problem: ProofWriterExample, query: dict, complement: dict, verdict: str) -> dict:
    facts, rules = _source_items(problem.context)
    nodes = [
        {"id": "query", "type": "QUERY", "label": problem.question, "status": query["status"], "grounding_status": query.get("grounding_status"), "formal_status": query.get("formal_status"), "proof_status": query.get("formal_status"), "system_status": query.get("system_status")},
        {"id": "complement", "type": "COMPLEMENT", "label": complement["hypothesis"], "status": complement["status"], "grounding_status": complement.get("grounding_status"), "formal_status": complement.get("formal_status"), "proof_status": complement.get("formal_status"), "system_status": complement.get("system_status")},
        {"id": "conclusion", "type": "CONCLUSION", "label": verdict, "status": verdict},
    ]
    edges = [
        {"source": "query", "target": "conclusion", "type": "supports"},
        {"source": "complement", "target": "conclusion", "type": "contradicts"},
    ]
    nodes.extend({"id": x["id"], "type": "FACT", "label": x["text"], "status": "GROUNDED"} for x in facts)
    nodes.extend({"id": x["id"], "type": "RULE", "label": x["text"], "status": "GROUNDED"} for x in rules)
    for role, candidate in (("query", query), ("complement", complement)):
        for grounding in candidate.get("grounding", []):
            if grounding["grounding"] == "GROUNDED_DIRECT_FACT":
                edges.append({"source": grounding["cited_source_sentence_id"], "target": role, "type": "directly_supports"})
    for prefix, artifact in (("query", query["artifact"]), ("complement", complement["artifact"])):
        for step in artifact.get("final_graph", {}).get("steps", []):
            if re.match(r"^(refine strategy|updated explanatory sentences)\b", step["text"], re.I):
                continue
            node_id = f"{prefix}-{step['revision_id']}"
            nodes.append({"id": node_id, "type": "EXPLANATION_STEP", "label": step["text"], "status": step["status"]})
            if (query if prefix == "query" else complement).get("acceptance_status") == "ACCEPTED":
                edges.append({"source": node_id, "target": prefix, "type": "supports"})
    return {"problem_id": problem.id, "nodes": nodes, "edges": edges}


def _html(problem_dir: Path) -> None:
    (problem_dir / "index.html").write_text("""<!doctype html><meta charset='utf-8'><title>ProofWriter</title>
<script src='https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js'></script>
<style>body{font:14px sans-serif}#graph{height:520px;border:1px solid #ccc}</style>
<h1 id='title'></h1><pre id='meta'></pre><div id='graph'></div>
<script>Promise.all(['artifact.json','graph.json'].map(x=>fetch(x).then(r=>r.json()))).then(([a,g])=>{
title.textContent=a.problem_id;meta.textContent=JSON.stringify({gold:a.gold_label,verdict:a.final_symbolic_verdict,status:a.final_status},null,2);
cytoscape({container:document.getElementById('graph'),elements:[...g.nodes.map(n=>({data:n})),...g.edges.map(e=>({data:{id:e.source+'-'+e.target+'-'+e.type,...e}}))],style:[{selector:'node',style:{label:'data(label)','background-color':'#4b7bec','color':'#111','text-wrap':'wrap','text-max-width':180}},{selector:'edge',style:{label:'data(type)','curve-style':'bezier','target-arrow-shape':'triangle'}}],layout:{name:'cose'}})})</script>""", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    examples = load_proofwriter_file(args.dataset)[:20]
    selected = examples[args.start_index : args.start_index + args.limit]
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for problem in selected:
        problem_dir = root / problem.id
        artifact_path = problem_dir / "artifact.json"
        if args.resume and artifact_path.exists():
            rows.append(json.loads(artifact_path.read_text())["summary"])
            continue
        problem_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        statement = _statement(problem.question)
        hypotheses = {"query": statement, "complement": _complement(statement)}
        results = {}
        for role, hypothesis in hypotheses.items():
            run_id = f"{problem.id}-{role}-gpt-4-1"
            partial = problem_dir / role / "partial.json"
            partial.parent.mkdir(parents=True, exist_ok=True)
            facts, _ = _source_items(problem.context)
            direct = next((x for x in facts if _normal(x["text"]) == _normal(hypothesis)), None)
            if direct:
                payload = _direct_fact_artifact(hypothesis, direct)
                grounding = _grounding(payload, problem.context, hypothesis, role)
                results[role] = {"formal_status": "DIRECT_FACT", "grounding_status": "GROUNDED_DIRECT_FACT", "acceptance_status": "ACCEPTED", "system_status": "OK", "status": "ACCEPTED", "hypothesis": hypothesis, "artifact": payload, "grounding": grounding}
                partial.write_text(json.dumps(results[role], ensure_ascii=False, indent=2), encoding="utf-8")
                continue
            queue = mp.get_context("spawn").Queue()
            process = mp.get_context("spawn").Process(target=_candidate_worker, args=(queue, problem.context, hypothesis, str(problem_dir / role), run_id))
            process.start(); process.join(args.candidate_timeout_seconds)
            if process.is_alive():
                os.killpg(process.pid, signal.SIGTERM)
                process.join(10)
                if process.is_alive():
                    os.killpg(process.pid, signal.SIGKILL)
                    process.join(5)
                results[role] = {"formal_status": "NOT_COMPLETED", "grounding_status": "AMBIGUOUS", "acceptance_status": "NOT_EVALUATED", "system_status": "TIMEOUT", "status": "TIMEOUT", "hypothesis": hypothesis, "artifact": {"status": "ERROR", "error": "candidate timeout", "total_openai_api_calls": 0, "total_timing_seconds": args.candidate_timeout_seconds}, "grounding": []}
                partial.write_text(json.dumps(results[role], ensure_ascii=False, indent=2), encoding="utf-8")
                continue
            result = queue.get() if not queue.empty() else {"ok": False, "error": "candidate exited without result"}
            if not result.get("ok"):
                results[role] = {"formal_status": "NOT_COMPLETED", "grounding_status": "AMBIGUOUS", "acceptance_status": "NOT_EVALUATED", "system_status": "ERROR", "status": "ERROR", "hypothesis": hypothesis, "artifact": {"status": "ERROR", "error": result.get("error"), "total_openai_api_calls": 0, "total_timing_seconds": 0}, "grounding": []}
            else:
                payload = result["artifact"]
                grounding = _grounding(payload, problem.context, hypothesis, role)
                formal = _candidate_status(payload)
                grounding_status = "GROUNDED" if grounding and all(x["grounding"].startswith("GROUNDED") for x in grounding) else (grounding[0]["grounding"] if grounding else "AMBIGUOUS")
                accepted = _acceptance(formal, grounding)
                error_text = str(payload.get("error") or "")
                system = "FORMALISATION_ADAPTER_ERROR" if "FORMALISATION_ADAPTER_ERROR" in error_text else ("ERROR" if payload.get("status") == "ERROR" else "OK")
                results[role] = {"formal_status": formal, "grounding_status": grounding_status, "acceptance_status": accepted if system == "OK" else "NOT_EVALUATED", "system_status": system, "status": accepted if system == "OK" else system, "hypothesis": hypothesis, "artifact": payload, "artifact_path": result["path"], "grounding": grounding}
            partial.write_text(json.dumps(results[role], ensure_ascii=False, indent=2), encoding="utf-8")
        q, c = results["query"], results["complement"]
        verdict = _verdict(q["status"], c["status"])
        system_status = _system_status(q, c)
        predicted_label = {"SUPPORTED": "TRUE", "CONTRADICTED": "FALSE"}.get(verdict, verdict)
        summary = {"problem_id": problem.id, "gold_label": problem.gold_label, "predicted_verdict": verdict, "correct": predicted_label == problem.gold_label, "query_status": q["status"], "complement_status": c["status"], "query_formal_status": q.get("formal_status"), "complement_formal_status": c.get("formal_status"), "query_grounding_status": q.get("grounding_status"), "complement_grounding_status": c.get("grounding_status"), "query_acceptance_status": q.get("acceptance_status"), "complement_acceptance_status": c.get("acceptance_status"), "system_status": system_status, "final_status": system_status if system_status != "OK" else ("VALID" if verdict in {"SUPPORTED", "CONTRADICTED", "INCONSISTENT"} else "REJECTED"), "iterations": sum(x["artifact"].get("iteration_count", 0) for x in results.values()), "api_calls": sum(x["artifact"].get("total_openai_api_calls", 0) for x in results.values()), "runtime": time.perf_counter() - started, "failure_category": system_status if system_status != "OK" else ("GROUNDING" if any(x.get("grounding") and any(not y["grounding"].startswith("GROUNDED") for y in x["grounding"]) for x in results.values()) else None), "artifact_link": str(artifact_path), "graph_link": str(problem_dir / "graph.json")}
        combined = {"problem_id": problem.id, "context": problem.context, "question": problem.question, "gold_label": problem.gold_label, "initial_proposed_label": q["artifact"].get("initial_candidate", {}).get("proposed_answer"), "semantic_verdict": verdict, "system_status": system_status, "final_symbolic_verdict": verdict, "final_status": summary["final_status"], "gold_match": summary["correct"], "query_candidate": q, "complement_candidate": c, "summary": summary}
        artifact_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        (problem_dir / "graph.json").write_text(json.dumps(_graph(problem, q, c, verdict), ensure_ascii=False, indent=2), encoding="utf-8")
        _html(problem_dir)
        rows.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
    all_rows = rows
    (root / "summary.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]) if all_rows else ["problem_id"]); writer.writeheader(); writer.writerows(all_rows)
    (root / "index.html").write_text("<meta charset='utf-8'><h1>ProofWriter gpt-4.1 batch</h1><pre id='out'></pre><script>fetch('summary.json').then(r=>r.json()).then(x=>out.textContent=JSON.stringify(x,null,2))</script>", encoding="utf-8")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=REPOSITORY_ROOT / "external/Logic-LLM/data/ProofWriter/dev.json")
    p.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "outputs/proofwriter_batch_gpt_4_1")
    p.add_argument("--limit", type=int, default=20); p.add_argument("--start-index", type=int, default=0); p.add_argument("--resume", action="store_true"); p.add_argument("--candidate-timeout-seconds", type=int, default=300); p.add_argument("--problem-timeout-seconds", type=int, default=600)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
