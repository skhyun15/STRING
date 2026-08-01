"""Serve the natural-language STRING demo and its small local audit endpoint."""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
sys.path.insert(0, str(ROOT / "scripts"))

from openai import OpenAI

from run_bioasq_audit import (
    ALLOWED_ACCEPTANCE,
    ALLOWED_BRIDGE,
    ALLOWED_GROUNDING,
    ModelOutputValidationError,
    normalize_audit,
    validate_audit_output,
    validate_generation_output,
    validate_raw_audit_output,
)


SUPPORTED_MODELS = {"gpt-4.1", "gpt-5.6"}
MAX_REQUEST_BYTES = 1_500_000
MAX_CLAIMS = 3


class RequestValidationError(ValueError):
    code = "INVALID_REQUEST"

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": str(self)}
        if self.field:
            result["field"] = self.field
        return result


def validate_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RequestValidationError("request body must be a JSON object")

    source_text = payload.get("source_text")
    if not isinstance(source_text, str) or not source_text.strip():
        raise RequestValidationError("source_text must be a non-empty string", field="source_text")

    target_claim = payload.get("target_claim")
    if target_claim is not None and not isinstance(target_claim, str):
        raise RequestValidationError("target_claim must be a string or null", field="target_claim")

    audit_focus = payload.get("audit_focus")
    if audit_focus is not None and not isinstance(audit_focus, str):
        raise RequestValidationError("audit_focus must be a string or null", field="audit_focus")

    model = payload.get("model")
    if not isinstance(model, str) or model not in SUPPORTED_MODELS:
        allowed = ", ".join(sorted(SUPPORTED_MODELS))
        raise RequestValidationError(f"model must be one of: {allowed}", field="model")

    return {
        "source_text": source_text,
        "target_claim": target_claim.strip() if target_claim else None,
        "audit_focus": audit_focus.strip() if audit_focus else None,
        "model": model,
    }


def _paragraph_spans(source_text: str) -> list[str]:
    pattern = re.compile(r"(?s)(?:^|\n[ \t]*\n)(\S.*?)(?=\n[ \t]*\n|\Z)")
    return [match.group(1) for match in pattern.finditer(source_text) if match.group(1).strip()]


def _sentence_spans(paragraph: str) -> list[str]:
    boundaries = list(re.finditer(r"(?<=[.!?])\s+", paragraph))
    if not boundaries:
        return [paragraph]
    spans: list[str] = []
    start = 0
    for boundary in boundaries:
        span = paragraph[start:boundary.start()]
        if span.strip():
            spans.append(span)
        start = boundary.end()
    tail = paragraph[start:]
    if tail.strip():
        spans.append(tail)
    return spans or [paragraph]


def split_source_text(source_text: str) -> list[dict[str, str]]:
    """Split paragraphs into exact, deterministic sentence spans."""
    paragraphs = _paragraph_spans(source_text)
    if not paragraphs:
        paragraphs = [source_text]
    spans: list[str] = []
    for paragraph in paragraphs:
        spans.extend(_sentence_spans(paragraph))
    return [{"id": f"source-{index}", "text": text} for index, text in enumerate(spans, 1)]


def _evidence_text(spans: list[dict[str, str]]) -> str:
    return "\n".join(f"{span['id']}: {span['text']}" for span in spans)


def generation_prompt(spans: list[dict[str, str]], target_claim: str | None, audit_focus: str | None) -> str:
    target = target_claim or "No single claim was supplied. Extract at most three audit-worthy claims from the text. Prioritize causal, necessary/sufficient, temporal, universal, and strongly worded claims."
    focus = audit_focus or "No special audit focus was supplied."
    return f"""You are the first stage of a source-grounded reasoning audit.
Use only the supplied source spans. Do not browse, use outside facts, or repair missing premises with domain knowledge.

Source spans:
{_evidence_text(spans)}

Claim to audit:
{target}

Audit focus:
{focus}

Return JSON only with this shape:
{{
  "initial_answer": "a concise interpretation of what the supplied text supports, or an uncertainty statement",
  "atomic_claims": [
    {{"claim_id": "claim-1", "text": "one atomic claim", "evidence_ids": ["source-1"]}}
  ]
}}
Each evidence_ids value must contain only IDs from the supplied spans. Keep the claim list small (no more than three). If a supplied claim is not supported by the text, still represent it so the audit stage can reject or refine it."""


def audit_prompt(
    spans: list[dict[str, str]],
    target_claim: str | None,
    audit_focus: str | None,
    generated: dict[str, Any],
) -> str:
    target = target_claim or "No single target claim; audit the extracted claims below."
    focus = audit_focus or "No special audit focus was supplied."
    focus_json = json.dumps(audit_focus or "", ensure_ascii=False)
    return f"""You are the grounding and refinement stage of a reasoning audit.
Use only the source spans below and the generated candidate. Do not browse, add outside facts, or turn general knowledge into a premise.

Source spans:
{_evidence_text(spans)}

Target claim:
{target}

Audit focus:
{focus}

Initial generated candidate:
{json.dumps(generated, ensure_ascii=False)}

For every atomic claim, classify how the exact source spans support it. Use only these grounding_status values:
DIRECTLY_SUPPORTED, COMPOSITIONALLY_SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, UNGROUNDED, OVERSTATED, UNKNOWN.
Use only these acceptance_status values:
ACCEPTED, ACCEPTED_WITH_ASSUMPTION, REJECTED, NOT_EVALUATED.
Use only these bridge_assumption.status values when a bridge exists:
EXPLICIT, REASONABLE_BUT_UNSTATED, UNSUPPORTED.

Return JSON only with this shape:
{{
  "claims": [
    {{
      "claim_id": "claim-1",
      "text": "audited atomic claim",
      "evidence_ids": ["source-1"],
      "grounding_status": "DIRECTLY_SUPPORTED",
      "acceptance_status": "ACCEPTED",
      "rationale": "brief source-grounded reason",
      "overstatement": null,
      "contradiction": null,
      "assumptions": [],
      "evidence_subclaims": [],
      "intermediate_inference": null,
      "bridge_assumption": {{}}
      }}
  ],
  "refined_answer": "a cautious conclusion from accepted claims only",
  "refined_claims": [
    {{"claim_id": "refined-claim-1", "text": "weakened or clarified claim", "evidence_ids": ["source-1"], "grounding_status": "DIRECTLY_SUPPORTED", "acceptance_status": "ACCEPTED", "previous_claim_id": "claim-1"}}
  ],
  "refinement_summary": "what was weakened, rejected, or left uncertain",
  "refinement_metadata": {{"focus": {focus_json}}},
  "conclusion_status": {{"grounding_status": "UNKNOWN", "acceptance_status": "REJECTED"}}
}}
The bridge_assumption object may be {{}} when no bridge is needed. A rejected, contradicted, overstatement, ungrounded, or assumption-dependent claim must not be presented as an unconditional conclusion. Only fully accepted claims may support the final conclusion. Keep refined_claims small and make every previous_claim_id refer to a claim above."""


def _call(client: OpenAI, model: str, prompt: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    arguments: dict[str, Any] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    if model == "gpt-4.1":
        arguments["temperature"] = 0
    response = client.chat.completions.create(**arguments)
    raw = response.choices[0].message.content or ""
    usage = response.usage.model_dump() if response.usage else {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ModelOutputValidationError(f"model did not return valid JSON: {error.msg}", path="$") from error
    return parsed, raw, usage


def _normalize_safe(audited: dict[str, Any], valid_ids: set[str]) -> dict[str, Any]:
    normalize_audit(audited, valid_ids)
    for claim in audited.get("claims", []) + audited.get("refined_claims", []):
        if claim.get("grounding_status") not in {"DIRECTLY_SUPPORTED", "COMPOSITIONALLY_SUPPORTED"}:
            claim["acceptance_status"] = "REJECTED"
        if not claim.get("evidence_ids"):
            claim["acceptance_status"] = "REJECTED"
    accepted = [
        claim for claim in (audited.get("refined_claims") or audited.get("claims", []))
        if claim.get("acceptance_status") == "ACCEPTED"
        and claim.get("grounding_status") in {"DIRECTLY_SUPPORTED", "COMPOSITIONALLY_SUPPORTED"}
    ]
    intermediate_statuses = [
        claim.get("intermediate_acceptance_status")
        for claim in audited.get("claims", [])
        if claim.get("intermediate_inference")
    ]
    if not accepted:
        conclusion_acceptance = "REJECTED"
    elif "REJECTED" in intermediate_statuses:
        conclusion_acceptance = "REJECTED"
    elif "ACCEPTED_WITH_ASSUMPTION" in intermediate_statuses:
        conclusion_acceptance = "ACCEPTED_WITH_ASSUMPTION"
    else:
        conclusion_acceptance = "ACCEPTED"
    audited["conclusion_status"] = {
        "grounding_status": (
            "COMPOSITIONALLY_SUPPORTED" if len(accepted) > 1
            else accepted[0]["grounding_status"] if accepted
            else "UNKNOWN"
        ),
        "acceptance_status": conclusion_acceptance,
        "verifier_status": "NOT_APPLICABLE",
    }
    return audited


def build_graph(spans: list[dict[str, str]], target_claim: str | None, generated: dict[str, Any], audited: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    question_label = target_claim or "Main audit-worthy claims in supplied source text"
    nodes: list[dict[str, Any]] = [{
        "id": "question",
        "type": "QUESTION",
        "label": question_label,
        "grounding_status": "NOT_APPLICABLE",
        "acceptance_status": "NOT_EVALUATED",
        "verifier_status": "NOT_APPLICABLE",
    }]
    edges: list[dict[str, Any]] = []
    edge_keys: set[tuple[str, str, str]] = set()

    def add_edge(source: str, target: str, relation: str) -> None:
        key = (source, target, relation)
        if key not in edge_keys:
            edge_keys.add(key)
            edges.append({"source": source, "target": target, "type": relation})

    for span in spans:
        nodes.append({
            "id": span["id"], "type": "EVIDENCE_SNIPPET", "label": span["text"],
            "grounding_status": "SOURCE", "acceptance_status": "ACCEPTED",
            "verifier_status": "NOT_APPLICABLE",
        })

    def add_claim(claim: dict[str, Any], node_type: str, node_id: str) -> None:
        nodes.append({
            "id": node_id, "type": node_type, "label": claim["text"],
            "grounding_status": claim["grounding_status"],
            "acceptance_status": claim["acceptance_status"],
            "verifier_status": "NOT_APPLICABLE",
            "rationale": claim.get("rationale"),
            "overstatement": claim.get("overstatement"),
            "contradiction": claim.get("contradiction"),
        })
        for evidence_id in claim.get("evidence_ids", []):
            add_edge(evidence_id, node_id, "grounded_in")
        bridge = claim.get("bridge_assumption") or {}
        assumptions = []
        if bridge.get("text"):
            assumptions.append({"text": bridge["text"], "status": bridge.get("status")})
        claim_assumptions = claim.get("assumptions", [])
        if isinstance(claim_assumptions, list):
            assumptions.extend({"text": text, "status": "UNSUPPORTED"} for text in claim_assumptions if text)
        seen_assumptions: set[str] = set()
        for assumption_index, assumption in enumerate(assumptions, 1):
            if assumption["text"] in seen_assumptions:
                continue
            seen_assumptions.add(assumption["text"])
            assumption_id = f"assumption-{node_id}-{assumption_index}"
            nodes.append({
                "id": assumption_id, "type": "ASSUMPTION", "label": assumption["text"],
                "assumption_status": assumption.get("status"),
                "grounding_status": "UNGROUNDED" if assumption.get("status") == "UNSUPPORTED" else "UNKNOWN",
                "acceptance_status": claim.get("intermediate_acceptance_status", claim["acceptance_status"]),
                "verifier_status": "NOT_APPLICABLE",
            })
            add_edge(node_id, assumption_id, "depends_on")

    for claim in audited.get("claims", []):
        add_claim(claim, "ANSWER_CLAIM", claim["claim_id"])
    for claim in audited.get("refined_claims", []):
        node_id = f"refined-{claim['claim_id']}"
        add_claim(claim, "REFINED_CLAIM", node_id)
        previous = claim.get("previous_claim_id")
        if previous:
            add_edge(previous, node_id, "revised_to")

    conclusion = audited["conclusion_status"]
    nodes.append({
        "id": "conclusion", "type": "CONCLUSION", "label": audited.get("refined_answer") or "No supported conclusion",
        "grounding_status": conclusion["grounding_status"],
        "acceptance_status": conclusion["acceptance_status"],
        "verifier_status": "NOT_APPLICABLE",
    })
    effective = audited.get("refined_claims") or audited.get("claims", [])
    for claim in effective:
        if claim.get("acceptance_status") == "ACCEPTED":
            source = f"refined-{claim['claim_id']}" if audited.get("refined_claims") else claim["claim_id"]
            add_edge(source, "conclusion", "supports")

    node_ids = {node["id"] for node in nodes}
    if any(edge["source"] not in node_ids or edge["target"] not in node_ids for edge in edges):
        raise RuntimeError("graph construction produced an invalid edge endpoint")
    return {"nodes": nodes, "edges": edges}


def _usage_total(usages: list[dict[str, Any]]) -> dict[str, int]:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {key: sum(int(usage.get(key, 0) or 0) for usage in usages) for key in keys}


def run_audit(request: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    spans = split_source_text(request["source_text"])
    valid_ids = {span["id"] for span in spans}
    client = OpenAI()
    generated, raw_generation, generation_usage = _call(
        client, request["model"], generation_prompt(spans, request["target_claim"], request["audit_focus"])
    )
    validate_generation_output(generated, valid_ids)
    if len(generated["atomic_claims"]) > MAX_CLAIMS:
        raise ModelOutputValidationError(f"must contain no more than {MAX_CLAIMS} claims", path="$.atomic_claims")

    audited, raw_audit, audit_usage = _call(
        client,
        request["model"],
        audit_prompt(spans, request["target_claim"], request["audit_focus"], generated),
    )
    validate_raw_audit_output(audited)
    audited = _normalize_safe(audited, valid_ids)
    validate_audit_output(audited, valid_ids)
    graph = build_graph(spans, request["target_claim"], generated, audited)
    runtime = time.perf_counter() - started
    usages = [generation_usage, audit_usage]
    run = {
        "api_calls": 2,
        "token_usage": _usage_total(usages),
        "runtime_seconds": runtime,
    }
    artifact = {
        "schema_version": 1,
        "run_id": f"natural-language-{uuid.uuid4().hex[:12]}",
        "model": request["model"],
        "input": {
            "source_text": request["source_text"],
            "target_claim": request["target_claim"],
            "audit_focus": request["audit_focus"],
            "source_spans": spans,
        },
        "generated": generated,
        "grounding": audited["claims"],
        "refinement": {
            "summary": audited.get("refinement_summary"),
            "metadata": audited.get("refinement_metadata"),
            "refined_answer": audited.get("refined_answer"),
            "refined_claims": audited.get("refined_claims", []),
        },
        "conclusion_status": audited["conclusion_status"],
        "final_accepted_answer": audited.get("refined_answer"),
        "verifier_status": "NOT_APPLICABLE",
        "formal_status": "NOT_APPLICABLE",
        "proof_status": "NOT_APPLICABLE",
        "system_status": "OK",
        "api_calls": run["api_calls"],
        "token_usage": run["token_usage"],
        "runtime_seconds": runtime,
        "raw_model_responses": {"generation": raw_generation, "audit": raw_audit},
    }
    return {"artifact": artifact, "graph": graph, "run": run}


class AuditHandler(SimpleHTTPRequestHandler):
    """Static demo handler plus JSON API; request bodies are never logged."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(DEMO), **kwargs)

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/api/audit":
            self._send_json(404, {"error": {"code": "NOT_FOUND", "message": "Use POST /api/audit."}})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": {"code": "INVALID_REQUEST", "message": "Content-Length must be an integer."}})
            return
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": {"code": "REQUEST_TOO_LARGE", "message": "Request body is empty or too large."}})
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
            request = validate_request(payload)
            result = run_audit(request)
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"code": "INVALID_JSON", "message": "Request body must contain valid JSON."}})
            return
        except RequestValidationError as error:
            self._send_json(400, {"error": error.as_dict()})
            return
        except ModelOutputValidationError as error:
            self._send_json(502, {"error": {"code": error.code, "message": str(error), "path": error.path}})
            return
        except Exception:
            # Keep upstream and credential details out of responses and logs.
            self._send_json(502, {"error": {"code": "AUDIT_BACKEND_ERROR", "message": "The audit backend could not complete the model request."}})
            return
        self._send_json(200, result)

    def log_message(self, format: str, *args: Any) -> None:
        # Method/status only; never include request bodies or exception details.
        sys.stderr.write(f"STRING demo {self.command} {self.path} {args[1] if len(args) > 1 else ''}\n")


def main() -> int:
    host = os.environ.get("STRING_DEMO_HOST", "127.0.0.1")
    port = int(os.environ.get("STRING_DEMO_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), AuditHandler)
    print(f"STRING natural-language demo: http://{host}:{port}/natural-language.html")
    print("API endpoint: POST /api/audit")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
