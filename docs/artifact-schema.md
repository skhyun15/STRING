# Artifact and Graph Contract

This document defines durable expectations rather than a single frozen JSON
schema. Dataset adapters may add fields, but must preserve the audit dimensions
below and increment `schema_version` for incompatible changes.

## Run artifact

Every run artifact should preserve:

- schema version, stable run/problem identifier, model, and relevant budgets;
- original source input and hypothesis/question;
- initial candidate and every checked iteration or audited claim;
- raw generated content and normalized/postprocessed content where applicable;
- grounding, formal, proof, acceptance, system, and semantic statuses as separate
  fields;
- exact error category/message and first failing formal line when available;
- API-call and token accounting, timestamps, per-stage elapsed time, and total
  runtime;
- output/theory references and solver cleanup result;
- final result derived only from checked and accepted state.

Model-response debug logs may contain prompts and raw completions but must never
contain API keys, secret configuration, authorization headers, or unrelated
environment data. Gold evaluation answers must be stored separately from
generation inputs and marked post-hoc.

Write partial artifacts at safe boundaries. A timeout or exception must retain
the latest completed phase instead of replacing it with an empty generic error.
Resume logic must recognize completed candidates/problems and avoid duplicate API
or Isabelle work.

## Iteration records

A symbolic verification iteration should include its input explanation and graph,
formalisation, syntax result/feedback, rough inference, generated proof, logical
validity, proof feedback, optional refinement output, API counts, theory path,
timings, and cleanup/error state. A refinement belongs to the producing iteration
as output and to the next iteration as verification input.

## Claim-evidence records

Each audited claim should include a stable claim ID, text, evidence IDs, grounding
status, relation strength, rationale, acceptance status, and verifier status.
Compositional claims additionally preserve evidence-specific subclaims,
intermediate inference, bridge text, whether the bridge is explicit, and bridge
status. Atomic-claim acceptance, intermediate-inference acceptance, and
conclusion acceptance are recorded separately so an unstated bridge does not
downgrade directly supported evidence. Before/after claim IDs and `revised_to`
relations must make refinement traceable.

## Graph contract

`graph.json` contains typed `nodes` and `edges` with stable IDs. Node status fields
must match the artifact record from which the node was derived. The conclusion
must agree with the artifact's semantic/final result.

ProofWriter graphs split the source into individual `FACT` and `RULE` nodes and
connect only grounded, accepted logical content. BioASQ graphs use evidence,
claim, inference, assumption, refined-claim, and conclusion nodes as applicable.
Every multi-evidence claim must have links to all cited evidence nodes.

Refinement instructions, strategy prose, critiques, headings, and rejected text
are metadata, not logical nodes. They cannot receive or emit a `supports` edge.
Assumptions must remain visible; unsupported bridges cannot be hidden inside an
intermediate inference or conclusion.

## Presentation

Per-problem HTML reads `artifact.json` and `graph.json` rather than duplicating
their state. It should foreground the reasoning/evidence graph and revision diff,
while verbose formalisation, proof, and feedback may be collapsed. Batch indexes
summarize generated per-problem artifacts without becoming a second source of
truth.

## Validation checklist

- JSON files parse and expected HTML files reference them.
- Artifact and graph statuses and conclusions agree.
- Source/evidence IDs resolve and typed node/edge counts are plausible.
- Refinement metadata is separated from logical claims.
- Partial/error artifacts retain the last completed stage.
- API calls, tokens, timings, and runtime are internally consistent where present.
- No credential or secret configuration appears in outputs.
