# STRING Repository Instructions

These instructions apply to the entire repository. Keep them focused on durable
project behavior; do not add problem IDs, transient run results, local secret
locations, or one-off debugging notes.

## Project purpose

STRING audits natural-language reasoning by preserving the path from source
evidence through explanations and formal verification. Its outputs must make it
possible to distinguish semantic support, source grounding, formal proof
results, and runtime failures. The system is an audit pipeline, not an authority
that can manufacture missing premises or domain facts.

## Runtime architecture

The default symbolic path is:

1. Load source, hypothesis, and optional initial explanation.
2. Generate an initial explanation when none is supplied.
3. Build a reasoning graph from logical explanation steps.
4. Run the upstream `AutoFormalisationModel`.
5. syntax-check the generated theory with Isabelle.
6. Run `RoughInferenceModel` and generate an Isabelle proof when syntax permits.
7. Ask `IsabelleSolver` for the proof result.
8. If verification fails and refinement budget remains, refine the explanation
   and re-run the complete formalisation and verification path.
9. Persist iteration, timing, debug, graph, cleanup, and final-status data.

`max_iterations` is a refinement budget. The initial verification is additional,
so every generated refinement must receive a subsequent verification attempt.
`final_validity` comes only from the latest iteration actually checked by
Isabelle, never from unverified refinement prose.

See [docs/architecture.md](docs/architecture.md) for component boundaries.

## Component responsibilities

- OpenAI models generate explanations, formalisation candidates, rough
  inference, proofs, refinements, and structured evidence audits. Model output
  is untrusted data and must be validated before acceptance.
- The upstream Explanation-Refiner owns its prompts and LLM/Isabelle reasoning
  implementation. STRING adapts, instruments, checkpoints, and presents it;
  do not silently replace its logical pipeline or rewrite its source prompts.
- Isabelle checks formal syntax and derivability of formalised dependencies. It
  does not establish that an LLM formalisation faithfully represents the source,
  and it must not be used to decide biomedical truth.

## Dataset roles

- ProofWriter is a controlled symbolic reasoning evaluation, not categorically a
  closed-world task. Respect each example's semantics. Under open-world
  assumption (OWA), failure to prove a query does not make it false and does not
  prove its complement; always verify the query and complement independently.
- BioASQ is a claim-evidence audit. Use only the supplied snippets during answer
  generation and refinement. Gold ideal/exact answers are post-hoc evaluation
  data and must never enter generation prompts. Isabelle is optional and limited
  to formal dependency structure; biomedical interpretation remains an evidence
  grounding judgment.

## Required work order

This is a time-boxed hackathon project. Prioritize a working, demonstrable
end-to-end path over exhaustive engineering work.

1. Read only the files directly relevant to the requested change.
2. Inspect one real input or existing artifact when it is needed to understand
   the current schema or behavior.
3. Make the smallest change that produces the requested visible or functional
   result.
4. Reuse the current architecture and existing outputs rather than redesigning
   unrelated components.
5. Validate with the cheapest meaningful check:
   - UI-only changes: open the generated page and verify the affected examples;
   - runner or adapter changes: run one narrow smoke example;
   - pure formatting or copy changes: inspect the resulting file;
   - high-risk status, grounding, or artifact changes: run the relevant focused
     tests.
6. Do not add tests by default. Add or update a focused regression test only
   when changing status semantics, grounding gates, artifact/graph consistency,
   parsing of model output, or a previously reproduced failure mode.
7. Do not run the full test suite unless explicitly requested or the change
   affects shared core behavior.
8. Stop after the requested result works. Do not expand the task into cleanup,
   refactoring, benchmarking, or unrelated diagnosis.

## Prohibited behavior

- Never print, copy into artifacts, commit, or hard-code credentials.
- Never generate a new credential when an existing configured credential is to
  be reused.
- Do not modify upstream prompts or sources unless explicitly authorized.
- Do not hard-code answers, model-specific expected logic, or benchmark cases.
- Do not use deterministic fallback reasoning where a live model run is required.
- Do not introduce facts, rules, bridge premises, or biomedical knowledge absent
  from the provided source evidence.
- Do not treat a successful Isabelle proof as sufficient when grounding failed.
- Do not collapse timeout, integration, formalisation, or adapter failures into a
  semantic rejection or `UNKNOWN` verdict.
- Do not include critique, strategy, headings, or refinement instructions as
  logical graph steps or create support edges from rejected text.
- Do not overwrite prior artifacts or start a full benchmark unless requested.
- Do not perform external search for evidence-restricted dataset tasks.
- Do not add tests, run the full test suite, or perform broad repository analysis
  merely to demonstrate diligence.
- Do not inspect unrelated files when the requested change can be completed from
  the directly relevant renderer, runner, or artifact.
- Do not turn a small UI or demo task into an architecture, schema, or cleanup
  project.

## Status semantics

Keep these dimensions separate wherever they apply:

- `grounding_status`: relationship between a claim and source evidence.
- `formal_status`: formalisation/proof outcome, including direct-fact handling.
- `proof_status`: Isabelle result when independently represented.
- `acceptance_status`: whether grounded evidence and formal conditions permit the
  candidate to contribute to a verdict.
- `system_status`: operational health, timeout, integration, or adapter failure.
- `semantic_verdict`: conclusion from accepted candidates only.

`UNKNOWN` is a semantic outcome only when required candidates completed normally
and neither was accepted. Operational or adapter failures produce
`semantic_verdict=UNRESOLVED` with a specific `system_status`. See
[docs/status-semantics.md](docs/status-semantics.md).

## Artifact and graph principles

- Artifacts are append-oriented audit records: preserve raw model responses,
  postprocessed responses, per-iteration inputs/outputs, token/API accounting,
  timing, exact errors, cleanup results, and source identifiers without secrets.
- Write checkpoints at iteration/candidate boundaries and retain partial
  artifacts on error or timeout. Resume must skip completed work.
- Graphs must be derived from the same accepted artifact state and agree with its
  grounding, formal, proof, acceptance, system, and conclusion statuses.
- Represent source facts/rules or evidence snippets individually. Preserve source
  IDs and typed edges. Multi-evidence claims must expose every evidence link,
  intermediate inference, and any bridge assumption.
- Store refinement metadata separately from logical claims. Only accepted logical
  content may support a conclusion.

See [docs/artifact-schema.md](docs/artifact-schema.md) for the durable schema
contract.

## Validation and completion criteria

Use risk-based validation rather than test-first development.

- Do not create unit tests for routine UI, styling, copy, layout, CLI wording,
  file-path, or one-off demo changes unless the user explicitly requests them.
- For frontend changes, successful rendering of the target BioASQ examples is
  the primary validation. Check that the page loads, text is readable, controls
  work, and no visible JavaScript error prevents rendering.
- For live pipeline changes, prefer one narrowly scoped smoke run over mocks or
  broad test suites.
- Run focused tests only when changing:
  - grounding or acceptance semantics;
  - formal, proof, system, or semantic status transitions;
  - model-output parsing and normalization;
  - artifact/graph consistency;
  - timeout, cleanup, resume, or error preservation.
- Run the full pytest suite only when explicitly requested, before a deliberate
  release checkpoint, or when shared core behavior has changed substantially.
- Do not spend time increasing coverage, refactoring tests, or testing unrelated
  behavior during a demo-critical task.
- A task is complete when the requested behavior is visibly or operationally
  confirmed and any important limitation is reported honestly.
- Never claim a live run, render, or external integration succeeded unless it
  was actually executed.

## Final report format

Keep the final report short and specific to the request.

Normally report only:

- whether the requested result works;
- changed files;
- the exact command or path used to verify it;
- any remaining blocker or visible limitation.

Include semantic statuses, API calls, token usage, runtime, cleanup, or test
results only when they are relevant to the requested task or were actually run.

Do not include unrelated implementation narration, exhaustive file inventories,
credentials, or raw secret-bearing configuration.

Never include credentials, raw secret-bearing configuration, screenshots unless
requested, or unrelated implementation narration.
