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

1. Read the relevant runner, adapter, tests, and existing artifact schema.
2. Inspect real input data and upstream implementation rather than guessing its
   schema or behavior.
3. Preserve existing outputs; use a new run ID or output directory unless the
   user explicitly requests replacement.
4. Make the smallest scoped change that preserves upstream behavior.
5. Add or update regression tests before relying on a live run.
6. Run mock/unit tests, then a narrowly scoped live smoke when requested.
7. Validate artifact, graph, status consistency, timing records, cleanup, and
   absence of credential leakage before broader execution.
8. Expand to a batch only after its smoke criteria pass and only when requested.

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

## Testing and completion criteria

- Add regression coverage for every fixed failure mode and retain existing
  behavior for other models and datasets.
- Tests must cover parsing/normalization, grounding gates, status transitions,
  graph/artifact consistency, refinement separation, and error preservation as
  applicable.
- For live OpenAI/Isabelle work, first prove the smallest smoke path. Confirm API
  call accounting, timings, generated files, process cleanup, and no lingering
  Isabelle server before considering it complete.
- A task is complete only when requested outputs exist, can be parsed/rendered,
  required assertions pass, tests pass, and failures are reported rather than
  hidden. Do not claim a live run occurred when only mocks were used.

## Final report format

Lead with the outcome. Report only requested fields, normally:

- semantic result and the separate grounding/formal/acceptance/system statuses;
- iteration or claim-level results needed to explain that outcome;
- API calls, token usage when available, and runtime for live runs;
- timeout/error IDs and cleanup state for batches;
- pytest command result;
- clickable paths to changed files and generated artifacts.

Never include credentials, raw secret-bearing configuration, screenshots unless
requested, or unrelated implementation narration.
