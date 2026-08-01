# STRING Architecture

## Boundaries

STRING is the orchestration and audit layer around model-generated reasoning and,
where appropriate, Isabelle verification. Repository code owns input adapters,
runtime isolation, instrumentation, checkpoints, status aggregation, artifacts,
graphs, and presentation. The upstream Explanation-Refiner checkout owns its
prompt-driven formalisation/refinement components and Isabelle integration.

OpenAI output is a candidate, not a verdict. Isabelle establishes only whether a
generated formal theory is syntactically acceptable and whether its stated goal
follows in that theory. Source grounding establishes whether that theory or claim
faithfully uses the supplied evidence. Acceptance requires the applicable gates,
not merely a model assertion or raw proof success.

## Symbolic explanation-refinement flow

The initial explanation is verified once. Each refinement consumes one unit of
the refinement budget and is verified in a new iteration:

```text
source + hypothesis
  -> initial explanation
  -> reasoning graph
  -> AutoFormalisationModel
  -> Isabelle syntax check
  -> RoughInferenceModel
  -> Isabelle proof generation
  -> IsabelleSolver proof check
  -> [failure + budget] ExplanationRefineModel
  -> next complete verification iteration
```

The adapter captures raw and postprocessed model output, per-stage timings,
generated theory files, feedback, and cleanup state. Generic response cleanup may
remove Markdown fencing, but must not rewrite logical expressions, declarations,
theorems, or proofs.

## ProofWriter path

ProofWriter evaluates controlled symbolic inference over a supplied context.
Parse source facts and rules separately, including abbreviated universal-rule
forms. Run query and complement as independent candidates. Before expensive model
or Isabelle work, an exact normalized source fact may be accepted through the
direct-fact path.

ProofWriter is not uniformly closed-world. In OWA data, an unproved query is not
false and its complement is not thereby proved. Independent candidate results,
not negation by failure, determine the verdict.

For non-direct candidates, explanations and refinements must cite existing source
items. New facts, rules, circular restatements, and unsupported bridge premises
fail grounding. A proved but ungrounded candidate is rejected and cannot affect
the semantic verdict. Candidate and problem timeouts preserve partial results,
terminate owned processes, and allow later batch work to continue or resume.

## BioASQ path

BioASQ audits whether generated biomedical claims are supported by provided gold
snippets. Generation receives the question and snippet IDs/text only. Atomic
claims are connected to evidence and classified as direct, compositional,
partial, contradicted, ungrounded, overstated, or unknown.

Composition must expose evidence-specific subclaims, intermediate inference, and
bridge assumptions. A directly supported atomic claim remains `ACCEPTED`; an
unstated but reasonable bridge changes only the dependent intermediate inference
and conclusion to `ACCEPTED_WITH_ASSUMPTION`. Refinement may
remove unsupported claims, weaken strength, repair evidence links, state
uncertainty, split claims, or expose assumptions; it may not add biomedical facts.

Ideal and exact answers are loaded for post-hoc comparison only after generation
and refinement. Isabelle is `NOT_APPLICABLE` for biomedical truth. It may be used
only when explicit propositional dependency structure can be formalised without
inventing biomedical premises.

## Operational safeguards

Use unique run directories. Record stage start/end events immediately so a killed
process still leaves useful timing evidence. Keep raw model responses in a debug
area stripped of credential/configuration data. Preserve partial artifacts and
graphs on failures. Always shut down the Isabelle session/server owned by a run;
timeouts must not strand child processes or the configured server port.
