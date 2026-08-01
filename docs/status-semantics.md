# Status Semantics

## Independent dimensions

Statuses answer different questions and must not be substituted for one another.

| Dimension | Question answered | Representative values |
| --- | --- | --- |
| `grounding_status` | Is the content supported by supplied sources? | `GROUNDED`, `GROUNDED_DIRECT_FACT`, `DIRECTLY_SUPPORTED`, `COMPOSITIONALLY_SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNGROUNDED`, `OVERSTATED`, `CONTRADICTED`, `AMBIGUOUS`, `UNKNOWN` |
| `formal_status` | What happened in formalisation/proof? | `DIRECT_FACT`, `PROVED`, `NOT_PROVED`, `SYNTAX_ERROR`, `NOT_COMPLETED` |
| `proof_status` | What did the proof checker report? | `PROVED`, `REJECTED`, `ERROR`, `NOT_APPLICABLE` |
| `acceptance_status` | May this candidate contribute to the conclusion? | `ACCEPTED`, `ACCEPTED_WITH_ASSUMPTION`, `REJECTED`, `NOT_EVALUATED` |
| `system_status` | Did the pipeline operate normally? | `OK`, `TIMEOUT`, `ERROR`, `FORMALISATION_ADAPTER_ERROR`, integration-specific errors |
| `semantic_verdict` | What follows from accepted candidates? | `SUPPORTED`, `CONTRADICTED`, `INCONSISTENT`, `UNKNOWN`, `UNRESOLVED` |

Dataset-specific grounding values may extend the representative list, but their
meaning must remain explicit in artifacts and graphs.

## ProofWriter verdict truth table

Compute verdicts from accepted grounded candidates, never raw Isabelle booleans:

| Query accepted | Complement accepted | Semantic verdict |
| --- | --- | --- |
| yes | no | `SUPPORTED` |
| no | yes | `CONTRADICTED` |
| yes | yes | `INCONSISTENT` |
| no | no, both completed normally | `UNKNOWN` |
| any required candidate timed out/errored | any | `UNRESOLVED` |

An ungrounded or circular explanation remains `REJECTED` even if Isabelle proves
its generated theorem. An errored candidate is `NOT_EVALUATED`, not a rejected
proof. Preserve the specific failure in `system_status` and error details.

## Explanation-refinement run status

- `VALID`: Isabelle verified the applicable final checked iteration.
- `REJECTED`: verification completed and the candidate was rejected without a
  pending unverified refinement.
- `MAX_ITERATIONS`: the refinement budget was exhausted after the final permitted
  checked candidate remained unverified.
- `ERROR`: an operational, adapter, model, parsing, or integration failure stopped
  normal completion.

`final_validity` is nullable. It records only the latest actual Isabelle proof
check: `true` for proved, `false` for checked and not proved, and `null` when no
proof result exists. Generating refinement text cannot change it.

## BioASQ acceptance

Grounding and acceptance are related but distinct. A directly supported atomic
claim is `ACCEPTED` on its own evidence and is not downgraded merely because a
later synthesis needs a bridge. A `REASONABLE_BUT_UNSTATED` bridge gives the
dependent intermediate inference and conclusion `ACCEPTED_WITH_ASSUMPTION`.
Unsupported, contradicted, materially overstated, or evidence-free claims are
rejected or remain unresolved according to the audit policy. `verifier_status`
is normally `NOT_APPLICABLE` unless only explicit logical dependency structure
is checked.
