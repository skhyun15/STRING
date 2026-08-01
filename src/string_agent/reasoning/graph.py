"""Reasoning graphs built from LLM explanation revisions.

This module deliberately knows nothing about ProofWriter grammar or proof search.
It only represents explanation sentences, their source grounding, and revision
relationships. Isabelle judgments are attached later by the upstream adapter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from enum import Enum


class StepStatus(Enum):
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    REFINED = "REFINED"
    REMOVED = "REMOVED"
    ERROR = "ERROR"


class RevisionRelation(Enum):
    INITIAL = "INITIAL"
    UNCHANGED = "UNCHANGED"
    MODIFIED = "MODIFIED"
    ADDED = "ADDED"


@dataclass(frozen=True)
class SourceGrounding:
    id: str
    text: str


@dataclass(frozen=True)
class ExplanationStep:
    stable_id: str
    revision_id: str
    revision: int
    text: str
    source_grounding_ids: tuple[str, ...]
    status: StepStatus
    isabelle_status: StepStatus
    revision_relation: RevisionRelation
    previous_revision_id: str | None = None


@dataclass(frozen=True)
class ReasoningGraph:
    revision: int
    explanation: str
    steps: tuple[ExplanationStep, ...]
    source_groundings: tuple[SourceGrounding, ...]

    def __post_init__(self) -> None:
        revision_ids = {step.revision_id for step in self.steps}
        if len(revision_ids) != len(self.steps):
            raise ValueError("explanation step revision ids must be unique")
        grounding_ids = {grounding.id for grounding in self.source_groundings}
        if any(
            grounding_id not in grounding_ids
            for step in self.steps
            for grounding_id in step.source_grounding_ids
        ):
            raise ValueError("explanation step references an unknown grounding")


_NUMBER_PREFIX = re.compile(r"^\s*\d+[.)]\s*")


def explanation_lines(explanation: str) -> tuple[str, ...]:
    """Extract display steps without interpreting their logical content."""

    return tuple(
        cleaned
        for line in explanation.splitlines()
        if (cleaned := _NUMBER_PREFIX.sub("", line).strip())
    )


def initial_reasoning_graph(
    explanation: str,
    step_specs: tuple[tuple[str, tuple[str, ...]], ...] | None = None,
) -> ReasoningGraph:
    """Create revision zero from LLM output or a user-supplied explanation."""

    if step_specs is None:
        step_specs = tuple((line, ()) for line in explanation_lines(explanation))
    if not step_specs:
        raise ValueError("an explanation must contain at least one step")

    grounding_ids: dict[str, str] = {}
    groundings: list[SourceGrounding] = []
    steps: list[ExplanationStep] = []
    for index, (text, sources) in enumerate(step_specs, start=1):
        source_ids: list[str] = []
        for source in sources:
            source_text = source.strip()
            if not source_text:
                continue
            grounding_id = grounding_ids.get(source_text)
            if grounding_id is None:
                grounding_id = f"source-{len(groundings) + 1}"
                grounding_ids[source_text] = grounding_id
                groundings.append(SourceGrounding(grounding_id, source_text))
            source_ids.append(grounding_id)
        stable_id = f"step-{index}"
        steps.append(
            ExplanationStep(
                stable_id=stable_id,
                revision_id=f"{stable_id}-r0",
                revision=0,
                text=text,
                source_grounding_ids=tuple(dict.fromkeys(source_ids)),
                status=StepStatus.UNVERIFIED,
                isabelle_status=StepStatus.UNVERIFIED,
                revision_relation=RevisionRelation.INITIAL,
            )
        )
    return ReasoningGraph(0, explanation, tuple(steps), tuple(groundings))


def revise_reasoning_graph(
    previous: ReasoningGraph, refined_explanation: str
) -> tuple[ReasoningGraph, ReasoningGraph]:
    """Link a textual refinement to the prior graph without semantic guessing.

    Exact equal lines retain their stable id. Replacements are linked by order
    within the diff block; additions get a fresh id. Removed/replaced prior
    steps are marked in the returned prior graph so the UI can show the diff.
    """

    old_steps = list(previous.steps)
    new_texts = list(explanation_lines(refined_explanation))
    if not new_texts:
        raise ValueError("refinement returned no explanation steps")
    matcher = SequenceMatcher(
        a=[step.text for step in old_steps], b=new_texts, autojunk=False
    )
    next_revision = previous.revision + 1
    next_stable_number = max(
        (int(step.stable_id.rsplit("-", 1)[1]) for step in old_steps),
        default=0,
    ) + 1
    new_steps: list[ExplanationStep] = []
    prior_updates: dict[int, StepStatus] = {}

    def add_step(
        text: str,
        relation: RevisionRelation,
        prior: ExplanationStep | None,
    ) -> None:
        nonlocal next_stable_number
        if prior is None:
            stable_id = f"step-{next_stable_number}"
            next_stable_number += 1
            grounding_ids: tuple[str, ...] = ()
            previous_revision_id = None
        else:
            stable_id = prior.stable_id
            grounding_ids = (
                prior.source_grounding_ids
                if relation is RevisionRelation.UNCHANGED
                else ()
            )
            previous_revision_id = prior.revision_id
        new_steps.append(
            ExplanationStep(
                stable_id=stable_id,
                revision_id=f"{stable_id}-r{next_revision}",
                revision=next_revision,
                text=text,
                source_grounding_ids=grounding_ids,
                status=StepStatus.REFINED,
                isabelle_status=StepStatus.UNVERIFIED,
                revision_relation=relation,
                previous_revision_id=previous_revision_id,
            )
        )

    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if tag == "equal":
            for offset, text in enumerate(new_texts[new_start:new_end]):
                add_step(text, RevisionRelation.UNCHANGED, old_steps[old_start + offset])
            continue
        if tag == "delete":
            for old_index in range(old_start, old_end):
                prior_updates[old_index] = StepStatus.REMOVED
            continue
        if tag == "insert":
            for text in new_texts[new_start:new_end]:
                add_step(text, RevisionRelation.ADDED, None)
            continue

        old_block = old_steps[old_start:old_end]
        new_block = new_texts[new_start:new_end]
        paired = min(len(old_block), len(new_block))
        for offset in range(paired):
            prior_updates[old_start + offset] = StepStatus.REFINED
            add_step(new_block[offset], RevisionRelation.MODIFIED, old_block[offset])
        for old_index in range(old_start + paired, old_end):
            prior_updates[old_index] = StepStatus.REMOVED
        for text in new_block[paired:]:
            add_step(text, RevisionRelation.ADDED, None)

    marked_previous = replace(
        previous,
        steps=tuple(
            replace(step, status=prior_updates.get(index, step.status))
            for index, step in enumerate(old_steps)
        ),
    )
    return (
        ReasoningGraph(
            next_revision,
            refined_explanation,
            tuple(new_steps),
            previous.source_groundings,
        ),
        marked_previous,
    )


def apply_isabelle_status(
    graph: ReasoningGraph,
    *,
    syntax_valid: bool | None,
    logical_validity: bool | None,
    used_explanation_indexes: frozenset[int] = frozenset(),
    failed_explanation_indexes: frozenset[int] = frozenset(),
    error: bool = False,
) -> ReasoningGraph:
    """Attach only judgments directly supported by this Isabelle attempt."""

    updated: list[ExplanationStep] = []
    for index, step in enumerate(graph.steps, start=1):
        if error:
            isabelle_status = StepStatus.ERROR
        elif syntax_valid is False:
            isabelle_status = StepStatus.SYNTAX_ERROR
        elif logical_validity is True and index in used_explanation_indexes:
            isabelle_status = StepStatus.VERIFIED
        elif logical_validity is False and index in failed_explanation_indexes:
            isabelle_status = StepStatus.REJECTED
        else:
            isabelle_status = StepStatus.UNVERIFIED
        status = (
            isabelle_status
            if step.status in {StepStatus.UNVERIFIED, StepStatus.REFINED}
            and isabelle_status is not StepStatus.UNVERIFIED
            else step.status
        )
        updated.append(
            replace(step, status=status, isabelle_status=isabelle_status)
        )
    return replace(graph, steps=tuple(updated))
