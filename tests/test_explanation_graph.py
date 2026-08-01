from string_agent.reasoning import (
    StepStatus,
    apply_isabelle_status,
    initial_reasoning_graph,
    revise_reasoning_graph,
)


def test_graph_comes_from_llm_steps_and_source_grounding():
    graph = initial_reasoning_graph(
        "1. First inference.\n2. Second inference.",
        (
            ("First inference.", ("Source one.",)),
            ("Second inference.", ("Source one.", "Source two.")),
        ),
    )

    assert [step.text for step in graph.steps] == [
        "First inference.",
        "Second inference.",
    ]
    assert len(graph.source_groundings) == 2
    assert graph.steps[0].source_grounding_ids == ("source-1",)


def test_revision_keeps_exact_ids_and_links_modified_steps():
    graph = initial_reasoning_graph("1. Keep this.\n2. Change this.")
    refined, marked = revise_reasoning_graph(
        graph, "1. Keep this.\n2. Changed sentence.\n3. Added sentence."
    )

    assert refined.steps[0].stable_id == "step-1"
    assert refined.steps[0].previous_revision_id == "step-1-r0"
    assert refined.steps[1].stable_id == "step-2"
    assert refined.steps[1].previous_revision_id == "step-2-r0"
    assert refined.steps[2].stable_id == "step-3"
    assert marked.steps[1].status is StepStatus.REFINED


def test_only_referenced_axioms_are_marked_verified():
    graph = initial_reasoning_graph("1. Used.\n2. Unused.")
    verified = apply_isabelle_status(
        graph,
        syntax_valid=True,
        logical_validity=True,
        used_explanation_indexes=frozenset({1}),
    )

    assert verified.steps[0].isabelle_status is StepStatus.VERIFIED
    assert verified.steps[1].isabelle_status is StepStatus.UNVERIFIED
