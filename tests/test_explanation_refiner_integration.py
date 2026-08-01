from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from string_agent.integration import (
    ExplanationRefiner,
    RunStatus,
    UpstreamBindings,
    artifact_to_dict,
)
from string_agent.integration import explanation_refiner as adapter


class FakeCompletions:
    def create(self, **kwargs):
        return kwargs


class FakeLlm:
    def __init__(self):
        self.client = type("Client", (), {})()
        self.client.chat = type("Chat", (), {})()
        self.client.chat.completions = FakeCompletions()
        self.prompt_model = type("Prompt", (), {"base_path": Path("prompts")})()

    def request(self, name):
        self.client.chat.completions.create(model=name)

    def generate(self, *, prompt_name, **kwargs):
        self.request(prompt_name)
        if prompt_name == "initial_explanation_prompt":
            return (
                '{"proposed_answer":"entailed","steps":['
                '{"text":"Tweety is a bird.","source_sentences":'
                '["Tweety is a bird."]},'
                '{"text":"Birds can fly.","source_sentences":'
                '["Birds can fly."]}]}'
            )
        raise AssertionError(f"unexpected direct generate call: {prompt_name}")


class FakeAutoFormalisationModel:
    events = []

    def __init__(self, llm):
        self.llm = llm
        self.events.append("AutoFormalisationModel.__init__")

    @staticmethod
    def _path(theory_name, data_name, model_name, iteration):
        return (
            Path("formalisation")
            / data_name
            / model_name
            / theory_name
            / f"{theory_name}_{iteration}.thy"
        )

    def formalise(
        self,
        theory_name,
        data_name,
        model_name,
        iteration,
        premise,
        explanation,
        hypothesis,
    ):
        self.events.append(("formalise", iteration, explanation))
        self.llm.request("formalise")
        code = (
            f"theory {theory_name}_{iteration}\nimports Main\nbegin\n"
            "axiomatization where explanation_1: \"True\"\n"
            "axiomatization where explanation_2: \"True\"\n"
            "theorem hypothesis: shows \"True\"\nproof -\n\nqed\nend\n"
        )
        path = self._path(theory_name, data_name, model_name, iteration)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        return code

    def get_isabelle_proof(
        self,
        rough_inference,
        isabelle_code,
        theory_name,
        data_name,
        model_name,
        iteration,
    ):
        self.events.append(("get_isabelle_proof", iteration, rough_inference))
        self.llm.request("proof")
        proof = isabelle_code.replace(
            "proof -\n\nqed", "proof -\n  using explanation_1 explanation_2 by blast\nqed"
        )
        self._path(theory_name, data_name, model_name, iteration).write_text(
            proof, encoding="utf-8"
        )
        unused = ["explanation_1"] if iteration == 0 else []
        return proof, unused


class FakeRoughInferenceModel:
    events = []

    def __init__(self, llm):
        self.llm = llm
        self.events.append("RoughInferenceModel.__init__")

    def get_rough_inference(self, premise, explanation, hypothesis):
        self.events.append(("rough", explanation))
        self.llm.request("rough")
        return f"strategy for: {explanation}"


class FakeExplanationRefineModel:
    events = []

    def __init__(self, llm):
        self.llm = llm
        self.events.append("ExplanationRefineModel.__init__")

    def refine(self, premise, explanation, hypothesis, isabelle_code, error_code):
        self.events.append(("refine", explanation, error_code))
        self.llm.request("refine")
        return "1. Tweety is a bird.\n2. Birds with healthy wings can fly."


class FakeIsabelleSolver:
    events = []
    outcomes = [(False, "using explanation_2 by blast", 0.02), (True, "", 0.01)]

    def __init__(self, **kwargs):
        self.events.append("IsabelleSolver.__init__")
        self.solve_index = 0

    def get_isabelle_syntax_output(self, code, theory, data, model, iteration):
        self.events.append(("syntax", iteration))
        return False

    def solve(self, theory, data, model, iteration, code, explanation):
        self.events.append(("solve", iteration, explanation))
        outcome = self.outcomes[self.solve_index]
        self.solve_index += 1
        return outcome

    def shutdown(self):
        self.events.append("shutdown")


def bindings(auto=FakeAutoFormalisationModel, solver=FakeIsabelleSolver):
    return UpstreamBindings(
        GPT=object,
        AutoFormalisationModel=auto,
        RoughInferenceModel=FakeRoughInferenceModel,
        ExplanationRefineModel=FakeExplanationRefineModel,
        IsabelleSolver=solver,
        filter_explanations=lambda refined, unused, previous: refined,
    )


@pytest.fixture(autouse=True)
def reset_fakes():
    for fake in (
        FakeAutoFormalisationModel,
        FakeRoughInferenceModel,
        FakeExplanationRefineModel,
        FakeIsabelleSolver,
    ):
        fake.events.clear()
    FakeIsabelleSolver.outcomes = [
        (False, "using explanation_2 by blast", 0.02),
        (True, "", 0.01),
    ]


def run_fake(tmp_path, **kwargs):
    refiner = ExplanationRefiner(
        max_iterations=kwargs.pop("max_iterations", 2),
        output_root=tmp_path,
        _bindings=kwargs.pop("_bindings", bindings()),
        _llm=kwargs.pop("_llm", FakeLlm()),
    )
    return refiner.run(
        premise="Tweety is a bird. Birds can fly.",
        hypothesis="Tweety can fly.",
        run_id="mock-run",
        **kwargs,
    )


def test_openai_configuration_uses_environment_and_ignores_config_key(tmp_path, monkeypatch):
    upstream_root = tmp_path / "upstream"
    upstream_root.mkdir()
    (upstream_root / "config.yaml").write_text(
        "gpt-4o-mini:\n  engine: gpt-4o-mini\n  api_key: test-key\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test-environment-key")
    engine, api_key = adapter._load_openai_configuration(
        upstream_root, "gpt-4.1"
    )

    assert engine == "gpt-4.1"
    assert api_key is None


def test_openai_configuration_requires_environment_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(adapter.CredentialConfigurationError, match="OPENAI_API_KEY") as error:
        adapter._load_openai_configuration(tmp_path, "gpt-4.1")
    assert error.value.as_dict() == {
        "code": "CREDENTIAL_CONFIGURATION_ERROR",
        "message": str(error.value),
    }


def test_openai_configuration_does_not_leak_environment_key(tmp_path, monkeypatch):
    secret = "test-environment-key"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    engine, placeholder = adapter._load_openai_configuration(tmp_path, "gpt-4.1")
    payload = repr({"engine": engine, "api_key": placeholder})
    assert secret not in payload


def test_openai_llm_constructor_does_not_receive_secret(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-environment-key")
    calls = []

    class FakeGPT:
        def __init__(self, *args):
            calls.append(args)

    adapter._construct_openai_llm(FakeGPT, "gpt-4.1")
    assert calls == [("gpt-4.1",)]


def test_gpt_5_4_compatibility_drops_unsupported_sampling_parameters():
    calls = []

    class Llm:
        def completion_with_backoff(self, **kwargs):
            calls.append(kwargs)
            return "ok"

    llm = Llm()
    adapter._apply_model_compatibility(llm, "gpt-5.4")
    llm.completion_with_backoff(
        model="gpt-5.4",
        temperature=0,
        frequency_penalty=0,
        max_tokens=4096,
        messages=[],
    )

    assert calls == [{"model": "gpt-5.4", "messages": []}]


def test_timing_events_are_appended_for_pipeline_phases(tmp_path):
    artifact, path = run_fake(tmp_path, initial_explanation="1. User step one.")
    events_path = path.parent / "timing_events.jsonl"
    events = [__import__("json").loads(line) for line in events_path.read_text().splitlines()]
    phases = {(event["phase"], event["event"]) for event in events}
    assert ("isabelle_process", "start") in phases
    assert ("isabelle_process", "ready") in phases
    assert ("autoformalisation", "start") in phases
    assert ("autoformalisation", "end") in phases
    assert ("rough_inference", "start") in phases
    assert ("proof_generation", "end") in phases
    assert ("proof_check", "start") in phases


def test_adapter_captures_logical_form_for_upstream_syntax_repairs():
    class Auto:
        def _get_davidsonian_form(self, value):
            return f"logical:{value}"

    auto = Auto()
    adapter._capture_logical_form(auto)
    assert auto._get_davidsonian_form("input") == "logical:input"
    assert auto.logical_form == "logical:input"


@pytest.mark.parametrize("tag", ["isabelle", "Isabelle", "thy"])
def test_markdown_fence_language_tags_are_removed(tag):
    source = f"```{tag}\ntheory Demo\nimports Main\nbegin\nend\n```"
    assert adapter._extract_markdown_code(source).startswith("theory Demo")
    assert "\nisabelle\n" not in adapter._extract_markdown_code(source)


def test_markdown_extraction_preserves_plain_text_and_body_identifier():
    plain = "theory Demo\nimports Main\nbegin\n(* isabelle *)\nend"
    assert adapter._extract_markdown_code(plain) == plain


def test_markdown_extraction_handles_explanations_and_multiple_fences():
    source = "before\n```thy\ntheory Demo\n```\nbetween\n```isabelle\nbegin\nend\n```\nafter"
    assert adapter._extract_markdown_code(source) == "theory Demo\n\nbegin\nend"


def test_calls_upstream_model_classes_and_refines_until_isabelle_valid(tmp_path):
    artifact, path = run_fake(tmp_path)

    assert artifact.status is RunStatus.VALID
    assert artifact.final_validity is True
    assert artifact.iteration_count == 2
    assert path.exists()
    assert "AutoFormalisationModel.__init__" in FakeAutoFormalisationModel.events
    assert "RoughInferenceModel.__init__" in FakeRoughInferenceModel.events
    assert "ExplanationRefineModel.__init__" in FakeExplanationRefineModel.events
    assert "IsabelleSolver.__init__" in FakeIsabelleSolver.events
    assert FakeIsabelleSolver.events[-1] == "shutdown"


def test_failed_proof_feedback_and_refinement_feed_the_next_iteration(tmp_path):
    artifact, _ = run_fake(tmp_path)

    first, second = artifact.iterations
    assert first.failed_proof_step == "using explanation_2 by blast"
    assert first.refined_explanation == second.input_explanation
    assert "using explanation_2 by blast" in FakeExplanationRefineModel.events[1][2]
    formalise_calls = [
        event for event in FakeAutoFormalisationModel.events if isinstance(event, tuple)
        and event[0] == "formalise"
    ]
    assert formalise_calls[1][2] == first.refined_explanation
    assert first.input_graph.steps[1].isabelle_status.value == "REJECTED"
    assert first.refined_graph.steps[1].previous_revision_id == "step-2-r0"


def test_max_iterations_is_refinement_budget_and_every_refinement_is_rechecked(
    tmp_path,
):
    FakeIsabelleSolver.outcomes = [
        (False, "using explanation_2 by blast", 0.02),
        (False, "using explanation_2 by blast", 0.02),
    ]

    artifact, _ = run_fake(tmp_path, max_iterations=1)

    assert artifact.iteration_count == 2
    assert artifact.iterations[0].refined_explanation == (
        artifact.iterations[1].input_explanation
    )
    assert artifact.iterations[1].generated_formalisation
    assert artifact.iterations[1].rough_inference
    assert artifact.iterations[1].generated_isabelle_proof
    assert artifact.iterations[1].logical_validity is False
    assert artifact.iterations[1].refined_explanation is None
    assert artifact.final_validity is False
    assert artifact.status is RunStatus.REJECTED


def test_zero_refinement_budget_does_not_create_unchecked_candidate(tmp_path):
    FakeIsabelleSolver.outcomes = [
        (False, "using explanation_2 by blast", 0.02),
    ]

    artifact, _ = run_fake(tmp_path, max_iterations=0)

    assert artifact.iteration_count == 1
    assert artifact.iterations[0].logical_validity is False
    assert artifact.iterations[0].refined_explanation is None
    assert artifact.final_explanation == artifact.iterations[0].input_explanation
    assert artifact.final_validity is False
    assert artifact.status is RunStatus.REJECTED


def test_iteration_artifacts_and_api_call_counts_are_preserved(tmp_path):
    artifact, _ = run_fake(tmp_path)
    first = artifact.iterations[0]

    assert artifact.initial_candidate.api_calls == 1
    assert first.generated_formalisation
    assert first.rough_inference
    assert first.generated_isabelle_proof
    assert first.unused_explanation_sentences == ("Tweety is a bird.",)
    assert first.openai_api_calls == 4
    assert artifact.total_openai_api_calls == 8
    assert first.isabelle_theory_path.exists()
    assert first.cleanup_result.succeeded is True
    assert first.timing_seconds["total"] >= 0


def test_user_explanation_is_preserved_and_skips_initial_generation(tmp_path):
    artifact, _ = run_fake(
        tmp_path,
        initial_explanation="1. User step one.\n2. User step two.",
    )

    assert artifact.initial_candidate.generated_by_openai is False
    assert artifact.initial_candidate.raw_response == (
        "1. User step one.\n2. User step two."
    )
    assert artifact.initial_candidate.api_calls == 0
    assert artifact.iterations[0].input_explanation.startswith("1. User step one.")


def test_upstream_failure_is_explicit_error_without_fallback(tmp_path):
    class FailingAuto(FakeAutoFormalisationModel):
        def formalise(self, *args, **kwargs):
            raise RuntimeError("upstream formalisation failed")

    artifact, _ = run_fake(tmp_path, _bindings=bindings(auto=FailingAuto))

    assert artifact.status is RunStatus.ERROR
    assert artifact.final_validity is None
    assert artifact.iteration_count == 1
    assert "upstream formalisation failed" in artifact.error
    assert artifact.iterations[0].input_graph.steps[0].isabelle_status.value == "ERROR"
    assert not any(
        isinstance(event, tuple) and event[0] == "solve"
        for event in FakeIsabelleSolver.events
    )


def test_json_schema_contains_ui_fields_and_no_secret(tmp_path):
    artifact, _ = run_fake(tmp_path)
    payload = artifact_to_dict(artifact)

    assert payload["premise"] == "Tweety is a bird. Birds can fly."
    assert payload["hypothesis"] == "Tweety can fly."
    assert payload["initial_candidate"]["graph"]["steps"]
    assert payload["iterations"][0]["syntax_feedback"] == []
    assert payload["iterations"][0]["refined_graph"]["steps"]
    assert payload["final_validity"] is True
    assert "api_key" not in repr(payload).lower()


def test_syntax_feedback_is_recorded_and_sent_to_refinement(tmp_path):
    class SyntaxFirstSolver(FakeIsabelleSolver):
        def get_isabelle_syntax_output(self, code, theory, data, model, iteration):
            self.events.append(("syntax", iteration))
            self._check_syntax_error()
            return iteration == 0

        def _check_syntax_error(self):
            if self.solve_index == 0 and not any(
                event == "syntax-recorded" for event in self.events
            ):
                self.events.append("syntax-recorded")
                return True, False, "Error on line 4: bad syntax", "bad code", "", 0.01
            return False, False, [], "", "", 0.01

        def solve(self, *args):
            self.solve_index += 1
            return True, "", 0.01

    artifact, _ = run_fake(
        tmp_path, _bindings=bindings(solver=SyntaxFirstSolver)
    )

    assert artifact.status is RunStatus.VALID
    assert artifact.iterations[0].syntax_validity is False
    assert "bad syntax" in "\n".join(artifact.iterations[0].syntax_feedback)
    refine_event = next(
        event
        for event in FakeExplanationRefineModel.events
        if isinstance(event, tuple) and event[0] == "refine"
    )
    assert "bad syntax" in refine_event[2]


def test_adapter_invokes_real_upstream_model_implementations(tmp_path):
    upstream_root = Path(__file__).resolve().parents[1] / "explanation_refinement"
    runtime = tmp_path / "binding-load"
    with adapter._upstream_runtime(
        upstream_root, runtime, Path("/home/nahyun0615/Isabelle2023")
    ):
        real_bindings = adapter._load_upstream_bindings()

    class ScriptedUpstreamLlm(FakeLlm):
        def generate(self, *, prompt_name, **kwargs):
            self.request(prompt_name)
            responses = {
                "initial_explanation_prompt": (
                    '{"proposed_answer":"entailed","steps":['
                    '{"text":"Tweety is a bird.","source_sentences":'
                    '["Tweety is a bird."]}]}'
                ),
                "get_event_prompt": (
                    "Hypothesis Sentence:\n1. Tweety can fly.\nHas Action: Yes\n"
                    "Actions: fly\nExplanation Sentence:\n1. Tweety is a bird.\n"
                    "Has Action: No\nActions: none\nPremise Sentence:\n"
                    "1. Tweety is a bird.\nHas Action: No\nActions: none"
                ),
                "get_davidsonian_form_prompt": (
                    "Hypothesis Sentence:\n1. Tweety can fly.\n"
                    "Logical form: True\n\nExplanation Sentence:\n"
                    "1. Tweety is a bird.\nLogical form: True\n\n"
                    "Premise Sentence:\n1. Tweety is a bird.\nLogical form: True"
                ),
                "get_isabelle_axiom_prompt": (
                    "begin\naxiomatization where\n  explanation_1: \"True\"\n"
                ),
                "get_isabelle_theorem_with_premise_prompt": (
                    "imports Main\n\nbegin\naxiomatization where\n"
                    "  explanation_1: \"True\"\n\ntheorem hypothesis:\n"
                    "  assumes asm: \"True\"\n  shows \"True\"\n"
                    "proof -\n\nqed\n\nend"
                ),
                "get_rough_inference_prompt": "Use explanation 1.",
                "get_isabelle_proof_prompt": (
                    "proof -\n  using explanation_1 by blast\nqed"
                ),
            }
            return responses[prompt_name]

    selected = replace(real_bindings, IsabelleSolver=FakeIsabelleSolver)
    FakeIsabelleSolver.outcomes = [(True, "", 0.01)]
    artifact, _ = run_fake(
        tmp_path,
        max_iterations=1,
        _bindings=selected,
        _llm=ScriptedUpstreamLlm(),
    )

    assert artifact.status is RunStatus.VALID
    assert real_bindings.AutoFormalisationModel.__module__ == (
        "models.autoformalisation_model"
    )
    assert real_bindings.RoughInferenceModel.__module__ == "models.rough_inference_model"
    assert real_bindings.ExplanationRefineModel.__module__ == "models.refine_model"
