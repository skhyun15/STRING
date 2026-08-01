"""STRING adapter for the upstream Explanation-Refiner LLM/Isabelle loop.

The upstream checkout is imported and called as-is. STRING owns orchestration,
artifact capture, and presentation data only; it does not provide an alternate
reasoning or verification backend.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from pathlib import Path
from threading import Lock
from types import MethodType
from typing import Any, Callable, Iterator

import yaml

from string_agent.reasoning import (
    ReasoningGraph,
    StepStatus,
    apply_isabelle_status,
    initial_reasoning_graph,
    revise_reasoning_graph,
)


_UPSTREAM_LOCK = Lock()
_EXPLANATION_REFERENCE = re.compile(r"\bexplanation_(\d+)\b")


class RunStatus(Enum):
    VALID = "VALID"
    REJECTED = "REJECTED"
    MAX_ITERATIONS = "MAX_ITERATIONS"
    ERROR = "ERROR"


class CredentialConfigurationError(RuntimeError):
    """Raised when the OpenAI credential is not supplied by the environment."""

    code = "CREDENTIAL_CONFIGURATION_ERROR"

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": str(self)}


@dataclass(frozen=True)
class CleanupResult:
    attempted: bool
    succeeded: bool | None
    message: str


@dataclass(frozen=True)
class InitialCandidate:
    proposed_answer: str | None
    explanation: str
    raw_response: str
    generated_by_openai: bool
    api_calls: int
    graph: ReasoningGraph


@dataclass(frozen=True)
class IterationArtifact:
    iteration_index: int
    input_explanation: str
    input_graph: ReasoningGraph
    generated_formalisation: str | None
    syntax_checked_formalisation: str | None
    syntax_validity: bool | None
    syntax_feedback: tuple[str, ...]
    rough_inference: str | None
    generated_isabelle_proof: str | None
    unused_explanation_sentences: tuple[str, ...]
    logical_validity: bool | None
    failed_proof_step: str | None
    error_code: str | None
    proof_feedback: str | None
    refined_explanation: str | None
    refined_graph: ReasoningGraph | None
    openai_api_calls: int
    cumulative_openai_api_calls: int
    isabelle_theory_path: Path | None
    timing_seconds: dict[str, float]
    cleanup_result: CleanupResult


@dataclass(frozen=True)
class RunArtifact:
    schema_version: int
    run_id: str
    premise: str
    hypothesis: str
    model: str
    max_iterations: int
    status: RunStatus
    initial_candidate: InitialCandidate | None
    iterations: tuple[IterationArtifact, ...]
    final_explanation: str | None
    final_graph: ReasoningGraph | None
    final_validity: bool | None
    total_openai_api_calls: int
    iteration_count: int
    error: str | None
    cleanup_result: CleanupResult
    started_at: str
    completed_at: str
    total_timing_seconds: float


@dataclass(frozen=True)
class UpstreamBindings:
    GPT: type
    AutoFormalisationModel: type
    RoughInferenceModel: type
    ExplanationRefineModel: type
    IsabelleSolver: type
    filter_explanations: Callable[[str, Any, str], str]
    retry_stop_after_attempt: Callable[[int], Any] | None = None


@dataclass
class _OpenAIRequestCounter:
    api_calls: int = 0

    def instrument(self, llm: object) -> None:
        completions = llm.client.chat.completions
        create = completions.create

        @wraps(create)
        def counted_create(*args: Any, **kwargs: Any) -> Any:
            self.api_calls += 1
            return create(*args, **kwargs)

        completions.create = counted_create


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_timing_event(path: Path, phase: str, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": _utc_now(), "monotonic": time.perf_counter(), "phase": phase, "event": event, **fields}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        file.flush()


def _safe_component(value: str, *, prefix: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"{prefix}_{cleaned}"
    return cleaned


def _string_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_upstream_root() -> Path:
    return Path(
        os.environ.get(
            "STRING_EXPLANATION_REFINER_ROOT",
            str(_string_root() / "explanation_refinement"),
        )
    ).resolve()


def _default_output_root() -> Path:
    return Path(
        os.environ.get(
            "STRING_EXPLANATION_REFINER_OUTPUT",
            str(_string_root() / "outputs" / "explanation_refiner"),
        )
    ).resolve()


def _default_isabelle_home() -> Path:
    return Path(
        os.environ.get("STRING_ISABELLE_HOME", "/home/nahyun0615/Isabelle2023")
    ).resolve()


@contextmanager
def _upstream_runtime(
    upstream_root: Path, runtime_root: Path, isabelle_home: Path
) -> Iterator[None]:
    """Supply upstream relative paths while leaving its checkout untouched."""

    runtime_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / "config.yaml").write_text(
        "isabelle:\n  master_dir: formalisation/\n", encoding="utf-8"
    )
    old_cwd = Path.cwd()
    old_path = os.environ.get("PATH")
    old_sys_path = list(sys.path)
    os.environ["PATH"] = (
        f"{isabelle_home / 'bin'}{os.pathsep}{old_path or ''}"
    )
    sys.path.insert(0, str(upstream_root))
    os.chdir(runtime_root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path
        if old_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = old_path


def _load_upstream_bindings() -> UpstreamBindings:
    # Upstream imports ``Mistral`` from the package root even though recent
    # mistralai releases expose the same SDK class from ``mistralai.client``.
    # The OpenAI path does not use it, but the eager import would otherwise
    # prevent the untouched upstream module from loading.
    mistralai = importlib.import_module("mistralai")
    if not hasattr(mistralai, "Mistral"):
        mistral_client = importlib.import_module("mistralai.client")
        mistralai.Mistral = mistral_client.Mistral
    generative = importlib.import_module("models.generative_model")
    autoformalisation = importlib.import_module("models.autoformalisation_model")
    rough = importlib.import_module("models.rough_inference_model")
    refine = importlib.import_module("models.refine_model")
    isabelle = importlib.import_module("models.isabelle_model")
    upstream_main = importlib.import_module("main")
    return UpstreamBindings(
        GPT=generative.GPT,
        AutoFormalisationModel=autoformalisation.AutoFormalisationModel,
        RoughInferenceModel=rough.RoughInferenceModel,
        ExplanationRefineModel=refine.ExplanationRefineModel,
        IsabelleSolver=isabelle.IsabelleSolver,
        filter_explanations=upstream_main.filter_explanations,
        retry_stop_after_attempt=getattr(generative.tenacity, "stop_after_attempt", None),
    )


def _load_openai_configuration(upstream_root: Path, model: str) -> tuple[str, None]:
    """Load non-secret model settings and require OPENAI_API_KEY in the environment.

    The second return value is retained as a compatibility placeholder for the
    upstream constructor; it is always ``None`` and never contains a secret.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        raise CredentialConfigurationError(
            "OPENAI_API_KEY is not set; configure the environment before using an OpenAI model"
        )
    config: dict[str, Any] = {}
    config_path = upstream_root / "config.yaml"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
        if isinstance(loaded, dict):
            config = loaded
    credential_model = "gpt-4o-mini" if model in {"gpt-4.1", "gpt-5.4"} else model
    model_config = (
        config.get(credential_model) if isinstance(config, dict) else None
    )
    model_config = model_config if isinstance(model_config, dict) else {}
    engine = model if model in {"gpt-4.1", "gpt-5.4"} else model_config.get("engine", model)
    if not isinstance(engine, str) or not engine.strip():
        raise ValueError(f"model {model!r} has no configured OpenAI engine")
    return engine, None


def _construct_openai_llm(gpt_class: type, engine: str) -> object:
    """Construct upstream GPT while letting the SDK read OPENAI_API_KEY."""
    try:
        return gpt_class(engine)
    except TypeError:
        # Older upstream constructors require a positional credential argument;
        # None tells OpenAI() to resolve OPENAI_API_KEY itself.
        return gpt_class(engine, None)


def _apply_model_compatibility(llm: object, model: str) -> None:
    """Adapt only request parameters rejected by frontier chat models."""
    if model != "gpt-5.4" or not hasattr(llm, "completion_with_backoff"):
        return
    original = llm.completion_with_backoff

    def compatible(*args: Any, **kwargs: Any) -> Any:
        for parameter in ("temperature", "frequency_penalty", "max_tokens"):
            kwargs.pop(parameter, None)
        return original(*args, **kwargs)

    llm.completion_with_backoff = compatible


_MARKDOWN_CODE_FENCE = re.compile(
    r"```[ \t]*(?:[A-Za-z0-9_-]+)?[ \t]*\r?\n?(.*?)```",
    re.DOTALL,
)


def _extract_markdown_code(result: str) -> str:
    """Extract fenced code while removing (but never inventing) language tags."""
    matches = _MARKDOWN_CODE_FENCE.findall(result)
    if not matches:
        return result
    return "\n\n".join(match.strip("\r\n") for match in matches).strip()


def _install_response_capture(llm: object, debug_path: Path) -> None:
    """Capture model text around generic Markdown extraction, without secrets."""
    original_completion = getattr(llm, "completion_with_backoff", None)
    original_extract = getattr(llm, "extract_code", None)
    if not callable(original_completion) or not callable(original_extract):
        return
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    def write_debug(raw: str) -> None:
        postprocessed = _extract_markdown_code(raw)
        with debug_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({
                "raw_response": raw,
                "postprocessed_response": postprocessed,
            }, ensure_ascii=False) + "\n")

    def captured_completion(*args: Any, **kwargs: Any) -> Any:
        response = original_completion(*args, **kwargs)
        if not kwargs.get("stream"):
            return response
        chunks = list(response)
        raw = "".join(
            str(chunk.choices[0].delta.content)
            for chunk in chunks
            if getattr(chunk, "choices", None)
            and chunk.choices[0].delta.content is not None
        )
        write_debug(raw)
        return chunks

    def captured_extract(result: str) -> str:
        postprocessed = _extract_markdown_code(result)
        return postprocessed

    llm.completion_with_backoff = captured_completion
    llm.extract_code = captured_extract


def _require_text(value: Any, phase: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{phase} returned no usable text")
    return value


def _capture_logical_form(autoformalisation: object) -> None:
    """Persist the generated logical form for upstream syntax repair calls."""
    generator = getattr(autoformalisation, "_get_davidsonian_form", None)
    if not callable(generator):
        return

    @wraps(generator)
    def captured(*args: Any, **kwargs: Any) -> Any:
        logical_form = generator(*args, **kwargs)
        setattr(autoformalisation, "logical_form", logical_form)
        return logical_form

    autoformalisation._get_davidsonian_form = captured


def _parse_initial_candidate(raw_response: str) -> tuple[str, str, tuple]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"initial explanation was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("initial explanation must be a JSON object")
    proposed_answer = payload.get("proposed_answer")
    steps = payload.get("steps")
    if not isinstance(proposed_answer, str) or not proposed_answer.strip():
        raise RuntimeError("initial explanation has no proposed_answer")
    if not isinstance(steps, list) or not steps:
        raise RuntimeError("initial explanation has no steps")

    step_specs: list[tuple[str, tuple[str, ...]]] = []
    explanation_lines: list[str] = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise RuntimeError(f"initial explanation step {index} is not an object")
        text = step.get("text")
        sources = step.get("source_sentences")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError(f"initial explanation step {index} has no text")
        if not isinstance(sources, list) or not all(
            isinstance(source, str) for source in sources
        ):
            raise RuntimeError(
                f"initial explanation step {index} has invalid source_sentences"
            )
        exact_text = text.strip()
        exact_sources = tuple(source.strip() for source in sources if source.strip())
        step_specs.append((exact_text, exact_sources))
        explanation_lines.append(f"{index}. {exact_text}")
    return proposed_answer, "\n".join(explanation_lines), tuple(step_specs)


def _explanation_indexes(text: str | None) -> frozenset[int]:
    if not text:
        return frozenset()
    return frozenset(int(match) for match in _EXPLANATION_REFERENCE.findall(text))


def _unused_sentence_texts(
    unused_references: tuple[str, ...], graph: ReasoningGraph
) -> tuple[str, ...]:
    indexes = _explanation_indexes("\n".join(unused_references))
    return tuple(
        graph.steps[index - 1].text
        for index in sorted(indexes)
        if 0 < index <= len(graph.steps)
    )


def _capture_syntax_feedback(solver: object) -> list[str]:
    feedback: list[str] = []
    attempt_count = 0
    checker = getattr(solver, "_check_syntax_error", None)
    if not callable(checker):
        return feedback

    @wraps(checker)
    def captured(*args: Any, **kwargs: Any) -> Any:
        nonlocal attempt_count
        attempt_count += 1
        prefix = f"Syntax attempt {attempt_count}"
        result = checker(*args, **kwargs)
        if isinstance(result, tuple) and len(result) >= 5:
            inner_error, contradiction_error, details, inner_code, contradiction = (
                result[:5]
            )
            if details:
                feedback.append(f"{prefix}: {details}")
            if inner_error and inner_code:
                feedback.append(f"{prefix} inner syntax code: {inner_code}")
            if contradiction_error and contradiction:
                feedback.append(f"{prefix} contradiction code: {contradiction}")
            if not inner_error and not contradiction_error:
                feedback.append(f"{prefix}: Isabelle syntax check passed")
        return result

    solver._check_syntax_error = captured
    return feedback


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def artifact_to_dict(artifact: RunArtifact) -> dict[str, Any]:
    return _json_value(artifact)


def artifact_to_json(artifact: RunArtifact, *, indent: int = 2) -> str:
    return json.dumps(
        artifact_to_dict(artifact), ensure_ascii=False, indent=indent, sort_keys=True
    ) + "\n"


class ExplanationRefiner:
    """Execute the sole STRING reasoning path and preserve every iteration."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        max_iterations: int = 3,
        upstream_root: Path | None = None,
        output_root: Path | None = None,
        isabelle_home: Path | None = None,
        port: int = 7777,
        watchdog_timeout: int = 65,
        _bindings: UpstreamBindings | None = None,
        _llm: object | None = None,
    ) -> None:
        if max_iterations < 0:
            raise ValueError("max_iterations must be at least 0")
        self.model = model
        self.max_iterations = max_iterations
        self.upstream_root = (upstream_root or _default_upstream_root()).resolve()
        self.output_root = (output_root or _default_output_root()).resolve()
        self.isabelle_home = (isabelle_home or _default_isabelle_home()).resolve()
        self.port = port
        self.watchdog_timeout = watchdog_timeout
        self._bindings = _bindings
        self._llm = _llm

    def run(
        self,
        *,
        premise: str,
        hypothesis: str,
        initial_explanation: str | None = None,
        run_id: str | None = None,
    ) -> tuple[RunArtifact, Path]:
        if not premise.strip():
            raise ValueError("premise must not be empty")
        if not hypothesis.strip():
            raise ValueError("hypothesis must not be empty")

        started_at = _utc_now()
        run_started = time.perf_counter()
        requested_run_id = run_id or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        safe_run_id = _safe_component(requested_run_id, prefix="run")
        runtime_root = self.output_root / "runs" / safe_run_id
        artifact_path = runtime_root / "artifact.json"
        timing_events_path = runtime_root / "timing_events.jsonl"
        data_name = "string_runs"
        model_path_name = _safe_component(self.model, prefix="model")
        theory_name = _safe_component(safe_run_id, prefix="String")
        iterations: list[IterationArtifact] = []
        initial_candidate: InitialCandidate | None = None
        current_explanation: str | None = initial_explanation
        current_graph: ReasoningGraph | None = None
        final_validity: bool | None = None
        run_error: str | None = None
        counter = _OpenAIRequestCounter()
        solver: object | None = None
        cleanup = CleanupResult(False, None, "Isabelle solver was not started")

        with _UPSTREAM_LOCK:
            with _upstream_runtime(
                self.upstream_root, runtime_root, self.isabelle_home
            ):
                try:
                    bindings = self._bindings or _load_upstream_bindings()
                    if self._llm is None:
                        engine, api_key = _load_openai_configuration(
                            self.upstream_root, self.model
                        )
                        llm = _construct_openai_llm(bindings.GPT, engine)
                    else:
                        llm = self._llm
                    _apply_model_compatibility(llm, self.model)
                    _install_response_capture(
                        llm, runtime_root / "debug" / "model_responses.jsonl"
                    )
                    counter.instrument(llm)
                    if (
                        bindings.retry_stop_after_attempt is not None
                        and hasattr(llm, "completion_with_backoff")
                        and hasattr(llm.completion_with_backoff, "retry_with")
                    ):
                        retried = llm.completion_with_backoff.retry_with(
                            stop=bindings.retry_stop_after_attempt(3), reraise=True
                        )
                        llm.completion_with_backoff = MethodType(retried, llm)

                    if initial_explanation is None:
                        _append_timing_event(timing_events_path, "initial_explanation", "start")
                        initial_call_start = counter.api_calls
                        prompt_model = llm.prompt_model
                        old_prompt_base = prompt_model.base_path
                        prompt_model.base_path = (
                            Path(__file__).resolve().parents[1] / "prompts"
                        )
                        try:
                            raw_initial = _require_text(
                                llm.generate(
                                    model_prompt_dir="explanation",
                                    prompt_name="initial_explanation_prompt",
                                    premise=premise,
                                    hypothesis=hypothesis,
                                ),
                                "initial explanation generation",
                            )
                        finally:
                            prompt_model.base_path = old_prompt_base
                        _append_timing_event(timing_events_path, "initial_explanation", "end", api_calls=counter.api_calls - initial_call_start)
                        proposed_answer, current_explanation, step_specs = (
                            _parse_initial_candidate(raw_initial)
                        )
                        current_graph = initial_reasoning_graph(
                            current_explanation, step_specs
                        )
                        initial_candidate = InitialCandidate(
                            proposed_answer=proposed_answer,
                            explanation=current_explanation,
                            raw_response=raw_initial,
                            generated_by_openai=True,
                            api_calls=counter.api_calls - initial_call_start,
                            graph=current_graph,
                        )
                    else:
                        current_explanation = _require_text(
                            initial_explanation, "provided initial explanation"
                        )
                        current_graph = initial_reasoning_graph(current_explanation)
                        initial_candidate = InitialCandidate(
                            proposed_answer=None,
                            explanation=current_explanation,
                            raw_response=current_explanation,
                            generated_by_openai=False,
                            api_calls=0,
                            graph=current_graph,
                        )

                    llm.prompt_model.base_path = self.upstream_root / "prompts"
                    autoformalisation = bindings.AutoFormalisationModel(llm)
                    _capture_logical_form(autoformalisation)
                    rough_inference_model = bindings.RoughInferenceModel(llm)
                    explanation_refine_model = bindings.ExplanationRefineModel(llm)
                    _append_timing_event(timing_events_path, "isabelle_process", "start")
                    solver = bindings.IsabelleSolver(
                        llm=llm,
                        isabelle_session="HOL",
                        port=self.port,
                        isabelle_name=f"string-{safe_run_id}",
                        watchdog_timeout=self.watchdog_timeout,
                        dirs=str(self.isabelle_home),
                    )
                    _append_timing_event(timing_events_path, "isabelle_process", "ready")
                    syntax_feedback_log = _capture_syntax_feedback(solver)

                    # ``max_iterations`` is the refinement budget. The initial
                    # candidate is always checked once, and each generated
                    # refinement receives its own subsequent verification.
                    for iteration_index in range(self.max_iterations + 1):
                        assert current_explanation is not None
                        assert current_graph is not None
                        iteration_started = time.perf_counter()
                        iteration_call_start = counter.api_calls
                        timing: dict[str, float] = {}
                        generated_formalisation: str | None = None
                        syntax_checked_formalisation: str | None = None
                        syntax_validity: bool | None = None
                        rough_inference: str | None = None
                        generated_proof: str | None = None
                        unused_references: tuple[str, ...] = ()
                        logical_validity: bool | None = None
                        failed_proof_step: str | None = None
                        error_code: str | None = None
                        proof_feedback: str | None = None
                        refined_explanation: str | None = None
                        refined_graph: ReasoningGraph | None = None
                        iteration_error: str | None = None
                        feedback_start = len(syntax_feedback_log)
                        theory_path = (
                            runtime_root
                            / "formalisation"
                            / data_name
                            / model_path_name
                            / theory_name
                            / f"{theory_name}_{iteration_index}.thy"
                        )

                        try:
                            phase = time.perf_counter()
                            _append_timing_event(timing_events_path, "autoformalisation", "start", iteration=iteration_index)
                            generated_formalisation = _require_text(
                                autoformalisation.formalise(
                                    theory_name,
                                    data_name,
                                    model_path_name,
                                    iteration_index,
                                    premise,
                                    current_explanation,
                                    hypothesis,
                                ),
                                "AutoFormalisationModel.formalise",
                            )
                            timing["formalisation"] = time.perf_counter() - phase
                            _append_timing_event(timing_events_path, "autoformalisation", "end", iteration=iteration_index, elapsed=timing["formalisation"])

                            phase = time.perf_counter()
                            _append_timing_event(timing_events_path, "syntax_check", "start", iteration=iteration_index)
                            has_syntax_error = solver.get_isabelle_syntax_output(
                                generated_formalisation,
                                theory_name,
                                data_name,
                                model_path_name,
                                iteration_index,
                            )
                            if not isinstance(has_syntax_error, bool):
                                raise RuntimeError(
                                    "IsabelleSolver syntax check returned a non-boolean"
                                )
                            syntax_validity = not has_syntax_error
                            timing["syntax_check"] = time.perf_counter() - phase
                            _append_timing_event(timing_events_path, "syntax_check", "end", iteration=iteration_index, elapsed=timing["syntax_check"])
                            if theory_path.exists():
                                syntax_checked_formalisation = theory_path.read_text(
                                    encoding="utf-8"
                                )
                            else:
                                syntax_checked_formalisation = generated_formalisation

                            if syntax_validity:
                                phase = time.perf_counter()
                                _append_timing_event(timing_events_path, "rough_inference", "start", iteration=iteration_index)
                                rough_inference = _require_text(
                                    rough_inference_model.get_rough_inference(
                                        premise, current_explanation, hypothesis
                                    ),
                                    "RoughInferenceModel.get_rough_inference",
                                )
                                timing["rough_inference"] = time.perf_counter() - phase
                                _append_timing_event(timing_events_path, "rough_inference", "end", iteration=iteration_index, elapsed=timing["rough_inference"])

                                phase = time.perf_counter()
                                _append_timing_event(timing_events_path, "proof_generation", "start", iteration=iteration_index)
                                proof_result = autoformalisation.get_isabelle_proof(
                                    rough_inference,
                                    syntax_checked_formalisation,
                                    theory_name,
                                    data_name,
                                    model_path_name,
                                    iteration_index,
                                )
                                if not isinstance(proof_result, tuple) or len(proof_result) != 2:
                                    raise RuntimeError(
                                        "AutoFormalisationModel.get_isabelle_proof returned an invalid result"
                                    )
                                generated_proof = _require_text(
                                    proof_result[0],
                                    "AutoFormalisationModel.get_isabelle_proof",
                                )
                                raw_unused = proof_result[1]
                                if isinstance(raw_unused, (list, tuple, set)):
                                    unused_references = tuple(str(item) for item in raw_unused)
                                elif raw_unused:
                                    unused_references = (str(raw_unused),)
                                timing["proof_generation"] = time.perf_counter() - phase
                                _append_timing_event(timing_events_path, "proof_generation", "end", iteration=iteration_index, elapsed=timing["proof_generation"])

                                phase = time.perf_counter()
                                _append_timing_event(timing_events_path, "proof_check", "start", iteration=iteration_index)
                                solve_result = solver.solve(
                                    theory_name,
                                    data_name,
                                    model_path_name,
                                    iteration_index,
                                    generated_proof,
                                    current_explanation,
                                )
                                if not isinstance(solve_result, tuple) or len(solve_result) != 3:
                                    raise RuntimeError(
                                        "IsabelleSolver.solve returned an invalid result"
                                    )
                                logical_validity = bool(solve_result[0])
                                # Only IsabelleSolver.solve may update the final
                                # validity. A newly generated, unchecked
                                # refinement never changes this value.
                                final_validity = logical_validity
                                error_code = str(solve_result[1] or "") or None
                                timing["isabelle_solve"] = float(solve_result[2])
                                timing["solve_wall"] = time.perf_counter() - phase
                                _append_timing_event(timing_events_path, "proof_check", "end", iteration=iteration_index, elapsed=timing["solve_wall"], logical_validity=logical_validity)
                                if not logical_validity:
                                    failed_proof_step = error_code or (
                                        "Isabelle did not establish the theorem"
                                    )
                                    proof_feedback = failed_proof_step
                                    if error_code == "no":
                                        syntax_validity = False
                                        proof_feedback = (
                                            "The generated Isabelle proof has invalid syntax"
                                        )
                            else:
                                error_code = "SYNTAX_ERROR"
                                feedback = syntax_feedback_log[feedback_start:]
                                failed_proof_step = (
                                    "\n".join(feedback)
                                    or "Isabelle rejected the generated theory syntax"
                                )
                                proof_feedback = failed_proof_step

                            used_indexes = _explanation_indexes(generated_proof)
                            failed_indexes = _explanation_indexes(failed_proof_step)
                            current_graph = apply_isabelle_status(
                                current_graph,
                                syntax_valid=syntax_validity,
                                logical_validity=logical_validity,
                                used_explanation_indexes=used_indexes,
                                failed_explanation_indexes=failed_indexes,
                            )

                            if not logical_validity and iteration_index < self.max_iterations:
                                combined_feedback = "\n".join(
                                    item
                                    for item in (
                                        proof_feedback,
                                        *syntax_feedback_log[feedback_start:],
                                    )
                                    if item
                                )
                                phase = time.perf_counter()
                                _append_timing_event(timing_events_path, "refinement", "start", iteration=iteration_index)
                                refined_explanation = _require_text(
                                    explanation_refine_model.refine(
                                        premise,
                                        current_explanation,
                                        hypothesis,
                                        generated_proof
                                        or syntax_checked_formalisation
                                        or generated_formalisation,
                                        combined_feedback,
                                    ),
                                    "ExplanationRefineModel.refine",
                                )
                                refined_explanation = _require_text(
                                    bindings.filter_explanations(
                                        refined_explanation,
                                        unused_references,
                                        current_explanation,
                                    ),
                                    "upstream filter_explanations",
                                )
                                refined_graph, current_graph = revise_reasoning_graph(
                                    current_graph, refined_explanation
                                )
                                timing["explanation_refinement"] = (
                                    time.perf_counter() - phase
                                )
                                _append_timing_event(timing_events_path, "refinement", "end", iteration=iteration_index, elapsed=timing["explanation_refinement"])
                        except Exception as exc:
                            iteration_error = f"{type(exc).__name__}: {exc}"
                            if isinstance(exc, AttributeError) and "logical_form" in str(exc):
                                iteration_error = f"FORMALISATION_ADAPTER_ERROR: {exc}"
                            run_error = iteration_error
                            current_graph = apply_isabelle_status(
                                current_graph,
                                syntax_valid=syntax_validity,
                                logical_validity=logical_validity,
                                error=True,
                            )
                            proof_feedback = proof_feedback or iteration_error

                        timing["total"] = time.perf_counter() - iteration_started
                        current_feedback = tuple(syntax_feedback_log[feedback_start:])
                        iterations.append(
                            IterationArtifact(
                                iteration_index=iteration_index,
                                input_explanation=current_explanation,
                                input_graph=current_graph,
                                generated_formalisation=generated_formalisation,
                                syntax_checked_formalisation=syntax_checked_formalisation,
                                syntax_validity=syntax_validity,
                                syntax_feedback=current_feedback,
                                rough_inference=rough_inference,
                                generated_isabelle_proof=generated_proof,
                                unused_explanation_sentences=_unused_sentence_texts(
                                    unused_references, current_graph
                                ),
                                logical_validity=logical_validity,
                                failed_proof_step=failed_proof_step,
                                error_code=error_code or iteration_error,
                                proof_feedback=proof_feedback,
                                refined_explanation=refined_explanation,
                                refined_graph=refined_graph,
                                openai_api_calls=counter.api_calls - iteration_call_start,
                                cumulative_openai_api_calls=counter.api_calls,
                                isabelle_theory_path=(
                                    theory_path if theory_path.exists() else None
                                ),
                                timing_seconds=timing,
                                cleanup_result=CleanupResult(
                                    False, None, "Pending Isabelle shutdown"
                                ),
                            )
                        )
                        if iteration_error or logical_validity:
                            break
                        if refined_explanation is None or refined_graph is None:
                            # Refinement budget exhausted. The current
                            # explanation was checked, so stop without creating
                            # an unverified candidate.
                            break
                        current_explanation = refined_explanation
                        current_graph = refined_graph
                except Exception as exc:
                    run_error = f"{type(exc).__name__}: {exc}"
                    if current_graph is not None:
                        current_graph = apply_isabelle_status(
                            current_graph,
                            syntax_valid=None,
                            logical_validity=None,
                            error=True,
                        )
                finally:
                    if solver is not None:
                        try:
                            solver.shutdown()
                        except Exception as exc:
                            cleanup = CleanupResult(True, False, str(exc))
                            run_error = run_error or f"Isabelle cleanup failed: {exc}"
                        else:
                            cleanup = CleanupResult(
                                True, True, "Isabelle session and server shut down"
                            )

        iterations = [replace(item, cleanup_result=cleanup) for item in iterations]
        if run_error is not None:
            status = RunStatus.ERROR
        elif final_validity is True:
            status = RunStatus.VALID
        elif final_validity is False:
            status = RunStatus.REJECTED
        else:
            status = RunStatus.MAX_ITERATIONS
        completed_at = _utc_now()
        artifact = RunArtifact(
            schema_version=1,
            run_id=safe_run_id,
            premise=premise,
            hypothesis=hypothesis,
            model=self.model,
            max_iterations=self.max_iterations,
            status=status,
            initial_candidate=initial_candidate,
            iterations=tuple(iterations),
            final_explanation=current_explanation,
            final_graph=current_graph,
            final_validity=final_validity,
            total_openai_api_calls=counter.api_calls,
            iteration_count=len(iterations),
            error=run_error,
            cleanup_result=cleanup,
            started_at=started_at,
            completed_at=completed_at,
            total_timing_seconds=time.perf_counter() - run_started,
        )
        runtime_root.mkdir(parents=True, exist_ok=True)
        serialized = artifact_to_json(artifact)
        artifact_path.write_text(serialized, encoding="utf-8")
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "latest.json").write_text(serialized, encoding="utf-8")
        return artifact, artifact_path


def run_explanation_refiner(
    *,
    premise: str,
    hypothesis: str,
    initial_explanation: str | None = None,
    model: str = "gpt-4o-mini",
    max_iterations: int = 3,
    run_id: str | None = None,
    output_root: Path | None = None,
) -> tuple[RunArtifact, Path]:
    return ExplanationRefiner(
        model=model,
        max_iterations=max_iterations,
        output_root=output_root,
    ).run(
        premise=premise,
        hypothesis=hypothesis,
        initial_explanation=initial_explanation,
        run_id=run_id,
    )
