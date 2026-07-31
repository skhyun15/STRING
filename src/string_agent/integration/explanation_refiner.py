"""Positive-query adapter for the upstream Explanation-Refiner pipeline."""

from __future__ import annotations

import importlib
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterator, Literal

import yaml

from string_agent.datasets import ProofWriterExample


VerificationStatus = Literal[
    "PROVED",
    "NOT_PROVED",
    "FORMALISATION_ERROR",
    "VERIFIER_ERROR",
]

QUESTION_PREFIX = (
    "Based on the above information, is the following statement true, "
    "false, or unknown?"
)
_CONTEXT_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_UPSTREAM_LOCK = Lock()


class QuestionFormatError(ValueError):
    """Raised when a ProofWriter question does not have the expected prefix."""


@dataclass(frozen=True)
class NaturalLanguageInput:
    """The complete, label-free input sent to Explanation-Refiner."""

    premise: str
    explanation: str
    hypothesis: str


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    theory_path: Path | None
    iterations: int
    error: str | None
    syntax_valid: bool | None = None
    proof_valid: bool | None = None


def extract_query_statement(question: str) -> str:
    """Extract the positive statement from a canonical ProofWriter question."""

    if not question.startswith(QUESTION_PREFIX):
        raise QuestionFormatError(
            "question must start with the canonical ProofWriter query prefix"
        )
    statement = question[len(QUESTION_PREFIX) :].strip()
    if not statement:
        raise QuestionFormatError("question does not contain a query statement")
    return statement


def _build_natural_language_input(example: ProofWriterExample) -> NaturalLanguageInput:
    context_sentences = [
        sentence.strip()
        for sentence in _CONTEXT_SENTENCE_BOUNDARY.split(example.context.strip())
        if sentence.strip()
    ]
    if not context_sentences:
        raise ValueError("ProofWriter context must not be empty")
    return NaturalLanguageInput(
        premise="none",
        explanation="\n".join(context_sentences),
        hypothesis=extract_query_statement(example.question),
    )


def _string_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _safe_theory_name(example_id: str) -> str:
    theory_name = re.sub(r"[^A-Za-z0-9_]", "_", example_id)
    if not theory_name or not theory_name[0].isalpha():
        theory_name = f"ProofWriter_{theory_name}"
    return theory_name


@contextmanager
def _upstream_runtime(upstream_root: Path, runtime_root: Path) -> Iterator[None]:
    """Provide upstream's relative paths without writing into its repository."""

    runtime_root.mkdir(parents=True, exist_ok=True)
    runtime_config = runtime_root / "config.yaml"
    runtime_config.write_text(
        "isabelle:\n  master_dir: formalisation/\n", encoding="utf-8"
    )

    old_cwd = Path.cwd()
    old_path = os.environ.get("PATH")
    old_sys_path = list(sys.path)
    isabelle_home = Path(
        os.environ.get("STRING_ISABELLE_HOME", "/home/nahyun0615/Isabelle2023")
    ).resolve()
    os.environ["PATH"] = f"{isabelle_home / 'bin'}{os.pathsep}{old_path or ''}"
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


def _load_upstream_config(upstream_root: Path) -> tuple[str, str]:
    with (upstream_root / "config.yaml").open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    model_config = config.get("gpt-4o-mini", {})
    engine = model_config.get("engine")
    api_key = model_config.get("api_key")
    if not isinstance(engine, str) or not engine:
        raise ValueError("upstream config has no gpt-4o-mini engine")
    if not isinstance(api_key, str) or not api_key:
        raise ValueError("upstream config has no gpt-4o-mini credential")
    return engine, api_key


def _verify_natural_language(
    natural_input: NaturalLanguageInput,
    example_id: str,
    *,
    max_iterations: int = 1,
) -> VerificationResult:
    if max_iterations != 1:
        raise ValueError("positive verification currently supports max_iterations=1 only")

    string_root = _string_root()
    upstream_root = Path(
        os.environ.get(
            "STRING_EXPLANATION_REFINER_ROOT",
            str(string_root / "explanation_refinement"),
        )
    ).resolve()
    runtime_root = Path(
        os.environ.get(
            "STRING_EXPLANATION_REFINER_OUTPUT",
            str(string_root / "outputs" / "explanation_refiner"),
        )
    ).resolve()
    theory_name = _safe_theory_name(example_id)
    data_name = "proofwriter_dev"
    model_name = "gpt-4o-mini"
    theory_path = (
        runtime_root
        / "formalisation"
        / data_name
        / model_name
        / theory_name
        / f"{theory_name}_0.thy"
    )

    try:
        engine, api_key = _load_upstream_config(upstream_root)
    except Exception as exc:
        return VerificationResult(
            "FORMALISATION_ERROR", None, 0, str(exc), None, None
        )

    solver = None
    result: VerificationResult | None = None
    with _UPSTREAM_LOCK:
        with _upstream_runtime(upstream_root, runtime_root):
            try:
                generative_model = importlib.import_module("models.generative_model")
                autoformalisation_module = importlib.import_module(
                    "models.autoformalisation_model"
                )
                rough_inference_module = importlib.import_module(
                    "models.rough_inference_model"
                )
                isabelle_module = importlib.import_module("models.isabelle_model")

                llm = generative_model.GPT(engine, api_key)
                llm.prompt_model.base_path = upstream_root / "prompts"
                autoformalisation = autoformalisation_module.AutoFormalisationModel(llm)
                rough_inference_model = rough_inference_module.RoughInferenceModel(llm)
            except Exception as exc:
                return VerificationResult(
                    "FORMALISATION_ERROR", None, 0, str(exc), None, None
                )

            try:
                solver = isabelle_module.IsabelleSolver(
                    llm=llm,
                    isabelle_session="HOL",
                    port=7777,
                    isabelle_name="string-proofwriter",
                    watchdog_timeout=65,
                    dirs=os.environ.get(
                        "STRING_ISABELLE_HOME", "/home/nahyun0615/Isabelle2023"
                    ),
                )
            except Exception as exc:
                return VerificationResult(
                    "VERIFIER_ERROR", None, 0, str(exc), None, None
                )

            try:
                try:
                    isabelle_code = autoformalisation.formalise(
                        theory_name,
                        data_name,
                        model_name,
                        0,
                        natural_input.premise,
                        natural_input.explanation,
                        natural_input.hypothesis,
                    )
                except Exception as exc:
                    result = VerificationResult(
                        "FORMALISATION_ERROR",
                        theory_path if theory_path.exists() else None,
                        1,
                        str(exc),
                        None,
                        None,
                    )
                else:
                    try:
                        has_syntax_error = solver.get_isabelle_syntax_output(
                            isabelle_code,
                            theory_name,
                            data_name,
                            model_name,
                            0,
                        )
                    except Exception as exc:
                        result = VerificationResult(
                            "VERIFIER_ERROR",
                            theory_path if theory_path.exists() else None,
                            1,
                            str(exc),
                            None,
                            None,
                        )
                    else:
                        if has_syntax_error:
                            result = VerificationResult(
                                "FORMALISATION_ERROR",
                                theory_path,
                                1,
                                "Isabelle rejected the generated theory syntax",
                                False,
                                None,
                            )
                        else:
                            try:
                                isabelle_code = theory_path.read_text(encoding="utf-8")
                                rough_inference = (
                                    rough_inference_model.get_rough_inference(
                                        natural_input.premise,
                                        natural_input.explanation,
                                        natural_input.hypothesis,
                                    )
                                )
                                isabelle_code, _ = autoformalisation.get_isabelle_proof(
                                    rough_inference,
                                    isabelle_code,
                                    theory_name,
                                    data_name,
                                    model_name,
                                    0,
                                )
                            except Exception as exc:
                                result = VerificationResult(
                                    "FORMALISATION_ERROR",
                                    theory_path,
                                    1,
                                    str(exc),
                                    True,
                                    None,
                                )
                            else:
                                try:
                                    is_valid, error_code, _ = solver.solve(
                                        theory_name,
                                        data_name,
                                        model_name,
                                        0,
                                        isabelle_code,
                                        natural_input.explanation,
                                    )
                                except Exception as exc:
                                    result = VerificationResult(
                                        "VERIFIER_ERROR",
                                        theory_path,
                                        1,
                                        str(exc),
                                        True,
                                        None,
                                    )
                                else:
                                    if is_valid:
                                        result = VerificationResult(
                                            "PROVED",
                                            theory_path,
                                            1,
                                            None,
                                            True,
                                            True,
                                        )
                                    elif error_code == "no":
                                        result = VerificationResult(
                                            "FORMALISATION_ERROR",
                                            theory_path,
                                            1,
                                            "generated proof has invalid Isabelle syntax",
                                            True,
                                            False,
                                        )
                                    else:
                                        result = VerificationResult(
                                            "NOT_PROVED",
                                            theory_path,
                                            1,
                                            error_code or "Isabelle did not prove the query",
                                            True,
                                            False,
                                        )
            finally:
                if solver is not None:
                    try:
                        solver.shutdown()
                    except Exception as exc:
                        result = VerificationResult(
                            "VERIFIER_ERROR",
                            theory_path if theory_path.exists() else None,
                            1,
                            f"failed to shut down Isabelle: {exc}",
                            result.syntax_valid if result else None,
                            result.proof_valid if result else None,
                        )

    if result is None:
        return VerificationResult(
            "VERIFIER_ERROR",
            theory_path if theory_path.exists() else None,
            1,
            "verification ended without a result",
            None,
            None,
        )
    return result


def verify_positive(example: ProofWriterExample) -> VerificationResult:
    """Verify Q only; the example's gold label never crosses this boundary."""

    try:
        natural_input = _build_natural_language_input(example)
    except (QuestionFormatError, ValueError) as exc:
        return VerificationResult(
            "FORMALISATION_ERROR", None, 0, str(exc), None, None
        )
    return _verify_natural_language(natural_input, example.id, max_iterations=1)
