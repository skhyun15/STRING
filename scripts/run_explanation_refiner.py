"""Run STRING's upstream Explanation-Refiner pipeline on one input."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from string_agent.integration import RunStatus, run_explanation_refiner


def _read_text(path: Path, label: str) -> str:
    try:
        value = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Unable to read {label} {path}: {exc}") from exc
    if not value.strip():
        raise SystemExit(f"{label} must not be empty: {path}")
    return value.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or accept an explanation, then verify/refine it through "
            "the upstream LLM-Isabelle loop."
        )
    )
    parser.add_argument("--premise-file", type=Path, required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--initial-explanation-file", type=Path)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help=(
            "Maximum number of refinements; total verification attempts can be "
            "this value plus the initial attempt."
        ),
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "explanation_refiner",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    premise = _read_text(args.premise_file, "premise file")
    initial_explanation = (
        None
        if args.initial_explanation_file is None
        else _read_text(args.initial_explanation_file, "initial explanation file")
    )
    artifact, artifact_path = run_explanation_refiner(
        premise=premise,
        hypothesis=args.hypothesis.strip(),
        initial_explanation=initial_explanation,
        model=args.model,
        max_iterations=args.max_iterations,
        run_id=args.run_id,
        output_root=args.output_dir,
    )
    print(
        json.dumps(
            {
                "status": artifact.status.value,
                "final_validity": artifact.final_validity,
                "iterations": artifact.iteration_count,
                "openai_api_calls": artifact.total_openai_api_calls,
                "artifact": str(artifact_path),
                "cleanup": artifact.cleanup_result.message,
                "error": artifact.error,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if artifact.status is RunStatus.ERROR else 0


if __name__ == "__main__":
    raise SystemExit(main())
