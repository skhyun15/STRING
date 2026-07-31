"""Smoke-load the Logic-LLM ProofWriter development split."""

from pathlib import Path

from string_agent.datasets.proofwriter import load_proofwriter_file


if __name__ == "__main__":
    dev_path = Path("external/Logic-LLM/data/ProofWriter/dev.json")
    examples = load_proofwriter_file(dev_path)
    print(f"Loaded {len(examples)} ProofWriter dev examples from {dev_path}")
