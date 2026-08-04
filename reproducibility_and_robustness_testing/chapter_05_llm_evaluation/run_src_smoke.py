"""
Chapter 05 optional-tools wiring smoke (no API calls).

Compiles/imports label_dataset.py and run_models.py, loads Chapter 04 schema
via AST, and loads llmn_extraction. Does not require the private corpus.

    python reproducibility_and_robustness_testing/chapter_05_llm_evaluation/run_src_smoke.py
"""

from __future__ import annotations

import importlib.util
import py_compile
import sys
from pathlib import Path

CHAPTER_TEST_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CHAPTER_TEST_ROOT.parents[1]
SRC_DIR = REPO_ROOT / "chapters" / "05_llm_evaluation" / "src"
LABEL_PATH = SRC_DIR / "label_dataset.py"
RUN_MODELS_PATH = SRC_DIR / "run_models.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses / package lookups behave.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    print("=" * 70)
    print("Chapter 05 src smoke (schema + path wiring, no API)")
    print(f"src: {SRC_DIR}")
    print("=" * 70)

    try:
        for path in (LABEL_PATH, RUN_MODELS_PATH):
            if not path.is_file():
                raise FileNotFoundError(f"Missing script: {path}")
            py_compile.compile(str(path), doraise=True)
            print(f"Compiled: {path.name}")

        label_dataset = _load_module("ch05_label_dataset", LABEL_PATH)
        schema = label_dataset.load_llm_schema()
        impacts = schema.get("impact_classes") or []
        recency = schema.get("recency_months") or []
        severity = schema.get("severity_classes") or []
        if len(impacts) < 1 or len(recency) < 1 or len(severity) < 1:
            raise ValueError(f"Schema lists empty or incomplete: {schema}")
        if not label_dataset.LLM_SCHEMA_PATH.exists():
            raise FileNotFoundError(
                f"Schema path does not exist: {label_dataset.LLM_SCHEMA_PATH}"
            )
        print(
            f"load_llm_schema OK: impacts={len(impacts)} "
            f"recency={len(recency)} severity={len(severity)}"
        )
        print(f"Schema file: {label_dataset.LLM_SCHEMA_PATH}")

        run_models = _load_module("ch05_run_models", RUN_MODELS_PATH)
        framework = run_models.CH04_FRAMEWORK
        extraction = framework / "llmn_extraction.py"
        if not extraction.is_file():
            raise FileNotFoundError(f"Missing Chapter 04 runner: {extraction}")
        llmn = run_models.load_llmn_extraction()
        if not hasattr(llmn, "run_extraction"):
            raise RuntimeError("llmn_extraction missing run_extraction")
        print(f"load_llmn_extraction OK: {extraction}")

        corpus = run_models.DEFAULT_SOURCE_CORPUS
        if corpus.is_file():
            print(f"Private corpus present (optional): {corpus}")
        else:
            print(
                "Note: private Chapter 04 corpus not on disk "
                f"({corpus.name}); full run_models.py runs are out of scope "
                "for this smoke check."
            )
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("TEST COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
