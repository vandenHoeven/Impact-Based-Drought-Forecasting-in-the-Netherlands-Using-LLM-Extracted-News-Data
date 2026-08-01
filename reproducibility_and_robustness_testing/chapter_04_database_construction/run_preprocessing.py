"""
Chapter 04.3 preprocessing check: run clean_archive on local data/raw.

Writes and verifies outputs under data/preprocessed/. Prints pipeline logs,
then TEST COMPLETE.

    python reproducibility_and_robustness_testing/chapter_04_database_construction/run_preprocessing.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

CHAPTER_TEST_ROOT = Path(__file__).resolve().parent
RAW_DIR = CHAPTER_TEST_ROOT / "data" / "raw"
OUTPUT_DIR = CHAPTER_TEST_ROOT / "data" / "preprocessed"
REPO_ROOT = CHAPTER_TEST_ROOT.parents[1]
CLEAN_ARCHIVE_PATH = (
    REPO_ROOT
    / "chapters"
    / "04_database_construction"
    / "04_3_LLM_Input_Preprocessing"
    / "clean_archive.py"
)

EXPECTED_OUTPUTS = (
    "all_articles_raw.json",
    "all_articles_deduplicated.json",
    "dedup_report.json",
    "newsjson.json",
)


def _load_clean_archive():
    spec = importlib.util.spec_from_file_location("clean_archive", CLEAN_ARCHIVE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {CLEAN_ARCHIVE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_outputs(output_dir: Path) -> None:
    for name in EXPECTED_OUTPUTS:
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing expected output: {path}")
        if path.stat().st_size <= 0:
            raise ValueError(f"Empty output file: {path}")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in {path}, got {type(payload).__name__}")
        # dedup_report may be empty if no duplicates; other files should have articles
        if name != "dedup_report.json" and len(payload) < 1:
            raise ValueError(f"Expected at least one record in {path}")
        print(f"Verified output: {path} ({len(payload)} records)")


def main() -> int:
    if not any(RAW_DIR.glob("*.zip")):
        print(f"Error: no ZIP files found in {RAW_DIR}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("Chapter 04 preprocessing robustness run")
    print(f"raw_dir:    {RAW_DIR}")
    print(f"output_dir: {OUTPUT_DIR}")
    print("=" * 70)

    try:
        clean_archive = _load_clean_archive()
        clean_archive.run_clean_archive(raw_dir=RAW_DIR, output_dir=OUTPUT_DIR)
        _verify_outputs(OUTPUT_DIR)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
