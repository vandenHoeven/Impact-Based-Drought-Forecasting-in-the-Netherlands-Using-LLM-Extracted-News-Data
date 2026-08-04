"""
Chapter 04.4 LLMn extraction check: run extraction on two fixture articles.

Prompts for GEMINI_API_KEY if unset. Leave the prompt blank to skip the live
API call (reported as SKIPPED by run_all_scripts.py). Reads the first two
articles from data/preprocessed/newsjson.json, writes results under
data/llm_extracted/, prints impacts, then TEST COMPLETE.

    python reproducibility_and_robustness_testing/chapter_04_database_construction/run_llmn_extraction.py
"""

from __future__ import annotations

import getpass
import json
import os
import sys
import traceback
from pathlib import Path

CHAPTER_TEST_ROOT = Path(__file__).resolve().parent
PREPROCESSED_PATH = CHAPTER_TEST_ROOT / "data" / "preprocessed" / "newsjson.json"
LLM_DIR = CHAPTER_TEST_ROOT / "data" / "llm_extracted"
INPUT_SLICE_PATH = LLM_DIR / "input_two_articles.json"
OUTPUT_PATH = LLM_DIR / "two_articles_with_llm_features.json"
REPO_ROOT = CHAPTER_TEST_ROOT.parents[1]
EXTRACTION_DIR = (
    REPO_ROOT
    / "chapters"
    / "04_database_construction"
    / "04_4_LLMn_Extraction_Framework"
)
N_ARTICLES = 2


def _ensure_api_key() -> bool:
    """Return True if a key is available; False if the user left the prompt blank."""
    if os.environ.get("GEMINI_API_KEY", "").strip():
        print("Using GEMINI_API_KEY from environment.")
        return True
    key = getpass.getpass("GEMINI_API_KEY (leave blank to skip live API check): ").strip()
    if not key:
        return False
    os.environ["GEMINI_API_KEY"] = key
    return True


def _load_llmn_extraction():
    extraction_dir = str(EXTRACTION_DIR)
    if extraction_dir not in sys.path:
        sys.path.insert(0, extraction_dir)
    import llmn_extraction  # noqa: PLC0415 — path set just above

    return llmn_extraction


def _load_two_articles(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        articles = json.load(handle)
    if not isinstance(articles, list) or len(articles) < N_ARTICLES:
        raise ValueError(
            f"Need at least {N_ARTICLES} articles in {path}, "
            f"got {0 if not isinstance(articles, list) else len(articles)}"
        )
    return articles[:N_ARTICLES]


def _print_results(output_articles: list[dict]) -> None:
    for article in output_articles:
        features = article.get("features") or {}
        article_id = article.get("id", "<missing id>")
        title = features.get("title", "<missing title>")
        error = features.get("llm_drought_impact_error") or ""
        impacts = features.get("llm_drought_impacts")
        print("=" * 70)
        print(f"Article id: {article_id}")
        print(f"Title: {title}")
        if error:
            print(f"Error: {error}")
        else:
            print("llm_drought_impacts:")
            print(json.dumps(impacts, indent=2, ensure_ascii=False))
        print("=" * 70)


def main() -> int:
    print("=" * 70)
    print("Chapter 04 LLMn extraction robustness run (2 articles)")
    print(f"source: {PREPROCESSED_PATH}")
    print(f"output: {OUTPUT_PATH}")
    print("=" * 70)

    try:
        if not PREPROCESSED_PATH.is_file():
            raise FileNotFoundError(
                f"Missing {PREPROCESSED_PATH}. Run run_preprocessing.py first."
            )

        if not _ensure_api_key():
            print(
                "CHECK_RESULT: SKIPPED - GEMINI_API_KEY was left blank; "
                "live LLMn check not run."
            )
            return 0

        articles = _load_two_articles(PREPROCESSED_PATH)
        LLM_DIR.mkdir(parents=True, exist_ok=True)
        with open(INPUT_SLICE_PATH, "w", encoding="utf-8") as handle:
            json.dump(articles, handle, indent=2, ensure_ascii=False)
        print(f"Wrote input slice: {INPUT_SLICE_PATH}")

        llmn = _load_llmn_extraction()
        summary = llmn.run_extraction(
            input_path=INPUT_SLICE_PATH,
            output_path=OUTPUT_PATH,
            limit=N_ARTICLES,
        )
        if summary is None:
            raise RuntimeError("run_extraction returned None")

        if not OUTPUT_PATH.is_file():
            raise FileNotFoundError(f"Missing extraction output: {OUTPUT_PATH}")

        with open(OUTPUT_PATH, encoding="utf-8") as handle:
            output_articles = json.load(handle)

        if len(output_articles) != N_ARTICLES:
            raise ValueError(
                f"Expected {N_ARTICLES} output articles, got {len(output_articles)}"
            )

        hard_failures = [
            a
            for a in output_articles
            if (a.get("features") or {}).get("llm_drought_impact_error")
        ]
        if hard_failures:
            ids = [a.get("id", "<missing id>") for a in hard_failures]
            raise RuntimeError(f"LLM hard-fail for article(s): {ids}")

        _print_results(output_articles)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print(f"CHECK_RESULT: FAIL - {exc}")
        return 1

    print("TEST COMPLETE")
    print("CHECK_RESULT: PASS - live 2-article LLMn extraction succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
