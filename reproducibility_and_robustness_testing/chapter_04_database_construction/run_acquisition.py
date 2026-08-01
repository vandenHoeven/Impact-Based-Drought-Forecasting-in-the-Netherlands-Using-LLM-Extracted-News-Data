"""
Chapter 04.2 acquisition check: ~20s headed Lexis viewer smoke.

Uses dummy random credentials (no prompts). Does not run the download pipeline.
Not part of run_all_scripts / default pytest.

    python reproducibility_and_robustness_testing/chapter_04_database_construction/run_acquisition.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRAPER_PATH = (
    REPO_ROOT
    / "chapters"
    / "04_database_construction"
    / "04_2_Automated Data Acquisition"
    / "lexis_nexis_scraper.py"
)


def main() -> int:
    if not SCRAPER_PATH.is_file():
        print(f"Error: scraper not found at {SCRAPER_PATH}", file=sys.stderr)
        return 1

    print("Launching Lexis viewer smoke (headed Chromium, ~20s)...")
    print(f"Script: {SCRAPER_PATH}")

    spec = importlib.util.spec_from_file_location("lexis_nexis_scraper", SCRAPER_PATH)
    if spec is None or spec.loader is None:
        print("Error: could not load scraper module", file=sys.stderr)
        return 1

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_lexis_viewer_smoke(duration_seconds=20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
