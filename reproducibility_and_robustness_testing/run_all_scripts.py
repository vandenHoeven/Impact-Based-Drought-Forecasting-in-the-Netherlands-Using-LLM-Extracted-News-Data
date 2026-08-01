"""
Run all reproducibility / robustness checks, including headed Lexis viewer smoke.

    python reproducibility_and_robustness_testing/run_all_scripts.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TESTING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTING_ROOT.parent
CHECK_IMPORTS = TESTING_ROOT / "check_imports.py"


def _run(cmd: list[str], label: str) -> int:
    print("=" * 70)
    print(label)
    print(" ".join(cmd))
    print("=" * 70)
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(completed.returncode)


def main() -> int:
    failures = 0

    failures += _run(
        [sys.executable, str(CHECK_IMPORTS)],
        "Import / compile smoke test",
    )

    preprocessing_runners = sorted(TESTING_ROOT.glob("chapter_*/run_preprocessing.py"))
    if not preprocessing_runners:
        print("No chapter_*/run_preprocessing.py files found.")
        failures += 1
    else:
        for runner in preprocessing_runners:
            failures += _run(
                [sys.executable, str(runner)],
                f"Preprocessing run: {runner.relative_to(REPO_ROOT).as_posix()}",
            )

    acquisition_runners = sorted(TESTING_ROOT.glob("chapter_*/run_acquisition.py"))
    if not acquisition_runners:
        print("No chapter_*/run_acquisition.py files found.")
        failures += 1
    else:
        for runner in acquisition_runners:
            failures += _run(
                [sys.executable, str(runner)],
                f"Viewer smoke (headed browser on screen): "
                f"{runner.relative_to(REPO_ROOT).as_posix()}",
            )

    if failures:
        print("FAIL: one or more checks failed.")
        return 1
    print("PASS: all reproducibility checks succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
