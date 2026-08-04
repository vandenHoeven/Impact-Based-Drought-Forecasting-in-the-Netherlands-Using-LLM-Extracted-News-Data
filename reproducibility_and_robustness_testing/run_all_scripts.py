"""
Run all reproducibility / robustness checks.

Order: imports → Chapter 05 offline/UI checks → Chapter 04 preprocessing →
2-article LLMn (optional if API key blank) → headed Lexis viewer smoke.

Prints a final PASS / SKIPPED / FAIL summary for every check.

    python reproducibility_and_robustness_testing/run_all_scripts.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

TESTING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTING_ROOT.parent
CHECK_IMPORTS = TESTING_ROOT / "check_imports.py"

CHECK_RESULT_RE = re.compile(
    r"^CHECK_RESULT:\s*(PASS|SKIPPED|FAIL)\b(.*)$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class CheckResult:
    label: str
    cmd: str
    status: str  # PASS | SKIPPED | FAIL
    exit_code: int
    detail: str = ""


def _classify_output(exit_code: int, output: str) -> tuple[str, str]:
    """Map exit code + optional CHECK_RESULT marker to status and detail."""
    matches = list(CHECK_RESULT_RE.finditer(output or ""))
    if matches:
        last = matches[-1]
        status = last.group(1).upper()
        detail = last.group(2).lstrip(" -:\t")
        if status == "SKIPPED":
            return "SKIPPED", detail
        if status == "FAIL" or exit_code != 0:
            return "FAIL", detail
        return "PASS", detail
    if exit_code == 0:
        return "PASS", ""
    return "FAIL", f"exit code {exit_code}"


def _run(cmd: list[str], label: str) -> CheckResult:
    print("=" * 70)
    print(label)
    print(" ".join(cmd))
    print("=" * 70)

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdin=None,  # inherit so getpass / prompts work
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chunks: list[str] = []
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        chunks.append(line)
        print(line, end="", flush=True)
    exit_code = int(proc.wait())
    output = "".join(chunks)
    status, detail = _classify_output(exit_code, output)
    # Non-zero exit always fails even if a PASS marker was printed earlier
    if exit_code != 0:
        status = "FAIL"
        if not detail:
            detail = f"exit code {exit_code}"

    print(f"[{status}] {label}" + (f" - {detail}" if detail else ""))
    return CheckResult(
        label=label,
        cmd=" ".join(cmd),
        status=status,
        exit_code=exit_code,
        detail=detail,
    )


def _run_glob(pattern: str, label_prefix: str) -> list[CheckResult]:
    runners = sorted(TESTING_ROOT.glob(pattern))
    if not runners:
        print(f"No {pattern} files found.")
        return [
            CheckResult(
                label=f"{label_prefix}: {pattern}",
                cmd="",
                status="FAIL",
                exit_code=1,
                detail=f"No {pattern} files found",
            )
        ]
    results: list[CheckResult] = []
    for runner in runners:
        results.append(
            _run(
                [sys.executable, str(runner)],
                f"{label_prefix}: {runner.relative_to(REPO_ROOT).as_posix()}",
            )
        )
    return results


def _print_summary(results: list[CheckResult]) -> None:
    print("=" * 70)
    print("Robustness-check summary")
    print("=" * 70)
    for item in results:
        suffix = f" - {item.detail}" if item.detail else ""
        print(f"  {item.status:7}  {item.label}{suffix}")

    failed = [r for r in results if r.status == "FAIL"]
    skipped = [r for r in results if r.status == "SKIPPED"]
    passed = [r for r in results if r.status == "PASS"]
    print("-" * 70)
    print(
        f"Totals: {len(passed)} passed, {len(skipped)} skipped, {len(failed)} failed "
        f"(of {len(results)} checks)"
    )
    if failed:
        print("Failed checks:")
        for item in failed:
            print(f"  - {item.label} (exit {item.exit_code})")
            if item.detail:
                print(f"    {item.detail}")


def main() -> int:
    results: list[CheckResult] = []

    results.append(
        _run(
            [sys.executable, str(CHECK_IMPORTS)],
            "Import / compile smoke test",
        )
    )

    results.extend(
        _run_glob(
            "chapter_*/run_evaluation_report.py",
            "Evaluation report (offline)",
        )
    )
    results.extend(
        _run_glob(
            "chapter_*/run_src_smoke.py",
            "Chapter src smoke",
        )
    )
    results.extend(
        _run_glob(
            "chapter_*/run_labeller_smoke.py",
            "Labeller UI smoke (Streamlit launch)",
        )
    )
    results.extend(
        _run_glob(
            "chapter_*/run_preprocessing.py",
            "Preprocessing run",
        )
    )
    results.extend(
        _run_glob(
            "chapter_*/run_llmn_extraction.py",
            "LLMn extraction (2 articles)",
        )
    )
    results.extend(
        _run_glob(
            "chapter_*/run_acquisition.py",
            "Viewer smoke (headed browser on screen)",
        )
    )

    _print_summary(results)

    failed = [r for r in results if r.status == "FAIL"]
    if failed:
        print("FAIL: one or more checks failed (see summary above).")
        return 1
    print("PASS: all reproducibility checks succeeded (skipped optional checks are OK).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
