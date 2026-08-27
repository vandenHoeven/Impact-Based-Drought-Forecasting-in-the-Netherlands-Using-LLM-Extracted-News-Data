"""
Chapter 06 Streamlit combined viewer UI smoke: launch → health OK → keep up 60s.

Uses geocoding smoke results under data/results/ (run run_geocoding.py first).
Opens the Streamlit UI on port 8506 for manual inspection, then terminates.

    python reproducibility_and_robustness_testing/chapter_06_spatial_postprocessing_visualization_dataset_reliability/run_viewer_smoke.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CHAPTER_TEST_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CHAPTER_TEST_ROOT.parents[1]
VIEWER_SCRIPT = (
    REPO_ROOT / "chapters" / "06_spatial_postprocessing_visualization_dataset_reliability" / "src" / "combined_viewer.py"
)
RESULTS_DIR = CHAPTER_TEST_ROOT / "data" / "results"
POINT_CSV = RESULTS_DIR / "impacts_geocoded_points.csv"
NUTS3_CSV = RESULTS_DIR / "impacts_nuts3.csv"
PORT = 8506
HEALTH_URL = f"http://127.0.0.1:{PORT}/_stcore/health"
UI_URL = f"http://127.0.0.1:{PORT}"
STARTUP_TIMEOUT_SECONDS = 60.0
VISIBLE_SECONDS = 60.0
POLL_INTERVAL_SECONDS = 0.5


def _wait_for_health(proc: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Streamlit exited early with code {proc.returncode} "
                f"before becoming healthy."
            )
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=2) as response:
                body = response.read().decode("utf-8", errors="replace").strip()
                if response.status == 200:
                    print(f"UI healthy at {HEALTH_URL} (response={body!r})")
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Streamlit did not become healthy within {STARTUP_TIMEOUT_SECONDS:.0f}s. "
        f"Last error: {last_error or 'n/a'}"
    )


def _terminate(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    print("Shutting down Streamlit viewer...")
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, AttributeError):
            proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def main() -> int:
    print("=" * 70)
    print("Chapter 06 viewer UI smoke (launch -> health -> 60s -> exit)")
    print(f"script:  {VIEWER_SCRIPT}")
    print(f"data:    {RESULTS_DIR}")
    print(f"port:    {PORT}")
    print("=" * 70)

    proc: subprocess.Popen[str] | None = None
    try:
        if not VIEWER_SCRIPT.is_file():
            raise FileNotFoundError(f"Missing viewer script: {VIEWER_SCRIPT}")
        if not POINT_CSV.is_file() or not NUTS3_CSV.is_file():
            raise FileNotFoundError(
                f"Missing smoke CSVs under {RESULTS_DIR}. "
                "Run run_geocoding.py first."
            )

        env = os.environ.copy()
        env["GEOCODING_VIEWER_DATA_DIR"] = str(RESULTS_DIR)
        env.setdefault("BROWSER", "none")

        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(VIEWER_SCRIPT),
            "--server.headless",
            "false",
            "--server.port",
            str(PORT),
            "--server.address",
            "127.0.0.1",
            "--browser.gatherUsageStats",
            "false",
            "--server.fileWatcherType",
            "none",
        ]
        print("Starting:", " ".join(cmd))
        popen_kwargs: dict = {
            "cwd": str(REPO_ROOT),
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
        }
        if sys.platform != "win32":
            popen_kwargs["start_new_session"] = True
        else:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(cmd, **popen_kwargs)
        _wait_for_health(proc)
        print(f"Open the viewer at: {UI_URL}")
        print(f"Keeping UI up for {VISIBLE_SECONDS:.0f}s so you can inspect it...")
        time.sleep(VISIBLE_SECONDS)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        if proc is not None:
            _terminate(proc)
        return 1

    _terminate(proc)
    print("TEST COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
