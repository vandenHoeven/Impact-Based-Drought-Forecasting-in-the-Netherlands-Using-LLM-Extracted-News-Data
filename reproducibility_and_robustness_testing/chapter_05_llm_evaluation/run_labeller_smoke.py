"""
Chapter 05 Streamlit labeller UI smoke: launch → health OK → shut down.

Uses a tiny synthetic article fixture (no private corpus). Opens the Streamlit
UI briefly on port 8505, then terminates the process.

    python reproducibility_and_robustness_testing/chapter_05_llm_evaluation/run_labeller_smoke.py
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
LABEL_SCRIPT = (
    REPO_ROOT / "chapters" / "05_llm_evaluation" / "src" / "label_dataset.py"
)
FIXTURE_PATH = CHAPTER_TEST_ROOT / "data" / "labeller_fixture" / "sample_articles.json"
STATE_DIR = CHAPTER_TEST_ROOT / "data" / "labeller_state"
PORT = 8505
HEALTH_URL = f"http://127.0.0.1:{PORT}/_stcore/health"
STARTUP_TIMEOUT_SECONDS = 45.0
VISIBLE_SECONDS = 20.0
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
    print("Shutting down Streamlit labeller...")
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
    print("Chapter 05 labeller UI smoke (launch -> health -> exit)")
    print(f"script:  {LABEL_SCRIPT}")
    print(f"fixture: {FIXTURE_PATH}")
    print(f"state:   {STATE_DIR}")
    print(f"port:    {PORT}")
    print("=" * 70)

    proc: subprocess.Popen[str] | None = None
    try:
        if not LABEL_SCRIPT.is_file():
            raise FileNotFoundError(f"Missing labeller script: {LABEL_SCRIPT}")
        if not FIXTURE_PATH.is_file():
            raise FileNotFoundError(f"Missing fixture: {FIXTURE_PATH}")

        STATE_DIR.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["EVAL_BUILDER_SOURCE_JSON"] = str(FIXTURE_PATH)
        env["EVAL_BUILDER_OUTPUT_DIR"] = str(STATE_DIR)
        # Avoid opening a second system browser if Streamlit auto-opens; headed
        # server still serves the UI for manual inspection during the wait.
        env.setdefault("BROWSER", "none")

        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(LABEL_SCRIPT),
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
        print(f"Keeping UI up for {VISIBLE_SECONDS:.0f}s so the window is visible...")
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
