"""
Smoke-test: compile chapter and testing Python files and verify third-party
imports resolve in the active environment.

Does not execute scrapers, notebooks, or pipelines.
"""

from __future__ import annotations

import ast
import importlib
import py_compile
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = [
    REPO_ROOT / "chapters",
    REPO_ROOT / "reproducibility_and_robustness_testing",
]
SKIP_DIR_NAMES = {"__pycache__", ".venv", "venv", ".git", "Reference"}

# pip / distribution name -> import name (extend as chapters are added)
IMPORT_NAME_OVERRIDES: dict[str, str] = {
    "Pillow": "PIL",
    "pillow": "PIL",
    "python-docx": "docx",
    "python-dotenv": "dotenv",
    "PyYAML": "yaml",
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "google-genai": "google.genai",
}


def iter_python_files() -> list[Path]:
    files: list[Path] = []
    for scan_dir in SCAN_DIRS:
        if not scan_dir.is_dir():
            continue
        for path in scan_dir.rglob("*.py"):
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            files.append(path)
    return sorted(files)


def collect_top_level_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return set()

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import
            if node.module:
                names.add(node.module.split(".", 1)[0])
    return names


def is_local_module(name: str, py_files: list[Path]) -> bool:
    """True if name matches a project .py file or package directory."""
    for path in py_files:
        if path.stem == name:
            return True
        if path.name == "__init__.py" and path.parent.name == name:
            return True
    for scan_dir in SCAN_DIRS:
        if (scan_dir / name).exists():
            return True
        if (REPO_ROOT / name).exists():
            return True
    return False


def resolve_import_name(name: str) -> str:
    return IMPORT_NAME_OVERRIDES.get(name, name)


def main() -> int:
    py_files = iter_python_files()
    if not py_files:
        print("No Python files found under chapters/ or reproducibility_and_robustness_testing/.")
        return 1

    compile_failures: list[str] = []
    third_party: set[str] = set()
    file_imports: dict[str, set[str]] = {}

    stdlib = set(sys.stdlib_module_names)

    for path in py_files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            compile_failures.append(f"{rel}: {exc.msg}")

        imports = collect_top_level_imports(path)
        file_imports[rel] = imports
        for name in imports:
            if name in stdlib or name == "__future__":
                continue
            if is_local_module(name, py_files):
                continue
            third_party.add(resolve_import_name(name))

    import_failures: list[str] = []
    import_ok: list[str] = []
    for name in sorted(third_party):
        try:
            importlib.import_module(name)
            import_ok.append(name)
        except Exception as exc:  # noqa: BLE001 - report any import failure
            import_failures.append(f"{name}: {type(exc).__name__}: {exc}")

    print(f"Scanned {len(py_files)} Python file(s).")
    print(f"Third-party packages to check: {len(third_party)}")
    if import_ok:
        print("OK imports:", ", ".join(import_ok))
    if compile_failures:
        print("\nSyntax / compile failures:")
        for line in compile_failures:
            print(f"  - {line}")
    if import_failures:
        print("\nImport failures (add missing packages to requirements.txt):")
        for line in import_failures:
            print(f"  - {line}")

    if compile_failures or import_failures:
        print("\nFAIL")
        return 1

    print("\nPASS: all files compile and third-party imports resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
