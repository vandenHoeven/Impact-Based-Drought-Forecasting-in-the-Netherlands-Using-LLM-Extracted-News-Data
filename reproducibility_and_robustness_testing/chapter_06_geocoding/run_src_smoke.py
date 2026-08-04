"""
Chapter 06 src wiring smoke (no Nominatim / no Streamlit server).

Compiles and imports geocoding modules; confirms NUTS geojson is present.

    python reproducibility_and_robustness_testing/chapter_06_geocoding/run_src_smoke.py
"""

from __future__ import annotations

import importlib.util
import py_compile
import sys
from pathlib import Path

CHAPTER_TEST_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CHAPTER_TEST_ROOT.parents[1]
SRC_DIR = REPO_ROOT / "chapters" / "06_geocoding" / "src"
GEOJSON_PATH = (
    REPO_ROOT / "chapters" / "06_geocoding" / "data" / "geo" / "nuts_nl_simplified.geojson"
)

COMPILE_MODULES = [
    "geocode_ranking.py",
    "geocoder.py",
    "location_aliases.py",
    "nuts_mapper.py",
    "point_coder.py",
    "nuts3_coder.py",
    "viewer_article_utils.py",
    "combined_viewer.py",
]

IMPORT_MODULES = [
    ("ch06_geocode_ranking", "geocode_ranking.py"),
    ("ch06_geocoder", "geocoder.py"),
    ("ch06_point_coder", "point_coder.py"),
    ("ch06_nuts3_coder", "nuts3_coder.py"),
]


def _load_module(name: str, path: Path):
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    print("=" * 70)
    print("Chapter 06 src smoke (compile/import, no network)")
    print(f"src: {SRC_DIR}")
    print("=" * 70)

    try:
        if not SRC_DIR.is_dir():
            raise FileNotFoundError(f"Missing chapter src: {SRC_DIR}")
        if not GEOJSON_PATH.is_file():
            raise FileNotFoundError(f"Missing NUTS geojson: {GEOJSON_PATH}")
        print(f"NUTS geojson present: {GEOJSON_PATH.name}")

        for filename in COMPILE_MODULES:
            path = SRC_DIR / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing script: {path}")
            py_compile.compile(str(path), doraise=True)
            print(f"Compiled: {filename}")

        for mod_name, filename in IMPORT_MODULES:
            module = _load_module(mod_name, SRC_DIR / filename)
            print(f"Imported: {filename} ({mod_name})")
            if filename == "point_coder.py" and not hasattr(module, "process_json"):
                raise RuntimeError("point_coder missing process_json")
            if filename == "nuts3_coder.py" and not hasattr(
                module, "process_geocoded_json"
            ):
                raise RuntimeError("nuts3_coder missing process_geocoded_json")
            if filename == "geocoder.py" and not hasattr(module, "geocode_location"):
                raise RuntimeError("geocoder missing geocode_location")
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("TEST COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
