"""
Chapters 08–09 baseline forecasting package smoke (offline; does not re-run Optuna).

Checks notebooks, inputs, meteo clip, processed panels, frozen AutoML results,
compile/import of src, and key ML dependencies.

    python reproducibility_and_robustness_testing/chapter_08_09_baseline_forecasting/run_src_smoke.py
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import py_compile
import sys
from pathlib import Path

CHAPTER_TEST_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CHAPTER_TEST_ROOT.parents[1]
PACKAGE_ROOT = REPO_ROOT / "chapters" / "08_09_baseline_forecasting"
SRC_DIR = PACKAGE_ROOT / "src"
NOTEBOOKS_DIR = PACKAGE_ROOT / "notebooks"
INPUT_DIR = PACKAGE_ROOT / "data" / "input"
METEO_NL_DIR = PACKAGE_ROOT / "data" / "meteo_nl"
PROCESSED_DIR = PACKAGE_ROOT / "data" / "processed"
AUTOML_DIR = PACKAGE_ROOT / "results" / "automl_results"
TABLES_DIR = PACKAGE_ROOT / "results" / "tables"

REQUIRED_NOTEBOOKS = (
    "00_subset_meteo_nl.ipynb",
    "01_build_statics.ipynb",
    "02_data_preparation.ipynb",
    "03_automl_nuts3.ipynb",
)

REQUIRED_INPUTS = (
    "impacts_nuts3.csv",
    "nuts_nl_simplified.geojson",
)

REQUIRED_PROCESSED = (
    "supervised_nuts3_forecast_2018_2025_complete.parquet",
    "feature_manifest.csv",
    "target_manifest.csv",
)

REQUIRED_AUTOML_JSON = (
    "run_summary.json",
    "best_config.json",
)

REQUIRED_AUTOML_CSV = (
    "test_macro_by_horizon.csv",
    "test_metrics_with_skill.csv",
    "selected_features.csv",
    "cv_trials.csv",
)

COMPILE_MODULES = (
    "automl_search.py",
    "run_automl.py",
    "soil_tif_utils.py",
)

REQUIRED_SYMBOLS = (
    "TOP_10_CLASSES",
    "FORECAST_HORIZONS",
    "make_optuna_objective",
    "load_feature_groups",
    "year_expanding_folds",
    "split_by_target_year",
    "assert_target_year_splits",
)

REQUIRED_IMPORTS = (
    "pandas",
    "numpy",
    "sklearn",
    "optuna",
    "xgboost",
    "catboost",
    "shap",
    "matplotlib",
)


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
    print("Chapters 08–09 baseline forecasting smoke (paths + frozen results + src)")
    print(f"package: {PACKAGE_ROOT}")
    print("=" * 70)

    try:
        if not PACKAGE_ROOT.is_dir():
            raise FileNotFoundError(f"Missing chapter package: {PACKAGE_ROOT}")

        for name in REQUIRED_NOTEBOOKS:
            path = NOTEBOOKS_DIR / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing notebook: {path}")
            print(f"Found notebook: {name}")

        for name in REQUIRED_INPUTS:
            path = INPUT_DIR / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing input: {path}")
            print(f"Found input: {name}")

        if not METEO_NL_DIR.is_dir():
            raise FileNotFoundError(f"Missing meteo_nl dir: {METEO_NL_DIR}")
        meteo_files = list(METEO_NL_DIR.rglob("*"))
        meteo_files = [p for p in meteo_files if p.is_file()]
        if not meteo_files:
            raise FileNotFoundError(f"meteo_nl has no files: {METEO_NL_DIR}")
        print(f"Found meteo_nl: {len(meteo_files)} files")

        for name in REQUIRED_PROCESSED:
            path = PROCESSED_DIR / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing processed artifact: {path}")
            print(f"Found processed: {name}")

        supervised_parquet = PROCESSED_DIR / "supervised_nuts3_forecast_2018_2025.parquet"
        supervised_csv = PROCESSED_DIR / "supervised_nuts3_forecast_2018_2025.csv"
        if not supervised_parquet.is_file() and not supervised_csv.is_file():
            raise FileNotFoundError(
                "Missing supervised panel "
                "(supervised_nuts3_forecast_2018_2025.parquet or .csv)"
            )
        print(
            "Found supervised panel: "
            + (
                supervised_parquet.name
                if supervised_parquet.is_file()
                else supervised_csv.name
            )
        )

        predictability = TABLES_DIR / "predictability_all15.csv"
        if not predictability.is_file():
            raise FileNotFoundError(f"Missing table: {predictability}")
        print(f"Found table: {predictability.name}")

        for name in REQUIRED_AUTOML_JSON:
            path = AUTOML_DIR / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing AutoML JSON: {path}")
            print(f"Found AutoML JSON: {name}")

        for name in REQUIRED_AUTOML_CSV:
            path = AUTOML_DIR / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing AutoML CSV: {path}")
            print(f"Found AutoML CSV: {name}")

        import pandas as pd  # noqa: PLC0415

        with (AUTOML_DIR / "run_summary.json").open(encoding="utf-8") as fh:
            summary = json.load(fh)
        for key in ("best_model", "n_trials"):
            if key not in summary:
                raise ValueError(f"run_summary.json missing key: {key}")
        print(
            f"Loaded run_summary.json: best_model={summary['best_model']!r}, "
            f"n_trials={summary['n_trials']}"
        )

        macro = pd.read_csv(AUTOML_DIR / "test_macro_by_horizon.csv", nrows=5)
        if macro.empty:
            raise ValueError("test_macro_by_horizon.csv has no rows")
        print(f"Loaded test_macro_by_horizon.csv sample: {len(macro)} rows")

        features = pd.read_csv(AUTOML_DIR / "selected_features.csv", nrows=5)
        if features.empty:
            raise ValueError("selected_features.csv has no rows")
        print(f"Loaded selected_features.csv sample: {len(features)} rows")

        impacts = pd.read_csv(INPUT_DIR / "impacts_nuts3.csv", nrows=5)
        if impacts.empty:
            raise ValueError(f"Impacts CSV has no rows: {INPUT_DIR / 'impacts_nuts3.csv'}")
        print(f"Loaded impacts CSV sample: {len(impacts)} rows")

        complete_path = PROCESSED_DIR / "supervised_nuts3_forecast_2018_2025_complete.parquet"
        panel = pd.read_parquet(complete_path)
        if panel.empty:
            raise ValueError("complete supervised panel parquet is empty")
        print(
            f"Loaded complete supervised panel: {len(panel)} rows, {panel.shape[1]} cols"
        )

        for filename in COMPILE_MODULES:
            path = SRC_DIR / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing script: {path}")
            py_compile.compile(str(path), doraise=True)
            print(f"Compiled: {filename}")

        automl_search = _load_module("ch09_automl_search", SRC_DIR / "automl_search.py")
        for symbol in REQUIRED_SYMBOLS:
            if not hasattr(automl_search, symbol):
                raise RuntimeError(f"automl_search missing symbol: {symbol}")
        print(
            f"Imported automl_search "
            f"(symbols: {', '.join(REQUIRED_SYMBOLS)})"
        )

        automl_search.assert_target_year_splits(panel, horizon=1)
        print("assert_target_year_splits(panel, horizon=1) OK")

        for mod_name in REQUIRED_IMPORTS:
            importlib.import_module(mod_name)
            print(f"Imported: {mod_name}")

    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        print(f"CHECK_RESULT: FAIL - {exc}")
        return 1

    print("TEST COMPLETE")
    print("CHECK_RESULT: PASS - Chapters 08–09 package layout, results, and imports OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
