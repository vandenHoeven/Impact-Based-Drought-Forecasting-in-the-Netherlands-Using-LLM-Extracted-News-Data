"""
Chapter 07 EDA package smoke (offline; does not re-run the notebook).

Checks inputs, frozen tables/figures, notebook presence, CSV load, and key imports.

    python reproducibility_and_robustness_testing/chapter_07_exploratory_data_analysis/run_src_smoke.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

CHAPTER_TEST_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CHAPTER_TEST_ROOT.parents[1]
PACKAGE_ROOT = REPO_ROOT / "chapters" / "07_exploratory_data_analysis"

NOTEBOOK_PATH = PACKAGE_ROOT / "notebooks" / "01_report_final_figures.ipynb"
INPUT_DIR = PACKAGE_ROOT / "data" / "input"
IMPACTS_CSV = INPUT_DIR / "impacts_nuts3.csv"
GEOJSON_PATH = INPUT_DIR / "nuts_nl_simplified.geojson"
KNMI_PATH = INPUT_DIR / "droogte_data_knmi.txt"
TABLES_DIR = PACKAGE_ROOT / "results" / "tables"
FIGURES_DIR = PACKAGE_ROOT / "results" / "figures"

REQUIRED_TABLES = (
    "classification_total_counts.csv",
    "cooccurrence_article_edges.csv",
    "cooccurrence_nuts3_month_edges.csv",
    "report_final_stats.csv",
)

REQUIRED_FIGURES = (
    "fig_choropleth_nuts3.png",
    "fig_class_distribution.png",
    "fig_cooccurrence_article_arc.png",
    "fig_cooccurrence_article_theme_circular.png",
    "fig_cooccurrence_nuts3_month_arc.png",
    "fig_cooccurrence_nuts3_month_theme_circular.png",
    "fig_meteo_comparison.png",
    "fig_monthly_seasonality.png",
    "fig_sector_distribution.png",
    "fig_yearly_volume.png",
)

REQUIRED_CSV_COLUMNS = (
    "classification",
    "recency_in_months",
    "nuts3_id",
)

REQUIRED_IMPORTS = (
    "pandas",
    "geopandas",
    "matplotlib",
    "seaborn",
    "networkx",
    "numpy",
)


def main() -> int:
    print("=" * 70)
    print("Chapter 07 EDA package smoke (paths + CSV + imports, no notebook run)")
    print(f"package: {PACKAGE_ROOT}")
    print("=" * 70)

    try:
        if not PACKAGE_ROOT.is_dir():
            raise FileNotFoundError(f"Missing chapter package: {PACKAGE_ROOT}")

        for path, label in (
            (NOTEBOOK_PATH, "notebook"),
            (IMPACTS_CSV, "impacts CSV"),
            (GEOJSON_PATH, "NUTS geojson"),
            (KNMI_PATH, "KNMI deficit text"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {label}: {path}")
            print(f"Found {label}: {path.name}")

        for name in REQUIRED_TABLES:
            path = TABLES_DIR / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing frozen table: {path}")
            print(f"Found table: {name}")

        for name in REQUIRED_FIGURES:
            path = FIGURES_DIR / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing frozen figure: {path}")
            print(f"Found figure: {name}")

        import pandas as pd  # noqa: PLC0415

        df = pd.read_csv(IMPACTS_CSV, nrows=5)
        if df.empty:
            raise ValueError(f"Impacts CSV has no rows: {IMPACTS_CSV}")
        missing_cols = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Impacts CSV missing columns: {missing_cols}")
        print(
            f"Loaded impacts CSV sample: {len(df)} rows "
            f"(columns include {list(REQUIRED_CSV_COLUMNS)})"
        )

        for mod_name in REQUIRED_IMPORTS:
            importlib.import_module(mod_name)
            print(f"Imported: {mod_name}")

    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        print(f"CHECK_RESULT: FAIL - {exc}")
        return 1

    print("TEST COMPLETE")
    print("CHECK_RESULT: PASS - Chapter 07 package layout and imports OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
