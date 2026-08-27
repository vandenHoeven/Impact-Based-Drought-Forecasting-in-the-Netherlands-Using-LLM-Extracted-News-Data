# Chapter 07: Drought impact EDA (report figures)

Self-contained package for the thesis Chapter 7 exploratory data analysis figures and Overleaf summary stats. Filters to current-month impacts (`recency_in_months == 0`), NUTS-3, years 2005–2025.

## What this does

Runs `notebooks/01_report_final_figures.ipynb` to regenerate the thesis EDA figures and summary tables from frozen NUTS-3 impact inputs, NUTS boundaries, and the KNMI precipitation-deficit series.

## Layout

```text
07_exploratory_data_analysis/
  notebooks/
    01_report_final_figures.ipynb   # all report figures + tables
  data/
    input/
      impacts_nuts3.csv             # geocoded + LLM-labelled impacts (~21 MB)
      nuts_nl_simplified.geojson    # NL NUTS boundaries
      droogte_data_knmi.txt         # KNMI Sep-30 precipitation deficit series
  results/
    figures/                        # 10 frozen PNGs (thesis figures)
    tables/                         # 4 CSVs incl. report_final_stats.csv
```

## How to run

```bash
cd chapters/07_exploratory_data_analysis
# open notebooks/01_report_final_figures.ipynb and Run All
# regenerates results/figures/*.png and results/tables/*.csv
```

Requires: `pandas`, `geopandas`, `matplotlib`, `seaborn`, `networkx`, `numpy`.

Update Overleaf numbers from `results/tables/report_final_stats.csv`.

## Data and provenance

| Use this | Path | Meaning |
| --- | --- | --- |
| **Thesis-final inputs** | `data/input/impacts_nuts3.csv`, `nuts_nl_simplified.geojson`, `droogte_data_knmi.txt` | Frozen inputs for the report notebook |
| **Thesis-final outputs** | `results/figures/`, `results/tables/` | Outputs of `01_report_final_figures.ipynb` |

Provenance detail:

| Artifact | Built from |
|----------|------------|
| `impacts_nuts3.csv` | Chapter 6 geocoding + LLM feature merge (same freeze as `08_09_baseline_forecasting/data/input/`) |
| `nuts_nl_simplified.geojson` | Chapter 6 NUTS-3 coder geo layer |
| `droogte_data_knmi.txt` | KNMI drought / precipitation-deficit series used in Ch6 viewers |

## Reproducibility and limits

- Thesis figures/tables ship frozen under `results/`; re-run the notebook to regenerate them from the shipped inputs.
- The robustness suite does **not** re-run the notebook; it checks package integrity and imports offline:

```text
python reproducibility_and_robustness_testing/chapter_07_exploratory_data_analysis/run_src_smoke.py
```

## Links

- Upstream geocoding / NUTS-3: [`../06_geocoding/README.md`](../06_geocoding/README.md)
- Downstream forecasting: [`../08_09_baseline_forecasting/README.md`](../08_09_baseline_forecasting/README.md)
- Robustness suite: [`../../reproducibility_and_robustness_testing/README.md`](../../reproducibility_and_robustness_testing/README.md)
