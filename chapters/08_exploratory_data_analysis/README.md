# Chapter 08 — Drought impact EDA (report figures)

Self-contained package for the thesis **Chapter 8** exploratory data analysis
figures and Overleaf summary stats. Filters to current-month impacts
(`recency_in_months == 0`), NUTS-3, years 2005–2025.

Location in this repo: `chapters/08_exploratory_data_analysis/`.

Upstream geocoding / NUTS-3: [`../06_geocoding/README.md`](../06_geocoding/README.md).

**Not included:** draft EDA notebooks (report 1–4, impact chains, ML EDA, LHM/LSW).

## Layout

```text
08_exploratory_data_analysis/
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
cd chapters/08_exploratory_data_analysis
# open notebooks/01_report_final_figures.ipynb and Run All
# regenerates results/figures/*.png and results/tables/*.csv
```

Requires: `pandas`, `geopandas`, `matplotlib`, `seaborn`, `networkx`, `numpy`.

## Provenance

| Artifact | Built from |
|----------|------------|
| `impacts_nuts3.csv` | Chapter 6 geocoding + LLM feature merge (same freeze as `09_baseline_forecasting/data/input/`) |
| `nuts_nl_simplified.geojson` | Chapter 6 NUTS-3 coder geo layer |
| `droogte_data_knmi.txt` | KNMI drought / precipitation-deficit series used in Ch6 viewers |
| `results/figures|tables/` | Outputs of `01_report_final_figures.ipynb` |

## What’s included / omitted

| Included | Omitted |
|----------|---------|
| Report notebook + 3 frozen inputs | Draft notebooks (`EDA-data_base_plots`, impact chains, ML EDA, LHM/`tkt`, …) |
| Frozen thesis figures/tables under `results/` | Old `images for report{, 2, 3, 4}/` and `outputs/` galleries |

Update Overleaf numbers from `results/tables/report_final_stats.csv`.
