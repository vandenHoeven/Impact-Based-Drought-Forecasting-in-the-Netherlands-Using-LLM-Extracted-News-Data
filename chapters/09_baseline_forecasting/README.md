# Chapter 09 — NUTS-3 drought impact AutoML (provenance-first)

Self-contained package for the thesis **Chapter 9** main experiment: multi-label
impact occurrence forecasting at NUTS-3, horizons \(h=0\ldots3\), top-10 classes,
development 2018–2023 / test 2024–2025, Optuna model + predictor-subset search.

Location in this repo: `chapters/09_baseline_forecasting/`.

Upstream geocoding labels: [`../06_geocoding/README.md`](../06_geocoding/README.md); EDA: [`../08_exploratory_data_analysis/README.md`](../08_exploratory_data_analysis/README.md).

**Not included:** NUTS-2 pipeline, class-aware / top-15 AutoML runners. Global SPEI/SPI archives (~50 GB) stay outside this package — only the NL meteo clip lives under `data/meteo_nl/`.

## Layout

```text
09_baseline_forecasting/
  notebooks/
    00_subset_meteo_nl.ipynb      # optional: global meteo → NL clip (~47 MB)
    01_build_statics.ipynb        # elevation + soil (+ land/pop) → static parquets
    02_data_preparation.ipynb     # meteo zonal + labels → supervised / dev / test panels
    03_automl_nuts3.ipynb         # Optuna CV + test + SHAP + diagnostics
  src/
    automl_search.py              # search space, CV, metrics, SHAP
    run_automl.py                 # headless twin of notebook 03
    soil_tif_utils.py
  data/
    meteo_nl/                     # NL-clipped SPEI/SPI/soil/CDD (in repo)
    static/                       # CLC + WorldPop TIFs (local only; gitignored)
    raw/                          # DEM + BRO soil (local only; gitignored)
    input/                        # impacts CSV + NUTS geojson (in repo)
    processed/                    # panels / manifests (in repo); EDA galleries local only
  results/
    automl_results/               # frozen thesis run
    figures/statics/              # curated elev + soil maps
    tables/                       # e.g. predictability_all15.csv
```

## Local-only large files (not in Git)

These are **copied into this folder on disk** for full rebuilds, but **gitignored** (GitHub 100 MB limit). Keep them locally; do not push:

| Local path | Approx. size | Role |
| --- | --- | --- |
| `data/raw/soil/BRO-SGM-Bodemkaart-V2025-01.gml` | ~299 MB | BRO soil map (statics notebook `01`) |
| `data/static/land/U2018_CLC2018_V2020_20u1.tif` | ~197 MB | CLC 2018 land cover |
| `data/raw/elevation/10_DEM_y50x0.tif` | ~125 MB | DEM for elevation statics |
| Other `data/static/**/*.tif` | varies | e.g. WorldPop population rasters |
| `data/processed/*.gpkg` | varies | BRO soil GPKG caches (rebuildable) |
| `data/processed/qa_predictor_july_maps/` | regenerable | July predictor QA PNG farm |
| `data/processed/soil_eda/`, `images_for_elevation_eda/` | regenerable | Full EDA galleries |

**Shipped for GitHub without those:** notebooks, `src/`, frozen `results/`, `data/meteo_nl/`, `data/input/`, small processed CSVs/parquets.

## Rebuild order

```bash
cd chapters/09_baseline_forecasting
# 1. OPTIONAL — only if data/meteo_nl/ is empty or you want to re-clip from a
#    local ~50 GB global Meteo archive. The hand-in copy already ships meteo_nl/.
#    Set CH09_METEO_GLOBAL_DIR (or METEO_GLOBAL_DIR_OVERRIDE in the notebook), then:
#    open notebooks/00_subset_meteo_nl.ipynb
#    If meteo_nl is already populated, the notebook sets SKIP_CLIP=True and does nothing.

# 2. statics (needs data/raw/elevation + soil, data/static rasters, input geojson)
#    open notebooks/01_build_statics.ipynb

# 3. panels (needs meteo_nl + statics + impacts)
#    open notebooks/02_data_preparation.ipynb

# 4. model (or use frozen results without re-running 100 trials)
python src/run_automl.py
#    or open notebooks/03_automl_nuts3.ipynb
```

To re-run notebook `00` from a global archive:

```text
# PowerShell example
$env:CH09_METEO_GLOBAL_DIR = "D:\path\to\Meteo data"
# then open notebooks/00_subset_meteo_nl.ipynb and Run All
```

## Provenance (SPI / SPEI / statics)

| Artifact | Built from |
|----------|------------|
| `data/meteo_nl/` | Global Ch7 `Meteo data/` via NL bbox + year≥2004 clip (`00_…`) |
| `meteo_nuts3_monthly_2018_2025.parquet` | Zonal means of SPI/SPEI 1–24, soil anomaly, CDD over NUTS-3 (`02_…`) |
| `static_land_cover_*` / `static_population_*` | CLC 2018 + WorldPop rasters in `data/static/` |
| `static_elevation_*` / `static_soil_*` | DEM + BRO soil under `data/raw/` (`01_…`) |
| `dev_*` / `test_*` panels | Meteo + statics + impact labels (`02_…`) |

## Frozen thesis results

Under `results/automl_results/` (selected Random Forest, CV / test macro PR-AUC ≈0.324 at \(h=1\)).
See `run_summary.json`, `best_config.json`, `test_macro_by_horizon.csv`,
`test_metrics_with_skill.csv`, SHAP and diagnostic figures.

Supplementary 15-class predictability freeze (no class-aware runner):
`results/tables/predictability_all15.csv`.

## Offline package smoke

```text
python reproducibility_and_robustness_testing/chapter_09_baseline_forecasting/run_src_smoke.py
```

Checks layout, frozen AutoML artifacts, and `src/` imports without re-running Optuna.
