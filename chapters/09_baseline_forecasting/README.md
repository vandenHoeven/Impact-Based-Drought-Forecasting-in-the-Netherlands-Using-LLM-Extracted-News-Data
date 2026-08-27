# Chapter 09: NUTS-3 drought impact AutoML

Self-contained package for the thesis Chapter 9 main experiment: multi-label impact occurrence forecasting at NUTS-3, horizons \(h=0\ldots3\), top-10 classes, **target-year** development (impact years 2018–2023) / test (impact years 2024–2025), Optuna model + predictor-subset search.

## What this does

1. Optionally clip global meteo archives to NL (`00_subset_meteo_nl.ipynb`).
2. Build static predictors from elevation / soil / land / population (`01_build_statics.ipynb`).
3. Build the supervised complete panel from meteo + statics + impact labels (`02_data_preparation.ipynb`).
4. Run Optuna CV + test + SHAP + diagnostics with target-year splits (`03_automl_nuts3.ipynb` or `src/run_automl.py`).

This chapter uses the NL meteo clip under `data/meteo_nl/` (global SPEI/SPI archives stay external; see [`DATA.md`](../../DATA.md)).

## Layout

```text
09_baseline_forecasting/
  notebooks/
    00_subset_meteo_nl.ipynb      # optional: global meteo → NL clip (~47 MB)
    01_build_statics.ipynb        # elevation + soil (+ land/pop) → static parquets
    02_data_preparation.ipynb     # meteo zonal + labels → supervised complete panel
    03_automl_nuts3.ipynb         # Optuna CV + test + SHAP + diagnostics (target-year splits)
  src/
    automl_search.py              # search space, target-year CV, metrics, SHAP
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

## How to run

```bash
# From the repo root: pin ML stack versions for bit-for-bit Optuna / model reproducibility
pip install -r chapters/09_baseline_forecasting/requirements.txt

cd chapters/09_baseline_forecasting
# 1. OPTIONAL: only if data/meteo_nl/ is empty or you want to re-clip from a
#    local ~50 GB global Meteo archive. This repo already ships meteo_nl/.
#    Set CH09_METEO_GLOBAL_DIR, then open notebooks/00_subset_meteo_nl.ipynb
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

Frozen thesis results live under `results/automl_results/` (selected configuration, CV / test macro AP at \(h=1\)). See `run_summary.json`, `best_config.json`, `test_macro_by_horizon.csv`, `test_metrics_with_skill.csv`, SHAP and diagnostic figures. Temporal splits use **year of target month** \(t+h\). Supplementary 15-class predictability freeze: `results/tables/predictability_all15.csv`.

## Data and provenance

| Use this | Path | Meaning |
| --- | --- | --- |
| **Thesis-final AutoML results** | `results/automl_results/` | Frozen thesis run (no Optuna re-fit needed) |
| **Shipped panels / meteo / inputs** | `data/meteo_nl/`, `data/input/`, processed CSVs/parquets | Enough for offline inspection and smoke checks |
| **Curated statics figures (shipped)** | `results/figures/statics/` | Selected soil/elevation EDA maps for the thesis |
| Local-only rasters (rebuild) | `data/raw/`, `data/static/**/*.tif` | DEM / BRO / CLC / WorldPop (gitignored) |
| Regenerable galleries | `data/processed/qa_predictor_july_maps/`, `soil_eda/`, … | Local only; rebuildable |

Local-only large files (keep on disk; do not push):

| Local path | Approx. size | Role |
| --- | --- | --- |
| `data/raw/soil/BRO-SGM-Bodemkaart-V2025-01.tif` | ~299 MB | BRO soil map (statics notebook `01`) |
| `data/static/land/U2018_CLC2018_V2020_20u1.tif` | ~197 MB | CLC 2018 land cover |
| `data/raw/elevation/10_DEM_y50x0.tif` | ~125 MB | Copernicus DEM (GLO-90) via [Eurostat GISCO](https://ec.europa.eu/eurostat/web/gisco/geodata/digital-elevation-model/copernicus#Elevation); elevation statics |
| Other `data/static/**/*.tif` | varies | e.g. WorldPop population rasters |
| `data/processed/*.gpkg` | varies | BRO soil GPKG caches (rebuildable) |

Provenance detail:

| Artifact | Built from |
|----------|------------|
| `data/meteo_nl/` | Global Ch7 `Meteo data/` via NL bbox + year≥2004 clip (`00_…`) |
| `meteo_nuts3_monthly_2018_2025.parquet` | Zonal means of SPI/SPEI 1–24, soil anomaly, CDD over NUTS-3 (`02_…`) |
| `static_land_cover_*` / `static_population_*` | CLC 2018 + WorldPop rasters in `data/static/` |
| `static_elevation_*` / `static_soil_*` | DEM + BRO soil under `data/raw/` (`01_…`) |
| `supervised_nuts3_forecast_2018_2025.parquet` / `_complete.parquet` | Meteo + statics + impact labels (`02_…`); AutoML splits by target year in memory |

## Reproducibility and limits

- Install pinned deps from [`requirements.txt`](requirements.txt) before re-running AutoML (`TPESampler(seed=42)`, model `random_state`/`random_seed=42`, CatBoost `thread_count=1`, sequential Optuna trials).
- Each full run writes an `environment` block into `results/automl_results/run_summary.json` (Python + package versions) so version drift is visible.
- SHAP `n_models` counts class-models where a feature's mean |SHAP| exceeds `0.05` (`SHAP_N_MODELS_THRESHOLD`); `n_models_evaluated` is the number of class models that produced SHAP values.
- Minor caveat: RF/XGBoost with `n_jobs=-1` are seed-deterministic on CPU, but float reduction order can theoretically differ across machines with different core counts.
- Full static rebuilds need local gitignored rasters under `data/`.
- The suite does **not** re-run Optuna / `run_automl.py` (100 trials); thesis metrics stay validated via frozen `results/automl_results/`.

Offline package smoke:

```text
python reproducibility_and_robustness_testing/chapter_09_baseline_forecasting/run_src_smoke.py
```

## Links

- Upstream geocoding labels: [`../06_geocoding/README.md`](../06_geocoding/README.md)
- Upstream EDA: [`../08_exploratory_data_analysis/README.md`](../08_exploratory_data_analysis/README.md)
- Robustness suite: [`../../reproducibility_and_robustness_testing/README.md`](../../reproducibility_and_robustness_testing/README.md)
