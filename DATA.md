# Data overview

All pipeline data lives under `chapters/*/data/` (and chapter `results/`). This file is the hand-in data catalogue at the repo root.

Full Lexis / newspaper article bodies **cannot be redistributed** (copyright). Large rasters and wildfire grids exceed GitHub’s size limits and stay local. A private ~50 GB global meteo archive on the author’s machine was used once to build the small NL clip that *is* shipped; that archive was **never** part of this repository or any hand-in.

## What can and cannot be handed in

**Shipped in this package:**

- Code, notebooks, and robustness smoke scripts / fixtures
- Chapter 05 annotations and metrics (no article bodies)
- Chapter 06 thesis-final point / NUTS-3 **CSVs**, NUTS geojson, KNMI deficit text, legacy package samples
- Chapter 08 report inputs and frozen `results/` figures/tables
- Chapter 09 NL meteo clip (`meteo_nl/`), processed panels/manifests, frozen AutoML results, curated statics figures under `results/figures/statics/`

**Not redistributed (copyright, size, secrets, or never a repo artifact):**

- Lexis ZIPs and full-article JSON (Ch04 `raw/` / `preprocessed/` / `llm_extracted/`; Ch06 nested geocode JSONs; Ch05 optional `data/runs/`)
- `.env` / API keys / Lexis credentials
- Multi‑GB wildfire occurrence grids
- Chapter 09 DEM / BRO / CLC / WorldPop rasters, GPKG caches, regenerable EDA galleries
- Global SPEI/SPI meteo archive (~50 GB): external only, never uploaded, never in git

## Dataset catalogue

Sizes are approximate (measured on disk where present, or as documented in chapter READMEs for local-only files). Paths under Chapter 06–09 are relative to that chapter’s folder unless written from repo root.

| Chapter | Dataset | Source | Size | In hand-in? | Path |
| --- | --- | --- | --- | --- | --- |
| 04 | Lexis raw ZIP archives | LexisNexis (subscription download) | varies per run | **No** (copyright) | `chapters/04_database_construction/data/raw/` |
| 04 | Cleaned / deduplicated article JSON | Derived: Ch04 `clean_archive.py` (MinHash + LSH) on Lexis ZIPs | varies | **No** (copyright) | `chapters/04_database_construction/data/preprocessed/` |
| 04 | LLM-extracted impact JSON (full article text) | Derived: LiteLLM / Gemini extraction (`llmn_extraction.py`) | varies | **No** (copyright) | `chapters/04_database_construction/data/llm_extracted/` |
| 05 | Evaluation set annotations | Manual event-level labelling | ~33 KB | **Yes** | `chapters/05_llm_evaluation/data/evaluation_set.json` |
| 05 | Model metrics (P/R/F1) | Offline matching of model outputs vs annotations | ~8 KB | **Yes** | `chapters/05_llm_evaluation/data/model_metrics.json` |
| 05 | Post-hoc impact / attribute labels | Manual post-hoc labelling | ~40 KB | **Yes** | `chapters/05_llm_evaluation/data/posthoc.json` |
| 05 | Optional API run dumps | Optional LiteLLM re-run (`run_models.py`) | varies | **No** (copyright) | `chapters/05_llm_evaluation/data/runs/` (gitignored) |
| 06 | Merged LLM + features input JSON (full text) | Derived from Ch04 LLM extraction (thesis-final flex run) | ~230 MB | **No** (copyright + size) | `chapters/06_geocoding/data/input/chapter7_merged_…_flex.json` |
| 06 | Legacy geocoding input sample | Legacy hand-in package sample (not thesis-final flex) | ~50 KB | **Yes** | `chapters/06_geocoding/data/input/impacts_for_geocoding.json` |
| 06 | Point-geocoded nested JSON (with article text) | Nominatim / OpenStreetMap + ranking | ~240 MB | **No** (copyright + size) | `chapters/06_geocoding/data/final/points/*_geocoded.json` |
| 06 | Point-geocoded flat CSV (+ filtered) | Derived from point-geocoded JSON | ~16 KB / ~14 KB | **Yes** | `chapters/06_geocoding/data/final/points/*.csv` |
| 06 | NUTS-3 assigned nested JSON (with article text) | Local spatial join of geocoded points to NUTS polygons | ~250 MB | **No** (copyright + size) | `chapters/06_geocoding/data/final/nuts3/*_nuts3.json` |
| 06 | NUTS-3 assigned flat CSV | Derived from NUTS-3 nested JSON | ~22 KB | **Yes** | `chapters/06_geocoding/data/final/nuts3/*.csv` |
| 06 | Legacy processed CSVs | Legacy hand-in package samples (script defaults) | a few KB each | **Yes** | `chapters/06_geocoding/data/processed/*.csv` |
| 06 | NUTS-3 boundaries (simplified) | Eurostat / Statistics Netherlands NUTS classification, simplified for this repo | ~290 KB | **Yes** | `chapters/06_geocoding/data/geo/nuts_nl_simplified.geojson` (also copied into Ch08/Ch09 inputs) |
| 06 | KNMI precipitation-deficit series | KNMI (Royal Netherlands Meteorological Institute) open drought / Sep-30 deficit series | ~0.6 MB | **Yes** | `chapters/06_geocoding/data/wildfire/droogte_data_knmi.txt` |
| 06 | Wildfire occurrence grid 2018–2020 | Observational wildfire occurrence data (upstream provider not documented in-repo) | ~680 MB | **No** (size) | `chapters/06_geocoding/data/wildfire/wildfires_2km_2018_2020.csv` |
| 06 | Wildfire decade grid 2017–2022 | Same observational wildfire source as above | ~1.3 GB | **No** (size) | `chapters/06_geocoding/data/wildfire/wildfires_2km_decade_2017-2022_no-military-bases.csv` |
| 08 | `impacts_nuts3.csv` | Ch06 NUTS-3 coder output + LLM feature merge (same freeze as Ch09 input) | ~21 MB | **Yes** | `chapters/08_exploratory_data_analysis/data/input/impacts_nuts3.csv` |
| 08 | Frozen EDA figures / tables | Generated by `01_report_final_figures.ipynb` | small PNGs / CSVs | **Yes** | `chapters/08_exploratory_data_analysis/results/` |
| 09 | NL meteo clip (SPEI/SPI/soil anomaly + ECAD CDD) | Clipped from the external global meteo archive below; ECAD indices from [ecad.eu](http://www.ecad.eu) | ~47 MB | **Yes** | `chapters/09_baseline_forecasting/data/meteo_nl/` |
| 09 | Global SPEI/SPI / meteo archive (clip source) | External climate archives on the author’s local machine only (never in git, never a hand-in artifact) | ~50 GB | **No** (external only) | none (set via `CH09_METEO_GLOBAL_DIR`; not in this repo) |
| 09 | DEM elevation raster | [Copernicus DEM](https://ec.europa.eu/eurostat/web/gisco/geodata/digital-elevation-model/copernicus#Elevation) (GLO-90, ESA/Airbus Defence and Space, TanDEM-X-derived, DLR), via Eurostat GISCO | ~125 MB | **No** (size) | `chapters/09_baseline_forecasting/data/raw/elevation/` |
| 09 | BRO soil map | BRO (Dutch Basisregistratie Ondergrond, national soil / subsurface registry) | ~299 MB | **No** (size) | `chapters/09_baseline_forecasting/data/raw/soil/` |
| 09 | CLC 2018 land cover | Copernicus CORINE Land Cover (CLC2018) | ~197 MB | **No** (size) | `chapters/09_baseline_forecasting/data/static/land/` |
| 09 | WorldPop population rasters | WorldPop.org gridded population | varies | **No** (size) | `chapters/09_baseline_forecasting/data/static/` |
| 09 | BRO soil GPKG caches | Derived / cached from the BRO soil map | varies | **No** (rebuildable) | `chapters/09_baseline_forecasting/data/processed/*.gpkg` |
| 09 | Dev / test panels + manifests | Derived: meteo zonal means + statics + impact labels (`02_data_preparation.ipynb`) | ~78 MB (processed tree; galleries excluded) | **Yes** (panels/manifests) | `chapters/09_baseline_forecasting/data/processed/` |
| 09 | Curated statics figures | Selected outputs from `soil_eda` / `images_for_elevation_eda` (Ch09 statics notebook `01`) | ~0.8 MB (4 PNGs) | **Yes** | `chapters/09_baseline_forecasting/results/figures/statics/` |
| 09 | Full regenerable EDA / QA galleries | Generated diagnostics from statics / predictor notebooks (bulk; not curated) | varies | **No** (local only) | `…/data/processed/qa_predictor_july_maps/`, `soil_eda/`, `images_for_elevation_eda/` |
| 09 | Frozen AutoML results | Optuna model + predictor-subset search (`run_automl.py` / notebook `03`) | small JSON/CSV + figures | **Yes** | `chapters/09_baseline_forecasting/results/automl_results/` |

## Chapter data folders

| Chapter | Path | Contents |
| --- | --- | --- |
| 04 Database construction | [`chapters/04_database_construction/data/`](chapters/04_database_construction/README.md) | `raw/` → `preprocessed/` → `llm_extracted/` (gitignored corpus) |
| 05 LLM evaluation | [`chapters/05_llm_evaluation/data/`](chapters/05_llm_evaluation/README.md) | Frozen evaluation JSON (no article bodies) |
| 06 Geocoding | [`chapters/06_geocoding/data/`](chapters/06_geocoding/README.md) | Input, geojson, `final/` CSVs, wildfire assets |
| 08 Exploratory data analysis | [`chapters/08_exploratory_data_analysis/data/`](chapters/08_exploratory_data_analysis/README.md) | Report inputs (`impacts_nuts3.csv`, geojson, KNMI) |
| 09 Baseline forecasting | [`chapters/09_baseline_forecasting/data/`](chapters/09_baseline_forecasting/README.md) | `meteo_nl/`, inputs, processed panels; large rasters local-only |

## Where details live

Each chapter README has a **Data and provenance** section. For thesis-final vs legacy geocoding files, see [`chapters/06_geocoding/data/final/README.md`](chapters/06_geocoding/data/final/README.md). Full local-only size tables for Ch06/Ch09 remain in those chapter READMEs.
