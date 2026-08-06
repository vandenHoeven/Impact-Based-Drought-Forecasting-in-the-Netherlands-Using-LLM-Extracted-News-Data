# Data overview

Most pipeline data lives under `chapters/*/data/`. This folder (`data/` at the repo root) is only a thin shared tree (`raw/`, `processed/`) and is usually empty in the hand-in package.

Full Lexis / newspaper corpora **cannot be redistributed** (copyright). The hand-in ships code, frozen metrics/figures, compact spatial tables, and small fixtures—not the thesis news corpus.

## What can and cannot be handed in

**Can hand in (shipped in this package):**

- Code, notebooks, and robustness smoke scripts
- Chapter 05 annotations and metrics (no article bodies)
- Chapter 06 thesis-final point / NUTS-3 **CSVs**, NUTS geojson, KNMI deficit text
- Chapter 08 report inputs and frozen `results/` figures/tables
- Chapter 09 NL meteo clip (`meteo_nl/`), processed panels, frozen AutoML results
- Small robustness fixtures (bodies truncated where needed)

**Cannot hand in:**

- Lexis ZIP archives and full article JSON (`raw/`, `preprocessed/`, `llm_extracted/`, nested geocode JSONs, Chapter 05 `data/runs/`)
- `.env` files and API / Lexis credentials
- Multi‑GB wildfire occurrence grids (`wildfires_2km_*.csv`)
- Chapter 09 DEM / BRO / CLC / WorldPop rasters and regenerable EDA galleries
- Global SPEI/SPI archives (~50 GB; outside this repo entirely)

## Dataset catalogue

| Dataset | Source | In hand-in? | Path | Notes |
| --- | --- | --- | --- | --- |
| LexisNexis news corpus | LexisNexis (subscription) | **No** | `chapters/04_database_construction/data/raw/` (and preprocessed / llm_extracted) | Copyright; gitignored |
| LLM impact extractions (full articles) | LiteLLM / provider APIs on cleaned Lexis text | **No** | `…/04_…/data/llm_extracted/`; Ch06 nested `*.json` | Bodies + nested article JSON stay local |
| Evaluation annotations / metrics | Manual event-level labels + offline matching | **Yes** | `chapters/05_llm_evaluation/data/*.json` | No article bodies |
| Point / NUTS-3 impact tables | Nominatim (OSM) + local NUTS assignment | **Yes** (CSV) | `chapters/06_geocoding/data/final/points|nuts3/*.csv` | Nested `*_geocoded.json` / `*_nuts3.json` are local only |
| NUTS boundaries | Simplified NL NUTS geojson (Ch06 coder layer) | **Yes** | `chapters/06_geocoding/data/geo/nuts_nl_simplified.geojson` (also copied into Ch08/Ch09 inputs) | Offline |
| KNMI precipitation deficit | KNMI drought / Sep-30 deficit series | **Yes** | `chapters/06_geocoding/data/wildfire/droogte_data_knmi.txt` (also Ch08 input) | Text series |
| Wildfire occurrence grids | Observational wildfire CSVs (viewer / EDA) | **No** | `chapters/06_geocoding/data/wildfire/wildfires_2km_*.csv` | Size (~0.7–1.3 GB); gitignored |
| SPEI / SPI / soil / CDD (NL clip) | Clipped from global meteo archive | **Yes** | `chapters/09_baseline_forecasting/data/meteo_nl/` | Global ~50 GB archive not in repo |
| Elevation / soil / land / population rasters | DEM, BRO soil, CLC 2018, WorldPop | **No** | `chapters/09_baseline_forecasting/data/raw/`, `data/static/` | Local rebuild only; gitignored |
| Ch08 EDA inputs + frozen figures/tables | Ch06 geocode freeze + KNMI + report notebook | **Yes** | `chapters/08_exploratory_data_analysis/data/input/`, `results/` | Offline reproducible from shipped inputs |
| Ch09 panels + AutoML metrics | Meteo + statics + impact labels; Optuna run | **Yes** (panels + frozen results) | `…/09_…/data/processed/`, `results/automl_results/` | Full static rebuild needs local rasters |

## Chapter data trees (preferred for pipelines)

| Chapter | Path | Contents |
| --- | --- | --- |
| 04 Database construction | [`chapters/04_database_construction/data/`](../chapters/04_database_construction/README.md) | `raw/` → `preprocessed/` → `llm_extracted/` (gitignored corpus) |
| 05 LLM evaluation | [`chapters/05_llm_evaluation/data/`](../chapters/05_llm_evaluation/README.md) | Frozen evaluation JSON (no article bodies) |
| 06 Geocoding | [`chapters/06_geocoding/data/`](../chapters/06_geocoding/README.md) | Input mentions, geojson, `final/` CSVs, wildfire assets |
| 08 Exploratory data analysis | [`chapters/08_exploratory_data_analysis/data/`](../chapters/08_exploratory_data_analysis/README.md) | Report inputs (`impacts_nuts3.csv`, geojson, KNMI series) |
| 09 Baseline forecasting | [`chapters/09_baseline_forecasting/data/`](../chapters/09_baseline_forecasting/README.md) | `meteo_nl/`, inputs, processed panels; large rasters local-only |

Repo-root placeholders:

- `raw/` — large raw inputs (usually gitignored); not used for the main chapter pipelines
- `processed/` — optional shared frozen panels not owned by a single chapter

## Where details live

Each chapter README has a **Data and provenance** section (`Use this | Path | Meaning`). For thesis-final vs legacy geocoding files, see [`chapters/06_geocoding/data/final/README.md`](../chapters/06_geocoding/data/final/README.md). Local-only file sizes for Ch06/Ch09 are listed in those chapter READMEs, not repeated here.
