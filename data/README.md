# Data directory

Repo-root shared data tree (not a chapter pipeline folder).

- `raw/`: large raw inputs (usually gitignored). Full Lexis / newspaper corpora are **not shipped** here for copyright reasons.
- `processed/`: shared frozen panels, impact databases, and geospatial outputs that are not chapter-local.

## Chapter data trees (preferred for pipelines)

| Chapter | Path | Contents |
| --- | --- | --- |
| 04 Database construction | [`chapters/04_database_construction/data/`](../chapters/04_database_construction/README.md) | `raw/` → `preprocessed/` → `llm_extracted/` |
| 05 LLM evaluation | [`chapters/05_llm_evaluation/data/`](../chapters/05_llm_evaluation/README.md) | Frozen evaluation JSON (no article bodies) |
| 06 Geocoding | [`chapters/06_geocoding/data/`](../chapters/06_geocoding/README.md) | Input mentions, geojson, `final/` points/NUTS-3, wildfire assets |
| 08 Exploratory data analysis | [`chapters/08_exploratory_data_analysis/data/`](../chapters/08_exploratory_data_analysis/README.md) | Report inputs (`impacts_nuts3.csv`, geojson, KNMI series) |
| 09 Baseline forecasting | [`chapters/09_baseline_forecasting/data/`](../chapters/09_baseline_forecasting/README.md) | `meteo_nl/`, inputs, processed panels; large rasters local-only |
