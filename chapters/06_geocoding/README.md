# Chapter 06 — Geocoding (point + NUTS-3)

Geocode drought-impact location mentions extracted in earlier chapters, assign Dutch NUTS-3 regions, and support a Streamlit map viewer plus wildfire validation EDA.

## What this does

1. **Point geocoding (`point_coder.py`)** — Calls Nominatim (OpenStreetMap) once per location mention, with ranking/filtering, and writes geocoded points.
2. **NUTS-3 assignment (`nuts3_coder.py`)** — Assigns NUTS-3 from those points using local `nuts_nl_simplified.geojson` (no network by default).
3. **Combined viewer (`combined_viewer.py`)** — Streamlit UI for points, NUTS-3 polygons, and optional wildfire overlays.
4. **Wildfire EDA (`notebooks/eda_wildfires.ipynb`)** — Uses frozen geocoded outputs and observational wildfire CSVs.

Thesis-final frozen files under `data/final/` support offline analysis without re-hitting Nominatim.

**Not included:** LLM extraction runner, NUTS-1/NUTS-2 coder outputs.

## Layout

```text
06_geocoding/
  data/
    geo/nuts_nl_simplified.geojson
    input/
      chapter7_merged_..._flex.json   # local only (gitignored)
      impacts_for_geocoding.json      # legacy package sample (in repo)
    final/                            # thesis-final coder outputs
      points/   *.csv in repo; *.json local only
      nuts3/    *.csv in repo; *.json local only
    processed/                        # legacy package CSVs (script defaults)
    wildfire/
      droogte_data_knmi.txt           # in repo
      wildfires_2km_*.csv             # local only (gitignored)
  notebooks/eda_wildfires.ipynb
  src/
  results/figures/
```

## How to run

From the repository root (with `.venv` activated and `pip install -r requirements.txt`):

```text
cd chapters/06_geocoding
python src/point_coder.py          # Nominatim + ranking → points (network)
python src/nuts3_coder.py          # NUTS-3 from geocoded points (offline)
streamlit run src/combined_viewer.py
```

| Step | Default input | Default output | Network |
|------|---------------|----------------|---------|
| `point_coder.py` | `data/input/impacts_for_geocoding.json` (legacy default) | `data/processed/impacts_geocoded.json` + `impacts_geocoded_points.csv` | Nominatim (~1 req/s) |
| `nuts3_coder.py` | `impacts_geocoded.json` if present, else `impacts_geocoded_points.csv` | `impacts_nuts3.json` + `impacts_nuts3.csv` | none |

To re-run against the thesis-final LLM input, pass that JSON via the scripts’ input arguments (see `--help`). Ready-made finals are already under `data/final/`.

Offline viewer/EDA: use `data/final/` (or legacy `data/processed/` if you intentionally want the package sample).

Legacy combined path (geocode + assign in one Nominatim pass — not required for the thesis pipeline):

```text
python src/nuts3_coder.py --from-llm-json
```

**Combined viewer.** By default the Data sidebar loads thesis-final CSVs under `data/final/points/` and `data/final/nuts3/`. Optionally switch to Upload / Local file (lists `data/final/` and legacy `data/processed/`). Also uses `data/geo/nuts_nl_simplified.geojson` and the decade wildfire CSV if present.

**Wildfire EDA.** Open `notebooks/eda_wildfires.ipynb`. Paths resolve to `data/wildfire/` and `data/processed/` by default; prefer thesis-final paths under `data/final/` when comparing to the thesis. One optional cell may look for a private Chapter 04 corpus; **skip that cell** if the file is missing.

**Thesis figures.** Under `results/figures/`. Some development-folder filenames contained `Nuts2` but show **NUTS-3** regions; packaged copies use `*Nuts3*` names (e.g. `Wildfires-July-2019-Nuts3.png`). Also included: Jul 2018 / Apr 2019 / Oct 2019 Nuts+point pairs, monthly geocoded vs observed wildfires, wildfire-risk-increase plot.

## Data and provenance

| Use this | Path | Meaning |
| --- | --- | --- |
| **Thesis-final coder outputs (in repo)** | [`data/final/`](data/final/README.md) `points/*.csv`, `nuts3/*.csv` | Point + NUTS-3 tables for the viewer / offline analysis |
| **Thesis-final LLM input / full JSON** | `data/input/chapter7_merged_..._flex.json` and `data/final/**/*.json` | **Local only** (gitignored; see below) |
| Legacy package samples | `data/input/impacts_for_geocoding.json`, `data/processed/impacts_*.csv` | Older hand-in samples — **not** the thesis-final flex run |
| Geo layer | `data/geo/nuts_nl_simplified.geojson` | NL NUTS boundaries (in repo) |
| KNMI deficit text | `data/wildfire/droogte_data_knmi.txt` | In repo |

Local-only large files (GitHub 100 MB limit / full article bodies; keep on disk, do not push):

| Local path | Approx. size | Role |
| --- | --- | --- |
| `data/input/chapter7_merged_with_llm_features_gemini_gemini-3.5-flash_flex.json` | ~230 MB | Full LLM-enriched articles (pre-geocode input) |
| `data/final/points/..._geocoded.json` | ~240 MB | Nested point-geocoded articles |
| `data/final/nuts3/..._nuts3.json` | ~250 MB | Nested NUTS-3 assigned articles |
| `data/wildfire/wildfires_2km_decade_2017-2022_no-military-bases.csv` | ~1.3 GB | Decade wildfire occurrence grid |
| `data/wildfire/wildfires_2km_2018_2020.csv` | ~680 MB | 2018–2020 wildfire subset |

**Shipped instead:** compact CSVs under `data/final/`, legacy `data/processed/` samples, geojson, and KNMI deficit text.

## Reproducibility and limits

- **Thesis-final NUTS-3 / points** under `data/final/` are frozen offline artefacts of the flex geocoding run.
- **Re-running Nominatim** may differ over time (rate limits, gazetteer drift).
- Robustness smokes (5-impact Nominatim + offline NUTS-3 + 60s viewer) live under [`../../reproducibility_and_robustness_testing/chapter_06_geocoding/`](../../reproducibility_and_robustness_testing/chapter_06_geocoding/); suite runners use a tiny fixture, not `data/final/`.

## Links

- Upstream extraction / evaluation: [`../04_database_construction/README.md`](../04_database_construction/README.md), [`../05_llm_evaluation/README.md`](../05_llm_evaluation/README.md)
- Downstream EDA: [`../08_exploratory_data_analysis/README.md`](../08_exploratory_data_analysis/README.md)
- Downstream forecasting: [`../09_baseline_forecasting/README.md`](../09_baseline_forecasting/README.md)
- Robustness suite: [`../../reproducibility_and_robustness_testing/README.md`](../../reproducibility_and_robustness_testing/README.md)
