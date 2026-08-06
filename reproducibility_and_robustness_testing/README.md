# Reproducibility and Robustness Testing

Procedure **checks** (smokes) for thesis chapter code—not a substitute for the full copyrighted corpus or bit-exact thesis LLMn outputs.

Each chapter README ends with a **Links** section pointing here. Prefer those READMEs for how to run the real pipeline; use this suite to verify the hand-in package still imports and that offline/fixture paths pass.

See also:
- [`chapters/04_database_construction/README.md`](../chapters/04_database_construction/README.md)
- [`chapters/05_llm_evaluation/README.md`](../chapters/05_llm_evaluation/README.md)
- [`chapters/06_geocoding/README.md`](../chapters/06_geocoding/README.md)
- [`chapters/08_exploratory_data_analysis/README.md`](../chapters/08_exploratory_data_analysis/README.md)
- [`chapters/09_baseline_forecasting/README.md`](../chapters/09_baseline_forecasting/README.md)

## Layout

```text
reproducibility_and_robustness_testing/
├── check_imports.py
├── run_all_scripts.py
├── chapter_04_database_construction/
│   ├── data/raw/
│   ├── data/preprocessed/
│   ├── data/llm_extracted/
│   ├── run_acquisition.py
│   ├── run_preprocessing.py
│   └── run_llmn_extraction.py
├── chapter_05_llm_evaluation/
│   ├── data/labeller_fixture/   # synthetic articles for UI smoke
│   ├── data/labeller_state/     # labeller runtime state (gitignored)
│   ├── data/results/            # recomputed tables from offline check
│   ├── run_evaluation_report.py
│   ├── run_src_smoke.py
│   └── run_labeller_smoke.py
├── chapter_06_geocoding/
│   ├── data/fixture/            # 5 real impacts from chapter input (bodies truncated)
│   ├── data/results/            # geocode + NUTS-3 smoke outputs (gitignored)
│   ├── run_geocoding.py
│   ├── run_viewer_smoke.py
│   └── run_src_smoke.py
├── chapter_08_exploratory_data_analysis/
│   └── run_src_smoke.py         # offline package + import smoke
└── chapter_09_baseline_forecasting/
    └── run_src_smoke.py         # offline package + frozen AutoML + src smoke
```

## Run

```text
# from repo root, with .venv activated
python reproducibility_and_robustness_testing/run_all_scripts.py

# Chapter 05 only
python reproducibility_and_robustness_testing/chapter_05_llm_evaluation/run_evaluation_report.py
python reproducibility_and_robustness_testing/chapter_05_llm_evaluation/run_src_smoke.py
python reproducibility_and_robustness_testing/chapter_05_llm_evaluation/run_labeller_smoke.py

# Chapter 06 only
python reproducibility_and_robustness_testing/chapter_06_geocoding/run_src_smoke.py
python reproducibility_and_robustness_testing/chapter_06_geocoding/run_geocoding.py
python reproducibility_and_robustness_testing/chapter_06_geocoding/run_viewer_smoke.py

# Chapter 08 only
python reproducibility_and_robustness_testing/chapter_08_exploratory_data_analysis/run_src_smoke.py

# Chapter 09 only
python reproducibility_and_robustness_testing/chapter_09_baseline_forecasting/run_src_smoke.py
```

`run_all_scripts.py` order: imports → Chapter 05 checks → Chapter 06 geocoding/viewer → Chapter 04 preprocessing → 2-article LLMn → headed Lexis acquisition.

At the end it prints a **Robustness-check summary** with `PASS` / `SKIPPED` / `FAIL` for every runner (plus totals and a list of failed checks). The suite exits non-zero only when one or more checks are `FAIL`; skipped optional checks do not fail the suite.

### Optional live API (Chapter 04 LLMn)

`run_llmn_extraction.py` prompts for `GEMINI_API_KEY` when unset:

- Enter a key → live 2-article extraction runs (`PASS` if successful).
- Leave blank / press Enter → check is **SKIPPED** (no API call); this does **not** fail `run_all_scripts.py`.
- If the key is already in the environment, the live call runs without prompting.

### Optional live Nominatim (Chapter 06 geocoding)

`run_geocoding.py` calls OpenStreetMap Nominatim for 5 real locations from the chapter input fixture:

- Network reachable → live geocode + offline NUTS-3 (`PASS` if all 5 get coordinates).
- Nominatim unreachable → check is **SKIPPED**; this does **not** fail `run_all_scripts.py`.
- Viewer smoke (`run_viewer_smoke.py`) expects those results under `chapter_06_geocoding/data/results/` (run geocoding first; the full suite does that automatically).

## Chapter 05 — what is tested and why

Chapter 05 thesis numbers come from **frozen annotations/metrics**, not from a live model call. The suite therefore focuses on offline consistency and tool wiring. There is **no live LLM API** here (that is covered by Chapter 04’s 2-article extraction).

| Check | What it tests | Why |
| --- | --- | --- |
| `run_evaluation_report.py` | Frozen JSON loads; stored F1 equals \(2PR/(P+R)\); recomputed tables match golden CSVs under `chapters/05_llm_evaluation/results/tables/` | Thesis tables must be reproducible offline without APIs or newspaper article bodies |
| `run_src_smoke.py` | `label_dataset.py` / `run_models.py` compile and import; schema AST from Chapter 04 `schemas.py`; `llmn_extraction` loads | Optional tools stay wired to Chapter 04 without calling providers or needing the private corpus |
| `run_labeller_smoke.py` | Streamlit labeller process starts on a local port, health endpoint returns OK, then the process is shut down | Confirms the annotation UI still launches (synthetic fixture only; no full corpus, no automated clicking) |

## Chapter 04 — what is tested and why

| Check | What it tests | Why |
| --- | --- | --- |
| `run_preprocessing.py` | `clean_archive` on a small fixture ZIP → preprocessed JSON | Cleaning / MinHash dedup procedure without sharing the thesis corpus |
| `run_llmn_extraction.py` | Live 2-article extraction (prompts for `GEMINI_API_KEY` if unset; blank = skip) | End-to-end schema + LiteLLM path when a key is provided; skippable offline |
| `run_acquisition.py` | ~20s headed Lexis viewer smoke with dummy credentials | Acquisition UI/browser path still opens |

Full Lexis downloads / full-corpus extraction / full evaluation re-runs use the chapter scripts with real credentials and the private corpus.

## Chapter 06 — what is tested and why

Geocoding code and frozen thesis outputs live under [`chapters/06_geocoding/`](../chapters/06_geocoding/README.md). Suite runners use a **tiny self-contained fixture** of 5 real non-geocoded impacts excerpted from `impacts_for_geocoding.json` (article bodies truncated); they do **not** load `data/final/` thesis CSVs.

| Check | What it tests | Why |
| --- | --- | --- |
| `run_src_smoke.py` | Compile/import geocoding modules; NUTS geojson present | Chapter wiring stays importable without network |
| `run_geocoding.py` | Nominatim on 5 real Dutch locations from the chapter input; print lat/lon/display; offline NUTS-3 via local geojson; write CSVs under `data/results/` | End-to-end point + NUTS path on real excerpts; skippable if Nominatim is unreachable |
| `run_viewer_smoke.py` | Streamlit `combined_viewer.py` on port 8506 for **60 seconds** (manual inspection), pointed at smoke CSVs via `GEOCODING_VIEWER_DATA_DIR` | Confirms the map UI still launches against the smoke outputs |

## Chapter 08 — what is tested and why

EDA figures/tables live under [`chapters/08_exploratory_data_analysis/`](../chapters/08_exploratory_data_analysis/README.md). The suite does **not** re-run the notebook (already validated manually).

| Check | What it tests | Why |
| --- | --- | --- |
| `run_src_smoke.py` | Notebook + inputs present; frozen `results/tables` (4 CSVs) and `results/figures` (10 PNGs); sample load of `impacts_nuts3.csv`; imports pandas/geopandas/matplotlib/seaborn/networkx/numpy | Confirms the hand-in package is intact and EDA deps are importable offline |

## Chapter 09 — what is tested and why

Baseline AutoML lives under [`chapters/09_baseline_forecasting/`](../chapters/09_baseline_forecasting/README.md). The suite does **not** re-run Optuna / `run_automl.py` (100 trials); thesis metrics stay validated via frozen `results/automl_results/`.

| Check | What it tests | Why |
| --- | --- | --- |
| `run_src_smoke.py` | Notebooks 00–03; inputs + `meteo_nl/`; processed panels/manifests; frozen AutoML JSON/CSVs; `py_compile` + import `automl_search`; ML deps (optuna/xgboost/catboost/shap/…) | Confirms the hand-in package and search code stay intact offline without a full model re-fit |
