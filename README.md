# Impact-Based-Drought-Forecasting-in-the-Netherlands-Using-LLM-Extracted-News-Data

Code and resources accompanying an MSc thesis on impact-based drought forecasting using LLM-extracted news data, developed to ensure transparent and reproducible results.

> **Data catalogue: what is / isn’t shipped**  
> Sources, sizes, and what is shipped for every major dataset: **[`DATA.md`](DATA.md)**

## Quick orientation

This repo is the **code + frozen artefacts** for the thesis pipeline.

| Goal | Where to look |
| --- | --- |
| Run procedure checks | `python reproducibility_and_robustness_testing/run_all_scripts.py` |
| Frozen Ch05 tables / figures | `chapters/05_llm_evaluation/results/` |
| Thesis-final geocoded points / NUTS-3 | `chapters/06_spatial_postprocessing_visualization_dataset_reliability/data/final/` |
| Frozen Ch07 EDA figures / tables | `chapters/07_exploratory_data_analysis/results/` |
| Frozen Ch08–09 AutoML metrics | `chapters/08_09_baseline_forecasting/results/automl_results/` |

Pipeline (data flow between chapters):

```mermaid
flowchart LR
  ch04[Ch04_extract] --> ch05[Ch05_evaluate]
  ch04 --> ch06[Ch06_spatial]
  ch06 --> ch07[Ch07_EDA]
  ch06 --> ch0809[Ch08_09_forecast]
```

## Repository structure

```text
.
├── README.md
├── .gitignore
├── DATA.md                            # data overview / catalogue
├── requirements.txt
├── chapters/
│   ├── 04_database_construction/      # Lexis → preprocess → LLMn
│   ├── 05_llm_evaluation/             # frozen eval metrics + report notebook
│   ├── 06_spatial_postprocessing_visualization_dataset_reliability/  # spatial post-processing + viz + reliability
│   ├── 07_exploratory_data_analysis/  # report EDA figures + tables
│   └── 08_09_baseline_forecasting/    # Ch08 methods + Ch09 NUTS-3 AutoML
└── reproducibility_and_robustness_testing/
    ├── check_imports.py
    ├── run_all_scripts.py             # PASS / SKIPPED / FAIL summary
    ├── chapter_04_database_construction/
    ├── chapter_05_llm_evaluation/
    ├── chapter_06_spatial_postprocessing_visualization_dataset_reliability/
    ├── chapter_07_exploratory_data_analysis/
    └── chapter_08_09_baseline_forecasting/
```

## Environment setup

```text
# create (first time)
python -m venv .venv

# activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# install or refresh dependencies after editing requirements.txt
python -m pip install -U pip
pip install -r requirements.txt

playwright install chromium   # once, after playwright is added

# automated procedure checks (imports → ch04–09 smokes; optional blank API key = skip)
python reproducibility_and_robustness_testing/run_all_scripts.py
```

`requirements.txt` covers Chapters 04–09 stacks (Playwright/LiteLLM, pandas/matplotlib/streamlit/geopandas, plus seaborn/networkx for Ch07 and scikit-learn/optuna/shap/xgboost/rasterio/pyarrow for Ch08–09).

## Chapter 04 (database construction)

Acquires LexisNexis article ZIPs, cleans and deduplicates them, then extracts structured drought impacts with a fixed schema via LiteLLM (multi-provider, batch-capable). Historical schema / batch post-processing under `04_4_LLMn_Extraction_Framework/Reference/` is archival documentation only (do not run).

Details: [`chapters/04_database_construction/README.md`](chapters/04_database_construction/README.md).

```text
python chapters/04_database_construction/04_2_automated_data_acquisition/lexis_nexis_scraper.py
python chapters/04_database_construction/04_3_LLM_Input_Preprocessing/clean_archive.py
python chapters/04_database_construction/04_4_LLMn_Extraction_Framework/llmn_extraction.py
```

## Chapter 05 (LLM evaluation)

Frozen annotations and metrics under [`chapters/05_llm_evaluation/`](chapters/05_llm_evaluation/README.md). Run `notebooks/report_chapter5.ipynb` to regenerate thesis tables/figures (no article bodies or API keys).

## Chapter 06 (spatial post-processing, visualization, dataset reliability)

Point geocoding (Nominatim) + NUTS-3 assignment, Streamlit viewer, and wildfire EDA under [`chapters/06_spatial_postprocessing_visualization_dataset_reliability/`](chapters/06_spatial_postprocessing_visualization_dataset_reliability/README.md). Frozen `data/final/` CSVs support offline viewing without re-running Nominatim.

```text
cd chapters/06_spatial_postprocessing_visualization_dataset_reliability
python src/point_coder.py
python src/nuts3_coder.py
streamlit run src/combined_viewer.py
```

## Chapter 07 (exploratory data analysis)

Report figures and Overleaf summary tables under [`chapters/07_exploratory_data_analysis/`](chapters/07_exploratory_data_analysis/README.md). Run `notebooks/01_report_final_figures.ipynb` (current-month NUTS-3 impacts, 2005–2025).

## Chapters 08–09 (baseline forecasting)

NUTS-3 multi-label AutoML (methods + forecasting) under [`chapters/08_09_baseline_forecasting/`](chapters/08_09_baseline_forecasting/README.md). Notebooks `00`–`03` + `src/run_automl.py`; frozen metrics in `results/automl_results/`. Large DEM/BRO/CLC rasters are kept **locally** for rebuilds but gitignored (see that README).

## Reproducibility notes

- Full Lexis news corpora **cannot be redistributed** (copyright). The repo ships code, fixtures, and frozen evaluation/geocoding outputs, not the full thesis corpus.
- Chapter 04 **acquisition procedure** and **preprocessing** are reproducible as code; full downloads need Lexis credentials.
- **LLMn outputs** depend on third-party APIs; exact extractions are hard to bit-reproduce. Leave the Chapter 04 suite API prompt blank to **skip** the live call without failing the suite.
- Chapter 06 **NUTS-3 / viewer** paths are offline-reproducible from frozen files; **Nominatim** may change over time.
- Chapters 07–09 ship frozen figures/metrics; full Ch08–09 static rebuilds need local gitignored rasters under `08_09_baseline_forecasting/data/`.

Procedure checks: [`reproducibility_and_robustness_testing/README.md`](reproducibility_and_robustness_testing/README.md).
