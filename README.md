# Impact-Based-Drought-Forecasting-in-the-Netherlands-Using-LLM-Extracted-News-Data

Code and resources accompanying an MSc thesis on impact-based drought forecasting using LLM-extracted news data, developed to ensure transparent and reproducible results.

## Repository structure

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── data/                              # repo-root shared data (see data/README.md)
├── chapters/
│   ├── 04_database_construction/      # Lexis → preprocess → LLMn
│   ├── 05_llm_evaluation/             # frozen eval metrics + report notebook
│   ├── 06_geocoding/                  # Nominatim points + NUTS-3 + viewer/EDA
│   ├── 07_visualization_reliability/
│   ├── 08_exploratory_data_analysis/
│   └── 09_baseline_forecasting/
└── reproducibility_and_robustness_testing/
    ├── check_imports.py
    ├── run_all_scripts.py             # PASS / SKIPPED / FAIL summary
    ├── chapter_04_database_construction/
    └── chapter_05_llm_evaluation/
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

# automated procedure checks (imports → ch05 → ch04; optional blank API key = skip)
python reproducibility_and_robustness_testing/run_all_scripts.py
```

`requirements.txt` covers Chapter 04–06 stacks (Playwright/LiteLLM, pandas/matplotlib/streamlit, and geopandas/shapely/pyproj/pydeck for geocoding).

## Chapter 04 (database construction)

Acquires LexisNexis article ZIPs, cleans and deduplicates them, then extracts structured drought impacts with a fixed schema via LiteLLM (multi-provider, batch-capable).

Details: [`chapters/04_database_construction/README.md`](chapters/04_database_construction/README.md).

```text
python chapters/04_database_construction/04_2_Automated Data Acquisition/lexis_nexis_scraper.py
python chapters/04_database_construction/04_3_LLM_Input_Preprocessing/clean_archive.py
python chapters/04_database_construction/04_4_LLMn_Extraction_Framework/llmn_extraction.py
```

## Chapter 05 (LLM evaluation)

Frozen annotations and metrics under [`chapters/05_llm_evaluation/`](chapters/05_llm_evaluation/README.md). Run `notebooks/report_chapter5.ipynb` to regenerate thesis tables/figures (no article bodies or API keys).

## Chapter 06 (geocoding)

Point geocoding (Nominatim) + NUTS-3 assignment, Streamlit viewer, and wildfire EDA under [`chapters/06_geocoding/`](chapters/06_geocoding/README.md). Frozen `data/processed/` CSVs support offline viewing without re-running Nominatim.

```text
cd chapters/06_geocoding
python src/point_coder.py
python src/nuts3_coder.py
streamlit run src/combined_viewer.py
```

## Reproducibility notes

- Full Lexis news corpora **cannot be redistributed** (copyright). The repo ships code, fixtures, and frozen evaluation/geocoding outputs—not the full thesis corpus.
- Chapter 04 **acquisition procedure** and **preprocessing** are reproducible as code; full downloads need Lexis credentials.
- **LLMn outputs** depend on third-party APIs; exact extractions are hard to bit-reproduce. Leave the Chapter 04 suite API prompt blank to **skip** the live call without failing the suite.
- Chapter 06 **NUTS-3 / viewer** paths are offline-reproducible from frozen files; **Nominatim** may change over time.

Procedure checks: [`reproducibility_and_robustness_testing/README.md`](reproducibility_and_robustness_testing/README.md).
