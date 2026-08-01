# Impact-Based-Drought-Forecasting-in-the-Netherlands-Using-LLM-Extracted-News-Data

Code and resources accompanying an MSc thesis on impact-based drought forecasting using LLM-extracted news data, developed to ensure transparent and reproducible results.

## Repository structure

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── chapters/
│   ├── 04_database_construction/   # see chapters/04_database_construction/README.md
│   ├── 05_llm_evaluation/
│   ├── 06_geocoding/
│   ├── 07_visualization_reliability/
│   ├── 08_exploratory_data_analysis/
│   └── 09_baseline_forecasting/
└── reproducibility_and_robustness_testing/
    ├── check_imports.py
    ├── run_all_scripts.py
    └── chapter_04_database_construction/
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

# automated procedure checks (imports → preprocessing → 2-article LLMn → viewer smoke)
python reproducibility_and_robustness_testing/run_all_scripts.py
```

## Chapter 04 (database construction)

Chapter 04 acquires LexisNexis article ZIPs, cleans and deduplicates them (accounting for acquisition artefacts), then extracts structured drought impacts with a fixed schema via LiteLLM so the same pipeline can target multiple providers and batch large corpora.

Details, layout, and run commands: [`chapters/04_database_construction/README.md`](chapters/04_database_construction/README.md).

Shared pipeline data lives under `chapters/04_database_construction/data/` (`raw/` → `preprocessed/` → `llm_extracted/`).

```text
python chapters/04_database_construction/04_2_Automated Data Acquisition/lexis_nexis_scraper.py
python chapters/04_database_construction/04_3_LLM_Input_Preprocessing/clean_archive.py
python chapters/04_database_construction/04_4_LLMn_Extraction_Framework/llmn_extraction.py
```

## Reproducibility notes

- Full Lexis news corpora **cannot be redistributed** (copyright). The repo ships code and a small test fixture, not the thesis corpus.
- The **acquisition procedure** and **preprocessing** are reproducible as code (same steps / same ZIPs → same cleaned JSON); full downloads still need Lexis credentials.
- **LLMn outputs** depend on third-party APIs that may change or disappear; exact extractions are hard to bit-reproduce. The schema/runner framework is largely time-invariant but may need updates when providers or model IDs change.

Procedure checks: [`reproducibility_and_robustness_testing/README.md`](reproducibility_and_robustness_testing/README.md).
