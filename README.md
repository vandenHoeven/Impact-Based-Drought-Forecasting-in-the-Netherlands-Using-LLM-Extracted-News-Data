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
│   ├── 04_database_construction/
│   │   ├── data/
│   │   │   ├── raw/
│   │   │   ├── preprocessed/
│   │   │   └── llm_extracted/
│   │   ├── 04_2_Automated Data Acquisition/
│   │   ├── 04_3_LLM_Input_Preprocessing/
│   │   └── 04_4_LLMn_Extraction_Framework/
│   ├── 05_llm_evaluation/
│   ├── 06_geocoding/
│   ├── 07_visualization_reliability/
│   ├── 08_exploratory_data_analysis/
│   └── 09_baseline_forecasting/
└── reproducibility_and_robustness_testing/
    ├── check_imports.py
    ├── run_all_scripts.py
    └── chapter_04_database_construction/
        ├── data/raw/
        ├── data/preprocessed/
        ├── run_acquisition.py
        └── run_preprocessing.py
```

## Environment setup and smoke test

After editing `requirements.txt` (or cloning on a fresh machine), create or refresh a virtual environment and verify that third-party imports used by all `.py` files resolve. This does **not** run scrapers, call APIs, or execute chapter pipelines.

```text
# create (first time)
python -m venv .venv

# activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# install or refresh dependencies after editing requirements.txt
python -m pip install -U pip
pip install -r requirements.txt

playwright install chromium   # once, after playwright is added

# automated reproducibility checks (imports + preprocessing + headed Lexis viewer smoke)
python reproducibility_and_robustness_testing/run_all_scripts.py

# 04_2 acquisition check (~20s headed Lexis viewer smoke; dummy credentials)
python reproducibility_and_robustness_testing/chapter_04_database_construction/run_acquisition.py

# 04_3 preprocessing only (visible clean_archive run → data/preprocessed/)
python reproducibility_and_robustness_testing/chapter_04_database_construction/run_preprocessing.py
```

## Chapter 04 shared data folder

All chapter 04 scripts read/write a single shared tree:

`chapters/04_database_construction/data/`

| Stage | Folder | Produced by |
| --- | --- | --- |
| Lexis ZIP archives | `data/raw/` | `04_2` scraper |
| Cleaned / deduplicated articles (`newsjson.json`, …) | `data/preprocessed/` | `04_3` `clean_archive.py` |
| LLM-enriched articles | `data/llm_extracted/` | `04_4` `llmn_extraction.py` |

No copying between subchapters is required.

```text
python chapters/04_database_construction/04_2_Automated Data Acquisition/lexis_nexis_scraper.py
python chapters/04_database_construction/04_3_LLM_Input_Preprocessing/clean_archive.py

# set GEMINI_API_KEY in 04_4_LLMn_Extraction_Framework/.env
python chapters/04_database_construction/04_4_LLMn_Extraction_Framework/llmn_extraction.py
```

Schema and system prompt: `04_4_LLMn_Extraction_Framework/schemas.py`.  
Optional: `--mode test-one --index 0` or `--limit N` on the extraction runner.

See also [`reproducibility_and_robustness_testing/README.md`](reproducibility_and_robustness_testing/README.md).
