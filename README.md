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
│   ├── 05_llm_evaluation/
│   ├── 06_geocoding/
│   ├── 07_visualization_reliability/
│   ├── 08_exploratory_data_analysis/
│   └── 09_baseline_forecasting/
├── scripts/
│   └── check_imports.py
└── docs/
    └── chapter_code_map.md
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

# verify: syntax + third-party imports used by all .py files
python scripts/check_imports.py
```
