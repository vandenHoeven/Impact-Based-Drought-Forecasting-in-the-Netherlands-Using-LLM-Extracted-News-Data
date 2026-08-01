# Reproducibility and Robustness Testing

Top-level entrypoints plus one folder per thesis chapter (data + tests together).

## Layout

```text
reproducibility_and_robustness_testing/
├── check_imports.py
├── run_all_scripts.py
└── chapter_04_database_construction/
    ├── data/raw/              # chapter fixture ZIPs
    ├── data/preprocessed/     # clean_archive outputs from the test run
    ├── run_acquisition.py     # ~20s headed Lexis viewer smoke
    └── run_preprocessing.py   # visible clean_archive run + verify
```

Later chapters: `chapter_05_.../`, `chapter_06_.../`, same pattern.

## Run

```text
# from repo root, with .venv activated
python reproducibility_and_robustness_testing/check_imports.py
python reproducibility_and_robustness_testing/run_all_scripts.py

# 04_2 acquisition: opens headed Chromium for ~20s with dummy credentials
python reproducibility_and_robustness_testing/chapter_04_database_construction/run_acquisition.py

# 04_3 preprocessing only (prints pipeline logs, writes data/preprocessed/, TEST COMPLETE)
python reproducibility_and_robustness_testing/chapter_04_database_construction/run_preprocessing.py
```

`run_all_scripts.py` runs import checks, each `chapter_*/run_preprocessing.py`, and each
`chapter_*/run_acquisition.py` (headed Chromium opens on your screen for ~20s).

## Chapter 04

| Check | Location |
| --- | --- |
| Acquisition viewer smoke (~20s, dummy login) | `run_acquisition.py` |
| Preprocessing on fixture ZIP | `run_preprocessing.py` (`data/raw` → `data/preprocessed`) |

Full Lexis downloads still use `chapters/.../lexis_nexis_scraper.py` with real credentials.

LLMn extraction tests are deferred.
