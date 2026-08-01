# Reproducibility and Robustness Testing

Procedure **checks** for thesis chapter code—not a substitute for the full copyrighted corpus or bit-exact thesis LLMn outputs.

See also the chapter pipeline overview: [`chapters/04_database_construction/README.md`](../chapters/04_database_construction/README.md).

## Layout

```text
reproducibility_and_robustness_testing/
├── check_imports.py
├── run_all_scripts.py
└── chapter_04_database_construction/
    ├── data/raw/              # small fixture ZIP(s) for procedure tests
    ├── data/preprocessed/     # clean_archive outputs from the test run
    ├── data/llm_extracted/    # 2-article LLMn extraction outputs
    ├── run_acquisition.py     # ~20s headed Lexis viewer smoke
    ├── run_preprocessing.py   # visible clean_archive run + verify
    └── run_llmn_extraction.py # 2-article live extraction (API key prompt)
```

Later chapters: `chapter_05_.../`, `chapter_06_.../`, same pattern.

## What these checks demonstrate

- **Acquisition smoke** — headed browser path for the Lexis scraper (dummy login, short run). Full corpus downloads stay out of git for copyright.
- **Preprocessing** — runs `clean_archive` on a small fixture ZIP so the cleaning / MinHash dedup procedure can be verified without sharing the thesis corpus.
- **LLMn (2 articles)** — live API call to exercise schema + LiteLLM runner end-to-end. This is a smoke test, **not** bit-exact reproduction of thesis extractions (providers may change or shut down).

## Run

```text
# from repo root, with .venv activated
python reproducibility_and_robustness_testing/check_imports.py
python reproducibility_and_robustness_testing/run_all_scripts.py

# individual checks
python reproducibility_and_robustness_testing/chapter_04_database_construction/run_preprocessing.py
python reproducibility_and_robustness_testing/chapter_04_database_construction/run_llmn_extraction.py
python reproducibility_and_robustness_testing/chapter_04_database_construction/run_acquisition.py
```

`run_all_scripts.py` runs imports, preprocessing, **2-article LLMn extraction** (prompts for `GEMINI_API_KEY` if unset), then the headed Lexis viewer smoke.

## Chapter 04

| Check | Location |
| --- | --- |
| Acquisition viewer smoke (~20s, dummy login) | `run_acquisition.py` |
| Preprocessing on fixture ZIP | `run_preprocessing.py` (`data/raw` → `data/preprocessed`) |
| LLMn extraction (2 articles, live API) | `run_llmn_extraction.py` → `data/llm_extracted/` |

Full Lexis downloads / full-corpus extraction use the chapter scripts with real credentials / full `newsjson.json`.
