# Reproducibility and Robustness Testing

Procedure **checks** for thesis chapter code—not a substitute for the full copyrighted corpus or bit-exact thesis LLMn outputs.

See also:
- [`chapters/04_database_construction/README.md`](../chapters/04_database_construction/README.md)
- [`chapters/05_llm_evaluation/README.md`](../chapters/05_llm_evaluation/README.md)

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
└── chapter_05_llm_evaluation/
    ├── data/labeller_fixture/   # synthetic articles for UI smoke
    ├── data/labeller_state/     # labeller runtime state (gitignored)
    ├── data/results/            # recomputed tables from offline check
    ├── run_evaluation_report.py
    ├── run_src_smoke.py
    └── run_labeller_smoke.py
```

## Run

```text
# from repo root, with .venv activated
python reproducibility_and_robustness_testing/run_all_scripts.py

# Chapter 05 only
python reproducibility_and_robustness_testing/chapter_05_llm_evaluation/run_evaluation_report.py
python reproducibility_and_robustness_testing/chapter_05_llm_evaluation/run_src_smoke.py
python reproducibility_and_robustness_testing/chapter_05_llm_evaluation/run_labeller_smoke.py
```

`run_all_scripts.py` order: imports → Chapter 05 checks → Chapter 04 preprocessing → 2-article LLMn → headed Lexis viewer.

At the end it prints a **Robustness-check summary** with `PASS` / `SKIPPED` / `FAIL` for every runner (plus totals and a list of failed checks). The suite exits non-zero only when one or more checks are `FAIL`; skipped optional checks do not fail the suite.

### Optional live API (Chapter 04 LLMn)

`run_llmn_extraction.py` prompts for `GEMINI_API_KEY` when unset:

- Enter a key → live 2-article extraction runs (`PASS` if successful).
- Leave blank / press Enter → check is **SKIPPED** (no API call); this does **not** fail `run_all_scripts.py`.
- If the key is already in the environment, the live call runs without prompting.

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
