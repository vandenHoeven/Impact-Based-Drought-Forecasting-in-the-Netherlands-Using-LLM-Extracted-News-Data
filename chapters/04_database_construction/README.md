# Chapter 04: Database construction

Builds the drought-impact news database: acquire LexisNexis archives, clean and deduplicate them, then extract structured drought impacts with an LLM.

## What this does

1. **Acquisition (`04_2`)**: Playwright session against LexisNexis; credentials at the terminal (not hard-coded). Saves page-range ZIP archives of DOCX articles into `data/raw/`.
2. **Preprocessing (`04_3`)**: `clean_archive.py` unpacks DOCX files, extracts title / dates / body text, and runs MinHash + LSH near-duplicate removal.
3. **LLMn extraction (`04_4`)**: Fixed Pydantic schema and system prompt (`schemas.py`); runner (`llmn_extraction.py`) calls models via LiteLLM with batching, checkpoints, and resume. Writes enriched articles under `data/llm_extracted/`.

## Layout

```text
04_database_construction/
├── data/                          # shared chapter outputs (gitignored contents)
│   ├── raw/                       # Lexis ZIP downloads
│   ├── preprocessed/              # cleaned / deduplicated article JSON
│   └── llm_extracted/             # LLM-enriched articles
├── 04_2_automated_data_acquisition/
│   └── lexis_nexis_scraper.py
├── 04_3_LLM_Input_Preprocessing/
│   └── clean_archive.py
└── 04_4_LLMn_Extraction_Framework/
    ├── schemas.py
    ├── llmn_extraction.py
    ├── .env.example
    └── Reference/                   # archival only — do not run
        ├── README.md
        ├── old_schema.py
        └── post-processing/
            ├── fuzzy_title_dedup.py
            └── post_proccesing_batches.py
```

All three stages share one data tree under `data/`.

## Historical reference (archival only)

[`04_4_LLMn_Extraction_Framework/Reference/`](04_4_LLMn_Extraction_Framework/Reference/) keeps an earlier flat extraction schema and the second-stage batch post-processing scripts used before production LLMn runs. **Do not run this code.** It is not part of the current pipeline, is not smoke-tested, and hard-coded paths still point at an old external layout. Details: [`04_4_LLMn_Extraction_Framework/Reference/README.md`](04_4_LLMn_Extraction_Framework/Reference/README.md).

## How to run

From the repository root, with `.venv` activated and dependencies installed (`pip install -r requirements.txt`; `playwright install chromium` once):

```text
python chapters/04_database_construction/04_2_automated_data_acquisition/lexis_nexis_scraper.py
python chapters/04_database_construction/04_3_LLM_Input_Preprocessing/clean_archive.py

# copy .env.example → .env and set GEMINI_API_KEY (or export it)
python chapters/04_database_construction/04_4_LLMn_Extraction_Framework/llmn_extraction.py
```

Optional extraction flags: `--mode test-one --index 0` or `--limit N`. Schema and prompt: `04_4_LLMn_Extraction_Framework/schemas.py`.

## Data and provenance

| Use this | Path | Meaning |
| --- | --- | --- |
| **Required outputs (local / gitignored)** | `data/raw/`, `data/preprocessed/`, `data/llm_extracted/` | Pipeline products; full Lexis corpus not shipped |
| **Schema / prompt** | `04_4_LLMn_Extraction_Framework/schemas.py` | Fixed extraction contract for LLMn |
| **Env template** | `04_4_LLMn_Extraction_Framework/.env.example` | Copy to `.env` for API keys |
| Robustness fixture | `reproducibility_and_robustness_testing/chapter_04_database_construction/data/` | Small ZIP / smoke outputs (not the thesis corpus) |

## Reproducibility and limits

- Full Lexis news corpora **cannot be redistributed** (copyright). This repository ships the code and a small robustness fixture, not the thesis corpus.
- The **acquisition procedure** is reproducible as code; full downloads need credentials and are not shared here.
- **Preprocessing is reproducible** given the same ZIP inputs.
- **LLMn outputs are hard to fully reproduce**: they depend on third-party model APIs that may change behaviour, pricing, or availability; exact outputs can also vary between runs.
- The **extraction framework** (schema, prompt, runner) is largely time-invariant as research code, but may need updates when providers, LiteLLM, or model IDs change.

## Links

- Downstream evaluation: [`../05_llm_evaluation/README.md`](../05_llm_evaluation/README.md)
- Downstream spatial post-processing: [`../06_spatial_postprocessing_visualization_dataset_reliability/README.md`](../06_spatial_postprocessing_visualization_dataset_reliability/README.md)
- Robustness suite (fixture ZIP, 2-article live extraction, headed Lexis smoke): [`../../reproducibility_and_robustness_testing/README.md`](../../reproducibility_and_robustness_testing/README.md)
