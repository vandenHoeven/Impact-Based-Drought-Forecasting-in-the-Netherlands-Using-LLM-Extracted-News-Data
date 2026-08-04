# Chapter 04 — Database construction

This chapter builds the drought-impact news database used later in the thesis: acquire LexisNexis archives, clean and deduplicate them, then extract structured drought impacts with an LLM.

## What the code does

**Acquisition (04_2).** Articles are downloaded with an automated Playwright browser session against LexisNexis. Credentials are entered at the terminal (not hard-coded). The scraper walks result pages and saves page-range ZIP archives of DOCX articles into the shared `data/raw/` folder. The **acquisition procedure itself is fully reproducible**—same scripted steps and download layout every time—provided you have a valid Lexis subscription. The robustness suite demonstrates a short headed viewer smoke of this path; it does not redistribute the full corpus.

**Preprocessing and deduplication (04_3).** After download, `clean_archive.py` unpacks DOCX files from every ZIP, extracts features such as title, publication dates, and cleaned body text, and runs **MinHash + LSH near-duplicate removal** over the whole corpus. That step accounts for **artefacts of the acquisition method**—overlapping page downloads, re-exported articles, and near-identical wire copies—so later LLM calls are not wasted on duplicates. Given the same ZIP inputs, this stage is deterministic and reproducible.

**LLMn extraction (04_4).** Structured extraction uses a fixed **Pydantic schema and system prompt** (`schemas.py`) that define drought-impact events and labels. The runner (`llmn_extraction.py`) calls models through **LiteLLM**, so the **same schema can be used across multiple providers and models** without rewriting the pipeline. The design supports **batch processing** of large article sets (concurrency, rate limits, checkpoints, and resume). Enriched articles are written under `data/llm_extracted/`.

## What is in this folder

```text
04_database_construction/
├── data/                          # shared chapter outputs (gitignored contents)
│   ├── raw/                       # Lexis ZIP downloads
│   ├── preprocessed/              # cleaned / deduplicated article JSON
│   └── llm_extracted/             # LLM-enriched articles
├── 04_2_Automated Data Acquisition/
│   └── lexis_nexis_scraper.py
├── 04_3_LLM_Input_Preprocessing/
│   └── clean_archive.py
└── 04_4_LLMn_Extraction_Framework/
    ├── schemas.py
    ├── llmn_extraction.py
    └── .env.example
```

All three stages share one data tree under `data/`. No copying between subchapters is required.

| Stage | Folder | Produced by |
| --- | --- | --- |
| Lexis ZIP archives | `data/raw/` | `04_2` scraper |
| Cleaned / deduplicated articles (`newsjson.json`, …) | `data/preprocessed/` | `04_3` `clean_archive.py` |
| LLM-enriched articles | `data/llm_extracted/` | `04_4` `llmn_extraction.py` |

## How to run

From the repository root, with `.venv` activated and dependencies installed (`pip install -r requirements.txt`; `playwright install chromium` once):

```text
python chapters/04_database_construction/04_2_Automated Data Acquisition/lexis_nexis_scraper.py
python chapters/04_database_construction/04_3_LLM_Input_Preprocessing/clean_archive.py

# copy .env.example → .env and set GEMINI_API_KEY (or export it)
python chapters/04_database_construction/04_4_LLMn_Extraction_Framework/llmn_extraction.py
```

Optional extraction flags: `--mode test-one --index 0` or `--limit N`. Schema and prompt: `04_4_LLMn_Extraction_Framework/schemas.py`.

## Reproducibility and limits

- Full Lexis news corpora **cannot be redistributed** (copyright). This repository ships the code and a small robustness fixture, not the thesis corpus.
- The **acquisition procedure** is reproducible as code; full downloads need credentials and are not shared here.
- **Preprocessing is reproducible** given the same ZIP inputs: the same cleaned / deduplicated JSON can be rebuilt.
- **LLMn outputs are hard to fully reproduce**: they depend on third-party model APIs that may change behaviour, pricing, or shut down services at any time; exact outputs can also vary between runs.
- The **extraction framework** (schema, prompt, runner) is largely time-invariant as research code, but may need occasional updates when providers, LiteLLM, or model IDs change.

Procedure checks (fixture ZIP, 2-article live extraction, headed Lexis smoke): see [`reproducibility_and_robustness_testing/README.md`](../../reproducibility_and_robustness_testing/README.md).

Next spatial step (geocoding extracted locations): [`../06_geocoding/README.md`](../06_geocoding/README.md). Evaluation of extraction quality: [`../05_llm_evaluation/README.md`](../05_llm_evaluation/README.md).
