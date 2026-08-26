# Historical reference (archival only)

This folder documents code from an earlier thesis workflow. **Do not run it.** It is not part of the current Chapter 04 pipeline and is excluded from the robustness / import smoke suite.

## Contents

| Path | What it was |
| --- | --- |
| `old_schema.py` | Earlier flat extraction schema: one impact label, severity, recency, and evidence per event (plus prompt). Kept for comparison with the current hierarchical schema in [`../schemas.py`](../schemas.py). |
| `post-processing/post_proccesing_batches.py` | Second-stage batch filters applied before production LLMn runs: missing/invalid `meta.date`, text longer than 50,000 characters, global exact-text MD5 duplicates, then fuzzy-title clustering with body-text near-duplicate removal. |
| `post-processing/fuzzy_title_dedup.py` | Helper used by the batch post-processor (title Jaccard clustering + iterative medoid text dedup). |

Hard-coded paths inside the post-processing scripts still point at an old external project layout (`Data/input/...`). They will not work in this repository.

## Current pipeline

Use only:

- [`../schemas.py`](../schemas.py) — live Pydantic schema and system prompt
- [`../llmn_extraction.py`](../llmn_extraction.py) — live extraction runner
- [`../../04_3_LLM_Input_Preprocessing/clean_archive.py`](../../04_3_LLM_Input_Preprocessing/clean_archive.py) — MinHash / LSH cleaning (first-stage dedup)

See [`../../README.md`](../../README.md).
