# Chapter 05 — LLM evaluation

Evaluation of structured drought-impact extraction quality on a manually annotated event-level dataset. Thesis tables and figures are reproduced from **frozen annotations/metrics** (no article bodies required).

## Layout

```text
05_llm_evaluation/
  data/
    evaluation_set.json   # 65 articles, 33 events (annotations)
    model_metrics.json    # event-level P/R/F1 per model
    posthoc.json          # post-hoc impact + attribute labels
  notebooks/
    report_chapter5.ipynb # ALL calculations, displays, and exports
  src/
    label_dataset.py      # Streamlit labeller (optional)
    run_models.py         # API runner (optional; needs keys + private corpus)
  results/                # written by the notebook
```

## Reproduce thesis numbers

Open and run **all cells** in `notebooks/report_chapter5.ipynb`.

The notebook includes **method narrative** (aligned with thesis Chapter 5) plus file provenance. It:

1. Loads the three JSON files under `data/`
2. Computes dataset counts and plots severity / recency / impact
3. Builds the performance table from `model_metrics.json` (and checks F1 = 2PR/(P+R))
4. Aggregates post-hoc impact precision and attribute tables
5. Writes outputs under `results/figures/` and `results/tables/`

No API keys or newspaper article bodies are needed for this path.

## Robustness / reproducibility checks

Automated checks live under [`reproducibility_and_robustness_testing/chapter_05_llm_evaluation/`](../../reproducibility_and_robustness_testing/chapter_05_llm_evaluation/). There is **no live LLM API** in this suite (Chapter 04 already covers extraction).

| Check | What it tests | Why |
| --- | --- | --- |
| `run_evaluation_report.py` | Frozen JSON + F1 consistency + golden CSV match | Thesis tables stay reproducible offline |
| `run_src_smoke.py` | Script import + Chapter 04 schema / `llmn_extraction` wiring | Optional tools still resolve without calling providers |
| `run_labeller_smoke.py` | Launch Streamlit UI → health OK → shut down | Annotation UI still starts (synthetic fixture; no full corpus) |

```text
python reproducibility_and_robustness_testing/chapter_05_llm_evaluation/run_evaluation_report.py
python reproducibility_and_robustness_testing/chapter_05_llm_evaluation/run_src_smoke.py
python reproducibility_and_robustness_testing/chapter_05_llm_evaluation/run_labeller_smoke.py
```

## Thesis mapping

| Report item | Where in notebook / output |
|-------------|----------------------------|
| Final evaluation dataset (§5.1.3) | Section 2 → `results/tables/dataset_summary.csv` |
| Fig. severity / recency / impact | Section 2 → `results/figures/*.png` |
| Event-level model table | Section 3 → `results/tables/model_performance.csv` |
| Model costs table | Section 3 → `results/tables/model_costs.csv` |
| Post-hoc impact precision | Section 4 → `results/tables/posthoc_impact_precision.csv` |
| Attribute Loc/Sev/Rec table | Section 5 → `results/tables/posthoc_attributes.csv` |
| Fuzzy-set concept figures | Overleaf only — see `results/figures/README.md` |

## Optional tools

These are **not** required to reproduce the thesis tables. They need the private Chapter 04 corpus (not in git) and API keys.

```bash
streamlit run chapters/05_llm_evaluation/src/label_dataset.py
python chapters/05_llm_evaluation/src/run_models.py --model "gemini/gemini-3.5-flash"
```

Defaults (overridable via env / CLI):

| Setting | Default |
| --- | --- |
| Schema | `chapters/04_database_construction/04_4_LLMn_Extraction_Framework/schemas.py` |
| Source corpus | `chapters/04_database_construction/data/preprocessed/all_articles_deduplicated.json` |
| Extraction runner | Chapter 04 `llmn_extraction.run_extraction` via LiteLLM |

Env overrides: `EVAL_BUILDER_LLM_SCHEMA`, `EVAL_BUILDER_SOURCE_JSON`, `EVAL_BUILDER_OUTPUT_DIR`, `DROUGHT_MONOREPO_ROOT`.  
Full article dumps from `run_models.py` go under `data/runs/` (gitignored).

See also Chapter 04: [`../04_database_construction/README.md`](../04_database_construction/README.md).

## Copyright

Newspaper article bodies are **not** included. Only annotations and metrics are frozen here. Event-level TP/FP/FN matching was done offline; `model_metrics.json` stores the resulting scores (the notebook verifies F1 from stored P and R).
