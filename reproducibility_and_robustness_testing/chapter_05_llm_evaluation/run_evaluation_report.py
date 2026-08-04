"""
Chapter 05 offline reproducibility check.

Recomputes thesis tables from frozen JSON under chapters/05_llm_evaluation/data/,
checks F1 consistency, compares to golden CSVs in results/tables/, and writes
recomputed copies under this folder's data/results/.

No API keys or article bodies required.

    python reproducibility_and_robustness_testing/chapter_05_llm_evaluation/run_evaluation_report.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

CHAPTER_TEST_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CHAPTER_TEST_ROOT.parents[1]
CHAPTER_ROOT = REPO_ROOT / "chapters" / "05_llm_evaluation"
DATA_DIR = CHAPTER_ROOT / "data"
GOLDEN_TAB = CHAPTER_ROOT / "results" / "tables"
OUT_DIR = CHAPTER_TEST_ROOT / "data" / "results"

EXPECTED_REVIEWS = 65
EXPECTED_EVENTS = 33
EXPECTED_RUNS = 5
F1_TOL = 1e-9
CSV_FLOAT_TOL = 1e-6

DISPLAY_NAMES = {
    "gemini_gemini-3.5-flash": "Gemini 3.5 Flash",
    "gemini_gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "anthropic_claude-sonnet-4-5": "Claude Sonnet 4.5",
    "gemini_gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
    "gpt-5.4-mini": "GPT-5.4 mini",
}

POSTHOC_DISPLAY = {
    "anthropic_claude-sonnet-4-5": "Claude Sonnet 4.5",
    "gemini_gemini-3.5-flash": "Gemini 3.5 Flash",
    "gemini_gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
    "gemini_gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
    "gpt-5.4-mini": "GPT-5.4 Mini",
}

# Thesis attribute table applies a manual spatial-usability filter (not recomputed from JSON).
THESIS_ATTRIBUTES = [
    {
        "model": "Gemini 3.5 Flash",
        "loc_tp": 17,
        "loc_fn": 5,
        "sev_tp": 24,
        "sev_fn": 0,
        "rec_tp": 14,
        "rec_fn": 0,
        "source": "thesis_table_after_spatial_usability_filter",
    },
    {
        "model": "Claude Sonnet 4.5",
        "loc_tp": 14,
        "loc_fn": 5,
        "sev_tp": 29,
        "sev_fn": 5,
        "rec_tp": 14,
        "rec_fn": 1,
        "source": "thesis_table_after_spatial_usability_filter",
    },
]


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _f1(p: float, r: float) -> float:
    if p + r == 0:
        return float("nan")
    return 2.0 * p * r / (p + r)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def _values_close(a: Any, b: Any, tol: float = CSV_FLOAT_TOL) -> bool:
    try:
        fa, fb = _as_float(a), _as_float(b)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()
    if math.isnan(fa) and math.isnan(fb):
        return True
    if math.isnan(fa) or math.isnan(fb):
        return False
    # Integers stored without decimals should still match
    return abs(fa - fb) <= tol


def _compare_tables(
    name: str,
    recomputed: list[dict[str, Any]],
    golden_path: Path,
    key_fields: list[str],
) -> None:
    if not golden_path.is_file():
        raise FileNotFoundError(f"Missing golden CSV: {golden_path}")
    golden = _read_csv(golden_path)
    if len(recomputed) != len(golden):
        raise ValueError(
            f"{name}: row count mismatch recomputed={len(recomputed)} golden={len(golden)}"
        )
    for i, (rec, gold) in enumerate(zip(recomputed, golden)):
        for field in key_fields:
            if field not in gold:
                raise ValueError(f"{name}: golden missing column {field}")
            if not _values_close(rec.get(field), gold.get(field)):
                raise ValueError(
                    f"{name}: mismatch row={i} field={field} "
                    f"recomputed={rec.get(field)!r} golden={gold.get(field)!r}"
                )
    print(f"Matched golden CSV: {golden_path.name} ({len(recomputed)} rows)")


def check_json_integrity(evaluation: dict, metrics: dict, posthoc: dict) -> None:
    for key in (
        "article_reviews",
        "event_records",
        "review_count",
        "relevant_count",
        "irrelevant_count",
        "uncertain_count",
        "event_summary",
    ):
        if key not in evaluation:
            raise ValueError(f"evaluation_set.json missing key: {key}")
    if "run_summaries" not in metrics:
        raise ValueError("model_metrics.json missing run_summaries")
    if "impact_labels" not in posthoc or "attribute_summary" not in posthoc:
        raise ValueError("posthoc.json missing impact_labels or attribute_summary")

    n_reviews = len(evaluation["article_reviews"])
    n_events = len(evaluation["event_records"])
    n_runs = len(metrics["run_summaries"])
    if n_reviews != EXPECTED_REVIEWS:
        raise ValueError(f"Expected {EXPECTED_REVIEWS} reviews, got {n_reviews}")
    if n_events != EXPECTED_EVENTS:
        raise ValueError(f"Expected {EXPECTED_EVENTS} events, got {n_events}")
    if n_runs != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} model runs, got {n_runs}")
    if evaluation.get("review_count") != n_reviews:
        raise ValueError("review_count does not match article_reviews length")
    print(
        f"JSON integrity OK: {n_reviews} reviews, {n_events} events, {n_runs} model runs"
    )


def check_f1_consistency(metrics: dict) -> None:
    failures: list[str] = []
    for row in metrics["run_summaries"]:
        model = row.get("model", "<unknown>")
        for prefix in ("location_impact", "location_impact_severity_recency"):
            p = float(row[f"{prefix}_precision"])
            r = float(row[f"{prefix}_recall"])
            f_stored = float(row[f"{prefix}_f1"])
            f_re = _f1(p, r)
            if not math.isclose(f_stored, f_re, rel_tol=0.0, abs_tol=F1_TOL):
                failures.append(
                    f"{model} {prefix}: stored={f_stored} from_PR={f_re} "
                    f"diff={abs(f_stored - f_re)}"
                )
    if failures:
        raise ValueError("F1 consistency failed:\n  " + "\n  ".join(failures))
    print("F1 consistency OK for all model runs (loc+impact and full-tuple)")


def build_dataset_summary(evaluation: dict) -> list[dict[str, Any]]:
    reviews = evaluation["article_reviews"]
    events = evaluation["event_records"]
    label_counts = Counter(str(r.get("label", "")) for r in reviews)
    impact_classes = {
        str(e.get("impact_class", "")).strip()
        for e in events
        if str(e.get("impact_class", "")).strip()
    }
    rows: list[dict[str, Any]] = [
        {"metric": "reviewed_articles", "value": len(reviews)},
        {"metric": "extracted_events", "value": len(events)},
        {"metric": "distinct_impact_classes", "value": len(impact_classes)},
    ]
    # Preserve notebook order: pandas value_counts (descending count, then first-seen)
    for lab, cnt in label_counts.most_common():
        if lab:
            rows.append({"metric": f"label_{lab}", "value": int(cnt)})
    return rows


def build_model_performance(metrics: dict) -> list[dict[str, Any]]:
    by_model = {row["model"]: row for row in metrics["run_summaries"]}
    rows: list[dict[str, Any]] = []
    for model_id, name in DISPLAY_NAMES.items():
        if model_id not in by_model:
            raise ValueError(f"Missing run summary for model_id={model_id}")
        r = by_model[model_id]
        rows.append(
            {
                "model": name,
                "model_id": model_id,
                "location_impact_precision": round(r["location_impact_precision"], 3),
                "location_impact_recall": round(r["location_impact_recall"], 3),
                "location_impact_f1": round(r["location_impact_f1"], 3),
                "location_impact_severity_recency_precision": round(
                    r["location_impact_severity_recency_precision"], 3
                ),
                "location_impact_severity_recency_recall": round(
                    r["location_impact_severity_recency_recall"], 3
                ),
                "location_impact_severity_recency_f1": round(
                    r["location_impact_severity_recency_f1"], 3
                ),
            }
        )
    return rows


def build_posthoc_impact_precision(posthoc: dict) -> list[dict[str, Any]]:
    entries = posthoc["impact_labels"]["review_entries"]
    agg = {m: {"tp": 0, "fp": 0} for m in POSTHOC_DISPLAY}
    for entry in entries:
        model = entry.get("model_name")
        if model not in agg:
            continue
        agg[model]["tp"] += int(entry.get("tp_count") or 0)
        agg[model]["fp"] += int(entry.get("fp_count") or 0)

    rows: list[dict[str, Any]] = []
    for model_id, name in POSTHOC_DISPLAY.items():
        tp, fp = agg[model_id]["tp"], agg[model_id]["fp"]
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        rows.append(
            {
                "model": name,
                "model_id": model_id,
                "TP": tp,
                "FP": fp,
                "precision": round(precision, 3),
            }
        )
    return rows


def build_posthoc_attributes_raw(posthoc: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in posthoc["attribute_summary"].get("model_summary", []):
        rows.append(
            {
                "model_id": item["model_name"],
                "loc_tp_raw": item["total_location_tp"],
                "loc_fn_raw": item["total_location_fn"],
                "sev_tp_raw": item["total_severity_tp"],
                "sev_fn_raw": item["total_severity_fn"],
                "rec_tp_raw": item["total_recency_tp"],
                "rec_fn_raw": item["total_recency_fn"],
                "note": "raw_json_before_thesis_spatial_usability_filter",
            }
        )
    return rows


def main() -> int:
    print("=" * 70)
    print("Chapter 05 evaluation report reproducibility check (offline)")
    print(f"data:   {DATA_DIR}")
    print(f"golden: {GOLDEN_TAB}")
    print(f"output: {OUT_DIR}")
    print("=" * 70)

    try:
        evaluation = _load_json(DATA_DIR / "evaluation_set.json")
        metrics = _load_json(DATA_DIR / "model_metrics.json")
        posthoc = _load_json(DATA_DIR / "posthoc.json")

        check_json_integrity(evaluation, metrics, posthoc)
        check_f1_consistency(metrics)

        dataset_summary = build_dataset_summary(evaluation)
        model_performance = build_model_performance(metrics)
        posthoc_precision = build_posthoc_impact_precision(posthoc)
        attributes_raw = build_posthoc_attributes_raw(posthoc)
        thesis_attributes = list(THESIS_ATTRIBUTES)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        _write_csv(
            OUT_DIR / "dataset_summary.csv",
            dataset_summary,
            ["metric", "value"],
        )
        _write_csv(
            OUT_DIR / "model_performance.csv",
            model_performance,
            [
                "model",
                "model_id",
                "location_impact_precision",
                "location_impact_recall",
                "location_impact_f1",
                "location_impact_severity_recency_precision",
                "location_impact_severity_recency_recall",
                "location_impact_severity_recency_f1",
            ],
        )
        _write_csv(
            OUT_DIR / "posthoc_impact_precision.csv",
            posthoc_precision,
            ["model", "model_id", "TP", "FP", "precision"],
        )
        _write_csv(
            OUT_DIR / "posthoc_attributes_raw.csv",
            attributes_raw,
            [
                "model_id",
                "loc_tp_raw",
                "loc_fn_raw",
                "sev_tp_raw",
                "sev_fn_raw",
                "rec_tp_raw",
                "rec_fn_raw",
                "note",
            ],
        )
        _write_csv(
            OUT_DIR / "posthoc_attributes.csv",
            thesis_attributes,
            ["model", "loc_tp", "loc_fn", "sev_tp", "sev_fn", "rec_tp", "rec_fn", "source"],
        )
        print(f"Wrote recomputed tables under {OUT_DIR}")

        golden_summary = {
            row["metric"]: row["value"] for row in _read_csv(GOLDEN_TAB / "dataset_summary.csv")
        }
        recomputed_summary = {row["metric"]: row["value"] for row in dataset_summary}
        if set(golden_summary) != set(recomputed_summary):
            raise ValueError(
                f"dataset_summary key mismatch: "
                f"extra={set(recomputed_summary) - set(golden_summary)} "
                f"missing={set(golden_summary) - set(recomputed_summary)}"
            )
        for key, value in recomputed_summary.items():
            if not _values_close(value, golden_summary[key]):
                raise ValueError(
                    f"dataset_summary mismatch for {key}: "
                    f"recomputed={value!r} golden={golden_summary[key]!r}"
                )
        print("Matched golden CSV: dataset_summary.csv")

        _compare_tables(
            "model_performance",
            model_performance,
            GOLDEN_TAB / "model_performance.csv",
            [
                "model",
                "model_id",
                "location_impact_precision",
                "location_impact_recall",
                "location_impact_f1",
                "location_impact_severity_recency_precision",
                "location_impact_severity_recency_recall",
                "location_impact_severity_recency_f1",
            ],
        )
        _compare_tables(
            "posthoc_impact_precision",
            posthoc_precision,
            GOLDEN_TAB / "posthoc_impact_precision.csv",
            ["model", "model_id", "TP", "FP", "precision"],
        )
        _compare_tables(
            "posthoc_attributes_raw",
            attributes_raw,
            GOLDEN_TAB / "posthoc_attributes_raw.csv",
            [
                "model_id",
                "loc_tp_raw",
                "loc_fn_raw",
                "sev_tp_raw",
                "sev_fn_raw",
                "rec_tp_raw",
                "rec_fn_raw",
                "note",
            ],
        )
        _compare_tables(
            "posthoc_attributes",
            thesis_attributes,
            GOLDEN_TAB / "posthoc_attributes.csv",
            ["model", "loc_tp", "loc_fn", "sev_tp", "sev_fn", "rec_tp", "rec_fn", "source"],
        )

        costs_path = GOLDEN_TAB / "model_costs.csv"
        if not costs_path.is_file() or costs_path.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing manual costs table: {costs_path}")
        print(f"Found manual costs table: {costs_path.name}")

    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("TEST COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
