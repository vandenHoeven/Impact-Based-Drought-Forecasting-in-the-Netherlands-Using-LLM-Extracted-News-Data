"""Restrict Chapter 5 post-hoc reviews to the nine shared articles.

The full manual archive remains in data/posthoc.json. This module filters
impact-validity and attribute rows to the articles reviewed in both passes,
recomputes tables, and reconstructs usable-location counts by subtracting
the potato-article location TPs from the published thesis usable totals
(not a re-filter of place names).

    python chapters/05_llm_evaluation/src/restrict_posthoc.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CHAPTER_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CHAPTER_ROOT / "data"
RESULTS_TAB = CHAPTER_ROOT / "results" / "tables"

SHARED_ARTICLE_IDS: frozenset[str] = frozenset(
    {
        "NEWS_2003_022690",
        "NEWS_2006_024445",
        "NEWS_2015_007768",
        "NEWS_2017_009248",
        "NEWS_2018_010689",
        "NEWS_2018_010835",
        "NEWS_2018_010479",
        "NEWS_2019_013729",
        "NEWS_2020_016427",
    }
)

DROPPED_ARTICLES: dict[str, str] = {
    "NEWS_2018_011316": (
        "Irrelevant lagoon / public-health edge case; present in impact-validity "
        "for four models, not in Claude impact rows"
    ),
    "NEWS_2020_018637": (
        "Potato-yield article; present in the attribute review only"
    ),
}

POTATO_ARTICLE_ID = "NEWS_2020_018637"

POSTHOC_DISPLAY: dict[str, str] = {
    "anthropic_claude-sonnet-4-5": "Claude Sonnet 4.5",
    "gemini_gemini-3.5-flash": "Gemini 3.5 Flash",
    "gemini_gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
    "gemini_gemini-3.1-pro-preview": "Gemini 3.1 Pro Preview",
    "gpt-5.4-mini": "GPT-5.4 Mini",
}

ATTRIBUTE_MODELS: tuple[str, ...] = (
    "gemini_gemini-3.5-flash",
    "anthropic_claude-sonnet-4-5",
)

# Published thesis usable-location table (full attribute sample, including potato).
PUBLISHED_USABLE_LOCATION: dict[str, dict[str, int]] = {
    "gemini_gemini-3.5-flash": {"loc_tp": 17, "loc_fn": 5},
    "anthropic_claude-sonnet-4-5": {"loc_tp": 14, "loc_fn": 5},
}

EXPECTED_POTATO_LOCATION_TP: dict[str, int] = {
    "gemini_gemini-3.5-flash": 3,
    "anthropic_claude-sonnet-4-5": 4,
}

EXPECTED_IMPACT: dict[str, tuple[int, int]] = {
    "anthropic_claude-sonnet-4-5": (26, 0),
    "gemini_gemini-3.5-flash": (26, 0),
    "gemini_gemini-3.1-pro-preview": (18, 3),
    "gemini_gemini-3.1-flash-lite": (16, 2),
    "gpt-5.4-mini": (27, 11),
}

EXPECTED_ATTRIBUTE_RAW: dict[str, dict[str, int]] = {
    "anthropic_claude-sonnet-4-5": {
        "loc_tp": 13,
        "loc_fn": 0,
        "loc_count": 15,
        "sev_tp": 21,
        "sev_fn": 5,
        "rec_tp": 11,
        "rec_fn": 0,
    },
    "gemini_gemini-3.5-flash": {
        "loc_tp": 13,
        "loc_fn": 0,
        "loc_count": 16,
        "sev_tp": 21,
        "sev_fn": 0,
        "rec_tp": 11,
        "rec_fn": 0,
    },
}

LOCATION_RECONSTRUCTION_NOTE = (
    "Usable-location TP is the published thesis usable count minus this model's "
    "location_tp on NEWS_2020_018637. FN is left at the published value (5). "
    "Place names were not re-filtered. Severity and recency are summed from the "
    "nine shared attribute_labels rows."
)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _article_id(entry: dict[str, Any]) -> str:
    return str(entry.get("article_id") or "").strip()


def filter_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in entries if _article_id(e) in SHARED_ARTICLE_IDS]


def _ids_by_model(entries: list[dict[str, Any]]) -> dict[str, set[str]]:
    by_model: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        model = str(entry.get("model_name") or "")
        aid = _article_id(entry)
        if model and aid:
            by_model[model].add(aid)
    return by_model


def potato_location_tp(full_attribute_entries: list[dict[str, Any]]) -> dict[str, int]:
    counts = {model: 0 for model in ATTRIBUTE_MODELS}
    for entry in full_attribute_entries:
        if _article_id(entry) != POTATO_ARTICLE_ID:
            continue
        model = str(entry.get("model_name") or "")
        if model in counts:
            counts[model] = int(entry.get("location_tp") or 0)
    return counts


def build_impact_precision(restricted_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    agg = {model: {"tp": 0, "fp": 0} for model in POSTHOC_DISPLAY}
    for entry in restricted_entries:
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


def _sum_attribute_fields(entries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {
        model: {
            "reviewed_rows": 0,
            "location_count": 0,
            "location_tp": 0,
            "location_fn": 0,
            "severity_tp": 0,
            "severity_fn": 0,
            "recency_tp": 0,
            "recency_fn": 0,
        }
        for model in ATTRIBUTE_MODELS
    }
    for entry in entries:
        model = str(entry.get("model_name") or "")
        if model not in totals:
            continue
        bucket = totals[model]
        bucket["reviewed_rows"] += 1
        bucket["location_count"] += int(entry.get("location_count") or 0)
        bucket["location_tp"] += int(entry.get("location_tp") or 0)
        bucket["location_fn"] += int(entry.get("location_fn") or 0)
        bucket["severity_tp"] += int(entry.get("severity_tp") or 0)
        bucket["severity_fn"] += int(entry.get("severity_fn") or 0)
        bucket["recency_tp"] += int(entry.get("recency_tp") or 0)
        bucket["recency_fn"] += int(entry.get("recency_fn") or 0)
    return totals


def _recall(tp: int, fn: int) -> float | None:
    denom = tp + fn
    if denom <= 0:
        return None
    return round(tp / denom, 4)


def build_attribute_raw_rows(totals: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_id in ATTRIBUTE_MODELS:
        item = totals[model_id]
        rows.append(
            {
                "model_id": model_id,
                "loc_tp_raw": item["location_tp"],
                "loc_fn_raw": item["location_fn"],
                "loc_count_raw": item["location_count"],
                "sev_tp_raw": item["severity_tp"],
                "sev_fn_raw": item["severity_fn"],
                "rec_tp_raw": item["recency_tp"],
                "rec_fn_raw": item["recency_fn"],
                "note": "raw_json_nine_shared_articles",
            }
        )
    return rows


def build_reconstructed_attributes(
    totals: dict[str, dict[str, int]],
    potato_tp: dict[str, int],
) -> list[dict[str, Any]]:
    display = {
        "gemini_gemini-3.5-flash": "Gemini 3.5 Flash",
        "anthropic_claude-sonnet-4-5": "Claude Sonnet 4.5",
    }
    rows: list[dict[str, Any]] = []
    for model_id, name in display.items():
        published = PUBLISHED_USABLE_LOCATION[model_id]
        loc_tp = published["loc_tp"] - potato_tp[model_id]
        loc_fn = published["loc_fn"]
        item = totals[model_id]
        rows.append(
            {
                "model": name,
                "loc_tp": loc_tp,
                "loc_fn": loc_fn,
                "sev_tp": item["severity_tp"],
                "sev_fn": item["severity_fn"],
                "rec_tp": item["recency_tp"],
                "rec_fn": item["recency_fn"],
                "source": "reconstructed_usable_location_minus_potato; sev_rec_from_nine_article_json",
            }
        )
    return rows


def build_attribute_summary(
    totals: dict[str, dict[str, int]],
    updated_at: str,
) -> dict[str, Any]:
    grand = {
        "location_count": 0,
        "location_tp": 0,
        "location_fn": 0,
        "severity_tp": 0,
        "severity_fn": 0,
        "recency_tp": 0,
        "recency_fn": 0,
        "total_rows": 0,
        "updated_at": updated_at,
    }
    model_summary: list[dict[str, Any]] = []
    for model_id in ATTRIBUTE_MODELS:
        item = totals[model_id]
        grand["location_count"] += item["location_count"]
        grand["location_tp"] += item["location_tp"]
        grand["location_fn"] += item["location_fn"]
        grand["severity_tp"] += item["severity_tp"]
        grand["severity_fn"] += item["severity_fn"]
        grand["recency_tp"] += item["recency_tp"]
        grand["recency_fn"] += item["recency_fn"]
        grand["total_rows"] += item["reviewed_rows"]
        loc_recall = _recall(item["location_tp"], item["location_fn"])
        sev_recall = _recall(item["severity_tp"], item["severity_fn"])
        rec_recall = _recall(item["recency_tp"], item["recency_fn"])
        recalls = [r for r in (loc_recall, sev_recall, rec_recall) if r is not None]
        overall = round(sum(recalls) / len(recalls), 4) if recalls else None
        model_summary.append(
            {
                "model_name": model_id,
                "run_count": 1,
                "reviewed_rows": item["reviewed_rows"],
                "total_location_count": item["location_count"],
                "total_location_tp": item["location_tp"],
                "total_location_fn": item["location_fn"],
                "total_severity_tp": item["severity_tp"],
                "total_severity_fn": item["severity_fn"],
                "total_recency_tp": item["recency_tp"],
                "total_recency_fn": item["recency_fn"],
                "location_recall": loc_recall,
                "severity_recall": sev_recall,
                "recency_recall": rec_recall,
                "overall_score": overall,
            }
        )
    return {"updated_at": updated_at, "totals": grand, "model_summary": model_summary}


def assert_restricted_counts(
    impact_entries: list[dict[str, Any]],
    attribute_entries: list[dict[str, Any]],
    impact_rows: list[dict[str, Any]],
    attribute_totals: dict[str, dict[str, int]],
    potato_tp: dict[str, int],
) -> None:
    expected_ids = set(SHARED_ARTICLE_IDS)
    impact_ids = _ids_by_model(impact_entries)
    for model_id in POSTHOC_DISPLAY:
        got = impact_ids.get(model_id, set())
        if got != expected_ids:
            raise ValueError(
                f"Impact labels for {model_id} cover {sorted(got)}, "
                f"expected {sorted(expected_ids)}"
            )

    attr_ids = _ids_by_model(attribute_entries)
    for model_id in ATTRIBUTE_MODELS:
        got = attr_ids.get(model_id, set())
        if got != expected_ids:
            raise ValueError(
                f"Attribute labels for {model_id} cover {sorted(got)}, "
                f"expected {sorted(expected_ids)}"
            )

    by_id = {row["model_id"]: row for row in impact_rows}
    for model_id, (tp, fp) in EXPECTED_IMPACT.items():
        row = by_id[model_id]
        if int(row["TP"]) != tp or int(row["FP"]) != fp:
            raise ValueError(
                f"Impact totals for {model_id}: got TP={row['TP']} FP={row['FP']}, "
                f"expected TP={tp} FP={fp}"
            )

    for model_id, expected in EXPECTED_ATTRIBUTE_RAW.items():
        item = attribute_totals[model_id]
        checks = {
            "location_tp": expected["loc_tp"],
            "location_fn": expected["loc_fn"],
            "location_count": expected["loc_count"],
            "severity_tp": expected["sev_tp"],
            "severity_fn": expected["sev_fn"],
            "recency_tp": expected["rec_tp"],
            "recency_fn": expected["rec_fn"],
        }
        for field, want in checks.items():
            got = item[field]
            if got != want:
                raise ValueError(
                    f"Attribute {field} for {model_id}: got {got}, expected {want}"
                )

    for model_id, want in EXPECTED_POTATO_LOCATION_TP.items():
        got = potato_tp.get(model_id, 0)
        if got != want:
            raise ValueError(
                f"Potato location_tp for {model_id}: got {got}, expected {want}"
            )


def build_restricted_payload(
    full_posthoc: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    impact_all = list(full_posthoc.get("impact_labels", {}).get("review_entries") or [])
    attribute_all = list(full_posthoc.get("attribute_labels", {}).get("review_entries") or [])
    impact_entries = filter_entries(impact_all)
    attribute_entries = filter_entries(attribute_all)
    potato_tp = potato_location_tp(attribute_all)
    impact_rows = build_impact_precision(impact_entries)
    attribute_totals = _sum_attribute_fields(attribute_entries)
    raw_rows = build_attribute_raw_rows(attribute_totals)
    reconstructed_rows = build_reconstructed_attributes(attribute_totals, potato_tp)
    assert_restricted_counts(
        impact_entries,
        attribute_entries,
        impact_rows,
        attribute_totals,
        potato_tp,
    )

    updated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "restriction": {
            "kept_article_ids": sorted(SHARED_ARTICLE_IDS),
            "dropped_articles": DROPPED_ARTICLES,
            "n_kept": len(SHARED_ARTICLE_IDS),
            "source_archive": "posthoc.json",
            "location_reconstruction": LOCATION_RECONSTRUCTION_NOTE,
            "potato_location_tp": potato_tp,
            "published_usable_location": PUBLISHED_USABLE_LOCATION,
        },
        "impact_labels": {
            "updated_at": updated_at,
            "review_entries": impact_entries,
        },
        "attribute_labels": {
            "updated_at": updated_at,
            "review_entries": attribute_entries,
        },
        "attribute_summary": build_attribute_summary(attribute_totals, updated_at),
        "tables": {
            "impact_precision": impact_rows,
            "attributes_raw": raw_rows,
            "attributes_reconstructed": reconstructed_rows,
        },
    }
    return payload, impact_rows, raw_rows, reconstructed_rows


def restrict_from_archive(full_posthoc: dict[str, Any] | None = None) -> dict[str, Any]:
    if full_posthoc is None:
        full_posthoc = load_json(DATA_DIR / "posthoc.json")
    payload, _, _, _ = build_restricted_payload(full_posthoc)
    return payload


def main() -> int:
    full_posthoc = load_json(DATA_DIR / "posthoc.json")
    payload, impact_rows, raw_rows, reconstructed_rows = build_restricted_payload(full_posthoc)

    restricted_path = DATA_DIR / "posthoc_restricted.json"
    save_json(restricted_path, payload)
    write_csv(
        RESULTS_TAB / "posthoc_impact_precision.csv",
        impact_rows,
        ["model", "model_id", "TP", "FP", "precision"],
    )
    write_csv(
        RESULTS_TAB / "posthoc_attributes_raw.csv",
        raw_rows,
        [
            "model_id",
            "loc_tp_raw",
            "loc_fn_raw",
            "loc_count_raw",
            "sev_tp_raw",
            "sev_fn_raw",
            "rec_tp_raw",
            "rec_fn_raw",
            "note",
        ],
    )
    write_csv(
        RESULTS_TAB / "posthoc_attributes.csv",
        reconstructed_rows,
        ["model", "loc_tp", "loc_fn", "sev_tp", "sev_fn", "rec_tp", "rec_fn", "source"],
    )

    print(f"Wrote {restricted_path}")
    print(f"Wrote {RESULTS_TAB / 'posthoc_impact_precision.csv'}")
    print(f"Wrote {RESULTS_TAB / 'posthoc_attributes_raw.csv'}")
    print(f"Wrote {RESULTS_TAB / 'posthoc_attributes.csv'}")
    for row in impact_rows:
        print(
            f"  impact {row['model']}: TP={row['TP']} FP={row['FP']} "
            f"P={row['precision']:.3f}"
        )
    for row in reconstructed_rows:
        print(
            f"  attr {row['model']}: loc={row['loc_tp']}/{row['loc_fn']} "
            f"sev={row['sev_tp']}/{row['sev_fn']} rec={row['rec_tp']}/{row['rec_fn']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
