"""Post-process raw LLM input batches.

Archival reference only — do not run. Paths still point at an old external
layout and will not work in this repository. See ../README.md.

Historical behaviour (documented for provenance):
Reads batches from Data/input/Raw batches/, applies filtering rules, and writes
cleaned batches to Data/input/Post-processed batches/.

Filters (in order):
1. Remove articles with missing/invalid meta.date
2. Remove articles with text length > 50,000 characters
3. Remove global exact-text duplicates (keep first in batch_01..batch_10 order)
4. Remove fuzzy title near-duplicates via iterative medoid dedup (global)
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from fuzzy_title_dedup import ArticleRecord, collect_removals

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

INPUT_DIR = SCRIPT_DIR / "Data" / "input" / "Raw batches"
OUTPUT_DIR = SCRIPT_DIR / "Data" / "input" / "Post-processed batches"
MAX_TEXT_CHARS = 50_000


def get_article_id(article: dict) -> str:
    article_id = article.get("id")
    if article_id:
        return str(article_id)
    return str(article.get("features", {}).get("article_id", ""))


def get_article_text(article: dict) -> str:
    features = article.get("features", {})
    return str(features.get("clean_text") or article.get("text_content") or "").strip()


def get_article_title(article: dict) -> str:
    features = article.get("features", {})
    return str(features.get("title") or "").strip()


def parse_article_date(article: dict) -> date | None:
    raw = article.get("meta", {}).get("date", "")
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def date_range_for_batch(articles: list[dict]) -> tuple[str | None, str | None]:
    dates = [parse_article_date(article) for article in articles]
    valid_dates = [value for value in dates if value is not None]
    if not valid_dates:
        return None, None
    return min(valid_dates).isoformat(), max(valid_dates).isoformat()


def load_json_list(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def write_json_list(path: Path, articles: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(articles, handle, ensure_ascii=False, indent=2)


def log_removal(step: str, article_id: str, batch_name: str, detail: str = "") -> None:
    suffix = f" ({detail})" if detail else ""
    print(f"[{step}] removed {article_id} from {batch_name}{suffix}")


def post_process_batches() -> tuple[dict[str, list[dict]], dict[str, int], dict[str, int]]:
    seen_text_hashes: set[str] = set()
    stats = {
        "removed_no_date": 0,
        "removed_long_text": 0,
        "removed_exact_text_duplicate": 0,
        "kept": 0,
    }
    raw_counts: dict[str, int] = {}
    output_batches: dict[str, list[dict]] = {}

    batch_paths = sorted(
        path for path in INPUT_DIR.glob("batch_*.json") if path.name != "batch_manifest.json"
    )
    if not batch_paths:
        raise FileNotFoundError(f"No raw batch files found in {INPUT_DIR}")

    for batch_path in batch_paths:
        batch_name = batch_path.name
        articles = load_json_list(batch_path)
        raw_counts[batch_name] = len(articles)
        batch_stats = {
            "removed_no_date": 0,
            "removed_long_text": 0,
            "removed_exact_text_duplicate": 0,
        }
        kept: list[dict] = []

        for article in articles:
            article_id = get_article_id(article)
            if parse_article_date(article) is None:
                log_removal("no_date", article_id, batch_name)
                stats["removed_no_date"] += 1
                batch_stats["removed_no_date"] += 1
                continue

            text = get_article_text(article)
            if len(text) > MAX_TEXT_CHARS:
                log_removal("long_text", article_id, batch_name, f"text_chars={len(text)}")
                stats["removed_long_text"] += 1
                batch_stats["removed_long_text"] += 1
                continue

            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest() if text else ""
            if text_hash and text_hash in seen_text_hashes:
                log_removal(
                    "exact_text_duplicate",
                    article_id,
                    batch_name,
                    "duplicate of earlier article",
                )
                stats["removed_exact_text_duplicate"] += 1
                batch_stats["removed_exact_text_duplicate"] += 1
                continue

            if text_hash:
                seen_text_hashes.add(text_hash)
            kept.append(article)
            stats["kept"] += 1

        output_batches[batch_name] = kept
        print(
            f"Finished {batch_name}: kept {len(kept)}, "
            f"removed no_date={batch_stats['removed_no_date']}, "
            f"long_text={batch_stats['removed_long_text']}, "
            f"exact_text_duplicate={batch_stats['removed_exact_text_duplicate']}"
        )

    return output_batches, stats, raw_counts


def article_to_record(article: dict, batch_name: str) -> ArticleRecord:
    article_id = get_article_id(article)
    text = get_article_text(article)
    title = get_article_title(article)
    parsed_date = parse_article_date(article)
    date_sort_key = parsed_date.isoformat() if parsed_date is not None else ""
    return ArticleRecord(
        id=article_id,
        batch=batch_name,
        title=title,
        title_norm=title.lower(),
        text=text,
        text_chars=len(text),
        date_sort_key=date_sort_key,
    )


def apply_fuzzy_title_dedup(
    output_batches: dict[str, list[dict]],
) -> tuple[dict[str, list[dict]], dict[str, int], dict]:
    records: list[ArticleRecord] = []
    for batch_name, articles in output_batches.items():
        for article in articles:
            records.append(article_to_record(article, batch_name))

    ids_to_remove, fuzzy_stats = collect_removals(records)
    if not ids_to_remove:
        return output_batches, {"removed_fuzzy_title_duplicate": 0}, fuzzy_stats

    batch_stats = {"removed_fuzzy_title_duplicate": 0}
    filtered_batches: dict[str, list[dict]] = {}

    for batch_name, articles in output_batches.items():
        kept: list[dict] = []
        for article in articles:
            article_id = get_article_id(article)
            if article_id in ids_to_remove:
                log_removal(
                    "fuzzy_title_duplicate",
                    article_id,
                    batch_name,
                    "near-duplicate in fuzzy title cluster",
                )
                batch_stats["removed_fuzzy_title_duplicate"] += 1
                continue
            kept.append(article)
        filtered_batches[batch_name] = kept

    return filtered_batches, batch_stats, fuzzy_stats


def validate_output(raw_ids: set[str], output_batches: dict[str, list[dict]]) -> None:
    seen_ids: set[str] = set()
    for filename, articles in output_batches.items():
        batch_ids = {get_article_id(article) for article in articles}
        overlap = seen_ids & batch_ids
        if overlap:
            raise ValueError(f"{filename} overlaps with earlier batches ({len(overlap)} duplicate IDs)")
        seen_ids |= batch_ids

    if not seen_ids <= raw_ids:
        extra = seen_ids - raw_ids
        raise ValueError(f"Output contains {len(extra)} IDs not present in raw batches")


def build_manifest(
    output_batches: dict[str, list[dict]],
    stats: dict[str, int],
    raw_counts: dict[str, int],
    fuzzy_stats: dict | None = None,
) -> dict:
    manifest_batches = []
    for filename, articles in sorted(output_batches.items()):
        date_min, date_max = date_range_for_batch(articles)
        manifest_batches.append(
            {
                "filename": filename,
                "count": len(articles),
                "raw_count": raw_counts[filename],
                "removed": raw_counts[filename] - len(articles),
                "date_min": date_min,
                "date_max": date_max,
            }
        )

    return {
        "source": str(INPUT_DIR.relative_to(PROJECT_ROOT)),
        "split_strategy": "post_processed_from_raw_batches",
        "post_processing": {
            "removed_no_date": stats["removed_no_date"],
            "removed_long_text": stats["removed_long_text"],
            "removed_exact_text_duplicate": stats["removed_exact_text_duplicate"],
            "removed_fuzzy_title_duplicate": stats.get("removed_fuzzy_title_duplicate", 0),
            "max_text_chars": MAX_TEXT_CHARS,
            "exact_text_dedup_scope": "global",
            "fuzzy_title_sim_threshold": (fuzzy_stats or {}).get("title_sim_threshold"),
            "fuzzy_text_dedup_threshold": (fuzzy_stats or {}).get("text_dedup_threshold"),
            "fuzzy_min_text_chars_for_match": (fuzzy_stats or {}).get(
                "min_text_chars_for_match"
            ),
            "fuzzy_title_clusters": (fuzzy_stats or {}).get("clusters"),
            "fuzzy_title_clusters_with_removals": (fuzzy_stats or {}).get(
                "clusters_with_removals"
            ),
        },
        "total_articles": stats["kept"],
        "batch_count": len(manifest_batches),
        "batches": manifest_batches,
    }


def cleanup_stale_batch_files() -> None:
    if not OUTPUT_DIR.exists():
        return
    for path in OUTPUT_DIR.glob("batch_*.json"):
        if path.name == "batch_manifest.json":
            continue
        path.unlink()


def print_summary(manifest: dict, stats: dict[str, int]) -> None:
    print()
    print("Post-processing complete")
    print(f"  removed_no_date: {stats['removed_no_date']}")
    print(f"  removed_long_text: {stats['removed_long_text']}")
    print(f"  removed_exact_text_duplicate: {stats['removed_exact_text_duplicate']}")
    print(
        f"  removed_fuzzy_title_duplicate: {stats.get('removed_fuzzy_title_duplicate', 0)}"
    )
    print(f"  kept: {stats['kept']}")
    print()
    print(f"{'Batch':<16} {'Raw':>6} {'Post':>6} {'Removed':>8}")
    print("-" * 40)
    for entry in manifest["batches"]:
        print(
            f"{entry['filename']:<16} {entry['raw_count']:>6} "
            f"{entry['count']:>6} {entry['removed']:>8}"
        )
    print()
    print(f"Output directory: {OUTPUT_DIR}")


def main() -> None:
    raise SystemExit(
        "Archival reference only; do not run. "
        "See ../README.md (Reference/) and chapters/04_database_construction/README.md."
    )
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Raw batch input folder not found: {INPUT_DIR}")

    raw_ids: set[str] = set()
    for batch_path in sorted(INPUT_DIR.glob("batch_*.json")):
        if batch_path.name == "batch_manifest.json":
            continue
        for article in load_json_list(batch_path):
            raw_ids.add(get_article_id(article))

    output_batches, stats, raw_counts = post_process_batches()
    output_batches, fuzzy_batch_stats, fuzzy_stats = apply_fuzzy_title_dedup(output_batches)
    stats.update(fuzzy_batch_stats)
    stats["kept"] -= stats.get("removed_fuzzy_title_duplicate", 0)
    validate_output(raw_ids, output_batches)
    manifest = build_manifest(output_batches, stats, raw_counts, fuzzy_stats=fuzzy_stats)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_stale_batch_files()
    for filename, articles in output_batches.items():
        write_json_list(OUTPUT_DIR / filename, articles)
    write_json_list(OUTPUT_DIR / "batch_manifest.json", manifest)

    print_summary(manifest, stats)


if __name__ == "__main__":
    main()
