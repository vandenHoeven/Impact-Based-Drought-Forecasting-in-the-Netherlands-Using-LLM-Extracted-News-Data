from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _SRC_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from geocode_ranking import enrich_geocode_fields
from geocoder import DEFAULT_DELAY, DEFAULT_LIMIT, geocode_location

PACKAGE_ROOT = _PACKAGE_ROOT
DEFAULT_INPUT = PACKAGE_ROOT / "data" / "input" / "impacts_for_geocoding.json"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "data" / "processed"
OUT_JSON_NAME = "impacts_geocoded.json"
OUT_CSV_NAME = "impacts_geocoded_points.csv"

GEO_FIELDS = [
    "geocoded_latitude",
    "geocoded_longitude",
    "geocoded_display_name",
    "geocoded_place_type",
    "geocoded_place_level",
    "geocoded_is_broad",
    "geocoded_country_code",
    "geocoded_osm_id",
    "geocoded_boundingbox",
    "geocoded_place_rank",
    "geocoded_is_low_quality",
]

CSV_FIELDS = [
    "id",
    "article_id",
    "model_name",
    "publication_date",
    "source_file",
    "impact_index",
    "location",
    "classification",
    "severity",
    "confidence",
    "evidence",
    "recency_in_months",
    *GEO_FIELDS,
]


def process_json(
    input_path: Path,
    delay: float,
    limit: int = DEFAULT_LIMIT,
) -> list[dict]:
    records = json.loads(input_path.read_text(encoding="utf-8"))

    cache: dict[str, dict | None] = {}
    total_impacts = sum(
        len(r.get("features", {}).get("llm_drought_impacts") or [])
        for r in records
    )
    done = 0

    for record in records:
        features = record.get("features") or {}
        impacts = features.get("llm_drought_impacts") or []
        for impact in impacts:
            location = (impact.get("location") or "").strip()
            if location not in cache:
                geo = geocode_location(location, delay, limit=limit)
                if geo:
                    geo.update(enrich_geocode_fields(geo.get("geocoded_place_type"), location=location))
                cache[location] = geo
                if geo:
                    place_type = geo.get("geocoded_place_type") or "unknown"
                    status = (
                        f"{geo['geocoded_latitude']}, {geo['geocoded_longitude']} ({place_type})"
                    )
                else:
                    status = "no result"
                print(f"  [{done + 1}/{total_impacts}] {location!r} -> {status}")
            else:
                geo = cache[location]

            if geo:
                impact.update(geo)
            else:
                for key in GEO_FIELDS:
                    impact[key] = None
            done += 1

    return records


def save_json(records: list[dict], output_path: Path) -> None:
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON saved -> {output_path}")


def save_csv(records: list[dict], output_path: Path) -> None:
    rows = []
    for record in records:
        features = record.get("features") or {}
        pub_date = features.get("publication_date", "")
        source_file = features.get("source_file", "")
        meta = record.get("meta") or {}

        impacts = features.get("llm_drought_impacts") or []
        for idx, impact in enumerate(impacts):
            for sub_impact in (impact.get("impacts") or [{}]):
                rows.append({
                    "id": record.get("id", ""),
                    "article_id": record.get("article_id", ""),
                    "model_name": record.get("model_name", ""),
                    "publication_date": pub_date or meta.get("date", ""),
                    "source_file": source_file,
                    "impact_index": idx,
                    "location": impact.get("location", ""),
                    "classification": sub_impact.get("classification", ""),
                    "severity": sub_impact.get("severity", ""),
                    "confidence": sub_impact.get("confidence", ""),
                    "evidence": sub_impact.get("evidence", ""),
                    "recency_in_months": impact.get("recency_in_months", ""),
                    **{k: impact.get(k) for k in GEO_FIELDS},
                })

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved  -> {output_path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geocode location mentions in LLM drought-impact JSON output (Nominatim)."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT,
        help="Path to the LLM output JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output files (default: chapter_06/data/processed).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Seconds to wait between Nominatim API calls (default: 1.0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Number of Nominatim candidates to fetch per query (default: 5).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing the flat CSV output.",
    )
    args = parser.parse_args()

    input_path: Path = args.input.resolve()
    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print(f"Delay:  {args.delay}s")
    print(f"Limit:  {args.limit}")
    print()

    records = process_json(input_path, args.delay, limit=args.limit)

    save_json(records, output_dir / OUT_JSON_NAME)

    if not args.no_csv:
        save_csv(records, output_dir / OUT_CSV_NAME)


# Example:
# python src/point_coder.py
# python src/point_coder.py data/input/impacts_for_geocoding.json --output-dir data/processed
if __name__ == "__main__":
    main()
