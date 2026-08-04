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
from geocoder import DEFAULT_DELAY, DEFAULT_LIMIT, classify_place_level, geocode_location
from nuts_mapper import NutsContext, assign_nuts, load_nuts_context

SCRIPT_DIR = _SRC_DIR
PACKAGE_ROOT = _PACKAGE_ROOT
DEFAULT_GEOCODED_JSON = PACKAGE_ROOT / "data" / "processed" / "impacts_geocoded.json"
DEFAULT_GEOCODED_CSV = PACKAGE_ROOT / "data" / "processed" / "impacts_geocoded_points.csv"
DEFAULT_LLM_JSON = PACKAGE_ROOT / "data" / "input" / "impacts_for_geocoding.json"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "data" / "processed"
DEFAULT_GEOJSON = PACKAGE_ROOT / "data" / "geo" / "nuts_nl_simplified.geojson"

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

NUTS_FIELDS = [
    "nuts_level",
    "nuts2_id",
    "nuts2_name",
    "nuts3_id",
    "nuts3_name",
    "nuts_match_distance_m",
    "source_nuts_level",
    "mention_weight",
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
    *NUTS_FIELDS,
]


def _empty_geo() -> dict[str, Any]:
    return {key: None for key in GEO_FIELDS}


def _empty_nuts() -> dict[str, Any]:
    return {
        "nuts_level": "none",
        "nuts2_id": "",
        "nuts2_name": "",
        "nuts3_id": "",
        "nuts3_name": "",
        "nuts_match_distance_m": "",
        "source_nuts_level": "none",
        "mention_weight": 1.0,
    }


def _format_status(
    geo: dict[str, Any] | None,
    assignment: dict[str, Any],
    assignments: list[dict[str, Any]] | None = None,
) -> str:
    if not geo:
        return "no result"
    lat = geo.get("geocoded_latitude")
    lon = geo.get("geocoded_longitude")
    place_type = geo.get("geocoded_place_type") or "unknown"
    level = assignment.get("nuts_level", "none")
    source = assignment.get("source_nuts_level", level)
    if level == "nuts3":
        nuts3_id = assignment.get("nuts3_id", "")
        nuts3_name = assignment.get("nuts3_name", "")
        if assignments and len(assignments) > 1:
            return (
                f"NL nuts3 x{len(assignments)} ({source}) "
                f"e.g. {nuts3_id} ({nuts3_name}) via {place_type} @ {lat}, {lon}"
            )
        return f"NL nuts3 {nuts3_id} ({nuts3_name}) [{source}] via {place_type} @ {lat}, {lon}"
    if level == "non_nl":
        return f"non_nl @ {lat}, {lon}"
    if level == "country":
        return f"country (no NUTS) @ {lat}, {lon}"
    if level == "nuts2":
        return f"nuts2 {assignment.get('nuts2_id', '')} via {place_type} @ {lat}, {lon}"
    return f"{level} @ {lat}, {lon} ({place_type})"


def _complete_geo_fields(
    geo: dict[str, Any] | None,
    location: str = "",
) -> dict[str, Any] | None:
    """Fill place_level / rank fields if missing (e.g. older point CSVs)."""
    if not geo:
        return None
    out = dict(geo)
    lat = out.get("geocoded_latitude")
    lon = out.get("geocoded_longitude")
    if lat in (None, "") or lon in (None, ""):
        return None

    place_type = out.get("geocoded_place_type") or ""
    if not out.get("geocoded_place_level"):
        place_level = classify_place_level(str(place_type))
        out["geocoded_place_level"] = place_level
        out["geocoded_is_broad"] = place_level in {"country", "region"}
    elif "geocoded_is_broad" not in out or out.get("geocoded_is_broad") in (None, ""):
        out["geocoded_is_broad"] = out.get("geocoded_place_level") in {"country", "region"}

    if out.get("geocoded_place_rank") in (None, "") or out.get("geocoded_is_low_quality") in (None, ""):
        out.update(enrich_geocode_fields(place_type or None, location=location))

    return out


def _geo_from_mapping(row: dict[str, Any], location: str = "") -> dict[str, Any] | None:
    geo = {key: row.get(key) for key in GEO_FIELDS if key in row or row.get(key) is not None}
    # Always pull core coords / type even if GEO_FIELDS absent in sparse rows
    for key in GEO_FIELDS:
        if key not in geo:
            geo[key] = row.get(key)
    return _complete_geo_fields(geo, location=location)


def _resolve_location(
    location: str,
    cache: dict[str, tuple[dict[str, Any] | None, list[dict[str, Any]]]],
    ctx: NutsContext,
    delay: float,
    limit: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if location not in cache:
        geo = geocode_location(location, delay=delay, limit=limit)
        if geo:
            geo.update(enrich_geocode_fields(geo.get("geocoded_place_type"), location=location))
        assignments = assign_nuts(geo, ctx, location=location)
        cache[location] = (geo, assignments)
    return cache[location]


def _resolve_from_geo(
    location: str,
    geo: dict[str, Any] | None,
    cache: dict[str, tuple[dict[str, Any] | None, list[dict[str, Any]]]],
    ctx: NutsContext,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if location not in cache:
        completed = _complete_geo_fields(geo, location=location)
        assignments = assign_nuts(completed, ctx, location=location)
        cache[location] = (completed, assignments)
    return cache[location]


def _apply_to_impact(
    impact: dict[str, Any],
    geo: dict[str, Any] | None,
    assignments: list[dict[str, Any]],
) -> None:
    if geo:
        impact.update(geo)
    else:
        impact.update(_empty_geo())

    primary = assignments[0] if assignments else _empty_nuts()
    impact.update(primary)

    if len(assignments) > 1:
        impact["nuts3_assignments"] = [
            {
                "nuts3_id": a.get("nuts3_id", ""),
                "nuts3_name": a.get("nuts3_name", ""),
                "mention_weight": a.get("mention_weight", 1.0),
            }
            for a in assignments
            if a.get("nuts3_id")
        ]
    elif "nuts3_assignments" in impact:
        del impact["nuts3_assignments"]


def process_geocoded_json(input_path: Path, ctx: NutsContext) -> list[dict]:
    """Assign NUTS-3 using geocode fields already present on each impact (no Nominatim)."""
    records = json.loads(input_path.read_text(encoding="utf-8"))

    cache: dict[str, tuple[dict[str, Any] | None, list[dict[str, Any]]]] = {}
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
            if location:
                if location not in cache:
                    geo = _geo_from_mapping(impact, location=location)
                    geo, assignments = _resolve_from_geo(location, geo, cache, ctx)
                    status = _format_status(geo, assignments[0], assignments)
                    print(f"  [{len(cache)}/{total_impacts}] {location!r} -> {status}")
                else:
                    geo, assignments = cache[location]
                _apply_to_impact(impact, geo, assignments)
            else:
                _apply_to_impact(impact, None, [_empty_nuts()])
            done += 1

    return records


def process_llm_json(
    input_path: Path,
    ctx: NutsContext,
    delay: float,
    limit: int = DEFAULT_LIMIT,
) -> list[dict]:
    """Legacy path: Nominatim geocode + NUTS assign (prefer point_coder then assign-only)."""
    records = json.loads(input_path.read_text(encoding="utf-8"))

    cache: dict[str, tuple[dict[str, Any] | None, list[dict[str, Any]]]] = {}
    total_impacts = sum(
        len(r.get("features", {}).get("llm_drought_impacts") or [])
        for r in records
    )

    for record in records:
        features = record.get("features") or {}
        impacts = features.get("llm_drought_impacts") or []
        for impact in impacts:
            location = (impact.get("location") or "").strip()
            if location:
                if location not in cache:
                    geo, assignments = _resolve_location(location, cache, ctx, delay, limit)
                    status = _format_status(geo, assignments[0], assignments)
                    print(f"  [{len(cache)}/{total_impacts}] {location!r} -> {status}")
                else:
                    geo, assignments = cache[location]
                _apply_to_impact(impact, geo, assignments)
            else:
                _apply_to_impact(impact, None, [_empty_nuts()])

    return records


def process_geocoded_csv(input_path: Path, ctx: NutsContext) -> list[dict[str, Any]]:
    """Assign NUTS-3 on a flat geocoded CSV; expand multi-NUTS rows like save_csv."""
    with input_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cache: dict[str, tuple[dict[str, Any] | None, list[dict[str, Any]]]] = {}
    unique_locations = {
        (row.get("location") or "").strip()
        for row in rows
        if (row.get("location") or "").strip()
    }
    print(f"Assigning NUTS-3 for {len(unique_locations)} unique locations "
          f"across {len(rows)} rows (no Nominatim)...")

    out_rows: list[dict[str, Any]] = []
    for row in rows:
        location = (row.get("location") or "").strip()
        if location:
            if location not in cache:
                geo = _geo_from_mapping(row, location=location)
                geo, assignments = _resolve_from_geo(location, geo, cache, ctx)
                status = _format_status(geo, assignments[0], assignments)
                print(f"  [{len(cache)}/{len(unique_locations)}] {location!r} -> {status}")
            else:
                geo, assignments = cache[location]
        else:
            geo, assignments = None, [_empty_nuts()]

        geo_fields = geo if geo else _empty_geo()
        for assignment in assignments:
            out_rows.append({
                **{k: row.get(k, "") for k in CSV_FIELDS if k not in GEO_FIELDS and k not in NUTS_FIELDS},
                **{k: geo_fields.get(k) for k in GEO_FIELDS},
                **{k: assignment.get(k, "") for k in NUTS_FIELDS},
            })

    return out_rows


def save_json(records: list[dict], output_path: Path) -> None:
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON saved -> {output_path}")


def save_csv(records: list[dict], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []

    for record in records:
        features = record.get("features") or {}
        pub_date = features.get("publication_date", "")
        source_file = features.get("source_file", "")
        meta = record.get("meta") or {}

        impacts = features.get("llm_drought_impacts") or []
        for idx, impact in enumerate(impacts):
            nuts3_assignments = impact.get("nuts3_assignments")
            if nuts3_assignments:
                assignment_list = [
                    {
                        "nuts3_id": item.get("nuts3_id", ""),
                        "nuts3_name": item.get("nuts3_name", ""),
                        "mention_weight": item.get("mention_weight", 1.0),
                        "nuts_level": "nuts3",
                        "nuts2_id": impact.get("nuts2_id", ""),
                        "nuts2_name": impact.get("nuts2_name", ""),
                        "nuts_match_distance_m": impact.get("nuts_match_distance_m", ""),
                        "source_nuts_level": impact.get("source_nuts_level", "nuts2"),
                    }
                    for item in nuts3_assignments
                ]
            else:
                assignment_list = [{
                    "nuts_level": impact.get("nuts_level", "none"),
                    "nuts2_id": impact.get("nuts2_id", ""),
                    "nuts2_name": impact.get("nuts2_name", ""),
                    "nuts3_id": impact.get("nuts3_id", ""),
                    "nuts3_name": impact.get("nuts3_name", ""),
                    "nuts_match_distance_m": impact.get("nuts_match_distance_m", ""),
                    "source_nuts_level": impact.get("source_nuts_level", ""),
                    "mention_weight": impact.get("mention_weight", 1.0),
                }]

            for sub_impact in (impact.get("impacts") or [{}]):
                for assignment in assignment_list:
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
                        **{k: assignment.get(k, "") for k in NUTS_FIELDS},
                    })

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved  -> {output_path} ({len(rows)} rows)")


def save_flat_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved  -> {output_path} ({len(rows)} rows)")


def _default_assign_input() -> Path:
    if DEFAULT_GEOCODED_JSON.exists():
        return DEFAULT_GEOCODED_JSON
    if DEFAULT_GEOCODED_CSV.exists():
        return DEFAULT_GEOCODED_CSV
    raise SystemExit(
        "No geocoded input found. Run point_coder first, or pass a path.\n"
        f"  Expected: {DEFAULT_GEOCODED_JSON}\n"
        f"        or: {DEFAULT_GEOCODED_CSV}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Assign NL NUTS-3 regions from geocoded points "
            "(default: no Nominatim; run point_coder first)."
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Geocoded JSON or CSV from point_coder "
            f"(default: {DEFAULT_GEOCODED_JSON.name} or {DEFAULT_GEOCODED_CSV.name})"
        ),
    )
    parser.add_argument(
        "--from-llm-json",
        action="store_true",
        help=(
            "Legacy: geocode with Nominatim then assign NUTS-3. "
            "Default input becomes the LLM JSON under data/input/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output files (default: chapter_06/data/processed)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help="Seconds between Nominatim API calls (only with --from-llm-json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Nominatim candidates per query (only with --from-llm-json)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing the flat CSV output",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_llm_json:
        input_path = (args.input or DEFAULT_LLM_JSON).resolve()
    else:
        input_path = (args.input or _default_assign_input()).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    if args.from_llm_json:
        print(f"Mode:   geocode + assign (Nominatim, delay={args.delay}s, limit={args.limit})")
    else:
        print("Mode:   assign-only (no Nominatim)")
    print()
    print("Loading NUTS boundaries...")
    ctx = load_nuts_context()
    print()

    suffix = input_path.suffix.lower()
    if args.from_llm_json:
        records = process_llm_json(input_path, ctx, args.delay, limit=args.limit)
        save_json(records, output_dir / "impacts_nuts3.json")
        if not args.no_csv:
            save_csv(records, output_dir / "impacts_nuts3.csv")
    elif suffix == ".json":
        records = process_geocoded_json(input_path, ctx)
        save_json(records, output_dir / "impacts_nuts3.json")
        if not args.no_csv:
            save_csv(records, output_dir / "impacts_nuts3.csv")
    elif suffix == ".csv":
        rows = process_geocoded_csv(input_path, ctx)
        if not args.no_csv:
            save_flat_csv(rows, output_dir / "impacts_nuts3.csv")
        else:
            print("CSV input with --no-csv: nothing to write (JSON nested form needs geocoded JSON).")
    else:
        raise SystemExit(f"Unsupported input type: {input_path.suffix} (use .json or .csv)")


if __name__ == "__main__":
    main()
