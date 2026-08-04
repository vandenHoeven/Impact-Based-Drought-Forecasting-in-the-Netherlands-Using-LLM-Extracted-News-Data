"""
Chapter 06 geocoding smoke: Nominatim on 5 fixture impacts, then offline NUTS-3.

Uses five real non-geocoded article/impact excerpts from
chapters/06_geocoding/data/input/impacts_for_geocoding.json (bodies truncated).
Needs network for Nominatim; unreachable Nominatim is reported as SKIPPED.

    python reproducibility_and_robustness_testing/chapter_06_geocoding/run_geocoding.py
"""

from __future__ import annotations

import json
import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path

CHAPTER_TEST_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CHAPTER_TEST_ROOT.parents[1]
SRC_DIR = REPO_ROOT / "chapters" / "06_geocoding" / "src"
GEOJSON_PATH = (
    REPO_ROOT / "chapters" / "06_geocoding" / "data" / "geo" / "nuts_nl_simplified.geojson"
)
FIXTURE_PATH = CHAPTER_TEST_ROOT / "data" / "fixture" / "sample_impacts.json"
RESULTS_DIR = CHAPTER_TEST_ROOT / "data" / "results"

OUT_GEOCODED_JSON = "impacts_geocoded.json"
OUT_GEOCODED_CSV = "impacts_geocoded_points.csv"
OUT_NUTS3_JSON = "impacts_nuts3.json"
OUT_NUTS3_CSV = "impacts_nuts3.csv"

NOMINATIM_PROBE_URL = (
    "https://nominatim.openstreetmap.org/search?"
    "q=Amsterdam&format=jsonv2&limit=1"
)
USER_AGENT = "drought-impact-location-classifier/1.0 (research-use-smoke)"
EXPECTED_IMPACTS = 5
NOMINATIM_DELAY = 1.0


def _load_src_modules():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    import nuts3_coder  # noqa: PLC0415
    import point_coder  # noqa: PLC0415
    from nuts_mapper import load_nuts_context  # noqa: PLC0415

    return point_coder, nuts3_coder, load_nuts_context


def _nominatim_reachable() -> bool:
    req = urllib.request.Request(
        NOMINATIM_PROBE_URL,
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
        return isinstance(payload, list) and len(payload) > 0
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError):
        return False


def _iter_impacts(records: list[dict]):
    for record in records:
        features = record.get("features") or {}
        for impact in features.get("llm_drought_impacts") or []:
            yield record, impact


def _print_geocode_summary(records: list[dict]) -> list[str]:
    """Print geocode fields; return locations that lack coordinates."""
    missing: list[str] = []
    print("-" * 70)
    print("Geocode outputs")
    print("-" * 70)
    for i, (record, impact) in enumerate(_iter_impacts(records), start=1):
        location = (impact.get("location") or "").strip() or "<empty>"
        lat = impact.get("geocoded_latitude")
        lon = impact.get("geocoded_longitude")
        display = impact.get("geocoded_display_name") or ""
        place_type = impact.get("geocoded_place_type") or ""
        country = impact.get("geocoded_country_code") or ""
        print(f"[{i}] id={record.get('id')} location={location!r}")
        if lat is None or lon is None:
            print("    -> FAILED (no coordinates)")
            missing.append(location)
        else:
            print(f"    lat/lon: {lat}, {lon}")
            print(f"    display: {display}")
            print(f"    place_type: {place_type}  country: {country}")
    return missing


def _print_nuts_summary(records: list[dict]) -> None:
    print("-" * 70)
    print("NUTS-3 outputs")
    print("-" * 70)
    for i, (record, impact) in enumerate(_iter_impacts(records), start=1):
        location = (impact.get("location") or "").strip() or "<empty>"
        nuts3_id = impact.get("nuts3_id") or ""
        nuts3_name = impact.get("nuts3_name") or ""
        level = impact.get("nuts_level") or "none"
        print(
            f"[{i}] id={record.get('id')} location={location!r} "
            f"-> level={level} nuts3={nuts3_id!r} ({nuts3_name})"
        )


def main() -> int:
    print("=" * 70)
    print("Chapter 06 geocoding smoke (5 impacts + offline NUTS-3)")
    print(f"fixture: {FIXTURE_PATH}")
    print(f"results: {RESULTS_DIR}")
    print(f"geojson: {GEOJSON_PATH}")
    print("=" * 70)

    try:
        if not FIXTURE_PATH.is_file():
            raise FileNotFoundError(f"Missing fixture: {FIXTURE_PATH}")
        if not SRC_DIR.is_dir():
            raise FileNotFoundError(f"Missing chapter src: {SRC_DIR}")
        if not GEOJSON_PATH.is_file():
            raise FileNotFoundError(f"Missing NUTS geojson: {GEOJSON_PATH}")

        if not _nominatim_reachable():
            print(
                "CHECK_RESULT: SKIPPED - Nominatim unreachable; "
                "live geocoding check not run."
            )
            return 0

        point_coder, nuts3_coder, load_nuts_context = _load_src_modules()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        print("\nNominatim geocoding...")
        records = point_coder.process_json(
            FIXTURE_PATH,
            delay=NOMINATIM_DELAY,
            limit=5,
        )
        n_impacts = sum(1 for _ in _iter_impacts(records))
        if n_impacts != EXPECTED_IMPACTS:
            raise ValueError(
                f"Expected {EXPECTED_IMPACTS} impacts in fixture, got {n_impacts}"
            )

        missing = _print_geocode_summary(records)
        if missing:
            raise RuntimeError(
                f"Geocoding failed for {len(missing)} location(s): {missing}"
            )

        geocoded_json = RESULTS_DIR / OUT_GEOCODED_JSON
        geocoded_csv = RESULTS_DIR / OUT_GEOCODED_CSV
        point_coder.save_json(records, geocoded_json)
        point_coder.save_csv(records, geocoded_csv)

        print("\nOffline NUTS-3 assignment...")
        ctx = load_nuts_context(geojson_path=GEOJSON_PATH)
        nuts_records = nuts3_coder.process_geocoded_json(geocoded_json, ctx)
        _print_nuts_summary(nuts_records)
        nuts3_coder.save_json(nuts_records, RESULTS_DIR / OUT_NUTS3_JSON)
        nuts3_coder.save_csv(nuts_records, RESULTS_DIR / OUT_NUTS3_CSV)

    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print(f"CHECK_RESULT: FAIL - {exc}")
        return 1

    print("TEST COMPLETE")
    print("CHECK_RESULT: PASS - 5 impacts geocoded and NUTS-3 assigned.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
