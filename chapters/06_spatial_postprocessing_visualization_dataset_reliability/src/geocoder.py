from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from geocode_ranking import enrich_geocode_fields, is_low_quality_type, place_type_rank

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "drought-impact-location-classifier/1.0 (research-use)"
DEFAULT_DELAY = 1.0
DEFAULT_LIMIT = 5

COUNTRY_TYPES = {"country"}
REGION_TYPES = {
    "state", "region", "province", "county", "state_district",
    "statistical", "safety_region", "protected_area",
}
CITY_TYPES = {"city", "town", "municipality"}
LOCAL_TYPES = {"village", "hamlet", "suburb", "neighbourhood", "quarter", "locality"}


def classify_place_level(addresstype: str) -> str:
    t = (addresstype or "").lower()
    if t in COUNTRY_TYPES:
        return "country"
    if t in REGION_TYPES:
        return "region"
    if t in CITY_TYPES:
        return "city"
    if t in LOCAL_TYPES:
        return "local"
    return "other"


def is_low_quality(candidate: dict[str, Any]) -> bool:
    return is_low_quality_type(candidate.get("addresstype") or "")


def pick_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    indexed = list(enumerate(candidates))
    indexed.sort(
        key=lambda item: (-place_type_rank(item[1].get("addresstype", "")), item[0]),
    )
    return indexed[0][1]


def _dedupe_by_osm_id(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[Any] = set()
    merged: list[dict[str, Any]] = []
    for candidate in candidates:
        osm_id = candidate.get("osm_id")
        if osm_id in seen:
            continue
        seen.add(osm_id)
        merged.append(candidate)
    return merged


def _format_geocode_result(result: dict[str, Any]) -> dict[str, Any]:
    country_code = (result.get("address") or {}).get("country_code", "")
    bb = result.get("boundingbox", [])
    place_type = result.get("addresstype", "")
    place_level = classify_place_level(place_type)
    rank_fields = enrich_geocode_fields(place_type)
    return {
        "geocoded_latitude": result.get("lat"),
        "geocoded_longitude": result.get("lon"),
        "geocoded_display_name": result.get("display_name"),
        "geocoded_place_type": place_type,
        "geocoded_place_level": place_level,
        "geocoded_is_broad": place_level in {"country", "region"},
        "geocoded_country_code": country_code,
        "geocoded_osm_id": result.get("osm_id"),
        "geocoded_boundingbox": "|".join(bb) if bb else "",
        **rank_fields,
    }


def _fetch_nominatim(
    mention: str,
    *,
    limit: int,
    delay: float,
    country_codes: str | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "q": mention,
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": limit,
    }
    if country_codes:
        params["countrycodes"] = country_codes

    url = f"{NOMINATIM_URL}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return []
    finally:
        time.sleep(delay)

    return payload if isinstance(payload, list) else []


def geocode_location(
    mention: str,
    delay: float = DEFAULT_DELAY,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any] | None:
    mention = mention.strip()
    if not mention:
        return None

    global_results = _fetch_nominatim(mention, limit=limit, delay=delay)
    best = pick_best_candidate(global_results)
    if not best:
        return None

    if is_low_quality(best):
        nl_results = _fetch_nominatim(
            mention, limit=limit, delay=delay, country_codes="nl",
        )
        merged = _dedupe_by_osm_id(global_results + nl_results)
        best = pick_best_candidate(merged) or best

    return _format_geocode_result(best)
