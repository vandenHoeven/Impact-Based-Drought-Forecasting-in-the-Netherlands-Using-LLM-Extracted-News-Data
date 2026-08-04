from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from location_aliases import (
    MACRO_REGIONS,
    NUTS1_ALIASES,
    NUTS2_ALIASES,
    NUTS3_ALIASES,
    normalize_location_name,
    normalize_nuts_name,
)

NETHERLANDS_COUNTRY_CODE = "nl"
NUTS_GEOJSON_URL = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/NUTS_RG_01M_2021_4326.geojson"
NEAREST_REGION_MAX_DISTANCE_M = 25_000

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
LOCAL_GEO_DIR = PACKAGE_ROOT / "data" / "geo"
LOCAL_NUTS_GEOJSON = LOCAL_GEO_DIR / "nuts_nl_simplified.geojson"


@dataclass
class NameMatch:
    level: Literal["nuts3", "nuts2", "macro"]
    nuts2_id: str = ""
    nuts2_name: str = ""
    nuts3_id: str = ""
    nuts3_name: str = ""
    macro_nuts2_ids: list[str] = field(default_factory=list)
    macro_nuts3_ids: list[str] = field(default_factory=list)


@dataclass
class NutsContext:
    nuts2: gpd.GeoDataFrame
    nuts3: gpd.GeoDataFrame
    nuts2_m: gpd.GeoDataFrame
    nuts3_m: gpd.GeoDataFrame
    nuts2_to_nuts3: dict[str, list[dict[str, Any]]]
    nuts2_by_name: dict[str, dict[str, str]] = field(default_factory=dict)
    nuts3_by_name: dict[str, dict[str, str]] = field(default_factory=dict)
    nuts2_by_id: dict[str, dict[str, str]] = field(default_factory=dict)
    nuts3_by_id: dict[str, dict[str, str]] = field(default_factory=dict)


def _geojson_needs_regeneration(geojson_path: Path) -> bool:
    if not geojson_path.exists():
        return True
    cached = gpd.read_file(geojson_path)
    return cached.empty or not (cached["LEVL_CODE"] == 1).any()


def _write_local_nuts_geojson(
    geojson_path: Path,
    simplify_tolerance_m: float = 250.0,
) -> Path:
    all_nuts = gpd.read_file(NUTS_GEOJSON_URL)
    nl_nuts = all_nuts[(all_nuts["CNTR_CODE"] == "NL") & (all_nuts["LEVL_CODE"].isin([1, 2, 3]))].copy()
    nl_nuts = nl_nuts[["CNTR_CODE", "LEVL_CODE", "NUTS_ID", "NAME_LATN", "geometry"]].reset_index(drop=True)

    nl_nuts_m = nl_nuts.to_crs(28992)
    nl_nuts_m["geometry"] = nl_nuts_m.geometry.simplify(simplify_tolerance_m, preserve_topology=True)
    nl_nuts_simple = nl_nuts_m.to_crs(4326)
    nl_nuts_simple.to_file(geojson_path, driver="GeoJSON")
    return geojson_path


def ensure_local_nuts_geojson(
    geojson_path: Path = LOCAL_NUTS_GEOJSON,
    simplify_tolerance_m: float = 250.0,
) -> Path:
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    if not _geojson_needs_regeneration(geojson_path):
        return geojson_path
    return _write_local_nuts_geojson(geojson_path, simplify_tolerance_m=simplify_tolerance_m)


def load_nuts_boundaries(
    nuts_geojson_path: Path,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nl_nuts = gpd.read_file(nuts_geojson_path)
    nuts2 = nl_nuts[nl_nuts["LEVL_CODE"] == 2].copy()
    nuts3 = nl_nuts[nl_nuts["LEVL_CODE"] == 3].copy()

    nuts2 = nuts2[["NUTS_ID", "NAME_LATN", "geometry"]].reset_index(drop=True)
    nuts3 = nuts3[["NUTS_ID", "NAME_LATN", "geometry"]].reset_index(drop=True)

    nuts2_m = nuts2.to_crs(28992)
    nuts3_m = nuts3.to_crs(28992)
    return nuts2, nuts3, nuts2_m, nuts3_m


def load_nuts1_boundaries(
    nuts_geojson_path: Path,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nl_nuts = gpd.read_file(nuts_geojson_path)
    nuts1 = nl_nuts[nl_nuts["LEVL_CODE"] == 1].copy()
    nuts1 = nuts1[["NUTS_ID", "NAME_LATN", "geometry"]].reset_index(drop=True)
    nuts1_m = nuts1.to_crs(28992)
    return nuts1, nuts1_m


def build_nuts2_to_nuts1_mapping(
    nuts1: gpd.GeoDataFrame,
    nuts2: gpd.GeoDataFrame,
) -> dict[str, str]:
    parent = nuts1[["NUTS_ID", "NAME_LATN", "geometry"]].rename(
        columns={"NUTS_ID": "nuts1_id", "NAME_LATN": "nuts1_name"},
    )

    nuts2_points = nuts2.copy()
    nuts2_points["geometry"] = nuts2_points.geometry.representative_point()
    child_points = nuts2_points[["NUTS_ID", "NAME_LATN", "geometry"]].rename(
        columns={"NUTS_ID": "nuts2_id", "NAME_LATN": "nuts2_name"},
    )

    try:
        joined = gpd.sjoin(child_points, parent, how="left", predicate="intersects")
    except Exception:
        records: list[dict[str, str]] = []
        for _, child_row in child_points.iterrows():
            matches = parent[parent.intersects(child_row.geometry)]
            if matches.empty:
                continue
            parent_row = matches.iloc[0]
            records.append({
                "nuts2_id": str(child_row["nuts2_id"]),
                "nuts1_id": str(parent_row["nuts1_id"]),
            })
        joined = pd.DataFrame(records)

    mapping: dict[str, str] = {}
    if joined.empty:
        return mapping

    joined = joined.dropna(subset=["nuts1_id"]).copy()
    for _, row in joined.drop_duplicates(subset=["nuts2_id"]).iterrows():
        mapping[str(row["nuts2_id"])] = str(row["nuts1_id"])
    return mapping


def build_nuts1_name_indexes(
    nuts1: gpd.GeoDataFrame,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    nuts1_by_name: dict[str, dict[str, str]] = {}
    nuts1_by_id: dict[str, dict[str, str]] = {}

    for _, row in nuts1.iterrows():
        nuts1_id = str(row["NUTS_ID"])
        nuts1_name = str(row["NAME_LATN"])
        entry = {"nuts1_id": nuts1_id, "nuts1_name": nuts1_name}
        nuts1_by_id[nuts1_id] = entry
        nuts1_by_name[normalize_nuts_name(nuts1_name)] = entry

    for alias, nuts1_id in NUTS1_ALIASES.items():
        if nuts1_id in nuts1_by_id:
            nuts1_by_name[alias] = nuts1_by_id[nuts1_id]

    return nuts1_by_name, nuts1_by_id


def build_nuts2_to_nuts3_mapping(
    nuts2: gpd.GeoDataFrame,
    nuts3: gpd.GeoDataFrame,
) -> dict[str, list[dict[str, Any]]]:
    parent = nuts2[["NUTS_ID", "NAME_LATN", "geometry"]].rename(
        columns={"NUTS_ID": "nuts2_id", "NAME_LATN": "nuts2_name"},
    )

    nuts3_points = nuts3.copy()
    nuts3_points["geometry"] = nuts3_points.geometry.representative_point()
    child_points = nuts3_points[["NUTS_ID", "NAME_LATN", "geometry"]].rename(
        columns={"NUTS_ID": "nuts3_id", "NAME_LATN": "nuts3_name"},
    )

    try:
        joined = gpd.sjoin(child_points, parent, how="left", predicate="intersects")
    except Exception:
        records: list[dict[str, Any]] = []
        for _, child_row in child_points.iterrows():
            matches = parent[parent.intersects(child_row.geometry)]
            if matches.empty:
                continue
            parent_row = matches.iloc[0]
            records.append({
                "nuts3_id": child_row["nuts3_id"],
                "nuts3_name": child_row["nuts3_name"],
                "nuts2_id": parent_row["nuts2_id"],
                "nuts2_name": parent_row["nuts2_name"],
            })
        joined = pd.DataFrame(records)

    mapping: dict[str, list[dict[str, Any]]] = {}
    if joined.empty:
        return mapping

    joined = joined.dropna(subset=["nuts2_id"]).copy()
    for nuts2_id, group in joined.groupby("nuts2_id", dropna=True):
        group = group.drop_duplicates(subset=["nuts3_id"])
        if group.empty:
            continue
        weight = 1.0 / len(group)
        mapping[str(nuts2_id)] = [
            {
                "nuts3_id": str(row["nuts3_id"]),
                "nuts3_name": str(row["nuts3_name"]),
                "nuts2_id": str(nuts2_id),
                "nuts2_name": str(group.iloc[0]["nuts2_name"]),
                "weight": weight,
            }
            for _, row in group.iterrows()
        ]

    return mapping


def build_name_indexes(
    nuts2: gpd.GeoDataFrame,
    nuts3: gpd.GeoDataFrame,
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
]:
    nuts2_by_name: dict[str, dict[str, str]] = {}
    nuts3_by_name: dict[str, dict[str, str]] = {}
    nuts2_by_id: dict[str, dict[str, str]] = {}
    nuts3_by_id: dict[str, dict[str, str]] = {}

    for _, row in nuts2.iterrows():
        nuts2_id = str(row["NUTS_ID"])
        nuts2_name = str(row["NAME_LATN"])
        entry = {"nuts2_id": nuts2_id, "nuts2_name": nuts2_name}
        nuts2_by_id[nuts2_id] = entry
        nuts2_by_name[normalize_nuts_name(nuts2_name)] = entry

    for _, row in nuts3.iterrows():
        nuts3_id = str(row["NUTS_ID"])
        nuts3_name = str(row["NAME_LATN"])
        entry = {"nuts3_id": nuts3_id, "nuts3_name": nuts3_name}
        nuts3_by_id[nuts3_id] = entry
        nuts3_by_name[normalize_nuts_name(nuts3_name)] = entry

    for alias, nuts2_id in NUTS2_ALIASES.items():
        if nuts2_id in nuts2_by_id:
            nuts2_by_name[alias] = nuts2_by_id[nuts2_id]

    for alias, nuts3_id in NUTS3_ALIASES.items():
        if nuts3_id in nuts3_by_id:
            nuts3_by_name[alias] = nuts3_by_id[nuts3_id]

    return nuts2_by_name, nuts3_by_name, nuts2_by_id, nuts3_by_id


def resolve_location_by_name(location: str, ctx: NutsContext) -> NameMatch | None:
    norm = normalize_location_name(location)
    if not norm:
        return None

    if norm in ctx.nuts3_by_name:
        row = ctx.nuts3_by_name[norm]
        return NameMatch(
            level="nuts3",
            nuts3_id=row["nuts3_id"],
            nuts3_name=row["nuts3_name"],
        )

    if norm in ctx.nuts2_by_name:
        row = ctx.nuts2_by_name[norm]
        return NameMatch(
            level="nuts2",
            nuts2_id=row["nuts2_id"],
            nuts2_name=row["nuts2_name"],
        )

    if norm in MACRO_REGIONS:
        spec = MACRO_REGIONS[norm]
        if "nuts3" in spec:
            return NameMatch(level="macro", macro_nuts3_ids=list(spec["nuts3"]))
        if "nuts2" in spec:
            return NameMatch(level="macro", macro_nuts2_ids=list(spec["nuts2"]))

    return None


def load_nuts_context(
    geojson_path: Path = LOCAL_NUTS_GEOJSON,
    simplify_tolerance_m: float = 250.0,
) -> NutsContext:
    local_geojson = ensure_local_nuts_geojson(
        geojson_path=geojson_path,
        simplify_tolerance_m=simplify_tolerance_m,
    )
    nuts2, nuts3, nuts2_m, nuts3_m = load_nuts_boundaries(local_geojson)
    nuts2_to_nuts3 = build_nuts2_to_nuts3_mapping(nuts2, nuts3)
    nuts2_by_name, nuts3_by_name, nuts2_by_id, nuts3_by_id = build_name_indexes(nuts2, nuts3)
    return NutsContext(
        nuts2=nuts2,
        nuts3=nuts3,
        nuts2_m=nuts2_m,
        nuts3_m=nuts3_m,
        nuts2_to_nuts3=nuts2_to_nuts3,
        nuts2_by_name=nuts2_by_name,
        nuts3_by_name=nuts3_by_name,
        nuts2_by_id=nuts2_by_id,
        nuts3_by_id=nuts3_by_id,
    )


def _parent_nuts2_for_nuts3(nuts3_id: str, ctx: NutsContext) -> tuple[str, str]:
    for nuts2_id, children in ctx.nuts2_to_nuts3.items():
        for child in children:
            if child["nuts3_id"] == nuts3_id:
                return nuts2_id, child.get("nuts2_name", ctx.nuts2_by_id.get(nuts2_id, {}).get("nuts2_name", ""))
    return "", ""


def map_point_to_nuts(
    latitude: str | float | None,
    longitude: str | float | None,
    ctx: NutsContext,
    max_distance_m: float = NEAREST_REGION_MAX_DISTANCE_M,
    is_broad_location: bool = False,
    prefer_nuts2_split: bool = False,
) -> dict[str, Any]:
    result = {
        "nuts_level": "none",
        "nuts2_id": "",
        "nuts2_name": "",
        "nuts3_id": "",
        "nuts3_name": "",
        "nuts_match_distance_m": "",
    }

    if latitude in (None, "") or longitude in (None, ""):
        return result

    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return result

    point = Point(lon, lat)
    point_series = gpd.GeoSeries([point], crs="EPSG:4326")
    try_nuts3_first = (not is_broad_location) or (is_broad_location and not prefer_nuts2_split)

    if try_nuts3_first:
        contains_nuts3 = ctx.nuts3[ctx.nuts3.contains(point)]
        if not contains_nuts3.empty:
            row = contains_nuts3.iloc[0]
            result["nuts_level"] = "nuts3"
            result["nuts3_id"] = row["NUTS_ID"]
            result["nuts3_name"] = row["NAME_LATN"]
            parent_nuts2 = ctx.nuts2[ctx.nuts2.contains(point)]
            if not parent_nuts2.empty:
                result["nuts2_id"] = parent_nuts2.iloc[0]["NUTS_ID"]
                result["nuts2_name"] = parent_nuts2.iloc[0]["NAME_LATN"]
            result["nuts_match_distance_m"] = 0
            return result

    contains_nuts2 = ctx.nuts2[ctx.nuts2.contains(point)]
    if not contains_nuts2.empty:
        row = contains_nuts2.iloc[0]
        result["nuts_level"] = "nuts2"
        result["nuts2_id"] = row["NUTS_ID"]
        result["nuts2_name"] = row["NAME_LATN"]
        result["nuts_match_distance_m"] = 0
        return result

    point_m = point_series.to_crs(28992).iloc[0]

    if try_nuts3_first and not ctx.nuts3_m.empty:
        distances_n3 = ctx.nuts3_m.distance(point_m)
        n3_idx = distances_n3.idxmin()
        n3_dist = float(distances_n3.loc[n3_idx])
        if n3_dist <= max_distance_m:
            n3_row = ctx.nuts3.iloc[n3_idx]
            result["nuts_level"] = "nuts3"
            result["nuts3_id"] = n3_row["NUTS_ID"]
            result["nuts3_name"] = n3_row["NAME_LATN"]
            result["nuts_match_distance_m"] = int(n3_dist)

            for _, n2_row in ctx.nuts2.iterrows():
                if n2_row.geometry.contains(n3_row.geometry.centroid):
                    result["nuts2_id"] = n2_row["NUTS_ID"]
                    result["nuts2_name"] = n2_row["NAME_LATN"]
                    break
            return result

    if not ctx.nuts2_m.empty:
        distances_n2 = ctx.nuts2_m.distance(point_m)
        n2_idx = distances_n2.idxmin()
        n2_dist = float(distances_n2.loc[n2_idx])
        if n2_dist <= max_distance_m:
            n2_row = ctx.nuts2.iloc[n2_idx]
            result["nuts_level"] = "nuts2"
            result["nuts2_id"] = n2_row["NUTS_ID"]
            result["nuts2_name"] = n2_row["NAME_LATN"]
            result["nuts_match_distance_m"] = int(n2_dist)

    return result


def _nuts_assignment(
    nuts_level: str,
    nuts_mapping: dict[str, Any] | None = None,
    *,
    source_nuts_level: str = "",
    mention_weight: float = 1.0,
) -> dict[str, Any]:
    base = {
        "nuts_level": nuts_level,
        "nuts2_id": "",
        "nuts2_name": "",
        "nuts3_id": "",
        "nuts3_name": "",
        "nuts_match_distance_m": "",
        "source_nuts_level": source_nuts_level or nuts_level,
        "mention_weight": mention_weight,
    }
    if nuts_mapping:
        for key in ("nuts2_id", "nuts2_name", "nuts3_id", "nuts3_name", "nuts_match_distance_m"):
            if key in nuts_mapping and nuts_mapping[key] != "":
                base[key] = nuts_mapping[key]
    return base


def _assignment_from_nuts3_id(
    nuts3_id: str,
    ctx: NutsContext,
    *,
    source_nuts_level: str = "nuts3",
    mention_weight: float = 1.0,
    nuts_match_distance_m: str | int = "",
) -> dict[str, Any]:
    row = ctx.nuts3_by_id.get(nuts3_id, {})
    nuts2_id, nuts2_name = _parent_nuts2_for_nuts3(nuts3_id, ctx)
    return _nuts_assignment(
        "nuts3",
        {
            "nuts3_id": nuts3_id,
            "nuts3_name": row.get("nuts3_name", ""),
            "nuts2_id": nuts2_id,
            "nuts2_name": nuts2_name,
            "nuts_match_distance_m": nuts_match_distance_m,
        },
        source_nuts_level=source_nuts_level,
        mention_weight=mention_weight,
    )


def _expand_nuts2_children(
    nuts2_id: str,
    ctx: NutsContext,
    *,
    source_nuts_level: str = "nuts2",
    base_mapping: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    children = ctx.nuts2_to_nuts3.get(str(nuts2_id), [])
    if not children:
        return [_nuts_assignment("nuts2", base_mapping, source_nuts_level=source_nuts_level)]
    return [
        _nuts_assignment(
            "nuts3",
            {
                **(base_mapping or {}),
                "nuts2_id": child.get("nuts2_id", nuts2_id),
                "nuts2_name": child.get("nuts2_name", ""),
                "nuts3_id": child["nuts3_id"],
                "nuts3_name": child["nuts3_name"],
            },
            source_nuts_level=source_nuts_level,
            mention_weight=child["weight"],
        )
        for child in children
    ]


def _expand_macro_match(name_match: NameMatch, ctx: NutsContext) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []

    if name_match.macro_nuts3_ids:
        for nuts3_id in name_match.macro_nuts3_ids:
            if nuts3_id in ctx.nuts3_by_id:
                targets.append({"nuts3_id": nuts3_id})
    elif name_match.macro_nuts2_ids:
        for nuts2_id in name_match.macro_nuts2_ids:
            for child in ctx.nuts2_to_nuts3.get(nuts2_id, []):
                targets.append(child)

    if not targets:
        return [_nuts_assignment("none")]

    weight = 1.0 / len(targets)
    return [
        _assignment_from_nuts3_id(
            target["nuts3_id"],
            ctx,
            source_nuts_level="macro",
            mention_weight=weight,
        )
        for target in targets
    ]


def assign_nuts(
    geocode_result: dict[str, Any] | None,
    ctx: NutsContext,
    location: str = "",
) -> list[dict[str, Any]]:
    if not geocode_result:
        return [_nuts_assignment("none")]

    lat = geocode_result.get("geocoded_latitude")
    lon = geocode_result.get("geocoded_longitude")
    country = (geocode_result.get("geocoded_country_code") or "").lower()
    place_level = geocode_result.get("geocoded_place_level", "")

    if lat in (None, "") or lon in (None, ""):
        return [_nuts_assignment("none")]

    if country and country != NETHERLANDS_COUNTRY_CODE:
        return [_nuts_assignment("non_nl")]

    if place_level == "country":
        return [_nuts_assignment("country")]

    name_match = resolve_location_by_name(location, ctx)
    if name_match:
        if name_match.level == "nuts3":
            return [
                _assignment_from_nuts3_id(
                    name_match.nuts3_id,
                    ctx,
                    source_nuts_level="nuts3",
                    mention_weight=1.0,
                )
            ]
        if name_match.level == "nuts2":
            return _expand_nuts2_children(name_match.nuts2_id, ctx, source_nuts_level="nuts2")
        if name_match.level == "macro":
            return _expand_macro_match(name_match, ctx)

    is_broad = bool(geocode_result.get("geocoded_is_broad"))
    nuts_mapping = map_point_to_nuts(
        lat,
        lon,
        ctx,
        is_broad_location=is_broad,
        prefer_nuts2_split=False,
    )

    if nuts_mapping.get("nuts_level") == "nuts2":
        return _expand_nuts2_children(
            str(nuts_mapping.get("nuts2_id", "")),
            ctx,
            source_nuts_level="nuts2",
            base_mapping=nuts_mapping,
        )

    source = nuts_mapping.get("nuts_level", "none")
    return [_nuts_assignment(source, nuts_mapping, source_nuts_level=source)]
