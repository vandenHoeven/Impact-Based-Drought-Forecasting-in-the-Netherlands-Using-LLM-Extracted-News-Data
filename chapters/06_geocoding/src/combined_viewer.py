from __future__ import annotations

import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import pydeck as pdk
import streamlit as st
from pyproj import Transformer

# ── Constants ─────────────────────────────────────────────────────────────────

NL_BOUNDS = {
    "min_lat": 50.75,
    "max_lat": 53.55,
    "min_lon": 3.35,
    "max_lon": 7.25,
}
NL_CENTER_CLOSE = {"latitude": 52.1326, "longitude": 5.2913, "zoom": 7.0}
NL_FIT_ZOOM = 6.1
MAP_DEFAULT_HEIGHT = 680
MAP_PRESET_FIT = "Fit Netherlands"
MAP_PRESET_CLOSE = "Closer (zoom 7)"
MAP_PRESET_CUSTOM = "Custom"
MAP_PRESET_ORDER = [MAP_PRESET_FIT, MAP_PRESET_CLOSE, MAP_PRESET_CUSTOM]

THEME_COLORS: dict[str, list[int]] = {
    "agriculture": [34, 139, 34, 200],
    "water": [30, 136, 229, 200],
    "fire_heat": [220, 53, 34, 200],
    "ecosystem": [0, 137, 123, 200],
    "economy": [123, 31, 162, 200],
    "social": [233, 30, 99, 200],
}
THEME_LABELS: dict[str, str] = {
    "agriculture": "Agriculture",
    "water": "Water",
    "fire_heat": "Fire & heat",
    "ecosystem": "Ecosystem",
    "economy": "Economy",
    "social": "Social",
}
THEME_ORDER = ["agriculture", "water", "fire_heat", "ecosystem", "economy", "social"]
CLASSIFICATION_THEME: dict[str, str] = {
    "Crop Failure & Yield Reduction": "agriculture",
    "Livestock Stress & Mortality": "agriculture",
    "Irrigation Shortage": "agriculture",
    "Agricultural Economic Loss": "agriculture",
    "Groundwater Depletion": "water",
    "Reservoir & Surface Water Shortage": "water",
    "Water Use Restrictions": "water",
    "Water Supply & Sanitation Issues": "water",
    "Hydropower Reduction": "water",
    "Thermal/Nuclear Cooling Constraints": "water",
    "Industrial Water Shortages": "water",
    "Inland Waterway Disruption": "water",
    "Wildfire Occurrence": "fire_heat",
    "Wildfire Risk Increase": "fire_heat",
    "Heat & Air Quality Health Impacts": "fire_heat",
    "Freshwater Ecosystem Degradation": "ecosystem",
    "Forest Dieback & Vegetation Stress": "ecosystem",
    "Wetland Loss": "ecosystem",
    "Broader Economic Disruption": "economy",
    "Social Impacts": "social",
}


def get_classification_color(classification: str) -> list[int]:
    cls = str(classification).strip() if classification is not None else ""
    if not cls or cls.lower() == "nan":
        return THEME_COLORS["economy"]
    theme = CLASSIFICATION_THEME.get(cls)
    if theme:
        return THEME_COLORS[theme]
    theme_keys = list(THEME_COLORS.keys())
    return THEME_COLORS[theme_keys[hash(cls) % len(theme_keys)]]


CLASSIFICATION_COLORS: dict[str, list[int]] = {
    cls: get_classification_color(cls) for cls in CLASSIFICATION_THEME
}
POINT_ALPHA_MIN = 80
POINT_ALPHA_MAX = 200
FIRE_COLOR = [138, 43, 226, 180]
EMPTY_REGION_COLOR = [235, 238, 245, 170]


def apply_dominance_alpha(color: list[int], dominant_share: float) -> list[int]:
    share = max(0.0, min(1.0, dominant_share))
    alpha = int(POINT_ALPHA_MIN + (POINT_ALPHA_MAX - POINT_ALPHA_MIN) * share)
    return [color[0], color[1], color[2], alpha]

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from geocode_ranking import enrich_geocode_fields, format_location_override_notes, format_rank_table_rows
from viewer_article_utils import attach_article_titles

# Optional override for robustness smoke tests (tiny self-contained CSV dir).
_VIEWER_DATA_OVERRIDE = (os.environ.get("GEOCODING_VIEWER_DATA_DIR") or "").strip()
_OVERRIDE_DIR = (
    Path(_VIEWER_DATA_OVERRIDE).expanduser().resolve() if _VIEWER_DATA_OVERRIDE else None
)
PROCESSED_DIR = _OVERRIDE_DIR if _OVERRIDE_DIR is not None else PACKAGE_ROOT / "data" / "processed"
FINAL_POINTS_DIR = PACKAGE_ROOT / "data" / "final" / "points"
FINAL_NUTS3_DIR = PACKAGE_ROOT / "data" / "final" / "nuts3"
_THESIS_POINT_CSV = (
    FINAL_POINTS_DIR
    / "chapter7_merged_with_llm_features_gemini_gemini-3.5-flash_flex_geocoded.csv"
)
_THESIS_NUTS3_CSV = (
    FINAL_NUTS3_DIR
    / "chapter7_merged_with_llm_features_gemini_gemini-3.5-flash_flex_nuts3.csv"
)
LOCAL_NUTS_GEOJSON = PACKAGE_ROOT / "data" / "geo" / "nuts_nl_simplified.geojson"
DEFAULT_MIN_GEOCODE_RANK = 40
# Thesis-final defaults; smoke override uses the small results CSVs instead.
DEFAULT_POINT_CSV = (
    PROCESSED_DIR / "impacts_geocoded_points.csv"
    if _OVERRIDE_DIR is not None
    else _THESIS_POINT_CSV
)
DEFAULT_NUTS3_CSV = (
    PROCESSED_DIR / "impacts_nuts3.csv"
    if _OVERRIDE_DIR is not None
    else _THESIS_NUTS3_CSV
)
DEFAULT_WILDFIRE_CSV = (
    PACKAGE_ROOT / "data" / "wildfire" / "wildfires_2km_decade_2017-2022_no-military-bases.csv"
)
# Optional KNMI deficit text (not required for core geocoding / maps)
KNMI_DEFICIT_PATH = PACKAGE_ROOT / "data" / "wildfire" / "droogte_data_knmi.txt"
KNMI_DEFICIT_BAR_COLOR = "#b0b0b0"
KNMI_DEFICIT_HIGHLIGHT_COLOR = "#377eb8"
KNMI_DEFICIT_YEAR_MIN = 2000
KNMI_DEFICIT_ZOOM_YEAR_MIN = 2015
KNMI_DEFICIT_ZOOM_YEAR_MAX = 2026

LAYER_ID_IMPACTS = "impacts"
LAYER_ID_NUTS3 = "nuts3_regions"
LAYER_ID_FIRES = "wildfires"
MAX_FIRE_MAP_POINTS = 50_000

MODE_POINT = "Point viewer"
MODE_NUTS3 = "NUTS3 viewer"

_RD_TO_WGS84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

# Discovery roots: thesis-final first, then legacy processed (or smoke override only).
CHAPTER_DIR = PACKAGE_ROOT
CHAPTER7_DIR = PACKAGE_ROOT  # unused outside this package
if _OVERRIDE_DIR is not None:
    POINT_DIR = PROCESSED_DIR
    NUTS3_DIR = PROCESSED_DIR
    CHAPTER7_POINT_DIR = PROCESSED_DIR
    CHAPTER7_NUTS3_DIR = PROCESSED_DIR
else:
    POINT_DIR = FINAL_POINTS_DIR
    NUTS3_DIR = FINAL_NUTS3_DIR
    CHAPTER7_POINT_DIR = PACKAGE_ROOT / "data" / "processed"
    CHAPTER7_NUTS3_DIR = PACKAGE_ROOT / "data" / "processed"

# ── Geocode rank helpers ──────────────────────────────────────────────────────


def _ensure_geocode_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    place_types = (
        df["geocoded_place_type"].fillna("").astype(str).str.strip()
        if "geocoded_place_type" in df.columns
        else pd.Series("", index=df.index)
    )
    locations = (
        df["location"].fillna("").astype(str).str.strip()
        if "location" in df.columns
        else pd.Series("", index=df.index)
    )

    ranks: list[int | None] = []
    low_flags: list[bool] = []
    for pt, loc in zip(place_types, locations):
        fields = enrich_geocode_fields(pt or None, location=loc or None)
        ranks.append(fields["geocoded_place_rank"])
        low_flags.append(bool(fields["geocoded_is_low_quality"]))

    df["geocoded_place_rank"] = pd.to_numeric(pd.Series(ranks, index=df.index), errors="coerce")
    df["geocoded_is_low_quality"] = pd.Series(low_flags, index=df.index).fillna(False).astype(bool)
    return df


def _render_geocode_rank_reference() -> None:
    st.caption("Reference: Nominatim place-type ranks used for geocode quality filtering.")
    st.dataframe(
        pd.DataFrame(format_rank_table_rows()),
        width="stretch",
        hide_index=True,
        height=320,
    )
    override_notes = format_location_override_notes()
    if override_notes:
        st.caption("Location overrides: " + "; ".join(override_notes))


# ── Shared data loading ───────────────────────────────────────────────────────


@st.cache_data
def load_geocoded_csv(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df = df.dropna(subset=["geocoded_latitude", "geocoded_longitude"])
    df["geocoded_latitude"] = df["geocoded_latitude"].astype(float)
    df["geocoded_longitude"] = df["geocoded_longitude"].astype(float)
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(1).astype(int)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").round(2)
    df["recency_in_months"] = pd.to_numeric(df["recency_in_months"], errors="coerce")
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df["year_month"] = df["publication_date"].dt.to_period("M")
    df["color"] = df["classification"].map(get_classification_color)
    return _ensure_geocode_rank_columns(df)


@st.cache_data
def load_nuts3_csv(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw))
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(1).astype(int)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").round(2)
    df["recency_in_months"] = pd.to_numeric(df["recency_in_months"], errors="coerce")
    if "mention_weight" in df.columns:
        df["mention_weight"] = pd.to_numeric(df["mention_weight"], errors="coerce").fillna(1.0)
    else:
        df["mention_weight"] = 1.0
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df["year_month"] = df["publication_date"].dt.to_period("M")
    df["color"] = df["classification"].map(get_classification_color)
    for col in ("nuts3_id", "nuts3_name", "nuts2_id", "nuts2_name", "nuts_level", "location"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
    return _ensure_geocode_rank_columns(df)


@st.cache_data
def load_nuts3_geojson(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    features = []
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        nuts3_id = str(props.get("NUTS_ID", "")).strip()
        if not nuts3_id:
            continue
        try:
            level_code = int(props.get("LEVL_CODE", -1))
        except (TypeError, ValueError):
            continue
        if level_code != 3:
            continue
        features.append({
            "type": "Feature",
            "geometry": feature.get("geometry"),
            "properties": {
                "nuts3_id": nuts3_id,
                "nuts3_name": str(props.get("NAME_LATN", "")).strip(),
            },
        })
    return {"type": "FeatureCollection", "features": features}


@st.cache_data
def load_wildfire_occurrences(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["x", "y", "time", "wildfires_sum"])
    df = df[df["wildfires_sum"] > 0].copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["x", "y", "time"])
    df["year_month"] = df["time"].dt.to_period("M")
    lons, lats = _RD_TO_WGS84.transform(df["x"].values, df["y"].values)
    df["latitude"] = lats
    df["longitude"] = lons
    df["radius"] = 800 + df["wildfires_sum"].clip(upper=10) * 400
    df["color"] = [FIRE_COLOR for _ in range(len(df))]
    df["month_label"] = df["time"].dt.strftime("%b %Y")
    return df.reset_index(drop=True)


@st.cache_data
def load_knmi_sep30_deficit(data_path: str) -> pd.DataFrame:
    raw = pd.read_csv(
        data_path,
        comment="#",
        sep=r"\s+",
        header=None,
        usecols=[0, 1],
        names=["date_code", "deficit_mm"],
        engine="python",
    )
    raw["deficit_mm"] = pd.to_numeric(raw["deficit_mm"], errors="coerce")
    raw["date"] = pd.to_datetime(raw["date_code"].astype(str), format="%Y%m%d", errors="coerce")
    raw = raw.dropna(subset=["date", "deficit_mm"]).sort_values("date")
    raw = raw.drop_duplicates(subset="date", keep="last")

    season = raw.loc[raw["date"].dt.month.between(4, 9)].copy()
    rows: list[dict[str, float | int]] = []
    for year, group in season.groupby(season["date"].dt.year):
        sep30_mask = (group["date"].dt.month == 9) & (group["date"].dt.day == 30)
        if sep30_mask.any():
            deficit_mm = float(group.loc[sep30_mask, "deficit_mm"].iloc[0])
        elif not group.empty:
            deficit_mm = float(group.sort_values("date")["deficit_mm"].iloc[-1])
        else:
            continue
        rows.append({"year": int(year), "deficit_mm": deficit_mm})

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def _plot_knmi_deficit_bars(
    annual: pd.DataFrame,
    *,
    highlight_year: int | None,
    year_min: int,
    year_max: int,
    title: str,
    x_tick_step: int,
) -> None:
    subset = annual.loc[annual["year"].between(year_min, year_max)].copy()
    if subset.empty:
        st.caption(f"No KNMI deficit data for {year_min}–{year_max}.")
        return

    years = subset["year"].to_numpy()
    values = subset["deficit_mm"].to_numpy()
    bar_colors = [
        KNMI_DEFICIT_HIGHLIGHT_COLOR if highlight_year is not None and y == highlight_year
        else KNMI_DEFICIT_BAR_COLOR
        for y in years
    ]

    fig, ax = plt.subplots(figsize=(6, 2.4), dpi=100)
    ax.bar(years, values, color=bar_colors, width=0.75, edgecolor="0.3", linewidth=0.4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Deficit (mm)")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(year_min - 0.6, year_max + 0.6)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(x_tick_step))
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def render_knmi_deficit_chart(*, highlight_year: int | None) -> None:
    if not KNMI_DEFICIT_PATH.exists():
        st.caption("KNMI deficit file not found.")
        return

    annual = load_knmi_sep30_deficit(str(KNMI_DEFICIT_PATH))
    if annual.empty:
        st.caption("No KNMI deficit data available.")
        return

    _plot_knmi_deficit_bars(
        annual,
        highlight_year=highlight_year,
        year_min=KNMI_DEFICIT_ZOOM_YEAR_MIN,
        year_max=KNMI_DEFICIT_ZOOM_YEAR_MAX,
        title=f"NL precipitation deficit (30 Sep), {KNMI_DEFICIT_ZOOM_YEAR_MIN}–{KNMI_DEFICIT_ZOOM_YEAR_MAX}",
        x_tick_step=1,
    )
    _plot_knmi_deficit_bars(
        annual,
        highlight_year=highlight_year,
        year_min=KNMI_DEFICIT_YEAR_MIN,
        year_max=int(annual["year"].max()),
        title=f"NL precipitation deficit (30 Sep), {KNMI_DEFICIT_YEAR_MIN}–present",
        x_tick_step=2,
    )

    st.caption("KNMI national cumulative deficit at end of growing season (1 Apr–30 Sep).")
    if highlight_year is not None and highlight_year in set(annual["year"]):
        selected_mm = float(annual.loc[annual["year"] == highlight_year, "deficit_mm"].iloc[0])
        st.caption(f"Selected year {highlight_year}: {selected_mm:.1f} mm")


def _build_article_lookup(records: list) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for rec in records:
        key = rec.get("id", "")
        if not key:
            continue
        features = rec.get("features") or {}
        text = features.get("clean_text") or rec.get("text_content", "")
        result[key] = {
            "clean_text": text,
            "title": features.get("title", ""),
        }
    return result


@st.cache_data
def load_json_articles(raw: bytes) -> dict[str, dict]:
    return _build_article_lookup(json.loads(raw.decode("utf-8")))


@st.cache_data
def load_json_articles_from_path(path: str) -> dict[str, dict]:
    json_path = Path(path)
    if not _is_valid_json_file(json_path):
        raise ValueError(f"Not a valid JSON articles file: {json_path}")
    with json_path.open(encoding="utf-8") as f:
        records = json.load(f)
    return _build_article_lookup(records)


def _existing_dirs(*paths: Path) -> list[Path]:
    return [path for path in paths if path.exists()]


def _point_data_dirs() -> list[Path]:
    return _existing_dirs(POINT_DIR, CHAPTER7_POINT_DIR)


def _nuts3_data_dirs() -> list[Path]:
    return _existing_dirs(NUTS3_DIR, CHAPTER7_NUTS3_DIR)


def _collect_files(dirs: list[Path], pattern: str) -> list[Path]:
    found: dict[str, Path] = {}
    for folder in dirs:
        for path in folder.glob(pattern):
            found[str(path.resolve())] = path
    return sorted(found.values(), key=lambda p: p.name.lower())


def _data_dir_label(path: Path) -> str:
    for label, root in (
        ("chapter 7", CHAPTER7_DIR),
        ("chapter 6", CHAPTER_DIR),
    ):
        try:
            rel = path.relative_to(root)
            return f"{label} / {rel.as_posix()}"
        except ValueError:
            continue
    return path.name


def _format_data_file_label(path: Path) -> str:
    return _data_dir_label(path)


def _is_valid_json_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8") as handle:
        head = handle.read(256).lstrip()
    return head.startswith("[") or head.startswith("{")


def _article_json_candidates(csv_path: Path) -> list[Path]:
    """Candidate article JSON paths for a geocoded CSV (incl. filtered variants)."""
    candidates: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            candidates.append(path)

    add(csv_path.with_suffix(".json"))
    stem = csv_path.stem
    if stem.endswith("_filtered"):
        add(csv_path.with_name(f"{stem[: -len('_filtered')]}.json"))
    return candidates


def find_paired_json(csv_path: Path) -> Path | None:
    for candidate in _article_json_candidates(csv_path):
        if _is_valid_json_file(candidate):
            return candidate
    return None


def find_json_by_filename(filename: str, viewer_mode: str) -> Path | None:
    if not filename or not filename.lower().endswith(".csv"):
        return None
    dirs = _nuts3_data_dirs() if viewer_mode == MODE_NUTS3 else _point_data_dirs()
    csv_name = Path(filename).name
    for folder in dirs:
        csv_path = folder / csv_name
        if csv_path.exists():
            paired = find_paired_json(csv_path)
            if paired is not None:
                return paired
        for candidate in _article_json_candidates(folder / csv_name):
            if _is_valid_json_file(candidate):
                return candidate
    return None


def find_local_csv(viewer_mode: str) -> list[Path]:
    if viewer_mode == MODE_NUTS3:
        return _collect_files(_nuts3_data_dirs(), "*_nuts3.csv")
    return _collect_files(_point_data_dirs(), "*_geocoded*.csv")


def find_local_json(viewer_mode: str) -> list[Path]:
    if viewer_mode == MODE_NUTS3:
        return _collect_files(_nuts3_data_dirs(), "*_nuts3.json")
    return _collect_files(_point_data_dirs(), "*_geocoded*.json")


# ── Point mode ────────────────────────────────────────────────────────────────


def aggregate_by_location(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for loc, grp in df.groupby("location", sort=False):
        first = grp.iloc[0]
        top = grp["classification"].value_counts()
        dominant = top.idxmax() if not top.empty else ""
        dominant_count = int(top.max()) if not top.empty else 0
        dominant_share = dominant_count / len(grp) if len(grp) else 1.0
        base_color = get_classification_color(dominant)
        rows.append({
            "location": loc,
            "geocoded_latitude": first["geocoded_latitude"],
            "geocoded_longitude": first["geocoded_longitude"],
            "geocoded_display_name": first.get("geocoded_display_name"),
            "geocoded_place_type": first.get("geocoded_place_type"),
            "geocoded_country_code": first.get("geocoded_country_code"),
            "impact_count": len(grp),
            "classifications": ", ".join(sorted(grp["classification"].dropna().unique())),
            "dominant_classification": dominant,
            "dominant_share": round(dominant_share, 2),
            "dominant_share_pct": f"{round(dominant_share * 100):.0f}%",
            "color": apply_dominance_alpha(base_color, dominant_share),
            "radius": 2500 + len(grp) * 1500,
        })
    agg = pd.DataFrame(rows).reset_index(drop=True)
    agg["_row_idx"] = agg.index
    return agg


def make_nl_view_state(preset: str) -> pdk.ViewState:
    if preset == MAP_PRESET_CLOSE:
        return pdk.ViewState(**NL_CENTER_CLOSE, pitch=0, bearing=0)
    if preset == MAP_PRESET_CUSTOM:
        custom = st.session_state.get("map_view_custom")
        if custom:
            return pdk.ViewState(
                latitude=custom["latitude"],
                longitude=custom["longitude"],
                zoom=custom["zoom"],
                pitch=0,
                bearing=0,
            )
    lat = (NL_BOUNDS["min_lat"] + NL_BOUNDS["max_lat"]) / 2
    lon = (NL_BOUNDS["min_lon"] + NL_BOUNDS["max_lon"]) / 2
    return pdk.ViewState(latitude=lat, longitude=lon, zoom=NL_FIT_ZOOM, pitch=0, bearing=0)


def _init_map_display_state() -> None:
    if "map_preset" not in st.session_state:
        st.session_state.map_preset = MAP_PRESET_FIT
    if "map_height" not in st.session_state:
        st.session_state.map_height = MAP_DEFAULT_HEIGHT


@dataclass
class MapDisplayState:
    preset: str
    height: int
    view_state: pdk.ViewState


def render_map_display_sidebar() -> MapDisplayState:
    _init_map_display_state()
    if st.session_state.map_preset not in MAP_PRESET_ORDER:
        st.session_state.map_preset = MAP_PRESET_FIT

    st.header("Map display")
    preset = st.selectbox(
        "View preset",
        MAP_PRESET_ORDER,
        index=MAP_PRESET_ORDER.index(st.session_state.map_preset),
    )
    if preset != st.session_state.map_preset:
        st.session_state.map_preset = preset

    height = st.slider(
        "Map height (px)",
        min_value=450,
        max_value=900,
        value=int(st.session_state.map_height),
    )
    st.session_state.map_height = height

    if preset == MAP_PRESET_CUSTOM:
        default_lat = (NL_BOUNDS["min_lat"] + NL_BOUNDS["max_lat"]) / 2
        default_lon = (NL_BOUNDS["min_lon"] + NL_BOUNDS["max_lon"]) / 2
        custom = st.session_state.get("map_view_custom") or {
            "latitude": default_lat,
            "longitude": default_lon,
            "zoom": NL_FIT_ZOOM,
        }
        c1, c2 = st.columns(2)
        custom["latitude"] = c1.number_input(
            "Latitude", value=float(custom["latitude"]), format="%.4f", key="map_custom_lat"
        )
        custom["longitude"] = c2.number_input(
            "Longitude", value=float(custom["longitude"]), format="%.4f", key="map_custom_lon"
        )
        custom["zoom"] = st.slider(
            "Zoom",
            min_value=4.5,
            max_value=12.0,
            value=float(custom["zoom"]),
            step=0.1,
            key="map_custom_zoom",
        )
        st.session_state.map_view_custom = custom

    if st.button("Reset map view", key="reset_map_view"):
        st.session_state.map_preset = MAP_PRESET_FIT
        st.session_state.map_height = MAP_DEFAULT_HEIGHT
        st.session_state.pop("map_view_custom", None)
        st.rerun()

    view_state = make_nl_view_state(st.session_state.map_preset)
    return MapDisplayState(
        preset=st.session_state.map_preset,
        height=height,
        view_state=view_state,
    )


def _render_pydeck(deck: pdk.Deck, *, key: str, height: int, layer_id: str) -> list:
    try:
        event = st.pydeck_chart(
            deck,
            on_select="rerun",
            selection_mode="single-object",
            key=key,
            width="stretch",
            height=height,
        )
        raw_objects = event.selection.objects if (event and event.selection) else {}
        return raw_objects.get(layer_id) or next(iter(raw_objects.values()), [])
    except TypeError:
        st.pydeck_chart(deck, width="stretch", height=height)
        return []


def build_point_map(
    impacts: pd.DataFrame,
    fires: pd.DataFrame | None = None,
    *,
    view_state: pdk.ViewState | None = None,
) -> pdk.Deck:
    layers: list[pdk.Layer] = []
    if not impacts.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id=LAYER_ID_IMPACTS,
                data=impacts,
                get_position=["geocoded_longitude", "geocoded_latitude"],
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                auto_highlight=True,
            )
        )
    if fires is not None and not fires.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id=LAYER_ID_FIRES,
                data=fires,
                get_position=["longitude", "latitude"],
                get_fill_color="color",
                get_radius="radius",
                pickable=False,
            )
        )
    tooltip = {
        "html": (
            "<b>{location}</b><br/>"
            "{impact_count} impact(s)<br/>"
            "Dominant: {dominant_classification} ({dominant_share_pct})<br/>"
            "{classifications}<br/>"
            "<i>{geocoded_display_name}</i><br/>"
            "<b>Wildfire</b> {month_label}<br/>"
            "Count: {wildfires_sum}"
        ),
        "style": {
            "backgroundColor": "rgba(30,30,30,0.85)",
            "color": "white",
            "padding": "6px",
            "fontSize": "13px",
        },
    }
    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state or make_nl_view_state(MAP_PRESET_FIT),
        map_style=pdk.map_styles.LIGHT,
        tooltip=tooltip,
    )


def show_location_panel(
    location: str,
    impacts: pd.DataFrame,
    article_lookup: dict[str, dict] | None,
) -> None:
    first = impacts.iloc[0]
    st.subheader(f"📍 {location}")
    meta_parts = [
        first.get("geocoded_display_name"),
        first.get("geocoded_place_type"),
        first.get("geocoded_country_code"),
    ]
    rank = first.get("geocoded_place_rank")
    if pd.notna(rank):
        meta_parts.append(f"rank {int(rank)}")
    st.caption(" · ".join(str(p) for p in meta_parts if p and pd.notna(p)))
    st.markdown(f"**{len(impacts)} impact{'s' if len(impacts) > 1 else ''} at this location**")
    display_cols = [
        "classification", "severity", "confidence", "evidence",
        "recency_in_months", "article_title",
    ]
    display_df = attach_article_titles(impacts, article_lookup)
    st.dataframe(
        display_df[[c for c in display_cols if c in display_df.columns]].reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )
    if article_lookup:
        seen: set[str] = set()
        for _, row in impacts.iterrows():
            aid = str(row.get("id") or row.get("article_id", ""))
            if not aid or aid in seen:
                continue
            seen.add(aid)
            article = article_lookup.get(aid)
            if article and article.get("clean_text"):
                label = f"📰 {article['title']}" if article.get("title") else f"📰 Article {aid}"
                with st.expander(label):
                    st.markdown(article["clean_text"])
            else:
                st.caption(f"No article text found for ID `{aid}`")


# ── NUTS3 mode ────────────────────────────────────────────────────────────────


def aggregate_by_nuts3(df: pd.DataFrame) -> pd.DataFrame:
    mapped = df[df["nuts_level"] == "nuts3"].copy()
    mapped = mapped[mapped["nuts3_id"] != ""]
    if mapped.empty:
        return pd.DataFrame()

    rows = []
    for nuts3_id, grp in mapped.groupby("nuts3_id", sort=False):
        first = grp.iloc[0]
        top = grp["classification"].value_counts()
        dominant = top.idxmax() if not top.empty else ""
        rows.append({
            "nuts3_id": nuts3_id,
            "nuts3_name": first.get("nuts3_name", ""),
            "nuts2_id": first.get("nuts2_id", ""),
            "nuts2_name": first.get("nuts2_name", ""),
            "impact_count": float(grp["mention_weight"].sum()),
            "row_count": len(grp),
            "article_count": grp["id"].nunique(),
            "classifications": ", ".join(sorted(grp["classification"].dropna().unique())),
            "dominant_classification": dominant,
            "color": get_classification_color(dominant),
        })
    return pd.DataFrame(rows).reset_index(drop=True)


def _heat_fill_color(impact_count: float, max_count: float) -> list[int]:
    if impact_count <= 0:
        return EMPTY_REGION_COLOR
    ratio = impact_count / max_count if max_count > 0 else 0.0
    if ratio < 0.25:
        return [254, 224, 210, 220]
    if ratio < 0.5:
        return [252, 141, 89, 230]
    if ratio < 0.75:
        return [227, 74, 51, 240]
    return [165, 15, 21, 255]


def build_region_geojson(nuts_geojson: dict, agg: pd.DataFrame) -> dict:
    agg_index = (
        agg.set_index("nuts3_id")[
            ["nuts3_name", "nuts2_id", "nuts2_name", "impact_count", "row_count",
             "article_count", "classifications", "dominant_classification"]
        ].to_dict("index")
        if not agg.empty
        else {}
    )
    max_count = float(agg["impact_count"].max()) if not agg.empty else 1.0
    features = []
    for idx, feature in enumerate(nuts_geojson.get("features", [])):
        props = dict(feature.get("properties") or {})
        nuts3_id = props.get("nuts3_id", "")
        row = agg_index.get(nuts3_id, {})
        impact_count = float(row.get("impact_count", 0.0))
        props.update({
            "nuts3_id": nuts3_id,
            "nuts3_name": row.get("nuts3_name") or props.get("nuts3_name", ""),
            "nuts2_id": row.get("nuts2_id", ""),
            "nuts2_name": row.get("nuts2_name", ""),
            "impact_count": impact_count,
            "row_count": int(row.get("row_count", 0)),
            "article_count": int(row.get("article_count", 0)),
            "classifications": row.get("classifications", ""),
            "dominant_classification": row.get("dominant_classification", ""),
            "fill_color": _heat_fill_color(impact_count, max_count),
            "_row_idx": idx,
        })
        features.append({
            "type": "Feature",
            "geometry": feature.get("geometry"),
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


def build_nuts3_map(
    regions_geojson: dict,
    fires: pd.DataFrame | None = None,
    *,
    view_state: pdk.ViewState | None = None,
) -> pdk.Deck:
    layers: list[pdk.Layer] = []
    if regions_geojson.get("features"):
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                regions_geojson,
                id=LAYER_ID_NUTS3,
                get_fill_color="properties.fill_color",
                stroked=True,
                filled=True,
                wireframe=False,
                get_line_color=[70, 70, 70, 200],
                line_width_min_pixels=1,
                pickable=True,
                auto_highlight=True,
            )
        )
    if fires is not None and not fires.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                id=LAYER_ID_FIRES,
                data=fires,
                get_position=["longitude", "latitude"],
                get_fill_color="color",
                get_radius="radius",
                pickable=False,
            )
        )
    tooltip = {
        "html": (
            "<b>{nuts3_name}</b> ({nuts3_id})<br/>"
            "{impact_count} weighted impact(s)<br/>"
            "{article_count} article(s)<br/>"
            "{classifications}<br/>"
            "<b>Wildfire</b> {month_label}<br/>"
            "Count: {wildfires_sum}"
        ),
        "style": {
            "backgroundColor": "rgba(30,30,30,0.85)",
            "color": "white",
            "padding": "6px",
            "fontSize": "13px",
        },
    }
    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state or make_nl_view_state(MAP_PRESET_FIT),
        map_style=pdk.map_styles.LIGHT,
        tooltip=tooltip,
    )


def show_region_panel(
    nuts3_id: str,
    nuts3_name: str,
    impacts: pd.DataFrame,
    article_lookup: dict[str, dict] | None,
) -> None:
    st.subheader(f"NUTS3: {nuts3_name or nuts3_id}")
    st.caption(f"Region ID: **{nuts3_id}**")
    weighted = impacts["mention_weight"].sum() if "mention_weight" in impacts.columns else len(impacts)
    st.markdown(
        f"**{weighted:.1f}** weighted impacts "
        f"({len(impacts)} row{'s' if len(impacts) != 1 else ''}, "
        f"{impacts['id'].nunique()} article{'s' if impacts['id'].nunique() != 1 else ''})"
    )
    display_cols = [
        "location", "classification", "severity", "confidence",
        "evidence", "recency_in_months", "mention_weight", "article_title",
        "geocoded_place_type", "geocoded_place_rank", "geocoded_is_low_quality",
    ]
    display_df = attach_article_titles(impacts, article_lookup)
    st.dataframe(
        display_df[[c for c in display_cols if c in display_df.columns]].reset_index(drop=True),
        width="stretch",
        hide_index=True,
    )
    if article_lookup:
        seen: set[str] = set()
        for _, row in impacts.iterrows():
            aid = str(row.get("id") or row.get("article_id", ""))
            if not aid or aid in seen:
                continue
            seen.add(aid)
            article = article_lookup.get(aid)
            if article and article.get("clean_text"):
                label = f"📰 {article['title']}" if article.get("title") else f"📰 Article {aid}"
                with st.expander(label):
                    st.markdown(article["clean_text"])
            else:
                st.caption(f"No article text found for ID `{aid}`")


# ── Shared map / filter helpers ───────────────────────────────────────────────


def prepare_fires_for_map(fires: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    if len(fires) <= MAX_FIRE_MAP_POINTS:
        return fires, None
    sampled = fires.sample(n=MAX_FIRE_MAP_POINTS, random_state=42)
    msg = (
        f"Subsampled {MAX_FIRE_MAP_POINTS:,} of {len(fires):,} fire points "
        "for map performance"
    )
    return sampled, msg


RECENCY_PRESET_ORDER = [
    "All", "0 months only", "0–1 months", "0–6 months",
    "6–12 months", "12–24 months", "24+ months", "Custom",
]
RECENCY_PRESET_RANGES: dict[str, tuple[int, int] | None] = {
    "All": None,
    "0 months only": (0, 0),
    "0–1 months": (0, 1),
    "0–6 months": (0, 6),
    "6–12 months": (6, 12),
    "12–24 months": (12, 24),
    "24+ months": None,
}


def _resolve_preset_range(preset: str, max_rec: int) -> tuple[int, int]:
    max_rec = max(int(max_rec), 1)
    if preset == "All":
        return 0, max_rec
    if preset == "24+ months":
        return min(24, max_rec), max_rec
    fixed = RECENCY_PRESET_RANGES.get(preset)
    if fixed is not None:
        lo, hi = fixed
        lo = min(max(0, lo), max_rec)
        hi = min(max(lo, hi), max_rec)
        return lo, hi
    prev = st.session_state.get("recency_range", (0, max_rec))
    lo = min(max(0, int(prev[0])), max_rec)
    hi = min(max(lo, int(prev[1])), max_rec)
    return lo, hi


def _preset_for_range(lo: int, hi: int, max_rec: int) -> str:
    for preset in RECENCY_PRESET_ORDER:
        if preset == "Custom":
            continue
        plo, phi = _resolve_preset_range(preset, max_rec)
        if (plo, phi) == (lo, hi):
            return preset
    return "Custom"


def _recency_mask(df: pd.DataFrame, lo: int, hi: int, include_unknown: bool) -> pd.Series:
    known = df["recency_in_months"].between(lo, hi)
    if include_unknown:
        return known | df["recency_in_months"].isna()
    return known


def render_recency_filter(df: pd.DataFrame) -> tuple[int, int, bool]:
    if df["recency_in_months"].notna().any():
        max_rec = int(df["recency_in_months"].max())
    else:
        max_rec = 24
    # Streamlit requires slider min < max; all-zero recency would otherwise crash.
    max_rec = max(max_rec, 1)

    if "recency_range" not in st.session_state:
        st.session_state.recency_range = (0, max_rec)
    else:
        lo, hi = st.session_state.recency_range
        lo = int(max(0, min(lo, max_rec)))
        hi = int(max(lo, min(hi, max_rec)))
        st.session_state.recency_range = (lo, hi)

    if "recency_preset" not in st.session_state:
        st.session_state.recency_preset = "All"
    if st.session_state.recency_preset not in RECENCY_PRESET_ORDER:
        st.session_state.recency_preset = "All"

    preset = st.selectbox(
        "Recency preset",
        RECENCY_PRESET_ORDER,
        index=RECENCY_PRESET_ORDER.index(st.session_state.recency_preset),
    )
    if preset != st.session_state.recency_preset:
        st.session_state.recency_preset = preset
        if preset != "Custom":
            st.session_state.recency_range = _resolve_preset_range(preset, max_rec)

    slider_range = st.slider("Recency (months)", 0, max_rec, value=st.session_state.recency_range)
    if slider_range != st.session_state.recency_range:
        st.session_state.recency_range = slider_range
        matched = _preset_for_range(slider_range[0], slider_range[1], max_rec)
        if matched != st.session_state.recency_preset:
            st.session_state.recency_preset = matched
            st.rerun()

    rec_lo, rec_hi = st.session_state.recency_range
    include_unknown = st.checkbox("Include unknown recency", value=True)
    n_match = int(_recency_mask(df, rec_lo, rec_hi, include_unknown).sum())
    st.caption(f"{n_match:,} impact rows match recency filter")
    return rec_lo, rec_hi, include_unknown


@dataclass
class FilterState:
    sel_models: list
    sel_classes: list
    sev_range: tuple[int, int]
    rec_lo: int
    rec_hi: int
    include_unknown_recency: bool
    show_wildfires: bool
    fires_df: pd.DataFrame | None
    time_mode: bool
    selected_month: object
    time_window: int
    map_nuts3_only: bool = True
    min_geocode_rank: int = DEFAULT_MIN_GEOCODE_RANK
    exclude_low_quality: bool = True


def render_data_sidebar(viewer_mode: str) -> tuple[bytes | None, dict | None]:
    st.header("Data")
    csv_label = "NUTS3 CSV" if viewer_mode == MODE_NUTS3 else "Geocoded CSV"
    default_csv = DEFAULT_NUTS3_CSV if viewer_mode == MODE_NUTS3 else DEFAULT_POINT_CSV
    csv_bytes: bytes | None = None
    article_lookup: dict | None = None
    chosen_csv_path: Path | None = None
    uploaded_csv_name: str | None = None

    csv_source = st.radio(
        "CSV source",
        ["Default file", "Upload file", "Local file"],
        horizontal=True,
        index=0,
    )
    if csv_source == "Default file":
        if default_csv.is_file():
            chosen_csv_path = default_csv
            csv_bytes = chosen_csv_path.read_bytes()
            st.caption(f"Default: {_format_data_file_label(default_csv)}")
        else:
            st.warning(f"Default CSV not found: {_format_data_file_label(default_csv)}")
    elif csv_source == "Upload file":
        uploaded_csv = st.file_uploader(csv_label, type="csv")
        if uploaded_csv:
            csv_bytes = uploaded_csv.read()
            uploaded_csv_name = uploaded_csv.name
    else:
        local_csvs = find_local_csv(viewer_mode)
        if local_csvs:
            default_index = 0
            default_resolved = str(default_csv.resolve()) if default_csv.exists() else ""
            for idx, path in enumerate(local_csvs):
                if str(path.resolve()) == default_resolved:
                    default_index = idx
                    break
            chosen_csv_path = st.selectbox(
                "Select file",
                local_csvs,
                index=default_index,
                format_func=_format_data_file_label,
            )
            csv_bytes = chosen_csv_path.read_bytes()
        else:
            pattern = "*_nuts3.csv" if viewer_mode == MODE_NUTS3 else "*_geocoded*.csv"
            st.warning(
                f"No {pattern} found in chapter 6 final/ or processed/ folders."
            )

    paired_json_path = (
        find_paired_json(chosen_csv_path)
        if chosen_csv_path is not None
        else find_json_by_filename(uploaded_csv_name or "", viewer_mode)
    )
    load_articles_default = paired_json_path is not None

    st.markdown("---")
    if st.checkbox("Load original articles (JSON)", value=load_articles_default):
        manual_json = st.checkbox(
            "Choose JSON manually",
            value=paired_json_path is None,
            help="Disable to use the JSON paired with the selected CSV filename.",
        )

        if not manual_json and paired_json_path is not None:
            with st.spinner("Loading paired articles (cached after first load)…"):
                try:
                    article_lookup = load_json_articles_from_path(str(paired_json_path))
                except (json.JSONDecodeError, ValueError) as exc:
                    st.error(
                        f"Could not load paired articles from {_format_data_file_label(paired_json_path)}: {exc}"
                    )
                    article_lookup = {}
                else:
                    st.caption(f"Paired articles: {_format_data_file_label(paired_json_path)}")
                    st.success(f"Loaded {len(article_lookup):,} articles")
        else:
            json_source = st.radio(
                "JSON source", ["Local file", "Upload file"], horizontal=True, key="json_radio"
            )
            if json_source == "Local file":
                local_jsons = find_local_json(viewer_mode)
                if local_jsons:
                    default_json_index = 0
                    if paired_json_path is not None:
                        resolved = str(paired_json_path.resolve())
                        for idx, path in enumerate(local_jsons):
                            if str(path.resolve()) == resolved:
                                default_json_index = idx
                                break
                    chosen_json = st.selectbox(
                        "Select JSON",
                        local_jsons,
                        index=default_json_index,
                        format_func=_format_data_file_label,
                        key="json_select",
                    )
                    with st.spinner("Loading articles (cached after first load)…"):
                        article_lookup = load_json_articles_from_path(str(chosen_json))
                    st.success(f"Loaded {len(article_lookup):,} articles")
                else:
                    pattern = "*_nuts3.json" if viewer_mode == MODE_NUTS3 else "*_geocoded*.json"
                    st.warning(
                        f"No {pattern} found in chapter 6 or chapter 7 geocoding folders."
                    )
            else:
                uploaded_json = st.file_uploader("Articles JSON", type="json")
                if uploaded_json:
                    with st.spinner("Loading…"):
                        article_lookup = load_json_articles(uploaded_json.read())

    return csv_bytes, article_lookup


def render_filter_sidebar(df: pd.DataFrame, *, nuts3_mode: bool) -> FilterState:
    st.markdown("---")
    st.header("Filters")

    all_models = sorted(df["model_name"].dropna().unique())
    sel_models = st.multiselect("Model", all_models, default=all_models)
    all_classes = sorted(df["classification"].dropna().unique())
    sel_classes = st.multiselect("Classification", all_classes, default=all_classes)

    min_sev = int(df["severity"].min())
    max_sev = int(df["severity"].max())
    sev_range = (
        st.slider("Severity", min_sev, max_sev, (min_sev, max_sev))
        if min_sev < max_sev
        else (min_sev, max_sev)
    )
    rec_lo, rec_hi, include_unknown_recency = render_recency_filter(df)
    map_nuts3_only = st.toggle("Map NL NUTS3 only", value=True) if nuts3_mode else True

    st.markdown("---")
    st.header("Geocode quality")
    _render_geocode_rank_reference()
    min_geocode_rank = st.slider(
        "Minimum geocode rank",
        0,
        100,
        DEFAULT_MIN_GEOCODE_RANK,
        help="Default 40. Village=50, canal/river=41, city/town=55+, region=90+. Tourism=9 (low-quality).",
    )
    exclude_low_quality = st.checkbox("Exclude low-quality geocodes", value=True)

    st.markdown("---")
    st.header("Wildfire layer")
    show_wildfires = st.toggle("Show wildfire occurrences", value=False)
    fires_df: pd.DataFrame | None = None
    if show_wildfires:
        if DEFAULT_WILDFIRE_CSV.exists():
            with st.spinner("Loading wildfire dataset (cached after first load)…"):
                fires_df = load_wildfire_occurrences(str(DEFAULT_WILDFIRE_CSV))
            st.caption(f"{len(fires_df):,} fire occurrence cells loaded")
        else:
            st.warning(f"Wildfire CSV not found: {DEFAULT_WILDFIRE_CSV}")
            show_wildfires = False

    st.markdown("---")
    st.header("Time Mode")
    time_mode = st.toggle("Enable time filter", value=False)
    selected_month = None
    time_window = 1
    if time_mode:
        impact_months = set(df["year_month"].dropna())
        fire_months = set(fires_df["year_month"].dropna()) if fires_df is not None else set()
        valid_months = sorted(impact_months | fire_months)
        if valid_months:
            month_labels = [m.to_timestamp().strftime("%b %Y") for m in valid_months]
            if len(month_labels) == 1:
                selected_label = month_labels[0]
                st.caption(f"Month: {selected_label}")
            else:
                selected_label = st.select_slider(
                    "Select month",
                    options=month_labels,
                    value=month_labels[len(month_labels) // 2],
                )
            selected_idx = month_labels.index(selected_label)
            selected_month = valid_months[selected_idx]
            time_window = st.select_slider(
                "Window (months each side)", options=[0, 1, 2, 3, 6], value=1
            )
            lo = selected_month - time_window
            hi = selected_month + time_window
            parts = [f"{df['year_month'].between(lo, hi).sum()} impact rows in window"]
            if fires_df is not None:
                parts.append(f"{fires_df['year_month'].between(lo, hi).sum()} fire cells in window")
            st.caption(" · ".join(parts))
            st.markdown("---")
            st.subheader("KNMI drought context")
            render_knmi_deficit_chart(highlight_year=int(selected_month.year))
        else:
            st.warning("No dated records in the impact or wildfire datasets.")
            time_mode = False

    preview_flt = FilterState(
        sel_models=sel_models,
        sel_classes=sel_classes,
        sev_range=sev_range,
        rec_lo=rec_lo,
        rec_hi=rec_hi,
        include_unknown_recency=include_unknown_recency,
        show_wildfires=show_wildfires,
        fires_df=fires_df,
        time_mode=time_mode,
        selected_month=selected_month,
        time_window=time_window,
        map_nuts3_only=map_nuts3_only,
        min_geocode_rank=min_geocode_rank,
        exclude_low_quality=exclude_low_quality,
    )
    geo_excluded, geo_scope = count_geocode_excluded(df, preview_flt)
    st.caption(
        f"{geo_excluded:,} of {geo_scope:,} content-filtered rows excluded by geocode quality"
    )

    return FilterState(
        sel_models=sel_models,
        sel_classes=sel_classes,
        sev_range=sev_range,
        rec_lo=rec_lo,
        rec_hi=rec_hi,
        include_unknown_recency=include_unknown_recency,
        show_wildfires=show_wildfires,
        fires_df=fires_df,
        time_mode=time_mode,
        selected_month=selected_month,
        time_window=time_window,
        map_nuts3_only=map_nuts3_only,
        min_geocode_rank=min_geocode_rank,
        exclude_low_quality=exclude_low_quality,
    )


def _content_filter_mask(df: pd.DataFrame, flt: FilterState) -> pd.Series:
    mask = (
        df["model_name"].isin(flt.sel_models)
        & df["classification"].isin(flt.sel_classes)
        & df["severity"].between(flt.sev_range[0], flt.sev_range[1])
        & _recency_mask(df, flt.rec_lo, flt.rec_hi, flt.include_unknown_recency)
    )
    if flt.time_mode and flt.selected_month is not None:
        lo = flt.selected_month - flt.time_window
        hi = flt.selected_month + flt.time_window
        mask &= df["year_month"].between(lo, hi)
    return mask


def _geocode_quality_mask(df: pd.DataFrame, flt: FilterState) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if flt.exclude_low_quality:
        mask &= ~df["geocoded_is_low_quality"].fillna(False)
    if flt.min_geocode_rank > 0:
        mask &= df["geocoded_place_rank"].isna() | (
            df["geocoded_place_rank"] >= flt.min_geocode_rank
        )
    return mask


def count_geocode_excluded(df: pd.DataFrame, flt: FilterState) -> tuple[int, int]:
    scoped = df[_content_filter_mask(df, flt)]
    excluded = int((~_geocode_quality_mask(scoped, flt)).sum())
    return excluded, len(scoped)


def _apply_filters(df: pd.DataFrame, flt: FilterState) -> pd.DataFrame:
    mask = _content_filter_mask(df, flt) & _geocode_quality_mask(df, flt)
    return df[mask].reset_index(drop=True)


def _filter_fires(flt: FilterState) -> tuple[pd.DataFrame, str | None]:
    if not flt.show_wildfires or flt.fires_df is None:
        return pd.DataFrame(), None
    fires = flt.fires_df.copy()
    if flt.time_mode and flt.selected_month is not None:
        lo = flt.selected_month - flt.time_window
        hi = flt.selected_month + flt.time_window
        fires = fires[fires["year_month"].between(lo, hi)].reset_index(drop=True)
    if fires.empty:
        return pd.DataFrame(), None
    return prepare_fires_for_map(fires)


def _time_label(flt: FilterState) -> str:
    if flt.time_mode and flt.selected_month is not None:
        lo = (flt.selected_month - flt.time_window).to_timestamp().strftime("%b %Y")
        hi = (flt.selected_month + flt.time_window).to_timestamp().strftime("%b %Y")
        return f" — {lo} → {hi}"
    return ""


def _classes_for_theme(theme: str) -> list[str]:
    return sorted(cls for cls, t in CLASSIFICATION_THEME.items() if t == theme)


def _render_theme_grouped_legend(cols: list) -> None:
    for i, theme in enumerate(THEME_ORDER):
        color = THEME_COLORS[theme]
        r, g, b = color[:3]
        classes = ", ".join(_classes_for_theme(theme))
        cols[i % len(cols)].markdown(
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
            f'background:rgb({r},{g},{b});margin-right:6px;vertical-align:middle;"></span>'
            f"**{THEME_LABELS[theme]}** — {classes}",
            unsafe_allow_html=True,
        )


def _render_classification_legend(show_wildfires: bool) -> None:
    st.caption(
        "Opacity reflects dominance: solid = one type at location; faded = mixed impacts."
    )
    cols = st.columns(2)
    _render_theme_grouped_legend(cols)
    if show_wildfires:
        r, g, b = FIRE_COLOR[:3]
        cols[0].markdown(
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
            f'background:rgb({r},{g},{b});margin-right:6px;vertical-align:middle;"></span>'
            "Wildfire occurrence (dataset)",
            unsafe_allow_html=True,
        )


# ── Mode runners ──────────────────────────────────────────────────────────────


def run_point_viewer(csv_bytes: bytes, article_lookup: dict | None) -> None:
    df = load_geocoded_csv(csv_bytes)
    with st.sidebar:
        map_disp = render_map_display_sidebar()
        st.markdown("---")
        flt = render_filter_sidebar(df, nuts3_mode=False)

    filtered = _apply_filters(df, flt)
    geo_excluded, _ = count_geocode_excluded(df, flt)
    agg = aggregate_by_location(filtered) if not filtered.empty else pd.DataFrame()
    filtered_fires, fire_subsample_msg = _filter_fires(flt)

    caption_parts = []
    if not agg.empty:
        caption_parts.append(f"**{len(agg)}** locations, **{len(filtered)}** total impacts")
    else:
        caption_parts.append("**0** impact locations")
    if geo_excluded:
        caption_parts.append(f"**{geo_excluded}** rows excluded by geocode quality")
    if flt.show_wildfires and flt.fires_df is not None:
        fires = flt.fires_df
        if flt.time_mode and flt.selected_month is not None:
            lo = flt.selected_month - flt.time_window
            hi = flt.selected_month + flt.time_window
            fires = fires[fires["year_month"].between(lo, hi)]
        caption_parts.append(f"**{len(fires)}** fire cells")
    st.caption(", ".join(caption_parts) + _time_label(flt))
    if fire_subsample_msg:
        st.caption(fire_subsample_msg)
    if has_impacts := not filtered.empty:
        st.caption(
            "Point color = most frequent impact type at that location. "
            "**Fainter dots** mean that type is a smaller share of impacts there — "
            "click a point for all impacts."
        )

    has_fires = flt.show_wildfires and not filtered_fires.empty
    if not has_impacts and not has_fires:
        st.warning("No points match the current filters.")
        return

    deck = build_point_map(
        agg if has_impacts else pd.DataFrame(),
        filtered_fires if has_fires else None,
        view_state=map_disp.view_state,
    )
    selected_objects = _render_pydeck(
        deck, key="point_map", height=map_disp.height, layer_id=LAYER_ID_IMPACTS
    )

    with st.expander("Legend", expanded=False):
        _render_classification_legend(flt.show_wildfires)

    st.markdown("---")
    if selected_objects and has_impacts:
        obj = selected_objects[0]
        row_idx = obj.get("_row_idx")
        if row_idx is not None and int(row_idx) < len(agg):
            location = agg.iloc[int(row_idx)]["location"]
        else:
            location = obj.get("location", "")
        show_location_panel(location, filtered[filtered["location"] == location], article_lookup)
    elif has_impacts:
        st.markdown("*Click a point on the map to see all impacts at that location.*")

    with st.expander("Raw data table", expanded=False):
        cols = [
            "location", "classification", "severity", "confidence", "evidence",
            "recency_in_months", "geocoded_latitude", "geocoded_longitude",
            "geocoded_display_name", "geocoded_place_type", "geocoded_place_rank",
            "geocoded_is_low_quality", "model_name",
        ]
        st.dataframe(filtered[[c for c in cols if c in filtered.columns]], width="stretch")


def run_nuts3_viewer(csv_bytes: bytes, article_lookup: dict | None) -> None:
    df = load_nuts3_csv(csv_bytes)
    nuts_geojson = load_nuts3_geojson(str(LOCAL_NUTS_GEOJSON))
    with st.sidebar:
        map_disp = render_map_display_sidebar()
        st.markdown("---")
        flt = render_filter_sidebar(df, nuts3_mode=True)

    filtered = _apply_filters(df, flt)
    geo_excluded, _ = count_geocode_excluded(df, flt)
    map_df = filtered[filtered["nuts_level"] == "nuts3"] if flt.map_nuts3_only else filtered
    agg = aggregate_by_nuts3(map_df)
    filtered_fires, fire_subsample_msg = _filter_fires(flt)

    weighted_total = float(map_df["mention_weight"].sum()) if not map_df.empty else 0.0
    caption_parts = [
        f"**{len(agg)}** NUTS3 regions with data",
        f"**{weighted_total:.0f}** weighted impacts",
        f"**{len(filtered)}** total filtered rows",
    ]
    if geo_excluded:
        caption_parts.append(f"**{geo_excluded}** rows excluded by geocode quality")
    if flt.show_wildfires and flt.fires_df is not None:
        fires = flt.fires_df
        if flt.time_mode and flt.selected_month is not None:
            lo = flt.selected_month - flt.time_window
            hi = flt.selected_month + flt.time_window
            fires = fires[fires["year_month"].between(lo, hi)]
        caption_parts.append(f"**{len(fires)}** fire cells")
    st.caption(", ".join(caption_parts) + _time_label(flt))
    if fire_subsample_msg:
        st.caption(fire_subsample_msg)

    has_regions = bool(nuts_geojson.get("features"))
    has_fires = flt.show_wildfires and not filtered_fires.empty
    if not has_regions and not has_fires:
        st.warning("No map data available.")
        return

    regions_geojson = build_region_geojson(nuts_geojson, agg)
    deck = build_nuts3_map(
        regions_geojson,
        filtered_fires if has_fires else None,
        view_state=map_disp.view_state,
    )

    selected_nuts3_id: str | None = None
    selected_objects = _render_pydeck(
        deck, key="nuts3_map", height=map_disp.height, layer_id=LAYER_ID_NUTS3
    )
    if selected_objects:
        obj = selected_objects[0]
        props = obj.get("properties") or {}
        selected_nuts3_id = props.get("nuts3_id") or obj.get("nuts3_id")

    with st.expander("Legend", expanded=False):
        cols = st.columns(2)
        cols[0].markdown("**Impact intensity (choropleth)**")
        for label, color in [
            ("No impacts", EMPTY_REGION_COLOR),
            ("Low", [254, 224, 210, 220]),
            ("Medium-low", [252, 141, 89, 230]),
            ("Medium-high", [227, 74, 51, 240]),
            ("High", [165, 15, 21, 255]),
        ]:
            r, g, b = color[:3]
            cols[0].markdown(
                f'<span style="display:inline-block;width:12px;height:12px;border-radius:2px;'
                f'background:rgb({r},{g},{b});margin-right:6px;vertical-align:middle;"></span>'
                f"{label}",
                unsafe_allow_html=True,
            )
        cols[1].markdown("**Classification colors (detail table)**")
        _render_theme_grouped_legend([cols[1]])
        if flt.show_wildfires:
            r, g, b = FIRE_COLOR[:3]
            cols[0].markdown(
                f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
                f'background:rgb({r},{g},{b});margin-right:6px;vertical-align:middle;"></span>'
                "Wildfire occurrence (dataset)",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    if selected_nuts3_id and not map_df.empty:
        region_impacts = map_df[map_df["nuts3_id"] == selected_nuts3_id]
        if not region_impacts.empty:
            show_region_panel(
                selected_nuts3_id,
                region_impacts.iloc[0].get("nuts3_name", ""),
                region_impacts,
                article_lookup,
            )
        else:
            st.markdown(f"*No impacts in current filters for region `{selected_nuts3_id}`.*")
    elif not map_df.empty:
        st.markdown("*Click a NUTS3 region on the map to see all impacts in that area.*")

    with st.expander("Raw data table", expanded=False):
        cols = [
            "location", "nuts3_id", "nuts3_name", "nuts_level", "classification",
            "severity", "confidence", "evidence", "recency_in_months",
            "mention_weight", "geocoded_display_name", "geocoded_place_type",
            "geocoded_place_rank", "geocoded_is_low_quality", "model_name",
        ]
        st.dataframe(filtered[[c for c in cols if c in filtered.columns]], width="stretch")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Drought Impact Viewer",
        page_icon="🗺️",
        layout="wide",
    )
    st.title("Drought Impact Viewer")

    with st.sidebar:
        viewer_mode = st.radio(
            "Viewer mode",
            [MODE_POINT, MODE_NUTS3],
            horizontal=True,
        )
        st.markdown("---")
        csv_bytes, article_lookup = render_data_sidebar(viewer_mode)

    if csv_bytes is None:
        st.info("Load a CSV file using the sidebar to get started.")
        return

    if viewer_mode == MODE_NUTS3:
        if not LOCAL_NUTS_GEOJSON.exists():
            st.error(f"NUTS GeoJSON not found: {LOCAL_NUTS_GEOJSON}. Run nuts3_coder.py first.")
            return
        run_nuts3_viewer(csv_bytes, article_lookup)
    else:
        run_point_viewer(csv_bytes, article_lookup)


if __name__ == "__main__":
    main()

# cd "Thesis specific chapters\chapter 6 geocoding\combined nuts-point vieuwer"
# streamlit run combined_vieuwer.py
