"""Helpers for Copernicus soil moisture anomaly GeoTIFFs (water/nodata masking)."""

from __future__ import annotations

import numpy as np
from rasterio.io import MemoryFile

SOIL_DATA_FOLDER = "Soil moisture index anomoly 2004-2026"
SOIL_COVERAGE_START = "2004-01-01"


def sanitize_soil_grid(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    """Mask water/non-soil fill values (nodata ~1e20) and non-finite pixels."""
    out = np.asarray(arr, dtype=float)
    if nodata is not None:
        out[np.isclose(out, nodata) | (out > 1e10)] = np.nan
    else:
        out[out > 1e10] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def read_soil_tif_bytes(data: bytes) -> tuple[np.ndarray, object, float | None, object]:
    """Read one soil GeoTIFF from raw bytes; return masked array, bounds, nodata, transform."""
    with MemoryFile(data) as mem:
        with mem.open() as src:
            nodata = src.nodata
            arr = sanitize_soil_grid(src.read(1), nodata)
            extent = src.bounds
            transform = src.transform
    return arr, extent, nodata, transform
