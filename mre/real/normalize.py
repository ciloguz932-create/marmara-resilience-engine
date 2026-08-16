"""Reproject and clean the raw OSM layers into analysis-ready GeoDataFrames.

Takes the EPSG:4326 layers from ``mre.real.osm`` and produces EPSG:32635
(the engine's CRS) layers with invalid geometries repaired, empties and
duplicates dropped, and a cleaning report recording exactly what was removed —
so nothing disappears silently. No attribute is invented; cleaning only ever
*removes* unusable geometry, never fills a missing value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd

__all__ = ["CleaningReport", "normalize_layers"]


@dataclass(slots=True)
class CleaningReport:
    """What normalisation removed, per layer. All-zero on clean input."""

    target_crs: str
    per_layer: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, layer: str, *, input_n: int, invalid_fixed: int,
               dropped_empty: int, dropped_duplicate: int, output_n: int) -> None:
        self.per_layer[layer] = {
            "input_features": input_n,
            "invalid_geometry_repaired": invalid_fixed,
            "dropped_empty_or_null": dropped_empty,
            "dropped_duplicate_osm_id": dropped_duplicate,
            "output_features": output_n,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"target_crs": self.target_crs, "layers": self.per_layer}


def _clean(gdf: gpd.GeoDataFrame, target_crs: str, report: CleaningReport, layer: str) -> gpd.GeoDataFrame:
    input_n = len(gdf)
    if input_n == 0:
        report.record(layer, input_n=0, invalid_fixed=0, dropped_empty=0, dropped_duplicate=0, output_n=0)
        return gdf.to_crs(target_crs) if gdf.crs else gdf

    gdf = gdf.to_crs(target_crs)

    # Drop null / empty geometry.
    non_empty = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    dropped_empty = input_n - len(non_empty)

    # Repair invalid geometry with a zero-width buffer (a standard fix). Count
    # how many needed it.
    invalid_mask = ~non_empty.geometry.is_valid
    invalid_fixed = int(invalid_mask.sum())
    if invalid_fixed:
        non_empty.loc[invalid_mask, non_empty.geometry.name] = non_empty.loc[
            invalid_mask, non_empty.geometry.name
        ].buffer(0)
        # A buffer(0) can still leave an empty geometry for degenerate input; drop those.
        still_bad = non_empty.geometry.is_empty | ~non_empty.geometry.is_valid
        if still_bad.any():
            dropped_empty += int(still_bad.sum())
            non_empty = non_empty[~still_bad].copy()

    # Drop duplicate OSM ids (keep first), if the column is present.
    dropped_duplicate = 0
    if "osm_id" in non_empty.columns:
        before = len(non_empty)
        non_empty = non_empty.drop_duplicates(subset="osm_id", keep="first").copy()
        dropped_duplicate = before - len(non_empty)

    non_empty = non_empty.reset_index(drop=True)
    report.record(
        layer, input_n=input_n, invalid_fixed=invalid_fixed,
        dropped_empty=dropped_empty, dropped_duplicate=dropped_duplicate, output_n=len(non_empty),
    )
    return non_empty


def normalize_layers(
    buildings: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    hospitals: gpd.GeoDataFrame,
    target_crs: str = "EPSG:32635",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, CleaningReport]:
    """Reproject to ``target_crs`` and clean all three layers."""
    report = CleaningReport(target_crs=target_crs)
    b = _clean(buildings, target_crs, report, "buildings")
    r = _clean(roads, target_crs, report, "roads")
    h = _clean(hospitals, target_crs, report, "hospitals")
    return b, r, h, report
