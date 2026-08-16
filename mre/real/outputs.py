"""Write real-pilot GIS/JSON outputs with honest, auditable attribute columns.

Distinct from ``mre.outputs`` (which serialises the synthetic city): every layer
here carries the *source* of each value — ``PROTOTYPE_UNIFORM`` vulnerability,
``UNKNOWN`` structural type/year/capacity, ``PROTOTYPE`` emergency capacity,
``UNIFORM_PROXY`` demand — so a reader can audit exactly what is real and what is
prototype. Real building footprints are written as real polygons, not squares.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point

from mre.models import DamageState
from mre.real.adapter import RealCity

__all__ = [
    "real_buildings_gdf",
    "real_roads_gdf",
    "real_hospitals_gdf",
    "write_real_city_layers",
    "write_real_hazard_layer",
    "write_real_results_layer",
    "write_real_scenario_json",
]

_REAL_DISCLAIMER = (
    "REAL-DATA PILOT — RESEARCH PROTOTYPE, NOT A REAL-WORLD PREDICTION. Geometry "
    "is real (OpenStreetMap, © OpenStreetMap contributors, ODbL). Fragility and "
    "intensity are PROTOTYPE values applied to real geometry; structural "
    "attributes and hospital capacities are UNKNOWN; demand is a UNIFORM proxy. "
    "This is not a real damage, loss, or casualty estimate for any real "
    "building, road, or hospital. See docs/SCIENTIFIC_ASSUMPTIONS.md §10."
)


def real_buildings_gdf(rc: RealCity) -> gpd.GeoDataFrame:
    """Real footprints (polygons) with honest per-building attribute columns."""
    frame = gpd.GeoDataFrame(rc.building_attributes, geometry=list(rc.building_polygons), crs=rc.city.crs)
    return frame


def real_roads_gdf(rc: RealCity) -> gpd.GeoDataFrame:
    """Road links as junction-to-junction lines (a documented simplification of
    the real polyline) with honest attribute columns."""
    nodes = rc.city.nodes
    geometries = [
        LineString([nodes[link.from_node], nodes[link.to_node]]) for link in rc.city.roads
    ]
    frame = gpd.GeoDataFrame(rc.road_attributes, geometry=geometries, crs=rc.city.crs)
    return frame


def real_hospitals_gdf(rc: RealCity) -> gpd.GeoDataFrame:
    geometries = [Point(h.easting, h.northing) for h in rc.city.hospitals]
    frame = gpd.GeoDataFrame(rc.hospital_attributes, geometry=geometries, crs=rc.city.crs)
    return frame


def write_real_city_layers(rc: RealCity, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buildings_path = out_dir / "buildings.gpkg"
    roads_path = out_dir / "roads.gpkg"
    hospitals_path = out_dir / "hospitals.gpkg"
    real_buildings_gdf(rc).to_file(buildings_path, layer="buildings", driver="GPKG")
    real_roads_gdf(rc).to_file(roads_path, layer="roads", driver="GPKG")
    real_hospitals_gdf(rc).to_file(hospitals_path, layer="hospitals", driver="GPKG")
    return {"buildings_gpkg": buildings_path, "roads_gpkg": roads_path, "hospitals_gpkg": hospitals_path}


def write_real_hazard_layer(rc: RealCity, scenario_result: Any, out_dir: Path) -> dict[str, Path]:
    """hazard.gpkg — real footprints carrying the scenario proxy intensity and the
    single-realisation sampled damage state (prototype fragility on real geometry)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = real_buildings_gdf(rc)
    frame["intensity_proxy"] = scenario_result.intensity
    frame["intensity_note"] = "scenario-derived proxy; NOT PGA/PGV/SA(T)/MMI"
    frame["damage_state"] = [s.name for s in scenario_result.damage.state_enums()]
    frame["expected_damage_index"] = scenario_result.damage.expected_damage_index
    frame["damage_note"] = "PROTOTYPE fragility on real geometry; not a real damage estimate"
    path = out_dir / "hazard.gpkg"
    frame.to_file(path, layer="buildings_hazard", driver="GPKG")
    return {"hazard_gpkg": path}


def write_real_results_layer(rc: RealCity, mc_result: Any, out_dir: Path) -> dict[str, Path]:
    """results.gpkg — Monte Carlo spatial frequencies on real geometry:
    per-building collapse frequency, per-link closed/degraded frequency, and
    per-proxy-unit unreachability frequency."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.gpkg"

    buildings = real_buildings_gdf(rc)
    buildings["collapse_frequency"] = mc_result.building_collapse_frequency
    buildings["mean_expected_damage_index"] = mc_result.building_mean_edi
    buildings.to_file(path, layer="buildings_frequency", driver="GPKG")

    roads = real_roads_gdf(rc)
    roads["closed_frequency"] = mc_result.link_closed_frequency
    roads["degraded_frequency"] = mc_result.link_degraded_frequency
    roads.to_file(path, layer="roads_frequency", driver="GPKG")

    population = gpd.GeoDataFrame(
        {
            "population_id": [p.population_id for p in rc.city.population],
            "demand_weight": [p.population_count for p in rc.city.population],
            "population_source": rc.population_source,
            "unreachable_frequency": mc_result.unit_unreachable_frequency,
            "mean_travel_time_min": mc_result.unit_mean_travel_time_min,
        },
        geometry=[Point(p.easting, p.northing) for p in rc.city.population],
        crs=rc.city.crs,
    )
    population.to_file(path, layer="demand_proxy_frequency", driver="GPKG")

    real_hospitals_gdf(rc).to_file(path, layer="hospitals", driver="GPKG")
    return {"results_gpkg": path}


def write_real_scenario_json(rc: RealCity, scenario_result: Any, config: dict, scenario_provenance: dict, out_dir: Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hazard = config["hazard"]
    summary = scenario_result.summary()
    payload = {
        "mode": "REAL_PILOT",
        "scenario_provenance": scenario_provenance,
        "hazard_parameters": {
            "scenario_id": hazard["scenario_id"],
            "magnitude": hazard["magnitude"],
            "model": hazard["model"],
            "source_line": hazard["source_line"],
            "intensity_at_source": hazard["intensity_at_source"],
            "decay_per_km": hazard["decay_per_km"],
            "parameter_note": (
                "intensity_at_source and decay_per_km are PROTOTYPE proxy tuning "
                "parameters, NOT calibrated attenuation. The fault system/segment "
                "location is real; the proxy field magnitude is not a real "
                "ground-motion estimate."
            ),
        },
        "realisation_0": {
            "intensity_proxy": summary["intensity"],
            "damage_state_counts": summary["damage_state_counts"],
            "mean_expected_damage_index": summary["mean_expected_damage_index"],
            "links": summary["links"],
        },
        "disclaimer": _REAL_DISCLAIMER,
    }
    path = out_dir / "scenario.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"scenario_json": path}
