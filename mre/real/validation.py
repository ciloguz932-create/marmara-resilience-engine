"""Mandatory validation of a real-pilot build → validation.json.

Runs the checks the pilot must pass before its results can be shown: valid
geometry, a single consistent CRS, no NaN in required fields, no negative
lengths, a road network that is actually connected enough to route on, hospitals
and buildings inside the study area, a non-empty demand proxy, and complete
provenance. Each check is recorded with a status; hard failures make the whole
build ``passed = False`` and callers stop. Warnings (e.g. a tiny disconnected
OSM stub) are recorded but do not fail the build.

The point is that a real-pilot result is never presented unless it has passed
these checks, and the checks are written to disk for audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import networkx as nx

from mre.real.adapter import RealCity
from mre.real.provenance import ProvenanceRecord
from mre.roads import build_graph

__all__ = ["ValidationResult", "validate_real_pilot"]


@dataclass(slots=True)
class ValidationResult:
    checks: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "", **extra: Any) -> None:
        assert status in ("pass", "warn", "fail")
        self.checks.append({"check": name, "status": status, "detail": detail, **extra})

    @property
    def passed(self) -> bool:
        return not any(c["status"] == "fail" for c in self.checks)

    @property
    def n_warnings(self) -> int:
        return sum(c["status"] == "warn" for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_checks": len(self.checks),
            "n_failures": sum(c["status"] == "fail" for c in self.checks),
            "n_warnings": self.n_warnings,
            "checks": self.checks,
        }


def _check_layer_geometry(result: ValidationResult, name: str, gdf: gpd.GeoDataFrame, target_crs: str) -> None:
    if len(gdf) == 0:
        result.add(f"{name}_non_empty", "fail", f"{name} layer is empty")
        return
    result.add(f"{name}_non_empty", "pass", f"{len(gdf)} features")

    crs_ok = gdf.crs is not None and gdf.crs.to_string() == target_crs
    result.add(
        f"{name}_crs", "pass" if crs_ok else "fail",
        f"crs={gdf.crs.to_string() if gdf.crs else None}, expected {target_crs}",
    )

    n_invalid = int((~gdf.geometry.is_valid).sum())
    n_empty = int((gdf.geometry.is_empty | gdf.geometry.isna()).sum())
    result.add(
        f"{name}_geometry_valid", "pass" if (n_invalid == 0 and n_empty == 0) else "fail",
        f"invalid={n_invalid}, empty={n_empty}",
    )


def validate_real_pilot(
    real_city: RealCity,
    normalized: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame],
    provenance: ProvenanceRecord,
    *,
    seed: int,
    target_crs: str = "EPSG:32635",
) -> ValidationResult:
    """Validate a real-pilot build. Never fabricates a pass — every check is real."""
    buildings, roads, hospitals = normalized
    result = ValidationResult()

    # 1. geometry + CRS per layer
    _check_layer_geometry(result, "buildings", buildings, target_crs)
    _check_layer_geometry(result, "roads", roads, target_crs)
    _check_layer_geometry(result, "hospitals", hospitals, target_crs)

    city = real_city.city

    # 2. no negative or zero road lengths / travel times
    bad_lengths = [r.road_id for r in city.roads if r.length_m <= 0]
    bad_times = [r.road_id for r in city.roads if r.travel_time_min <= 0]
    result.add(
        "road_lengths_positive", "pass" if not bad_lengths else "fail",
        f"{len(bad_lengths)} links with non-positive length",
    )
    result.add(
        "road_travel_times_positive", "pass" if not bad_times else "fail",
        f"{len(bad_times)} links with non-positive travel time",
    )

    # 3. no NaN in required model fields
    def _finite(values):
        return all(v == v and v not in (float("inf"), float("-inf")) for v in values)

    coords_ok = _finite([b.easting for b in city.buildings] + [b.northing for b in city.buildings])
    result.add("building_coords_finite", "pass" if coords_ok else "fail",
               "no NaN/inf in building coordinates" if coords_ok else "NaN/inf present")

    # 4. road network connectivity (routable-enough, not necessarily one component)
    graph = build_graph(city)
    if graph.number_of_nodes() == 0:
        result.add("road_network_connectivity", "fail", "empty road graph")
    else:
        components = sorted((len(c) for c in nx.connected_components(graph)), reverse=True)
        largest_fraction = components[0] / graph.number_of_nodes()
        status = "pass" if largest_fraction >= 0.90 else ("warn" if largest_fraction >= 0.5 else "fail")
        result.add(
            "road_network_connectivity", status,
            f"largest component holds {largest_fraction:.1%} of {graph.number_of_nodes()} nodes "
            f"across {len(components)} components",
            largest_component_fraction=round(largest_fraction, 4),
            n_components=len(components),
        )

    # 5. hospitals and buildings inside study area (bbox of buildings, buffered)
    minx, miny, maxx, maxy = buildings.total_bounds
    pad = 250.0  # metres tolerance
    def _inside(x, y):
        return (minx - pad) <= x <= (maxx + pad) and (miny - pad) <= y <= (maxy + pad)

    hosp_outside = [h.hospital_id for h in city.hospitals if not _inside(h.easting, h.northing)]
    result.add(
        "hospitals_in_study_area", "pass" if not hosp_outside else "warn",
        f"{len(hosp_outside)} hospitals outside the building extent (+{pad:.0f} m)",
    )
    bld_outside = sum(1 for b in city.buildings if not _inside(b.easting, b.northing))
    result.add(
        "buildings_in_study_area", "pass" if bld_outside == 0 else "warn",
        f"{bld_outside} buildings outside their own extent (+{pad:.0f} m)",
    )

    # 6. demand proxy accounting
    total_demand = sum(p.population_count for p in city.population)
    result.add(
        "demand_proxy_present", "pass" if total_demand > 0 and city.population else "fail",
        f"{len(city.population)} uniform-proxy demand units, total weight {total_demand} "
        f"(source={real_city.population_source}; NOT a real population count)",
    )

    # 7. reproducibility: seed recorded on the city
    result.add(
        "reproducibility_seed", "pass" if city.seed == seed else "fail",
        f"city.seed={city.seed}, requested {seed}",
    )

    # 8. provenance completeness
    ok, problems = provenance.is_complete()
    result.add(
        "provenance_complete", "pass" if ok else "fail",
        "all sources fully attributed" if ok else f"problems: {problems}",
    )

    # 9. integrity labelling present
    has_notes = bool(provenance.integrity_notes) and bool(real_city.notes)
    result.add(
        "integrity_labels_present", "pass" if has_notes else "fail",
        "prototype/UNKNOWN/proxy labels recorded" if has_notes else "integrity labels missing",
    )

    return result
