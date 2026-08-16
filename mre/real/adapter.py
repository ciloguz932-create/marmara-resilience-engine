"""Adapt normalised real GIS layers into the engine's ``mre.models`` types.

This is the real end of the ``CityBuilder`` seam (docs/ARCHITECTURE.md §4): it
produces the **same** ``SyntheticCity`` the synthetic builder produces, so the
Monte Carlo driver cannot tell whether its input was invented or observed.

What is real here and what is prototype — enforced, not hoped:

- **Real** (from OSM): building footprints and centroids, storey counts where
  ``building:levels`` is tagged, occupancy where the ``building`` tag is
  determinable, the road network and its OSM node topology, road classes, and
  hospital locations and names.
- **UNKNOWN** (recorded, never invented): construction year and structural
  system for every building; storey count / occupancy where the tag is absent;
  hospital capacity.
- **PROTOTYPE** (assigned, labelled): a *uniform* vulnerability index for every
  building (open data gives no basis for differential vulnerability, so damage
  varies only with the scenario intensity field, not with invented per-building
  fragility); a prototype hospital emergency capacity (so the accessibility
  model can run — note the *primary* accessibility objective, who can reach a
  hospital, does not depend on capacity at all); and free-flow travel times from
  the engine's prototype class speeds.

The honest per-entity source flags travel alongside the city in ``RealCity`` and
are written into the real-pilot GPKG so a reader can audit every value's origin.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd

from mre.data import nearest_node
from mre.models import (
    Building,
    Hospital,
    Occupancy,
    PopulationUnit,
    RoadClass,
    RoadLink,
    StructuralType,
    SyntheticCity,
)
from mre.real import UNKNOWN

__all__ = ["RealCity", "build_real_city"]

# Uniform PROTOTYPE vulnerability index for every real building. 1.0 is the
# reference building (fragility medians divided by 1.0 = unchanged). Uniform
# because open data carries no structural attributes to differentiate it — so
# the damage field reflects the real intensity gradient, never an invented
# per-building fragility. Labelled PROTOTYPE_UNIFORM in every output.
PROTOTYPE_UNIFORM_VULNERABILITY = 1.0

# PROTOTYPE emergency capacity assigned to every real hospital, since OSM does
# not carry capacity. Labelled PROTOTYPE in every output. The primary
# accessibility objective (reachability) does not use it; only the secondary
# service-pressure / utilisation metric does, and that is reported as prototype.
PROTOTYPE_HOSPITAL_EMERGENCY_CAPACITY = 50

# OSM building= tag -> engine Occupancy, where determinable. Absent/generic tags
# (e.g. building=yes) map to None -> occupancy recorded UNKNOWN.
_OCCUPANCY_MAP = {
    "apartments": Occupancy.RESIDENTIAL, "residential": Occupancy.RESIDENTIAL,
    "house": Occupancy.RESIDENTIAL, "detached": Occupancy.RESIDENTIAL,
    "dormitory": Occupancy.RESIDENTIAL, "terrace": Occupancy.RESIDENTIAL,
    "bungalow": Occupancy.RESIDENTIAL, "semidetached_house": Occupancy.RESIDENTIAL,
    "commercial": Occupancy.COMMERCIAL, "retail": Occupancy.COMMERCIAL,
    "office": Occupancy.COMMERCIAL, "shop": Occupancy.COMMERCIAL,
    "supermarket": Occupancy.COMMERCIAL, "hotel": Occupancy.COMMERCIAL,
    "kiosk": Occupancy.COMMERCIAL,
    "industrial": Occupancy.INDUSTRIAL, "warehouse": Occupancy.INDUSTRIAL,
    "factory": Occupancy.INDUSTRIAL, "manufacture": Occupancy.INDUSTRIAL,
    "school": Occupancy.PUBLIC, "university": Occupancy.PUBLIC,
    "college": Occupancy.PUBLIC, "kindergarten": Occupancy.PUBLIC,
    "hospital": Occupancy.PUBLIC, "mosque": Occupancy.PUBLIC,
    "church": Occupancy.PUBLIC, "public": Occupancy.PUBLIC,
    "civic": Occupancy.PUBLIC, "government": Occupancy.PUBLIC,
    "chapel": Occupancy.PUBLIC, "cathedral": Occupancy.PUBLIC,
}


@dataclass(slots=True)
class RealCity:
    """A real-data city plus the per-entity provenance the outputs must carry."""

    city: SyntheticCity
    # Per-building honest attributes, in city.buildings order.
    building_attributes: list[dict[str, Any]] = field(default_factory=list)
    road_attributes: list[dict[str, Any]] = field(default_factory=list)
    hospital_attributes: list[dict[str, Any]] = field(default_factory=list)
    # Real footprint polygons (EPSG:32635), in city.buildings order — for
    # visualization that extrudes the REAL footprint, not a square.
    building_polygons: list[Any] = field(default_factory=list)
    population_source: str = "UNIFORM_PROXY"
    notes: tuple[str, ...] = ()


def _occupancy_for(building_tag: str) -> tuple[Occupancy, str]:
    """(engine placeholder occupancy, honest source string).

    Where the OSM building tag is determinable, both agree. Where it is not, the
    honest string is ``UNKNOWN`` and the engine placeholder is RESIDENTIAL (the
    modal urban occupancy) — a value that drives no real-pilot result (real
    demand is the uniform population proxy, not building occupancy). The written
    GPKG carries the honest string, never the placeholder.
    """
    mapped = _OCCUPANCY_MAP.get(str(building_tag).lower())
    if mapped is not None:
        return mapped, mapped.value
    return Occupancy.RESIDENTIAL, UNKNOWN


def _build_buildings(buildings: gpd.GeoDataFrame) -> tuple[list[Building], list[dict], list]:
    models: list[Building] = []
    attrs: list[dict] = []
    polys: list = []
    for i, row in enumerate(buildings.itertuples(index=False)):
        geom = row.geometry
        centroid = geom.centroid
        area = float(geom.area)
        side = math.sqrt(area) if area > 0 else 1.0  # area-equivalent square (documented)

        levels_raw = getattr(row, "building_levels", UNKNOWN)
        floors_known = levels_raw not in (UNKNOWN, None) and str(levels_raw) != "UNKNOWN"
        try:
            floors = int(float(levels_raw)) if floors_known else 0
        except (ValueError, TypeError):
            floors, floors_known = 0, False
        floors = max(0, min(floors, 60))  # clamp obviously bad tags; 0 == unknown

        building_tag = getattr(row, "building", UNKNOWN)
        occupancy_enum, occupancy_source = _occupancy_for(building_tag)

        osm_id = getattr(row, "osm_id", i)
        building_id = f"OSM{osm_id}"

        models.append(
            Building(
                building_id=building_id,
                easting=float(centroid.x),
                northing=float(centroid.y),
                floors=floors if floors_known else 1,  # engine placeholder if unknown
                construction_year=0,  # UNKNOWN — sentinel, drives nothing real here
                structural_type=StructuralType.OTHER,  # UNKNOWN in the real pilot
                occupancy=occupancy_enum,
                occupants=0,  # UNKNOWN — real demand uses the population proxy
                vulnerability_index=PROTOTYPE_UNIFORM_VULNERABILITY,
                footprint_side_m=float(side),
            )
        )
        attrs.append(
            {
                "building_id": building_id,
                "osm_id": osm_id,
                "osm_building": str(building_tag),
                "floors": int(floors) if floors_known else UNKNOWN,
                "floors_source": "OSM:building:levels" if floors_known else UNKNOWN,
                "occupancy": occupancy_source,
                "construction_year": UNKNOWN,
                "structural_type": UNKNOWN,
                "vulnerability_index": PROTOTYPE_UNIFORM_VULNERABILITY,
                "vulnerability_source": "PROTOTYPE_UNIFORM",
                "footprint_area_m2": round(area, 2),
            }
        )
        polys.append(geom)
    return models, attrs, polys


def _parse_node_ids(value: Any) -> list[int]:
    """node_ids may survive a GeoJSON/GPKG round-trip as a JSON string OR as a
    native list/ndarray, depending on the driver. Accept both."""
    if isinstance(value, str):
        value = json.loads(value)
    return [int(v) for v in value]


def _junction_nodes(roads: gpd.GeoDataFrame) -> set[int]:
    """OSM node ids that are true graph junctions: shared by >=2 ways, or a way
    endpoint, or repeated within a way. Interior pass-through nodes are not
    junctions and are collapsed into a single link span."""
    occurrence: dict[int, int] = {}
    junctions: set[int] = set()
    for row in roads.itertuples(index=False):
        node_ids = _parse_node_ids(row.node_ids)
        if node_ids:
            junctions.add(node_ids[0])
            junctions.add(node_ids[-1])
        seen_in_way: dict[int, int] = {}
        for nid in node_ids:
            occurrence[nid] = occurrence.get(nid, 0) + 1
            seen_in_way[nid] = seen_in_way.get(nid, 0) + 1
            if seen_in_way[nid] > 1:
                junctions.add(nid)
    for nid, count in occurrence.items():
        if count >= 2:
            junctions.add(nid)
    return junctions


def _build_roads(
    roads: gpd.GeoDataFrame, config: dict[str, Any]
) -> tuple[list[RoadLink], dict[int, tuple[float, float]], list[dict]]:
    """Split real OSM ways at junctions into RoadLinks; return links, node
    coordinate map (compact ids), and honest per-link attributes."""
    speed_kmh = config["synthetic_city"]["roads"]["speed_kmh"]
    susceptibility_cfg = config["synthetic_city"]["roads"]["susceptibility"]
    baseline_closure = float(config["roads"]["disruption"]["baseline_closure_probability"])

    junctions = _junction_nodes(roads)

    # Collect the real coordinate of every OSM node we keep, from way geometries.
    osm_coord: dict[int, tuple[float, float]] = {}
    for row in roads.itertuples(index=False):
        node_ids = _parse_node_ids(row.node_ids)
        coords = list(row.geometry.coords)
        for nid, xy in zip(node_ids, coords):
            osm_coord[nid] = (float(xy[0]), float(xy[1]))

    # Compact integer ids for the junction nodes only.
    compact: dict[int, int] = {osm: idx for idx, osm in enumerate(sorted(junctions))}
    nodes: dict[int, tuple[float, float]] = {compact[o]: osm_coord[o] for o in junctions if o in osm_coord}

    links: list[RoadLink] = []
    attrs: list[dict] = []
    seen_pairs: set[tuple[int, int]] = set()
    link_counter = 0

    for row in roads.itertuples(index=False):
        node_ids = _parse_node_ids(row.node_ids)
        coords = list(row.geometry.coords)
        if len(node_ids) != len(coords) or len(node_ids) < 2:
            continue
        road_class = RoadClass(row.road_class)
        speed = float(speed_kmh[road_class.value])
        suscept = float(susceptibility_cfg[road_class.value])

        # Walk the way, accumulating length until the next junction node.
        span_start = 0
        accumulated = 0.0
        for k in range(1, len(node_ids)):
            x0, y0 = coords[k - 1]
            x1, y1 = coords[k]
            accumulated += math.hypot(x1 - x0, y1 - y0)
            if node_ids[k] in junctions:
                a_osm, b_osm = node_ids[span_start], node_ids[k]
                if a_osm != b_osm and a_osm in compact and b_osm in compact and accumulated > 0:
                    a, b = compact[a_osm], compact[b_osm]
                    pair = (min(a, b), max(a, b))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        travel_time = accumulated / (speed * 1000.0 / 60.0)
                        road_id = f"OSM{row.osm_id}_{link_counter}"
                        links.append(
                            RoadLink(
                                road_id=road_id,
                                from_node=a,
                                to_node=b,
                                length_m=float(accumulated),
                                road_class=road_class,
                                travel_time_min=float(travel_time),
                                closure_probability=baseline_closure,
                                criticality=0.0,  # not computed for the pilot
                                susceptibility=suscept,
                            )
                        )
                        attrs.append(
                            {
                                "road_id": road_id,
                                "osm_id": int(row.osm_id),
                                "highway": getattr(row, "highway", UNKNOWN),
                                "road_class": road_class.value,
                                "road_class_source": "OSM:highway",
                                "length_m": round(float(accumulated), 2),
                                "travel_time_source": "PROTOTYPE_CLASS_SPEED",
                                "criticality": "NOT_COMPUTED",
                            }
                        )
                        link_counter += 1
                span_start = k
                accumulated = 0.0

    return links, nodes, attrs


def _build_hospitals(
    hospitals: gpd.GeoDataFrame, nodes: dict[int, tuple[float, float]]
) -> tuple[list[Hospital], list[dict]]:
    models: list[Hospital] = []
    attrs: list[dict] = []
    for row in hospitals.itertuples(index=False):
        pt = row.geometry
        node_id = nearest_node(nodes, float(pt.x), float(pt.y)) if nodes else 0
        osm_id = getattr(row, "osm_id", 0)
        hospital_id = f"OSM{osm_id}"
        models.append(
            Hospital(
                hospital_id=hospital_id,
                easting=float(pt.x),
                northing=float(pt.y),
                capacity=PROTOTYPE_HOSPITAL_EMERGENCY_CAPACITY * 5,  # placeholder total
                emergency_capacity=PROTOTYPE_HOSPITAL_EMERGENCY_CAPACITY,
                node_id=node_id,
            )
        )
        attrs.append(
            {
                "hospital_id": hospital_id,
                "osm_id": osm_id,
                "name": getattr(row, "name", UNKNOWN),
                "emergency_tag": getattr(row, "emergency", UNKNOWN),
                "capacity": UNKNOWN,
                "emergency_capacity": PROTOTYPE_HOSPITAL_EMERGENCY_CAPACITY,
                "emergency_capacity_source": "PROTOTYPE",
            }
        )
    return models, attrs


def _build_population_proxy(
    buildings: gpd.GeoDataFrame, nodes: dict[int, tuple[float, float]], config: dict[str, Any]
) -> list[PopulationUnit]:
    """A UNIFORM demand proxy on a regular grid over the study extent.

    Real population is not in open data, so demand is a uniform grid — every
    unit carries the same weight, labelled population_source=UNIFORM_PROXY in the
    provenance and outputs. This keeps demand decoupled from buildings (so a
    later intervention comparison does not confound damage and demand) exactly as
    the synthetic design requires, while never pretending to be a real census.
    """
    minx, miny, maxx, maxy = buildings.total_bounds
    grid = config["synthetic_city"]["population"]["grid"]
    nx, ny = int(grid[0]), int(grid[1])
    units: list[PopulationUnit] = []
    UNIFORM_WEIGHT = 100  # arbitrary uniform proxy weight; NOT a person count
    idx = 0
    for ix in range(nx):
        for iy in range(ny):
            e = minx + (maxx - minx) * (ix + 0.5) / nx
            n = miny + (maxy - miny) * (iy + 0.5) / ny
            node_id = nearest_node(nodes, e, n) if nodes else 0
            units.append(
                PopulationUnit(
                    population_id=f"PROXY{idx:04d}",
                    easting=float(e),
                    northing=float(n),
                    population_count=UNIFORM_WEIGHT,
                    node_id=node_id,
                )
            )
            idx += 1
    return units


def build_real_city(
    buildings: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    hospitals: gpd.GeoDataFrame,
    config: dict[str, Any],
    *,
    seed: int,
    crs: str = "EPSG:32635",
) -> RealCity:
    """Assemble a ``SyntheticCity`` from normalised real GIS layers (EPSG:32635)."""
    building_models, building_attrs, building_polys = _build_buildings(buildings)
    road_models, nodes, road_attrs = _build_roads(roads, config)
    hospital_models, hospital_attrs = _build_hospitals(hospitals, nodes)
    population = _build_population_proxy(buildings, nodes, config)

    city = SyntheticCity(
        crs=crs,
        seed=seed,
        buildings=tuple(building_models),
        hospitals=tuple(hospital_models),
        roads=tuple(road_models),
        population=tuple(population),
        nodes=nodes,
    )
    return RealCity(
        city=city,
        building_attributes=building_attrs,
        road_attributes=road_attrs,
        hospital_attributes=hospital_attrs,
        building_polygons=building_polys,
        population_source="UNIFORM_PROXY",
        notes=(
            "Vulnerability is PROTOTYPE_UNIFORM; construction year and structural "
            "type are UNKNOWN; population is a UNIFORM_PROXY grid; hospital "
            "emergency capacity is PROTOTYPE.",
        ),
    )
