"""Phase 6 — real-data pilot layer, tested OFFLINE.

No network. Tiny in-memory GeoDataFrames stand in for the OSM extract, so the
adapter, normalisation, validation, provenance, and honest-labelling logic are
exercised without touching Overpass. The scientific-integrity contract is what
is under test: real values are carried, absent values become ``UNKNOWN`` and are
never invented, and prototype/proxy assignments are labelled as such.
"""

from __future__ import annotations

import json

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

from mre.config import load_config
from mre.models import StructuralType
from mre.real import UNKNOWN, CityMode
from mre.real.adapter import (
    PROTOTYPE_HOSPITAL_EMERGENCY_CAPACITY,
    PROTOTYPE_UNIFORM_VULNERABILITY,
    build_real_city,
)
from mre.real.normalize import normalize_layers
from mre.real.osm import _levels, _tag
from mre.real.outputs import real_buildings_gdf, real_hospitals_gdf, real_roads_gdf
from mre.real.pilot import build_pilot_provenance
from mre.real.validation import validate_real_pilot

# A small node grid in WGS84 near the pilot; ids mimic OSM node ids.
_NODES = {
    1: (28.9040, 40.9960), 2: (28.9060, 40.9960), 3: (28.9060, 40.9980),
    4: (28.9040, 40.9980), 5: (28.9080, 40.9960),
}


def _line(node_ids):
    return LineString([_NODES[n] for n in node_ids])


@pytest.fixture
def raw_layers():
    buildings = gpd.GeoDataFrame(
        {
            "osm_id": [101, 102, 103],
            "osm_type": ["way", "way", "way"],
            "building": ["apartments", "yes", "school"],
            "building_levels": [5, UNKNOWN, 3],
            "construction_year": [UNKNOWN, UNKNOWN, UNKNOWN],
            "structural_type": [UNKNOWN, UNKNOWN, UNKNOWN],
            "name": [UNKNOWN, UNKNOWN, "Test School"],
            "geometry": [
                Polygon([(28.9042, 40.9962), (28.9046, 40.9962), (28.9046, 40.9966), (28.9042, 40.9966)]),
                Polygon([(28.9052, 40.9962), (28.9056, 40.9962), (28.9056, 40.9966), (28.9052, 40.9966)]),
                Polygon([(28.9062, 40.9972), (28.9066, 40.9972), (28.9066, 40.9976), (28.9062, 40.9976)]),
            ],
        },
        geometry="geometry", crs="EPSG:4326",
    )
    roads = gpd.GeoDataFrame(
        {
            "osm_id": [201, 202, 203],
            "highway": ["primary", "residential", "residential"],
            "road_class": ["ARTERIAL", "LOCAL", "LOCAL"],
            "name": [UNKNOWN, UNKNOWN, UNKNOWN],
            "maxspeed": [UNKNOWN, UNKNOWN, UNKNOWN],
            "oneway": [UNKNOWN, UNKNOWN, UNKNOWN],
            "node_ids": [json.dumps([1, 2, 3]), json.dumps([3, 4]), json.dumps([2, 5])],
            "geometry": [_line([1, 2, 3]), _line([3, 4]), _line([2, 5])],
        },
        geometry="geometry", crs="EPSG:4326",
    )
    hospitals = gpd.GeoDataFrame(
        {
            "osm_id": [301, 302],
            "osm_type": ["node", "node"],
            "name": ["Devlet Hastanesi", UNKNOWN],
            "emergency": ["yes", UNKNOWN],
            "capacity": [UNKNOWN, UNKNOWN],
            "beds": [UNKNOWN, UNKNOWN],
            "geometry": [Point(28.9050, 40.9968), Point(28.9064, 40.9974)],
        },
        geometry="geometry", crs="EPSG:4326",
    )
    return buildings, roads, hospitals


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def normalized(raw_layers):
    b, r, h = raw_layers
    bn, rn, hn, report = normalize_layers(b, r, h)
    return bn, rn, hn, report


@pytest.fixture
def real_city(normalized, config):
    bn, rn, hn, _ = normalized
    return build_real_city(bn, rn, hn, config, seed=20260810)


# --- constants / vocabulary -----------------------------------------------


def test_unknown_sentinel_and_mode():
    assert UNKNOWN == "UNKNOWN"
    assert CityMode.SYNTHETIC.value == "SYNTHETIC"
    assert CityMode.REAL_PILOT.value == "REAL_PILOT"


def test_osm_pure_helpers_never_guess():
    assert _tag({"name": "X"}, "name") == "X"
    assert _tag({}, "name") == UNKNOWN
    assert _levels({"building:levels": "6"}) == 6
    assert _levels({}) == UNKNOWN
    assert _levels({"building:levels": "not-a-number"}) == UNKNOWN


# --- normalisation ---------------------------------------------------------


def test_normalize_reprojects_to_target_crs(normalized):
    bn, rn, hn, report = normalized
    for gdf in (bn, rn, hn):
        assert gdf.crs.to_string() == "EPSG:32635"
        assert gdf.crs.is_projected
    assert report.per_layer["buildings"]["output_features"] == 3


def test_normalize_reports_cleaning_counts(normalized):
    _, _, _, report = normalized
    layers = report.to_dict()["layers"]
    assert set(layers) == {"buildings", "roads", "hospitals"}
    assert layers["roads"]["input_features"] == 3


# --- adapter: real values kept, missing marked UNKNOWN, prototype labelled ---


def test_building_floors_real_where_present_unknown_otherwise(real_city):
    attrs = {a["osm_id"]: a for a in real_city.building_attributes}
    assert attrs[101]["floors"] == 5           # real building:levels
    assert attrs[101]["floors_source"] == "OSM:building:levels"
    assert attrs[102]["floors"] == UNKNOWN     # no tag -> UNKNOWN, never guessed
    assert attrs[102]["floors_source"] == UNKNOWN


def test_occupancy_real_where_determinable_unknown_for_generic(real_city):
    attrs = {a["osm_id"]: a for a in real_city.building_attributes}
    assert attrs[101]["occupancy"] == "RESIDENTIAL"   # building=apartments
    assert attrs[103]["occupancy"] == "PUBLIC"        # building=school
    assert attrs[102]["occupancy"] == UNKNOWN         # building=yes -> unknown


def test_structural_and_year_are_always_unknown_never_invented(real_city):
    for a in real_city.building_attributes:
        assert a["construction_year"] == UNKNOWN
        assert a["structural_type"] == UNKNOWN


def test_vulnerability_is_uniform_prototype_and_labelled(real_city):
    for a in real_city.building_attributes:
        assert a["vulnerability_index"] == PROTOTYPE_UNIFORM_VULNERABILITY
        assert a["vulnerability_source"] == "PROTOTYPE_UNIFORM"
    for b in real_city.city.buildings:
        assert b.vulnerability_index == PROTOTYPE_UNIFORM_VULNERABILITY
        assert b.structural_type is StructuralType.OTHER


def test_hospitals_keep_real_names_capacity_unknown_emergency_prototype(real_city):
    attrs = {a["osm_id"]: a for a in real_city.hospital_attributes}
    assert attrs[301]["name"] == "Devlet Hastanesi"   # real name preserved
    assert attrs[301]["capacity"] == UNKNOWN          # not invented
    assert attrs[301]["emergency_capacity_source"] == "PROTOTYPE"
    for h in real_city.city.hospitals:
        assert h.emergency_capacity == PROTOTYPE_HOSPITAL_EMERGENCY_CAPACITY


def test_population_is_uniform_proxy_not_real(real_city):
    assert real_city.population_source == "UNIFORM_PROXY"
    weights = {p.population_count for p in real_city.city.population}
    assert len(weights) == 1  # every unit identical weight -> a proxy, not a census


def test_road_topology_is_connected_from_shared_osm_nodes(real_city):
    import networkx as nx

    from mre.roads import build_graph

    graph = build_graph(real_city.city)
    assert graph.number_of_nodes() >= 4
    # nodes 1-2-3-4-5 are all linked through shared ids -> one component
    assert nx.number_connected_components(graph) == 1


def test_city_is_a_synthetic_city_the_engine_can_run(real_city, config):
    from mre.simulation import run_scenario

    result = run_scenario(real_city.city, config, realisation=0, seed=20260810)
    counts = result.summary()["damage_state_counts"]
    assert sum(counts.values()) == len(real_city.city.buildings)


# --- honest output columns -------------------------------------------------


def test_real_output_gdfs_carry_source_columns(real_city):
    b = real_buildings_gdf(real_city)
    assert "vulnerability_source" in b.columns
    assert (b["vulnerability_source"] == "PROTOTYPE_UNIFORM").all()
    assert (b["construction_year"] == UNKNOWN).all()
    assert b.crs.to_string() == "EPSG:32635"

    r = real_roads_gdf(real_city)
    assert "road_class_source" in r.columns
    h = real_hospitals_gdf(real_city)
    assert "emergency_capacity_source" in h.columns


# --- provenance + validation ----------------------------------------------


def test_provenance_is_complete_and_names_licence_and_dates():
    prov = build_pilot_provenance(10, 5, 2, acquired_at="2026-08-16T00:00:00+00:00")
    ok, problems = prov.is_complete()
    assert ok, problems
    payload = prov.to_dict()
    assert all(s["license"].startswith("Open Database License") for s in payload["sources"])
    assert payload["scenario_provenance"]["sources"]


def test_validation_passes_on_a_good_build(real_city, normalized):
    bn, rn, hn, _ = normalized
    prov = build_pilot_provenance(len(bn), len(rn), len(hn), acquired_at="2026-08-16T00:00:00+00:00")
    result = validate_real_pilot(real_city, (bn, rn, hn), prov, seed=20260810)
    assert result.passed, [c for c in result.checks if c["status"] == "fail"]


def test_validation_fails_on_empty_layer(real_city, normalized):
    bn, rn, hn, _ = normalized
    empty = bn.iloc[0:0].copy()
    prov = build_pilot_provenance(0, len(rn), len(hn), acquired_at="2026-08-16T00:00:00+00:00")
    result = validate_real_pilot(real_city, (empty, rn, hn), prov, seed=20260810)
    assert not result.passed
    assert any(c["check"] == "buildings_non_empty" and c["status"] == "fail" for c in result.checks)


def test_validation_flags_incomplete_provenance(real_city, normalized):
    from mre.real.provenance import ProvenanceRecord

    bn, rn, hn, _ = normalized
    bare = ProvenanceRecord(
        pilot_name="x", study_area="x", bbox_wgs84=(0, 0, 1, 1), target_crs="EPSG:32635",
    )  # no sources, no scenario
    result = validate_real_pilot(real_city, (bn, rn, hn), bare, seed=20260810)
    assert any(c["check"] == "provenance_complete" and c["status"] == "fail" for c in result.checks)
