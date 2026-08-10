"""Phase 3 — synthetic city generation."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from mre.data import build_synthetic_city, nearest_node
from mre.models import Occupancy, RoadClass, StructuralType
from mre.roads import build_graph

SEED = 20260810


# --- counts, identity, geometry ------------------------------------------


def test_building_count_matches_config(city, config):
    assert len(city.buildings) == config["synthetic_city"]["n_buildings"] == 1000


def test_hospital_and_population_counts(city, config):
    cfg = config["synthetic_city"]
    gx, gy = cfg["population"]["grid"]
    assert len(city.hospitals) == cfg["n_hospitals"] == 3
    assert len(city.population) == gx * gy


def test_ids_are_unique(city):
    for items, attribute in [
        (city.buildings, "building_id"),
        (city.hospitals, "hospital_id"),
        (city.roads, "road_id"),
        (city.population, "population_id"),
    ]:
        ids = [getattr(i, attribute) for i in items]
        assert len(ids) == len(set(ids)), f"duplicate {attribute}"


def test_footprints_are_valid_squares(city):
    for building in city.buildings:
        footprint = building.footprint()
        assert footprint.is_valid
        assert footprint.area == pytest.approx(building.footprint_side_m**2)


def test_everything_lies_inside_the_study_extent(city, config):
    cfg = config["synthetic_city"]
    extent_x, extent_y = cfg["extent_m"]
    min_e, min_n = cfg["origin_easting"], cfg["origin_northing"]
    max_e, max_n = min_e + extent_x, min_n + extent_y

    for building in city.buildings:
        assert min_e <= building.easting <= max_e
        assert min_n <= building.northing <= max_n
    for unit in city.population:
        assert min_e <= unit.easting <= max_e
        assert min_n <= unit.northing <= max_n


def test_crs_is_the_configured_projected_crs(city, config):
    assert city.crs == config["project"]["crs"]


# --- heterogeneity --------------------------------------------------------


def test_vulnerability_is_heterogeneous_not_constant(city):
    """A synthetic *population*, not a synthetic archetype."""
    values = np.array([b.vulnerability_index for b in city.buildings])
    assert values.std() > 0.05
    assert len(np.unique(values)) > 900


def test_vulnerability_respects_configured_clip(city, config):
    low, high = config["synthetic_city"]["buildings"]["vulnerability"]["clip"]
    values = np.array([b.vulnerability_index for b in city.buildings])
    assert values.min() >= low
    assert values.max() <= high


def test_identical_attributes_still_differ_in_vulnerability(city, config):
    """Two buildings sharing type, era, and height must not share a
    vulnerability -- the lognormal term guarantees it.

    Values sitting exactly on the clip bounds are excluded: clipping is
    deliberate, and it legitimately creates ties (6 buildings in the default
    city).
    """
    low, high = config["synthetic_city"]["buildings"]["vulnerability"]["clip"]

    groups: dict[tuple, list[float]] = {}
    for b in city.buildings:
        if b.vulnerability_index <= low + 1e-12 or b.vulnerability_index >= high - 1e-12:
            continue
        key = (b.structural_type, b.floors, b.construction_year // 25)
        groups.setdefault(key, []).append(b.vulnerability_index)

    populated = [v for v in groups.values() if len(v) > 2]
    assert populated, "no comparable group found"
    assert all(len(set(v)) == len(v) for v in populated)


def test_masonry_is_more_vulnerable_than_steel_on_average(city):
    """Sanity check on the invented type factors, not a validation claim."""
    masonry = [b.vulnerability_index for b in city.buildings if b.structural_type is StructuralType.MASONRY]
    steel = [b.vulnerability_index for b in city.buildings if b.structural_type is StructuralType.STEEL]
    assert masonry and steel
    assert np.mean(masonry) > np.mean(steel)


def test_older_buildings_are_more_vulnerable_on_average(city, config):
    early, _late = config["synthetic_city"]["buildings"]["vulnerability"]["year_breakpoints"]
    old = [b.vulnerability_index for b in city.buildings if b.construction_year < early]
    new = [b.vulnerability_index for b in city.buildings if b.construction_year >= 1999]
    assert old and new
    assert np.mean(old) > np.mean(new)


def test_attributes_span_their_configured_domains(city, config):
    cfg = config["synthetic_city"]["buildings"]
    floors = np.array([b.floors for b in city.buildings])
    years = np.array([b.construction_year for b in city.buildings])

    assert floors.min() >= cfg["floors_range"][0]
    assert floors.max() <= cfg["floors_range"][1]
    assert len(np.unique(floors)) > 5
    assert years.min() >= cfg["construction_year_range"][0]
    assert years.max() <= cfg["construction_year_range"][1]

    assert len({b.structural_type for b in city.buildings}) == len(StructuralType)
    assert len({b.occupancy for b in city.buildings}) == len(Occupancy)


# --- population, decoupled ------------------------------------------------


def test_population_is_positive_and_near_target(city, config):
    counts = np.array([p.population_count for p in city.population])
    assert (counts > 0).all()
    target = config["synthetic_city"]["population"]["total_population"]
    assert counts.sum() == pytest.approx(target, rel=0.05)


def test_population_units_are_not_derived_from_buildings(city):
    """Decoupled by construction: neither the count nor the positions may
    coincide with buildings, or damage would contaminate demand."""
    assert len(city.population) != len(city.buildings)
    building_points = {(b.easting, b.northing) for b in city.buildings}
    assert not any((p.easting, p.northing) in building_points for p in city.population)


def test_population_and_hospitals_attach_to_real_nodes(city):
    for unit in city.population:
        assert unit.node_id in city.nodes
    for hospital in city.hospitals:
        assert hospital.node_id in city.nodes


def test_hospital_emergency_capacity_is_a_subset_of_capacity(city):
    for hospital in city.hospitals:
        assert 0 < hospital.emergency_capacity < hospital.capacity


def test_hospitals_are_not_co_located(city):
    points = [(h.easting, h.northing) for h in city.hospitals]
    assert len(set(points)) == len(points)


# --- road network ---------------------------------------------------------


def test_undamaged_network_is_connected(city):
    graph = build_graph(city)
    assert graph.number_of_nodes() == len(city.nodes)
    assert graph.number_of_edges() == len(city.roads)
    assert nx.is_connected(graph)


def test_dropout_actually_removed_edges(city, config):
    """Dropout must break the perfect grid, while keeping it connected."""
    nx_cells, ny_cells = config["synthetic_city"]["road_grid"]
    full_grid_edges = 2 * nx_cells * (ny_cells + 1)
    assert len(city.roads) < full_grid_edges


def test_link_attributes_are_well_formed(city):
    for link in city.roads:
        assert link.length_m > 0
        assert link.travel_time_min > 0
        assert 0.0 <= link.criticality <= 1.0
        assert link.susceptibility > 0
        assert 0.0 <= link.closure_probability <= 1.0
        assert link.road_class in set(RoadClass)
        assert link.from_node in city.nodes and link.to_node in city.nodes


def test_criticality_is_not_uniform(city):
    values = np.array([r.criticality for r in city.roads])
    assert values.std() > 0
    assert values.max() > values.min()


def test_travel_time_is_consistent_with_length_and_class(city, config):
    speeds = config["synthetic_city"]["roads"]["speed_kmh"]
    for link in city.roads[:50]:
        expected = (link.length_m / 1000.0) / speeds[link.road_class.value] * 60.0
        assert link.travel_time_min == pytest.approx(expected)


def test_nearest_node_picks_the_closest(city):
    node_id, (easting, northing) = next(iter(city.nodes.items()))
    assert nearest_node(city.nodes, easting, northing) == node_id


# --- reproducibility ------------------------------------------------------


def _fingerprint(city) -> tuple:
    return (
        tuple((b.building_id, b.easting, b.northing, b.vulnerability_index) for b in city.buildings),
        tuple((r.road_id, r.from_node, r.to_node, r.travel_time_min) for r in city.roads),
        tuple((h.hospital_id, h.easting, h.capacity) for h in city.hospitals),
        tuple((p.population_id, p.population_count) for p in city.population),
    )


def test_same_seed_produces_an_identical_city(config):
    assert _fingerprint(build_synthetic_city(config, seed=SEED)) == _fingerprint(
        build_synthetic_city(config, seed=SEED)
    )


def test_different_seed_produces_a_different_city(config):
    assert _fingerprint(build_synthetic_city(config, seed=SEED)) != _fingerprint(
        build_synthetic_city(config, seed=SEED + 1)
    )


def test_seed_defaults_to_config(config):
    default = build_synthetic_city(config)
    explicit = build_synthetic_city(config, seed=config["random"]["seed"])
    assert _fingerprint(default) == _fingerprint(explicit)


def test_named_streams_isolate_stages(mutable_config):
    """Changing the building count must leave the road network bit-identical.

    This is what named streams buy: stages can be varied independently, so a
    later experiment is a controlled one.
    """
    baseline = build_synthetic_city(mutable_config, seed=SEED)
    mutable_config["synthetic_city"]["n_buildings"] = 500
    varied = build_synthetic_city(mutable_config, seed=SEED)

    assert len(varied.buildings) == 500
    assert [(r.road_id, r.from_node, r.to_node) for r in baseline.roads] == [
        (r.road_id, r.from_node, r.to_node) for r in varied.roads
    ]
    assert [p.population_count for p in baseline.population] == [
        p.population_count for p in varied.population
    ]
