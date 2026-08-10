"""Phase 4 — hospital accessibility.

The invariant this file exists to protect: **unreachable population must never
leak into the travel-time statistics as an infinity or a NaN.** Every other
test here is secondary to that.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from mre.hospitals import (
    NearestHospitalAccessibility,
    compare_accessibility,
    travel_times_to_nearest_hospital,
    weighted_median,
)
from mre.roads import build_graph

SEED = 20260810
TOLERANCE = 0.01


@pytest.fixture
def model(config) -> NearestHospitalAccessibility:
    return NearestHospitalAccessibility(config)


@pytest.fixture
def graph(city) -> nx.Graph:
    return build_graph(city)


@pytest.fixture
def baseline(model, graph, city):
    return model.evaluate(graph, city)


def _severed(graph: nx.Graph, city) -> nx.Graph:
    """Isolate the node of one population unit, guaranteeing unreachability."""
    damaged = graph.copy()
    target = city.population[0].node_id
    damaged.remove_edges_from(list(damaged.edges(target)))
    return damaged


# --- the core invariant ---------------------------------------------------


def test_unreachable_population_never_enters_the_mean(model, graph, city):
    """A severed unit must be counted separately, not averaged as infinity."""
    result = model.evaluate(_severed(graph, city), city)

    assert result.population_unreachable > 0
    assert math.isfinite(result.mean_travel_time_min)
    assert math.isfinite(result.median_travel_time_min)
    assert not math.isnan(result.mean_travel_time_min)


def test_mean_equals_population_weighted_mean_over_reachable_only(model, graph, city):
    result = model.evaluate(_severed(graph, city), city)

    times = result.travel_time_min
    counts = np.array([p.population_count for p in city.population], dtype=float)
    reachable = np.isfinite(times)

    expected = np.average(times[reachable], weights=counts[reachable])
    assert result.mean_travel_time_min == pytest.approx(expected)
    # And the excluded units are genuinely accounted for elsewhere.
    assert result.population_unreachable == int(counts[~reachable].sum())


def test_reachable_and_unreachable_partition_the_population(model, graph, city):
    total = sum(p.population_count for p in city.population)
    for candidate in (graph, _severed(graph, city)):
        result = model.evaluate(candidate, city)
        assert result.population_reachable + result.population_unreachable == total


def test_travel_time_array_is_infinite_exactly_where_unreachable(model, graph, city):
    result = model.evaluate(_severed(graph, city), city)
    for index, time in enumerate(result.travel_time_min):
        assert (result.assigned_hospital[index] is None) == (not math.isfinite(time))


def test_no_nan_in_reported_statistics(model, graph, city):
    for candidate in (graph, _severed(graph, city)):
        result = model.evaluate(candidate, city)
        assert not math.isnan(result.mean_travel_time_min)
        assert not math.isnan(result.median_travel_time_min)
        assert math.isfinite(result.mean_travel_time_min)
        assert math.isfinite(result.median_travel_time_min)


def test_totally_disconnected_city_reports_nan_not_a_silent_zero(model, city):
    """Degenerate case: nobody reachable. Must be NaN so callers must handle
    it, never 0.0, which would read as 'instant access'."""
    result = model.evaluate(nx.Graph(), city)

    assert result.population_reachable == 0
    assert math.isnan(result.mean_travel_time_min)
    assert result.mean_travel_time_min != 0.0
    assert not np.isfinite(result.travel_time_min).any()


def test_units_on_a_hospital_node_stay_reachable_without_any_roads(model, city):
    """A unit co-located with a hospital is at the hospital: travel time 0,
    reachable even with every road destroyed. Not a bug -- and it is why
    'no edges' is not the same degenerate case as 'no graph'."""
    edgeless = nx.Graph()
    edgeless.add_nodes_from(city.nodes)
    result = model.evaluate(edgeless, city)

    hospital_nodes = {h.node_id for h in city.hospitals}
    on_hospital = np.array([p.node_id in hospital_nodes for p in city.population])
    assert on_hospital.any(), "fixture city has no unit co-located with a hospital"

    np.testing.assert_array_equal(np.isfinite(result.travel_time_min), on_hospital)
    assert result.mean_travel_time_min == pytest.approx(0.0)
    assert result.population_reachable == sum(
        p.population_count for p, flag in zip(city.population, on_hospital) if flag
    )


# --- baseline behaviour ---------------------------------------------------


def test_baseline_network_reaches_everyone(baseline, city):
    assert baseline.population_unreachable == 0
    assert baseline.population_reachable == sum(p.population_count for p in city.population)
    assert np.isfinite(baseline.travel_time_min).all()


def test_baseline_is_deterministic(model, graph, city):
    first = model.evaluate(graph, city)
    second = model.evaluate(graph, city)
    np.testing.assert_array_equal(first.travel_time_min, second.travel_time_min)
    assert first.hospital_load == second.hospital_load


def test_hospital_nodes_have_zero_travel_time(graph, city):
    times, _ = travel_times_to_nearest_hospital(graph, city)
    for index, unit in enumerate(city.population):
        if unit.node_id in {h.node_id for h in city.hospitals}:
            assert times[index] == pytest.approx(0.0)


def test_no_hospitals_yields_all_unreachable_without_raising(model, graph, city):
    import dataclasses

    hospital_free = dataclasses.replace(city, hospitals=())
    result = model.evaluate(graph, hospital_free)
    assert result.population_reachable == 0
    assert not np.isfinite(result.travel_time_min).any()
    assert result.hospital_load == {}


# --- documented model constraints ----------------------------------------


def test_hospital_loads_sum_to_reachable_population(model, graph, city):
    """Assignment is exhaustive over reachable units: every reachable person
    is assigned to exactly one hospital."""
    for candidate in (graph, _severed(graph, city)):
        result = model.evaluate(candidate, city)
        assert sum(result.hospital_load.values()) == result.population_reachable


def test_utilisation_is_load_over_emergency_capacity(baseline, city):
    for hospital in city.hospitals:
        expected = baseline.hospital_load[hospital.hospital_id] / hospital.emergency_capacity
        assert baseline.hospital_utilisation[hospital.hospital_id] == pytest.approx(expected)


def test_utilisation_is_unbounded_by_design(baseline):
    """A demand-pressure indicator, not a queueing result: values above 1 are
    expected and are not a bug. In this synthetic city they are large because
    population and capacity were chosen independently."""
    assert all(v >= 0 for v in baseline.hospital_utilisation.values())
    assert max(baseline.hospital_utilisation.values()) > 1.0


def test_every_hospital_appears_in_the_load_table(baseline, city):
    assert set(baseline.hospital_load) == {h.hospital_id for h in city.hospitals}
    assert set(baseline.hospital_utilisation) == {h.hospital_id for h in city.hospitals}


def test_assignment_is_to_the_nearest_hospital(baseline, graph, city):
    """Spot-check: no other hospital is closer than the assigned one."""
    for index in range(0, len(city.population), 40):
        unit = city.population[index]
        assigned = baseline.assigned_hospital[index]
        if assigned is None:
            continue
        distances = {
            h.hospital_id: nx.shortest_path_length(
                graph, unit.node_id, h.node_id, weight="travel_time_min"
            )
            for h in city.hospitals
        }
        assert distances[assigned] == pytest.approx(min(distances.values()))


# --- the max_travel_time_min threshold -----------------------------------


def test_threshold_censors_distant_units(mutable_config, graph, city):
    """Above the threshold counts as unreachable, exactly like no path."""
    mutable_config["accessibility"]["max_travel_time_min"] = 3.0
    strict = NearestHospitalAccessibility(mutable_config).evaluate(graph, city)
    generous = NearestHospitalAccessibility(
        mutable_config | {"accessibility": {**mutable_config["accessibility"], "max_travel_time_min": 600.0}}
    ).evaluate(graph, city)

    assert strict.population_unreachable > generous.population_unreachable
    assert (strict.travel_time_min[np.isfinite(strict.travel_time_min)] <= 3.0).all()


def test_non_positive_threshold_is_rejected(mutable_config):
    mutable_config["accessibility"]["max_travel_time_min"] = 0.0
    with pytest.raises(ValueError, match="max_travel_time_min"):
        NearestHospitalAccessibility(mutable_config)


# --- weighted statistics --------------------------------------------------


def test_weighted_median_respects_weights():
    values = np.array([1.0, 2.0, 100.0])
    assert weighted_median(values, np.array([1.0, 1.0, 1.0])) == 2.0
    # A dominant weight on the smallest value drags the median down.
    assert weighted_median(values, np.array([100.0, 1.0, 1.0])) == 1.0


def test_weighted_median_of_empty_is_nan():
    assert math.isnan(weighted_median(np.array([]), np.array([])))


def test_population_weighting_actually_applies(model, graph, city):
    """An unweighted mean would differ; confirm we are not silently computing
    one."""
    result = model.evaluate(graph, city)
    times = result.travel_time_min
    unweighted = float(times.mean())
    assert result.mean_travel_time_min != pytest.approx(unweighted, abs=1e-9)


# --- paired comparison ----------------------------------------------------


def test_comparing_baseline_with_itself_shows_no_disruption(baseline, city):
    comparison = compare_accessibility(baseline, baseline, city, tolerance_min=TOLERANCE)
    assert comparison.n_disrupted_routes == 0
    assert comparison.n_newly_unreachable_units == 0
    assert comparison.delta_mean_travel_time_min == pytest.approx(0.0)


def test_severed_network_registers_disruption(model, baseline, graph, city):
    post = model.evaluate(_severed(graph, city), city)
    comparison = compare_accessibility(baseline, post, city, tolerance_min=TOLERANCE)

    assert comparison.n_disrupted_routes > 0
    assert comparison.n_newly_unreachable_units >= 1
    assert comparison.population_newly_unreachable > 0
    assert comparison.delta_population_unreachable > 0


def test_already_unreachable_units_are_not_counted_as_newly_disrupted(model, graph, city):
    """A unit unreachable before and after did not change, so it must not be
    double-counted as disruption."""
    severed = _severed(graph, city)
    before = model.evaluate(severed, city)
    comparison = compare_accessibility(before, before, city, tolerance_min=TOLERANCE)
    assert before.population_unreachable > 0
    assert comparison.n_disrupted_routes == 0


def test_tolerance_suppresses_float_noise(baseline, city):
    """A negligible increase is not a disruption."""
    import dataclasses

    nudged = dataclasses.replace(
        baseline, travel_time_min=baseline.travel_time_min + 1e-9
    )
    comparison = compare_accessibility(baseline, nudged, city, tolerance_min=TOLERANCE)
    assert comparison.n_disrupted_routes == 0


def test_slower_routes_count_as_disrupted(baseline, city):
    """Doubling every travel time disrupts every route EXCEPT those already at
    zero -- units sitting on a hospital node, where 2 x 0 is still 0."""
    import dataclasses

    slower = dataclasses.replace(baseline, travel_time_min=baseline.travel_time_min * 2.0)
    comparison = compare_accessibility(baseline, slower, city, tolerance_min=TOLERANCE)

    expected = int((baseline.travel_time_min > TOLERANCE).sum())
    assert comparison.n_disrupted_routes == expected
    assert expected == len(city.population) - int(
        (baseline.travel_time_min <= TOLERANCE).sum()
    )
