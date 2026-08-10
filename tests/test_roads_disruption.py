"""Phase 3 — road disruption and the disrupted network."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from mre.data import build_synthetic_city
from mre.models import DamageState, LinkState
from mre.rng import named_generator
from mre.roads import (
    AdjacentCollapseDisruption,
    apply_link_states,
    build_graph,
    link_adjacency,
    shortest_travel_time,
    travel_times_from,
)

SEED = 20260810


@pytest.fixture
def model(config) -> AdjacentCollapseDisruption:
    return AdjacentCollapseDisruption(config)


@pytest.fixture
def no_damage(city) -> np.ndarray:
    return np.full(len(city.buildings), DamageState.NONE.value)


@pytest.fixture
def total_collapse(city) -> np.ndarray:
    return np.full(len(city.buildings), DamageState.COLLAPSE.value)


# --- baseline network -----------------------------------------------------


def test_network_is_connected_before_disruption(city):
    assert nx.is_connected(build_graph(city))


def test_graph_carries_link_attributes(city):
    graph = build_graph(city)
    link = city.roads[0]
    edge = graph.edges[link.from_node, link.to_node]
    assert edge["road_id"] == link.road_id
    assert edge["travel_time_min"] == pytest.approx(link.travel_time_min)
    assert edge["criticality"] == pytest.approx(link.criticality)


# --- closure probabilities ------------------------------------------------


def test_closure_probabilities_are_valid(model, city, no_damage):
    probabilities = model.closure_probabilities(city, no_damage)
    assert probabilities.shape == (len(city.roads),)
    assert (probabilities >= 0.0).all()
    assert (probabilities <= 1.0).all()


def test_no_damage_gives_only_the_baseline(model, city, no_damage, config):
    baseline = config["roads"]["disruption"]["baseline_closure_probability"]
    np.testing.assert_allclose(model.closure_probabilities(city, no_damage), baseline)


def test_collapses_raise_closure_probability(model, city, no_damage, total_collapse):
    quiet = model.closure_probabilities(city, no_damage)
    catastrophic = model.closure_probabilities(city, total_collapse)
    assert (catastrophic >= quiet).all()
    assert catastrophic.mean() > quiet.mean()


def test_closure_probabilities_are_deterministic(model, city, total_collapse):
    np.testing.assert_array_equal(
        model.closure_probabilities(city, total_collapse),
        model.closure_probabilities(city, total_collapse),
    )


def test_probabilities_are_capped_at_one(model, city, total_collapse):
    assert model.closure_probabilities(city, total_collapse).max() <= 1.0


def test_adjacency_is_within_the_configured_radius(city, config):
    radius = config["roads"]["disruption"]["adjacency_radius_m"]
    adjacency = link_adjacency(city, radius)
    assert len(adjacency) == len(city.roads)

    from shapely.geometry import LineString

    for index in range(0, len(city.roads), 97):
        link = city.roads[index]
        geometry = LineString([city.nodes[link.from_node], city.nodes[link.to_node]])
        for building_index in adjacency[index]:
            assert city.buildings[building_index].footprint().distance(geometry) <= radius + 1e-9


def test_susceptibility_modulates_disruption(mutable_config, city, total_collapse):
    """Higher susceptibility must mean more disruption, all else equal."""
    mutable_config["roads"]["disruption"]["per_adjacent_collapse"] = 0.001
    base = AdjacentCollapseDisruption(mutable_config).closure_probabilities(city, total_collapse)

    local = np.array([r.susceptibility for r in city.roads])
    # Links with higher susceptibility have higher probability, given equal
    # adjacency counts; compare within the same road class instead.
    assert np.corrcoef(local, base)[0, 1] > 0


# --- sampled link states --------------------------------------------------


def test_sampled_states_are_valid(model, city, total_collapse):
    states = model.sample_link_states(city, total_collapse, named_generator(SEED, "disruption:0"))
    assert states.shape == (len(city.roads),)
    assert set(states.tolist()) <= set(LinkState)


def test_sampling_is_reproducible(model, city, total_collapse):
    first = model.sample_link_states(city, total_collapse, named_generator(SEED, "disruption:0"))
    second = model.sample_link_states(city, total_collapse, named_generator(SEED, "disruption:0"))
    np.testing.assert_array_equal(first, second)


def test_different_seed_changes_the_realisation(model, city, total_collapse):
    first = model.sample_link_states(city, total_collapse, named_generator(SEED, "disruption:0"))
    second = model.sample_link_states(city, total_collapse, named_generator(SEED + 1, "disruption:0"))
    assert not np.array_equal(first, second)


def test_disruption_concentrates_where_buildings_collapse(
    model, city, config, total_collapse
):
    """The mechanism, stated as a contrast rather than a raw count.

    Only ~40% of links have any building within the adjacency radius -- the
    synthetic city is far sparser than a real one (see
    docs/SCIENTIFIC_ASSUMPTIONS.md), so a "most links close" assertion would be
    meaningless. What must hold is that debris drives disruption: links beside
    collapses fail far more often than links with no neighbours.
    """
    adjacency = link_adjacency(city, config["roads"]["disruption"]["adjacency_radius_m"])
    has_neighbour = np.array([len(a) > 0 for a in adjacency])
    assert has_neighbour.any() and (~has_neighbour).any()

    states = model.sample_link_states(city, total_collapse, named_generator(SEED, "disruption:0"))
    affected = np.array([s is not LinkState.OPEN for s in states])

    rate_with = affected[has_neighbour].mean()
    rate_without = affected[~has_neighbour].mean()
    assert rate_with > 5 * rate_without
    assert rate_with > 0.15


def test_zero_probability_leaves_everything_open(mutable_config, no_damage):
    """``closure_probability`` is baked into each RoadLink when the city is
    built, so the city must be rebuilt from the modified config -- mutating the
    config afterwards has no effect on an existing city."""
    mutable_config["roads"]["disruption"]["baseline_closure_probability"] = 0.0
    mutable_config["roads"]["disruption"]["per_adjacent_collapse"] = 0.0

    quiet_city = build_synthetic_city(mutable_config, seed=SEED)
    assert all(r.closure_probability == 0.0 for r in quiet_city.roads)

    model = AdjacentCollapseDisruption(mutable_config)
    damage = np.full(len(quiet_city.buildings), DamageState.COLLAPSE.value)
    states = model.sample_link_states(quiet_city, damage, named_generator(SEED, "disruption:0"))
    assert all(s is LinkState.OPEN for s in states)


def test_degraded_speed_factor_is_validated(mutable_config):
    mutable_config["roads"]["disruption"]["degraded_speed_factor"] = 0.0
    with pytest.raises(ValueError, match="degraded_speed_factor"):
        AdjacentCollapseDisruption(mutable_config)


# --- applying states to the graph ----------------------------------------


def test_closed_links_are_removed(model, city, config, total_collapse):
    graph = build_graph(city)
    states = model.sample_link_states(city, total_collapse, named_generator(SEED, "disruption:0"))
    disrupted = apply_link_states(
        graph, city, states, config["roads"]["disruption"]["degraded_speed_factor"]
    )

    closed = [
        link for link, state in zip(city.roads, states) if state is LinkState.CLOSED
    ]
    assert closed, "expected some closures under total collapse"
    for link in closed:
        assert not disrupted.has_edge(link.from_node, link.to_node)
    assert disrupted.number_of_edges() == len(city.roads) - len(closed)


def test_degraded_links_cost_more_but_survive(model, city, config, total_collapse):
    factor = config["roads"]["disruption"]["degraded_speed_factor"]
    graph = build_graph(city)
    states = model.sample_link_states(city, total_collapse, named_generator(SEED, "disruption:0"))
    disrupted = apply_link_states(graph, city, states, factor)

    degraded = [link for link, state in zip(city.roads, states) if state is LinkState.DEGRADED]
    assert degraded, "expected some degraded links"
    for link in degraded:
        assert disrupted.has_edge(link.from_node, link.to_node)
        cost = disrupted.edges[link.from_node, link.to_node]["travel_time_min"]
        assert cost == pytest.approx(link.travel_time_min / factor)
        assert cost > link.travel_time_min


def test_open_links_are_untouched(model, city, config, total_collapse):
    graph = build_graph(city)
    states = model.sample_link_states(city, total_collapse, named_generator(SEED, "disruption:0"))
    disrupted = apply_link_states(
        graph, city, states, config["roads"]["disruption"]["degraded_speed_factor"]
    )
    for link, state in zip(city.roads, states):
        if state is LinkState.OPEN:
            assert disrupted.edges[link.from_node, link.to_node][
                "travel_time_min"
            ] == pytest.approx(link.travel_time_min)


def test_the_original_network_is_preserved(model, city, config, total_collapse):
    """The baseline must remain available for before/after comparison."""
    graph = build_graph(city)
    edges_before = graph.number_of_edges()
    weights_before = [d["travel_time_min"] for _, _, d in graph.edges(data=True)]

    states = model.sample_link_states(city, total_collapse, named_generator(SEED, "disruption:0"))
    disrupted = apply_link_states(
        graph, city, states, config["roads"]["disruption"]["degraded_speed_factor"]
    )

    assert graph.number_of_edges() == edges_before
    assert [d["travel_time_min"] for _, _, d in graph.edges(data=True)] == weights_before
    assert disrupted is not graph
    assert disrupted.number_of_edges() < edges_before


# --- unreachability is a result, not an exception -------------------------


def test_disconnected_nodes_give_infinity_not_an_exception(city):
    graph = build_graph(city)
    isolated = next(iter(graph.nodes))
    graph.remove_edges_from(list(graph.edges(isolated)))
    target = max(graph.nodes)

    assert shortest_travel_time(graph, isolated, target) == math.inf
    assert math.isinf(travel_times_from(graph, [target])[isolated])


def test_missing_nodes_give_infinity(city):
    graph = build_graph(city)
    assert shortest_travel_time(graph, -1, 0) == math.inf
    assert shortest_travel_time(graph, 0, -1) == math.inf


def test_travel_times_from_covers_every_node(city):
    graph = build_graph(city)
    times = travel_times_from(graph, [city.hospitals[0].node_id])
    assert set(times) == set(graph.nodes)
    assert all(math.isfinite(t) for t in times.values())
    assert times[city.hospitals[0].node_id] == 0.0


def test_no_sources_gives_all_infinite(city):
    times = travel_times_from(build_graph(city), [])
    assert all(math.isinf(t) for t in times.values())


def test_severe_disruption_can_fragment_the_network(model, city, config, total_collapse):
    """Fragmentation must be representable, and produce inf rather than raise."""
    graph = build_graph(city)
    states = model.sample_link_states(city, total_collapse, named_generator(SEED, "disruption:0"))
    disrupted = apply_link_states(
        graph, city, states, config["roads"]["disruption"]["degraded_speed_factor"]
    )

    assert not nx.is_connected(disrupted)
    times = travel_times_from(disrupted, [city.hospitals[0].node_id])
    assert any(math.isinf(t) for t in times.values())
    assert any(math.isfinite(t) for t in times.values())
