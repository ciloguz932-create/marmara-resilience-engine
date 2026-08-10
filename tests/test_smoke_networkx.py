"""Phase 2, smoke test 4 — NetworkX.

Proves the graph mechanics the accessibility stage depends on: weighted
shortest paths, multi-source Dijkstra from hospital nodes, edge removal
(closure), edge re-weighting (degradation), and disconnection producing
*unreachable* rather than an exception.

OSMnx is deliberately absent — the synthetic slice must run with no network
access. This test uses only NetworkX.
"""

from __future__ import annotations

import math

import networkx as nx
import pytest

import mre  # noqa: F401
from mre.config import load_config

CONFIG = load_config()


@pytest.fixture
def grid() -> nx.Graph:
    """5x5 grid, every edge 200 m at 30 km/h => 0.4 min."""
    graph = nx.grid_2d_graph(5, 5)
    nx.set_edge_attributes(graph, 0.4, "travel_time_min")
    nx.set_edge_attributes(graph, 200.0, "length_m")
    return graph


def test_networkx_available():
    assert nx.__version__.startswith("3.")


def test_grid_is_connected(grid):
    assert nx.is_connected(grid)
    assert grid.number_of_nodes() == 25
    assert grid.number_of_edges() == 40


def test_weighted_shortest_path(grid):
    """Corner to corner: 8 edges x 0.4 min."""
    cost = nx.shortest_path_length(grid, (0, 0), (4, 4), weight="travel_time_min")
    assert cost == pytest.approx(3.2)


def test_multi_source_dijkstra_from_hospitals(grid):
    """The accessibility primitive: distance from every node to its nearest
    hospital, in one pass."""
    hospitals = [(0, 0), (4, 4)]
    lengths, paths = nx.multi_source_dijkstra(grid, sources=hospitals, weight="travel_time_min")

    assert len(lengths) == 25
    assert lengths[(0, 0)] == 0.0
    assert lengths[(4, 4)] == 0.0
    assert lengths[(2, 2)] == pytest.approx(1.6)
    # Each path starts at whichever hospital is nearest.
    assert paths[(1, 0)][0] in hospitals


def test_link_closure_lengthens_the_route(grid):
    """Removing edges = CLOSED links. The detour must cost more."""
    before = nx.shortest_path_length(grid, (0, 0), (0, 2), weight="travel_time_min")
    grid.remove_edge((0, 0), (0, 1))
    after = nx.shortest_path_length(grid, (0, 0), (0, 2), weight="travel_time_min")
    assert after > before


def test_link_degradation_reweights_without_removal(grid):
    """DEGRADED links stay in the graph at reduced speed."""
    factor = CONFIG["roads"]["disruption"]["degraded_speed_factor"]
    before = nx.shortest_path_length(grid, (0, 0), (0, 1), weight="travel_time_min")

    for u, v in [((0, 0), (0, 1))]:
        grid[u][v]["travel_time_min"] /= factor

    after = nx.shortest_path_length(grid, (0, 0), (0, 1), weight="travel_time_min")
    assert after == pytest.approx(before / factor)
    assert grid.has_edge((0, 0), (0, 1))


def test_disconnection_yields_unreachable_not_an_exception(grid):
    """Isolating a node must surface as 'unreachable', because that is a
    reported metric, not an error condition."""
    isolated = (0, 0)
    grid.remove_edges_from(list(grid.edges(isolated)))

    lengths, _ = nx.multi_source_dijkstra(grid, sources=[(4, 4)], weight="travel_time_min")
    assert isolated not in lengths

    travel_time = lengths.get(isolated, math.inf)
    assert math.isinf(travel_time)
    assert not nx.has_path(grid, isolated, (4, 4))


def test_unreachable_threshold_from_config(grid):
    """Nodes beyond max_travel_time_min are counted separately rather than
    dragging the mean upward."""
    limit = CONFIG["accessibility"]["max_travel_time_min"]
    assert limit > 0
    lengths, _ = nx.multi_source_dijkstra(grid, sources=[(0, 0)], weight="travel_time_min")
    reachable = [t for t in lengths.values() if t <= limit]
    assert len(reachable) == 25


def test_betweenness_centrality_ranks_links(grid):
    """RoadLink.criticality will be derived from edge betweenness; central
    edges must outrank peripheral ones."""
    centrality = nx.edge_betweenness_centrality(grid, weight="travel_time_min")
    central = centrality[((2, 1), (2, 2))] if ((2, 1), (2, 2)) in centrality else centrality[((2, 2), (2, 1))]
    corner_key = ((0, 0), (0, 1)) if ((0, 0), (0, 1)) in centrality else ((0, 1), (0, 0))
    assert central > centrality[corner_key]
