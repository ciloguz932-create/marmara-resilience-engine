"""Road network construction and post-earthquake disruption — PROTOTYPE.

NetworkX is the graph backend. OSMnx is an optional FUTURE integration and is
deliberately not a dependency: the synthetic slice must run with no network
access and no downloads.

Disruption formulation. For link *l*:

    n_l      = collapsed buildings whose footprint lies within R of the link
    p_l      = clip( baseline_l + k * n_l * susceptibility_l , 0, 1 )
    affected ~ Bernoulli(p_l)
    state    = DEGRADED with probability q, else CLOSED   (given affected)

``R`` is ``adjacency_radius_m``, ``k`` is ``per_adjacent_collapse``, and ``q``
is ``probability_degraded_given_affected``. CLOSED links are removed from the
graph; DEGRADED links stay with travel time divided by
``degraded_speed_factor``.

The original network is never mutated -- ``apply_link_states`` returns a new
graph, so the baseline stays available for comparison.

All parameters are INVENTED. Debris blockage is modelled by proximity count
alone; bridges, tunnels and embankments are not distinct assets; there is no
time dimension and no debris clearance. See docs/SCIENTIFIC_ASSUMPTIONS.md.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import networkx as nx
import numpy as np
from shapely.geometry import LineString
from shapely.strtree import STRtree

from mre.models import DamageState, LinkState, SyntheticCity

__all__ = [
    "DisruptionModel",
    "AdjacentCollapseDisruption",
    "build_graph",
    "apply_link_states",
    "link_adjacency",
    "shortest_travel_time",
    "travel_times_from",
]


def build_graph(city: SyntheticCity) -> nx.Graph:
    """The undamaged network, edges weighted by free-flow travel time."""
    graph = nx.Graph()
    for node_id, (easting, northing) in city.nodes.items():
        graph.add_node(node_id, easting=easting, northing=northing)
    for link in city.roads:
        graph.add_edge(
            link.from_node,
            link.to_node,
            road_id=link.road_id,
            road_class=link.road_class.value,
            length_m=link.length_m,
            travel_time_min=link.travel_time_min,
            criticality=link.criticality,
            susceptibility=link.susceptibility,
        )
    return graph


def link_adjacency(city: SyntheticCity, radius_m: float) -> list[np.ndarray]:
    """Building indices within ``radius_m`` of each link, by footprint.

    Static for a given city, so the disruption model computes it once and
    reuses it across realisations. STRtree keeps this tractable for 1,000+
    buildings against 800+ links.
    """
    footprints = [b.footprint() for b in city.buildings]
    tree = STRtree(footprints)

    adjacency = []
    for link in city.roads:
        geometry = LineString([city.nodes[link.from_node], city.nodes[link.to_node]])
        candidates = tree.query(geometry.buffer(radius_m))
        # query() is bounding-box based; confirm true distance.
        hits = [
            int(i) for i in candidates if footprints[int(i)].distance(geometry) <= radius_m
        ]
        adjacency.append(np.array(sorted(hits), dtype=int))
    return adjacency


class DisruptionModel(Protocol):
    def closure_probabilities(
        self, city: SyntheticCity, damage_states: np.ndarray
    ) -> np.ndarray: ...

    def sample_link_states(
        self,
        city: SyntheticCity,
        damage_states: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray: ...


class AdjacentCollapseDisruption:
    """Debris-driven disruption from collapsed buildings near each link."""

    def __init__(self, config: dict[str, Any]) -> None:
        disruption = config["roads"]["disruption"]
        self.radius_m = float(disruption["adjacency_radius_m"])
        self.per_adjacent_collapse = float(disruption["per_adjacent_collapse"])
        self.degraded_speed_factor = float(disruption["degraded_speed_factor"])
        self.probability_degraded = float(disruption["probability_degraded_given_affected"])
        if not 0.0 < self.degraded_speed_factor <= 1.0:
            raise ValueError("degraded_speed_factor must be in (0, 1]")
        self._adjacency_cache: dict[int, list[np.ndarray]] = {}

    def adjacency(self, city: SyntheticCity) -> list[np.ndarray]:
        key = id(city)
        if key not in self._adjacency_cache:
            self._adjacency_cache[key] = link_adjacency(city, self.radius_m)
        return self._adjacency_cache[key]

    def closure_probabilities(
        self, city: SyntheticCity, damage_states: np.ndarray
    ) -> np.ndarray:
        """Per-link probability of being affected. Deterministic given damage."""
        collapsed = np.asarray(damage_states) == DamageState.COLLAPSE.value
        adjacency = self.adjacency(city)

        probabilities = np.empty(len(city.roads), dtype=float)
        for index, link in enumerate(city.roads):
            neighbours = adjacency[index]
            n_collapsed = int(collapsed[neighbours].sum()) if neighbours.size else 0
            probabilities[index] = (
                link.closure_probability
                + self.per_adjacent_collapse * n_collapsed * link.susceptibility
            )
        return np.clip(probabilities, 0.0, 1.0)

    def sample_link_states(
        self,
        city: SyntheticCity,
        damage_states: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """One realisation: a ``LinkState`` per road link."""
        probabilities = self.closure_probabilities(city, damage_states)
        n_links = len(city.roads)

        affected = rng.random(n_links) < probabilities
        degraded_draw = rng.random(n_links) < self.probability_degraded

        states = np.full(n_links, LinkState.OPEN, dtype=object)
        states[affected & degraded_draw] = LinkState.DEGRADED
        states[affected & ~degraded_draw] = LinkState.CLOSED
        return states


def apply_link_states(
    graph: nx.Graph,
    city: SyntheticCity,
    link_states: np.ndarray,
    degraded_speed_factor: float,
) -> nx.Graph:
    """Return a NEW disrupted graph; ``graph`` is left untouched.

    CLOSED links are removed. DEGRADED links remain with travel time divided by
    ``degraded_speed_factor`` (a factor < 1 means slower, so dividing raises the
    cost).
    """
    disrupted = graph.copy()
    for link, state in zip(city.roads, link_states):
        if state is LinkState.CLOSED:
            if disrupted.has_edge(link.from_node, link.to_node):
                disrupted.remove_edge(link.from_node, link.to_node)
        elif state is LinkState.DEGRADED:
            edge = disrupted.edges[link.from_node, link.to_node]
            edge["travel_time_min"] = link.travel_time_min / degraded_speed_factor
            edge["link_state"] = LinkState.DEGRADED.value
    return disrupted


def shortest_travel_time(graph: nx.Graph, source: int, target: int) -> float:
    """Travel time in minutes, or ``inf`` when no path exists.

    Disconnection is a RESULT, not an error: an unreachable node is exactly
    what the accessibility stage will need to count.
    """
    if source not in graph or target not in graph:
        return math.inf
    try:
        return float(nx.shortest_path_length(graph, source, target, weight="travel_time_min"))
    except nx.NetworkXNoPath:
        return math.inf


def travel_times_from(graph: nx.Graph, sources: list[int]) -> dict[int, float]:
    """Travel time from every node to its nearest source; ``inf`` if isolated."""
    present = [s for s in sources if s in graph]
    times = {node: math.inf for node in graph.nodes}
    if not present:
        return times
    lengths, _ = nx.multi_source_dijkstra(graph, sources=present, weight="travel_time_min")
    times.update({node: float(value) for node, value in lengths.items()})
    return times
