"""World construction: synthetic now, real data later.

The swap point for real data. Any future loader (OSM roads, building
inventories, census population) must return the same ``SyntheticCity`` shape so
the simulation core is unchanged.

Everything here is INVENTED. The city is not Istanbul, not simplified Istanbul,
and not anonymised Istanbul. See docs/SCIENTIFIC_ASSUMPTIONS.md.

Seeding: each sub-generator owns a named stream (``city.roads``,
``city.buildings``, ``city.hospitals``, ``city.population``). Changing the
building count therefore leaves the road layout bit-for-bit identical, which
makes controlled experiments possible.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol

import networkx as nx
import numpy as np

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
from mre.rng import named_generator

__all__ = ["CityBuilder", "build_synthetic_city", "nearest_node"]


class CityBuilder(Protocol):
    """Anything that can produce a world for the engine to simulate."""

    def build(self, config: dict[str, Any], seed: int) -> SyntheticCity: ...


def _weighted_choice(
    rng: np.random.Generator, weights: dict[str, float], size: int
) -> np.ndarray:
    """Sample keys from a ``{name: weight}`` mapping, order made deterministic
    by sorting the keys."""
    names = sorted(weights)
    probabilities = np.array([weights[n] for n in names], dtype=float)
    probabilities /= probabilities.sum()
    return rng.choice(names, size=size, p=probabilities)


def nearest_node(
    nodes: dict[int, tuple[float, float]], easting: float, northing: float
) -> int:
    """Nearest network node to a point. Used to attach hospitals and population
    units to the road graph."""
    ids = np.fromiter(nodes.keys(), dtype=int, count=len(nodes))
    coords = np.array([nodes[i] for i in ids], dtype=float)
    distances = np.hypot(coords[:, 0] - easting, coords[:, 1] - northing)
    return int(ids[int(np.argmin(distances))])


def _build_road_network(
    cfg: dict[str, Any], rng: np.random.Generator
) -> tuple[dict[int, tuple[float, float]], list[RoadLink]]:
    """A regular grid with random edge dropout, kept connected by construction.

    Dropout breaks the perfect regularity of a grid; each candidate removal is
    reverted if it would disconnect the network, so the undamaged city is
    always a single connected component. Real networks are of course not
    grids -- see docs/SCIENTIFIC_ASSUMPTIONS.md.
    """
    extent_x, extent_y = cfg["extent_m"]
    origin_e, origin_n = cfg["origin_easting"], cfg["origin_northing"]
    nx_cells, ny_cells = cfg["road_grid"]
    roads_cfg = cfg["roads"]

    dx, dy = extent_x / nx_cells, extent_y / ny_cells

    def node_id(i: int, j: int) -> int:
        return j * (nx_cells + 1) + i

    nodes = {
        node_id(i, j): (origin_e + i * dx, origin_n + j * dy)
        for i in range(nx_cells + 1)
        for j in range(ny_cells + 1)
    }

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for i in range(nx_cells):
        for j in range(ny_cells + 1):
            graph.add_edge(node_id(i, j), node_id(i + 1, j), length_m=dx)
    for i in range(nx_cells + 1):
        for j in range(ny_cells):
            graph.add_edge(node_id(i, j), node_id(i, j + 1), length_m=dy)

    candidates = sorted(tuple(sorted(e)) for e in graph.edges())
    target = int(len(candidates) * cfg["road_edge_dropout"])
    removed = 0
    for index in rng.permutation(len(candidates)):
        if removed >= target:
            break
        u, v = candidates[int(index)]
        attrs = graph.edges[u, v]
        graph.remove_edge(u, v)
        if nx.is_connected(graph):
            removed += 1
        else:
            graph.add_edge(u, v, **attrs)

    edges = sorted(tuple(sorted(e)) for e in graph.edges())
    classes = _weighted_choice(rng, roads_cfg["class_weights"], len(edges))

    links: list[RoadLink] = []
    for index, ((u, v), class_name) in enumerate(zip(edges, classes)):
        road_class = RoadClass(str(class_name))
        length_m = float(graph.edges[u, v]["length_m"])
        speed_kmh = roads_cfg["speed_kmh"][road_class.value]
        links.append(
            RoadLink(
                road_id=f"R{index:05d}",
                from_node=u,
                to_node=v,
                length_m=length_m,
                road_class=road_class,
                travel_time_min=(length_m / 1000.0) / speed_kmh * 60.0,
                closure_probability=float(
                    cfg["_baseline_closure_probability"]
                ),
                susceptibility=float(roads_cfg["susceptibility"][road_class.value]),
            )
        )
    return nodes, links


def _assign_criticality(links: list[RoadLink], nodes: dict) -> list[RoadLink]:
    """Edge betweenness on the undamaged network, used as link criticality.

    Betweenness answers "how many shortest paths rely on this link", which is
    the property that matters when a link is removed. It is descriptive, not a
    prediction of importance during a real emergency.
    """
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for link in links:
        graph.add_edge(link.from_node, link.to_node, travel_time_min=link.travel_time_min)

    centrality = nx.edge_betweenness_centrality(graph, weight="travel_time_min")
    resolved = []
    for link in links:
        key = (link.from_node, link.to_node)
        value = centrality.get(key, centrality.get((link.to_node, link.from_node), 0.0))
        resolved.append(dataclasses.replace(link, criticality=float(value)))
    return resolved


def _vulnerability_index(
    vuln_cfg: dict[str, Any],
    structural_types: np.ndarray,
    years: np.ndarray,
    floors: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Heterogeneous synthetic vulnerability.

        v = type_factor * year_factor * (floors / ref)^exp * lognormal(sigma)

    Never a constant: the lognormal term guarantees that two otherwise
    identical buildings still differ, which is the point of a synthetic
    *population* rather than a synthetic archetype.
    """
    type_factors = np.array(
        [vuln_cfg["structural_type_factor"][str(t)] for t in structural_types], dtype=float
    )

    early, late = vuln_cfg["year_breakpoints"]
    f_old, f_mid, f_new = vuln_cfg["year_factor"]
    year_factors = np.where(years < early, f_old, np.where(years < late, f_mid, f_new))

    floor_factors = (floors / vuln_cfg["floors_reference"]) ** vuln_cfg["floors_exponent"]
    noise = rng.lognormal(mean=0.0, sigma=vuln_cfg["sigma_ln"], size=len(years))

    low, high = vuln_cfg["clip"]
    return np.clip(type_factors * year_factors * floor_factors * noise, low, high)


def _build_buildings(cfg: dict[str, Any], rng: np.random.Generator) -> list[Building]:
    extent_x, extent_y = cfg["extent_m"]
    origin_e, origin_n = cfg["origin_easting"], cfg["origin_northing"]
    b_cfg = cfg["buildings"]
    n = cfg["n_buildings"]

    eastings = origin_e + rng.uniform(0.0, extent_x, n)
    northings = origin_n + rng.uniform(0.0, extent_y, n)

    floor_low, floor_high = b_cfg["floors_range"]
    floors = rng.integers(floor_low, floor_high + 1, n)

    year_low, year_high = b_cfg["construction_year_range"]
    years = rng.integers(year_low, year_high + 1, n)

    structural_types = _weighted_choice(rng, b_cfg["structural_type_weights"], n)
    occupancies = _weighted_choice(rng, b_cfg["occupancy_weights"], n)

    side_low, side_high = b_cfg["footprint_side_range"]
    sides = rng.uniform(side_low, side_high, n)

    occupants = rng.poisson(b_cfg["mean_occupants_per_floor"] * floors)
    vulnerability = _vulnerability_index(b_cfg["vulnerability"], structural_types, years, floors, rng)

    return [
        Building(
            building_id=f"B{i:05d}",
            easting=float(eastings[i]),
            northing=float(northings[i]),
            floors=int(floors[i]),
            construction_year=int(years[i]),
            structural_type=StructuralType(str(structural_types[i])),
            occupancy=Occupancy(str(occupancies[i])),
            occupants=int(occupants[i]),
            vulnerability_index=float(vulnerability[i]),
            footprint_side_m=float(sides[i]),
        )
        for i in range(n)
    ]


def _build_hospitals(
    cfg: dict[str, Any], nodes: dict[int, tuple[float, float]], rng: np.random.Generator
) -> list[Hospital]:
    """Hospitals spread across the extent with jitter, so they are neither
    co-located nor perfectly regular."""
    extent_x, extent_y = cfg["extent_m"]
    origin_e, origin_n = cfg["origin_easting"], cfg["origin_northing"]
    h_cfg = cfg["hospitals"]
    n = cfg["n_hospitals"]

    # Evenly spaced anchors along the diagonal, then jittered.
    fractions = np.linspace(0.2, 0.8, n)
    jitter_x = rng.uniform(-0.12, 0.12, n)
    jitter_y = rng.uniform(-0.12, 0.12, n)
    eastings = origin_e + np.clip(fractions + jitter_x, 0.05, 0.95) * extent_x
    northings = origin_n + np.clip(fractions[::-1] + jitter_y, 0.05, 0.95) * extent_y

    cap_low, cap_high = h_cfg["capacity_range"]
    capacities = rng.integers(cap_low, cap_high + 1, n)
    fraction = h_cfg["emergency_capacity_fraction"]

    return [
        Hospital(
            hospital_id=f"H{i:02d}",
            easting=float(eastings[i]),
            northing=float(northings[i]),
            capacity=int(capacities[i]),
            emergency_capacity=max(1, int(round(capacities[i] * fraction))),
            node_id=nearest_node(nodes, float(eastings[i]), float(northings[i])),
        )
        for i in range(n)
    ]


def _build_population(
    cfg: dict[str, Any], nodes: dict[int, tuple[float, float]], rng: np.random.Generator
) -> list[PopulationUnit]:
    """A separate population grid, DECOUPLED from buildings.

    Demand must never be a function of the damage model, or accessibility
    improvements and damage reductions become impossible to disentangle.
    """
    extent_x, extent_y = cfg["extent_m"]
    origin_e, origin_n = cfg["origin_easting"], cfg["origin_northing"]
    p_cfg = cfg["population"]
    gx, gy = p_cfg["grid"]

    cell_x, cell_y = extent_x / gx, extent_y / gy
    centres = [
        (origin_e + (i + 0.5) * cell_x, origin_n + (j + 0.5) * cell_y)
        for j in range(gy)
        for i in range(gx)
    ]

    weights = rng.lognormal(mean=0.0, sigma=p_cfg["count_sigma_ln"], size=len(centres))
    weights /= weights.sum()
    counts = np.maximum(1, np.round(weights * p_cfg["total_population"]).astype(int))

    return [
        PopulationUnit(
            population_id=f"P{i:04d}",
            easting=float(easting),
            northing=float(northing),
            population_count=int(counts[i]),
            node_id=nearest_node(nodes, float(easting), float(northing)),
        )
        for i, (easting, northing) in enumerate(centres)
    ]


def build_synthetic_city(config: dict[str, Any], seed: int | None = None) -> SyntheticCity:
    """Generate the synthetic city described by ``config['synthetic_city']``.

    Fully determined by ``seed``: same seed and config, same city.
    """
    seed = config["random"]["seed"] if seed is None else seed
    cfg = dict(config["synthetic_city"])
    cfg["_baseline_closure_probability"] = config["roads"]["disruption"][
        "baseline_closure_probability"
    ]

    nodes, links = _build_road_network(cfg, named_generator(seed, "city.roads"))
    links = _assign_criticality(links, nodes)

    buildings = _build_buildings(cfg, named_generator(seed, "city.buildings"))
    hospitals = _build_hospitals(cfg, nodes, named_generator(seed, "city.hospitals"))
    population = _build_population(cfg, nodes, named_generator(seed, "city.population"))

    return SyntheticCity(
        crs=config["project"]["crs"],
        seed=seed,
        buildings=tuple(buildings),
        hospitals=tuple(hospitals),
        roads=tuple(links),
        population=tuple(population),
        nodes=nodes,
    )
