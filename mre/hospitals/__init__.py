"""Emergency accessibility to hospitals — PROTOTYPE.

Modular by design: ``AccessibilityModel`` is the seam. MRE-001 ships a
shortest-travel-time model (multi-source Dijkstra from hospital nodes). A future
model can add capacity-constrained assignment or congestion without changing the
Monte Carlo driver.

**Unreachable population is handled explicitly and is NEVER folded into the mean
or median as an infinite travel time.** Units with no path, or with a travel
time above ``max_travel_time_min``, are excluded from the travel-time statistics
and counted separately. Averaging in an infinity would produce an infinite or
NaN mean, and averaging in the threshold value would silently understate the
loss; both hide exactly the effect the engine exists to measure.

Statistics are **population-weighted**: a unit of 5,000 people counts five times
a unit of 1,000.

Hospital utilisation is a **demand-pressure indicator**, not a queueing result.
Assignment is by travel time alone; capacity does not constrain it, so values
above 1.0 mean assigned demand exceeds emergency capacity, not that patients
were turned away.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import networkx as nx
import numpy as np

from mre.models import SyntheticCity

__all__ = [
    "AccessibilityResult",
    "AccessibilityComparison",
    "AccessibilityModel",
    "NearestHospitalAccessibility",
    "compare_accessibility",
    "travel_times_to_nearest_hospital",
    "weighted_median",
]


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Population-weighted median: the value at 50% of cumulative weight."""
    if values.size == 0 or weights.sum() <= 0:
        return math.nan
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, cumulative[-1] / 2.0, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


@dataclass(frozen=True, slots=True)
class AccessibilityResult:
    """Population-weighted accessibility for one network state.

    ``mean_travel_time_min`` and ``median_travel_time_min`` cover the REACHABLE
    population only; they are ``nan`` when nobody is reachable, which callers
    must handle rather than propagate.
    """

    mean_travel_time_min: float
    median_travel_time_min: float
    population_reachable: int
    population_unreachable: int
    # hospital_id -> assigned population (reachable units only)
    hospital_load: dict[str, int]
    # hospital_id -> assigned population / emergency_capacity
    hospital_utilisation: dict[str, float]
    # Per population unit, in city.population order; inf where unreachable.
    travel_time_min: np.ndarray
    # Per population unit; None where unreachable.
    assigned_hospital: tuple[str | None, ...]

    @property
    def population_total(self) -> int:
        return self.population_reachable + self.population_unreachable

    @property
    def fraction_unreachable(self) -> float:
        total = self.population_total
        return self.population_unreachable / total if total else math.nan

    @property
    def reachable_mask(self) -> np.ndarray:
        return np.isfinite(self.travel_time_min)


@dataclass(frozen=True, slots=True)
class AccessibilityComparison:
    """A PAIRED before/after comparison on the same city and same realisation.

    Never compare accessibility across unrelated realisations: the difference
    would mix the effect of the earthquake with the difference between two
    random draws.
    """

    baseline: AccessibilityResult
    post_event: AccessibilityResult
    n_disrupted_routes: int
    n_newly_unreachable_units: int
    population_newly_unreachable: int
    delta_mean_travel_time_min: float
    delta_median_travel_time_min: float
    delta_population_unreachable: int


class AccessibilityModel(Protocol):
    def evaluate(self, graph: nx.Graph, city: SyntheticCity) -> AccessibilityResult: ...


def travel_times_to_nearest_hospital(
    graph: nx.Graph, city: SyntheticCity
) -> tuple[np.ndarray, tuple[str | None, ...]]:
    """Minutes from each population unit to its nearest hospital, and which one.

    ``inf`` where unreachable. Disconnection is a result, not an exception.
    """
    node_to_hospital: dict[int, str] = {}
    for hospital in city.hospitals:
        # First hospital wins if two share a node; deterministic by city order.
        node_to_hospital.setdefault(hospital.node_id, hospital.hospital_id)

    sources = [n for n in node_to_hospital if n in graph]
    n_units = len(city.population)

    if not sources:
        return np.full(n_units, math.inf), tuple([None] * n_units)

    lengths, paths = nx.multi_source_dijkstra(graph, sources=sources, weight="travel_time_min")

    times = np.full(n_units, math.inf, dtype=float)
    assigned: list[str | None] = [None] * n_units
    for index, unit in enumerate(city.population):
        if unit.node_id in lengths:
            times[index] = float(lengths[unit.node_id])
            assigned[index] = node_to_hospital.get(paths[unit.node_id][0])
    return times, tuple(assigned)


class NearestHospitalAccessibility:
    """Assign each population unit to its shortest-travel-time hospital."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.max_travel_time_min = float(config["accessibility"]["max_travel_time_min"])
        if self.max_travel_time_min <= 0:
            raise ValueError("max_travel_time_min must be positive")

    def evaluate(self, graph: nx.Graph, city: SyntheticCity) -> AccessibilityResult:
        times, assigned = travel_times_to_nearest_hospital(graph, city)
        counts = np.array([p.population_count for p in city.population], dtype=float)

        # Beyond the threshold counts as unreachable, exactly like no path.
        reachable = np.isfinite(times) & (times <= self.max_travel_time_min)

        # Censor the travel time of threshold-exceeding units to inf, so the
        # array and the statistics agree on what "reachable" means.
        censored = np.where(reachable, times, math.inf)
        assigned_censored = tuple(
            hospital if reachable[i] else None for i, hospital in enumerate(assigned)
        )

        population_reachable = int(counts[reachable].sum())
        population_unreachable = int(counts[~reachable].sum())

        if reachable.any():
            mean_travel_time = float(
                np.average(times[reachable], weights=counts[reachable])
            )
            median_travel_time = weighted_median(times[reachable], counts[reachable])
        else:
            mean_travel_time = math.nan
            median_travel_time = math.nan

        load = {h.hospital_id: 0 for h in city.hospitals}
        for index, hospital_id in enumerate(assigned_censored):
            if hospital_id is not None:
                load[hospital_id] += int(counts[index])

        utilisation = {
            h.hospital_id: load[h.hospital_id] / h.emergency_capacity
            for h in city.hospitals
        }

        return AccessibilityResult(
            mean_travel_time_min=mean_travel_time,
            median_travel_time_min=median_travel_time,
            population_reachable=population_reachable,
            population_unreachable=population_unreachable,
            hospital_load=load,
            hospital_utilisation=utilisation,
            travel_time_min=censored,
            assigned_hospital=assigned_censored,
        )


def compare_accessibility(
    baseline: AccessibilityResult,
    post_event: AccessibilityResult,
    city: SyntheticCity,
    *,
    tolerance_min: float,
) -> AccessibilityComparison:
    """Paired before/after comparison for ONE realisation.

    A route is disrupted when its travel time rises by more than
    ``tolerance_min``, or when it becomes unreachable. Routes that were already
    unreachable before the event are not counted again -- they did not change.
    """
    counts = np.array([p.population_count for p in city.population], dtype=float)
    before, after = baseline.travel_time_min, post_event.travel_time_min

    was_reachable = np.isfinite(before)
    is_reachable = np.isfinite(after)

    newly_unreachable = was_reachable & ~is_reachable
    slower = was_reachable & is_reachable & (after > before + tolerance_min)

    return AccessibilityComparison(
        baseline=baseline,
        post_event=post_event,
        n_disrupted_routes=int((newly_unreachable | slower).sum()),
        n_newly_unreachable_units=int(newly_unreachable.sum()),
        population_newly_unreachable=int(counts[newly_unreachable].sum()),
        delta_mean_travel_time_min=post_event.mean_travel_time_min
        - baseline.mean_travel_time_min,
        delta_median_travel_time_min=post_event.median_travel_time_min
        - baseline.median_travel_time_min,
        delta_population_unreachable=post_event.population_unreachable
        - baseline.population_unreachable,
    )
