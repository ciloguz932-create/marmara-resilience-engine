"""Scenario driver and Monte Carlo orchestration.

    hazard.sample -> fragility.sample_damage -> disruption.sample_link_states
    -> apply_link_states -> accessibility.evaluate -> paired comparison

**Orchestration only.** This module owns looping, seeding, and accumulation. It
knows the hazard, fragility, disruption, and accessibility models solely through
their protocols, never as concrete classes, so any of them can be replaced
without touching this file.

Reproducibility. Each stage draws from its own named stream, derived from the
master seed and the realisation index:

    hazard:<i>      damage:<i>      disruption:<i>

Because streams are named rather than sequential, realisation *i* is identical
whether it is run alone or as part of a 1,000-realisation batch, and adding a
stage later cannot shift an existing one.

Baseline pairing. The pre-earthquake network is a property of the city, not of a
realisation, so baseline accessibility is deterministic and computed once. Every
realisation is compared against **that same baseline** — a paired comparison on
one city. Unrelated realisations are never compared against each other.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Sequence

import networkx as nx
import numpy as np

from mre.buildings import DamageResult, FragilityModel, PrototypeLognormalFragility
from mre.hazard import HazardField, hazard_field_from_config
from mre.hospitals import (
    AccessibilityComparison,
    AccessibilityModel,
    AccessibilityResult,
    NearestHospitalAccessibility,
    compare_accessibility,
)
from mre.models import DamageState, LinkState, SyntheticCity
from mre.rng import child_generators, named_generator
from mre.roads import (
    AdjacentCollapseDisruption,
    DisruptionModel,
    apply_link_states,
    build_graph,
)

__all__ = [
    "ScenarioResult",
    "RealisationRecord",
    "MonteCarloResult",
    "ModelSuite",
    "build_models",
    "run_scenario",
    "run_monte_carlo",
    "child_generators",
    "named_generator",
]


@dataclass(frozen=True, slots=True)
class ModelSuite:
    """The four replaceable scientific models, built once and reused.

    Constructing these per realisation would recompute the link/building
    adjacency index 1,000 times; more importantly, holding them here keeps the
    orchestrator ignorant of which concrete models it is running.
    """

    hazard: HazardField
    fragility: FragilityModel
    disruption: DisruptionModel
    accessibility: AccessibilityModel


def build_models(config: dict[str, Any]) -> ModelSuite:
    """Default prototype models, selected by config."""
    return ModelSuite(
        hazard=hazard_field_from_config(config),
        fragility=PrototypeLognormalFragility(config),
        disruption=AdjacentCollapseDisruption(config),
        accessibility=NearestHospitalAccessibility(config),
    )


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """One realisation, with enough provenance to reproduce it exactly."""

    scenario_id: str
    seed: int
    realisation: int
    # (n_buildings,) dimensionless intensity proxy -- NOT PGA/MMI.
    intensity: np.ndarray
    damage: DamageResult
    # (n_links,) LinkState
    link_states: np.ndarray
    # (n_links,) probability each link was affected, before sampling.
    link_closure_probabilities: np.ndarray
    baseline_graph: nx.Graph
    disrupted_graph: nx.Graph
    accessibility: AccessibilityComparison | None = None

    @property
    def n_closed(self) -> int:
        return int(sum(s is LinkState.CLOSED for s in self.link_states))

    @property
    def n_degraded(self) -> int:
        return int(sum(s is LinkState.DEGRADED for s in self.link_states))

    @property
    def n_open(self) -> int:
        return int(sum(s is LinkState.OPEN for s in self.link_states))

    def summary(self) -> dict[str, Any]:
        """Compact, JSON-safe description of the realisation."""
        counts = {
            state.name: int((self.damage.states == state.value).sum())
            for state in DamageState
        }
        summary: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "realisation": self.realisation,
            "n_buildings": int(len(self.damage.states)),
            "intensity": {
                "min": float(self.intensity.min()),
                "mean": float(self.intensity.mean()),
                "max": float(self.intensity.max()),
            },
            "damage_state_counts": counts,
            "mean_expected_damage_index": float(self.damage.expected_damage_index.mean()),
            "links": {
                "total": int(len(self.link_states)),
                "open": self.n_open,
                "degraded": self.n_degraded,
                "closed": self.n_closed,
            },
            "network_connected_after": bool(
                nx.is_connected(self.disrupted_graph)
                if self.disrupted_graph.number_of_nodes()
                else False
            ),
            "n_components_after": int(nx.number_connected_components(self.disrupted_graph)),
        }
        if self.accessibility is not None:
            comparison = self.accessibility
            summary["accessibility"] = {
                "baseline_mean_travel_time_min": comparison.baseline.mean_travel_time_min,
                "post_mean_travel_time_min": comparison.post_event.mean_travel_time_min,
                "delta_mean_travel_time_min": comparison.delta_mean_travel_time_min,
                "population_reachable": comparison.post_event.population_reachable,
                "population_unreachable": comparison.post_event.population_unreachable,
                "n_disrupted_routes": comparison.n_disrupted_routes,
                "hospital_utilisation": comparison.post_event.hospital_utilisation,
            }
        return summary


@dataclass(frozen=True, slots=True)
class RealisationRecord:
    """Flat, CSV-friendly metrics for one realisation.

    Deliberately small: a 1,000-realisation run keeps 1,000 of these, and must
    not retain graphs or per-building arrays. Spatial detail is accumulated as
    frequencies instead.
    """

    realisation: int
    intensity_mean: float
    intensity_max: float
    n_none: int
    n_slight: int
    n_moderate: int
    n_severe: int
    n_collapse: int
    mean_expected_damage_index: float
    n_open: int
    n_degraded: int
    n_closed: int
    n_components_after: int
    network_connected_after: bool
    mean_travel_time_min: float
    median_travel_time_min: float
    population_reachable: int
    population_unreachable: int
    n_disrupted_routes: int
    n_newly_unreachable_units: int
    population_newly_unreachable: int
    delta_mean_travel_time_min: float
    delta_median_travel_time_min: float
    hospital_load: dict[str, int] = field(default_factory=dict)
    hospital_utilisation: dict[str, float] = field(default_factory=dict)

    def flat_row(self) -> dict[str, Any]:
        """One CSV row; hospital dicts expanded to ``load_<id>`` columns."""
        row = {k: v for k, v in asdict(self).items() if not isinstance(v, dict)}
        for hospital_id, value in sorted(self.hospital_load.items()):
            row[f"load_{hospital_id}"] = value
        for hospital_id, value in sorted(self.hospital_utilisation.items()):
            row[f"utilisation_{hospital_id}"] = value
        return row


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """A full Monte Carlo run: the distribution, not a single outcome."""

    scenario_id: str
    seed: int
    n_simulations: int
    baseline: AccessibilityResult
    records: tuple[RealisationRecord, ...]
    # hospital_id -> emergency capacity, carried so summaries can report the
    # population/capacity scale without needing the city.
    hospital_emergency_capacity: dict[str, int]
    # Spatial frequencies accumulated across realisations.
    building_collapse_count: np.ndarray
    building_edi_sum: np.ndarray
    link_closed_count: np.ndarray
    link_degraded_count: np.ndarray
    unit_unreachable_count: np.ndarray
    unit_travel_time_sum: np.ndarray
    unit_reachable_count: np.ndarray

    def metric(self, name: str) -> np.ndarray:
        """One metric across all realisations, as a float array."""
        return np.array([getattr(r, name) for r in self.records], dtype=float)

    @property
    def building_collapse_frequency(self) -> np.ndarray:
        return self.building_collapse_count / self.n_simulations

    @property
    def building_mean_edi(self) -> np.ndarray:
        return self.building_edi_sum / self.n_simulations

    @property
    def link_closed_frequency(self) -> np.ndarray:
        return self.link_closed_count / self.n_simulations

    @property
    def link_degraded_frequency(self) -> np.ndarray:
        return self.link_degraded_count / self.n_simulations

    @property
    def unit_unreachable_frequency(self) -> np.ndarray:
        return self.unit_unreachable_count / self.n_simulations

    @property
    def unit_mean_travel_time_min(self) -> np.ndarray:
        """Mean over realisations where the unit was REACHABLE.

        ``nan`` for units never reachable -- deliberately not 0 and not inf,
        so downstream code must decide rather than silently average a
        placeholder.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(
                self.unit_reachable_count > 0,
                self.unit_travel_time_sum / np.maximum(self.unit_reachable_count, 1),
                np.nan,
            )


def run_scenario(
    city: SyntheticCity,
    config: dict[str, Any],
    *,
    realisation: int = 0,
    seed: int | None = None,
    hazard: HazardField | None = None,
    fragility: FragilityModel | None = None,
    disruption: DisruptionModel | None = None,
    accessibility: AccessibilityModel | None = None,
    baseline_graph: nx.Graph | None = None,
    baseline_accessibility: AccessibilityResult | None = None,
) -> ScenarioResult:
    """Run one realisation.

    The model arguments exist so tests -- and later, validated models -- can
    inject alternatives without this function knowing anything about them.
    ``baseline_graph`` and ``baseline_accessibility`` may be passed in to avoid
    recomputing city-level constants across a batch.
    """
    seed = city.seed if seed is None else seed

    hazard = hazard_field_from_config(config) if hazard is None else hazard
    fragility = PrototypeLognormalFragility(config) if fragility is None else fragility
    disruption = AdjacentCollapseDisruption(config) if disruption is None else disruption
    if accessibility is None:
        accessibility = NearestHospitalAccessibility(config)

    intensity = hazard.sample(city.buildings, named_generator(seed, f"hazard:{realisation}"))
    damage = fragility.sample_damage(
        city.buildings, intensity, named_generator(seed, f"damage:{realisation}")
    )
    link_probabilities = disruption.closure_probabilities(city, damage.states)
    link_states = disruption.sample_link_states(
        city, damage.states, named_generator(seed, f"disruption:{realisation}")
    )

    if baseline_graph is None:
        baseline_graph = build_graph(city)
    disrupted_graph = apply_link_states(
        baseline_graph,
        city,
        link_states,
        config["roads"]["disruption"]["degraded_speed_factor"],
    )

    if baseline_accessibility is None:
        baseline_accessibility = accessibility.evaluate(baseline_graph, city)
    post_event = accessibility.evaluate(disrupted_graph, city)
    comparison = compare_accessibility(
        baseline_accessibility,
        post_event,
        city,
        tolerance_min=config["accessibility"]["disrupted_route_tolerance_min"],
    )

    return ScenarioResult(
        scenario_id=config["hazard"]["scenario_id"],
        seed=seed,
        realisation=realisation,
        intensity=intensity,
        damage=damage,
        link_states=link_states,
        link_closure_probabilities=link_probabilities,
        baseline_graph=baseline_graph,
        disrupted_graph=disrupted_graph,
        accessibility=comparison,
    )


def _record(result: ScenarioResult) -> RealisationRecord:
    comparison = result.accessibility
    assert comparison is not None
    post = comparison.post_event
    counts = {
        state: int((result.damage.states == state.value).sum()) for state in DamageState
    }
    return RealisationRecord(
        realisation=result.realisation,
        intensity_mean=float(result.intensity.mean()),
        intensity_max=float(result.intensity.max()),
        n_none=counts[DamageState.NONE],
        n_slight=counts[DamageState.SLIGHT],
        n_moderate=counts[DamageState.MODERATE],
        n_severe=counts[DamageState.SEVERE],
        n_collapse=counts[DamageState.COLLAPSE],
        mean_expected_damage_index=float(result.damage.expected_damage_index.mean()),
        n_open=result.n_open,
        n_degraded=result.n_degraded,
        n_closed=result.n_closed,
        n_components_after=int(nx.number_connected_components(result.disrupted_graph)),
        network_connected_after=bool(nx.is_connected(result.disrupted_graph)),
        mean_travel_time_min=post.mean_travel_time_min,
        median_travel_time_min=post.median_travel_time_min,
        population_reachable=post.population_reachable,
        population_unreachable=post.population_unreachable,
        n_disrupted_routes=comparison.n_disrupted_routes,
        n_newly_unreachable_units=comparison.n_newly_unreachable_units,
        population_newly_unreachable=comparison.population_newly_unreachable,
        delta_mean_travel_time_min=comparison.delta_mean_travel_time_min,
        delta_median_travel_time_min=comparison.delta_median_travel_time_min,
        hospital_load=dict(post.hospital_load),
        hospital_utilisation=dict(post.hospital_utilisation),
    )


def iter_realisations(
    city: SyntheticCity,
    config: dict[str, Any],
    *,
    n_simulations: int,
    seed: int,
    models: ModelSuite | None = None,
    realisation_indices: Sequence[int] | None = None,
) -> Iterator[ScenarioResult]:
    """Yield realisations one at a time, so callers need not hold them all.

    By default the contiguous indices ``0 .. n_simulations-1`` are run. Passing
    ``realisation_indices`` runs exactly those realisation indices instead, in
    the given order -- used to evaluate an out-of-sample subset while preserving
    each index's identity (index *i* always draws the same stream). ``n_simulations``
    is then ignored.
    """
    models = build_models(config) if models is None else models
    baseline_graph = build_graph(city)
    baseline_accessibility = models.accessibility.evaluate(baseline_graph, city)

    indices = range(n_simulations) if realisation_indices is None else realisation_indices
    for index in indices:
        yield run_scenario(
            city,
            config,
            realisation=index,
            seed=seed,
            hazard=models.hazard,
            fragility=models.fragility,
            disruption=models.disruption,
            accessibility=models.accessibility,
            baseline_graph=baseline_graph,
            baseline_accessibility=baseline_accessibility,
        )


def run_monte_carlo(
    city: SyntheticCity,
    config: dict[str, Any],
    *,
    n_simulations: int | None = None,
    seed: int | None = None,
    models: ModelSuite | None = None,
    realisation_indices: Sequence[int] | None = None,
) -> MonteCarloResult:
    """Run ``n_simulations`` realisations and return the distributions.

    Graphs and per-building arrays are discarded as each realisation completes;
    only the flat record and running spatial frequencies are retained.

    ``realisation_indices`` runs exactly those realisation indices instead of the
    contiguous ``0 .. n_simulations-1`` range, in the given order. Spatial
    frequencies and the resulting ``n_simulations`` then reflect the size of that
    subset. This supports an out-of-sample train/evaluation split without
    breaking common random numbers -- each index keeps its own draws.
    """
    seed = city.seed if seed is None else seed
    if realisation_indices is not None:
        realisation_indices = tuple(int(i) for i in realisation_indices)
        if len(realisation_indices) < 1:
            raise ValueError("realisation_indices must be non-empty")
        n_simulations = len(realisation_indices)
    else:
        if n_simulations is None:
            n_simulations = config["monte_carlo"]["n_simulations"]
        if n_simulations < 1:
            raise ValueError("n_simulations must be at least 1")

    models = build_models(config) if models is None else models
    baseline_graph = build_graph(city)
    baseline = models.accessibility.evaluate(baseline_graph, city)

    n_buildings = len(city.buildings)
    n_links = len(city.roads)
    n_units = len(city.population)

    building_collapse_count = np.zeros(n_buildings, dtype=np.int64)
    building_edi_sum = np.zeros(n_buildings, dtype=float)
    link_closed_count = np.zeros(n_links, dtype=np.int64)
    link_degraded_count = np.zeros(n_links, dtype=np.int64)
    unit_unreachable_count = np.zeros(n_units, dtype=np.int64)
    unit_travel_time_sum = np.zeros(n_units, dtype=float)
    unit_reachable_count = np.zeros(n_units, dtype=np.int64)

    records: list[RealisationRecord] = []
    for result in iter_realisations(
        city,
        config,
        n_simulations=n_simulations,
        seed=seed,
        models=models,
        realisation_indices=realisation_indices,
    ):
        records.append(_record(result))

        building_collapse_count += result.damage.collapse_mask()
        building_edi_sum += result.damage.expected_damage_index
        link_closed_count += np.fromiter(
            (s is LinkState.CLOSED for s in result.link_states), dtype=bool, count=n_links
        )
        link_degraded_count += np.fromiter(
            (s is LinkState.DEGRADED for s in result.link_states), dtype=bool, count=n_links
        )

        times = result.accessibility.post_event.travel_time_min
        reachable = np.isfinite(times)
        unit_reachable_count += reachable
        unit_unreachable_count += ~reachable
        unit_travel_time_sum += np.where(reachable, times, 0.0)

    return MonteCarloResult(
        scenario_id=config["hazard"]["scenario_id"],
        seed=seed,
        n_simulations=n_simulations,
        baseline=baseline,
        records=tuple(records),
        hospital_emergency_capacity={
            h.hospital_id: h.emergency_capacity for h in city.hospitals
        },
        building_collapse_count=building_collapse_count,
        building_edi_sum=building_edi_sum,
        link_closed_count=link_closed_count,
        link_degraded_count=link_degraded_count,
        unit_unreachable_count=unit_unreachable_count,
        unit_travel_time_sum=unit_travel_time_sum,
        unit_reachable_count=unit_reachable_count,
    )
