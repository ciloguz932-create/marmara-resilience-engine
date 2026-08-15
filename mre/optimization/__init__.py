"""Prototype intervention optimization — Phase 5.

**NOT a scientifically validated optimization model.** This is a budget-
constrained comparison of a small number of candidate intervention portfolios,
each evaluated by re-running the existing Monte Carlo engine on a modified copy
of the synthetic city. Report it only as "Prototype intervention optimization".

What Phase 5 adds, and nothing more:

    baseline city ──(intervention transforms per-entity parameters)──▶ modified city
    modified city ──run_monte_carlo (SAME seed, SAME realisation streams)──▶ distribution
    paired benefit = baseline_unreachable[i] − portfolio_unreachable[i]   (per realisation)

The design rests on one property of the Phase 3–4 RNG architecture: every
stochastic stage draws a *fixed-size* sample from a stream named only by
``(seed, stage, realisation)`` — never by any parameter value. So changing a
building's fragility or a link's closure probability changes the *threshold* a
draw is compared against, never the *draw itself*. Running two portfolios at the
same seed is therefore automatically a **common-random-numbers / paired
simulation**: realisation *i* of every portfolio sees the identical intensity
field and the identical uniform draws, and the only difference is the modelled
effect of the intervention. This is what makes the benefit distribution a fair
comparison rather than a difference between two unrelated random samples.

Effects act on model PARAMETERS carried by the entities (per-building
vulnerability index, per-link closure probability / susceptibility, per-hospital
emergency capacity), never on results. The baseline city is never mutated;
``apply_to_city`` returns a modified copy.

Known limitations, stated up front:
  - enumerative search over a tiny candidate set — no optimality guarantee;
  - costs are invented unit costs, not engineering estimates;
  - effect multipliers are assumed, not measured;
  - HOSPITAL_SUPPORT does not change the primary objective at all under the
    current accessibility model (capacity does not gate assignment); it moves
    only the secondary service-pressure metric. This is surfaced, not hidden.
  - the ranking is a ranking under the prototype model, not under reality.

The enumeration is deliberately behind a strategy boundary
(``enumerate_feasible_portfolios``) so an MILP / optimizer can replace it later
without touching the evaluation or the domain model.

**Out-of-sample target selection (Phase 5.1).** Data-driven road-hardening
targets are chosen from a SELECTION subset of the realisations
(``selection_fraction``, default 30%), and every reported metric is computed on
the disjoint EVALUATION subset. This removes the in-sample bias of selecting
targets and scoring them on the same realisations. The split is a deterministic
seed-derived permutation (``split_realisations``); common random numbers are
preserved *within* the evaluation set. It is a prototype validation split, not a
statistical validation of the model.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from mre.hospitals import NearestHospitalAccessibility
from mre.models import InterventionType, SyntheticCity
from mre.rng import named_generator
from mre.roads import build_graph
from mre.simulation import MonteCarloResult, run_monte_carlo

__all__ = [
    "PRIMARY_METRIC",
    "Intervention",
    "Portfolio",
    "InterventionOutcome",
    "PortfolioComparison",
    "building_retrofit",
    "road_hardening",
    "hospital_support",
    "build_candidate_interventions",
    "enumerate_feasible_portfolios",
    "evaluate_portfolio",
    "compare_portfolios",
    "split_realisations",
]

# The single, interpretable primary objective: expected unreachable population.
# Lower is better; benefit is the reduction relative to the no-intervention
# baseline. Deliberately NOT a weighted composite score.
PRIMARY_METRIC = "population_unreachable"

# Default share of realisations used only to SELECT data-driven targets. The
# disjoint remainder is the out-of-sample EVALUATION set. Overridable via
# config[interventions][selection_fraction].
SELECTION_FRACTION = 0.30

_PROTOTYPE_EFFECT = "SYNTHETIC PROTOTYPE EFFECT — assumed multiplier, not measured."


# --------------------------------------------------------------------------- #
# Domain model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Intervention:
    """A single prototype action and the entities it applies to.

    ``effect`` holds the fully-resolved numeric parameters of the action, so
    ``apply_to_city`` needs no access to config. Effects are labelled synthetic
    on construction.
    """

    intervention_id: str
    intervention_type: InterventionType
    # building_id / road_id / hospital_id values this action targets.
    target_ids: tuple[str, ...]
    cost: float
    effect: dict[str, float]
    description: str = ""
    assumptions: str = _PROTOTYPE_EFFECT

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError(f"intervention cost must be >= 0, got {self.cost}")
        if not isinstance(self.intervention_type, InterventionType):
            raise TypeError("intervention_type must be an InterventionType")

    def validate_targets(self, city: SyntheticCity) -> None:
        """Raise if any target id is absent from the city.

        An intervention that references an entity the city does not contain is
        ill-formed, not a silent no-op.
        """
        present = _entity_ids(city, self.intervention_type)
        missing = [t for t in self.target_ids if t not in present]
        if missing:
            raise ValueError(
                f"intervention {self.intervention_id!r} targets "
                f"{len(missing)} id(s) absent from the city, e.g. {missing[:3]}"
            )

    def apply_to_city(self, city: SyntheticCity) -> SyntheticCity:
        """Return a modified copy of ``city`` expressing this action's effect.

        The baseline city is never mutated. Only the targeted entities differ;
        every other entity, and every geometry and RNG-relevant count, is
        preserved so that Monte Carlo draws stay paired across portfolios.
        """
        targets = set(self.target_ids)
        if not targets:
            return city

        if self.intervention_type is InterventionType.BUILDING_RETROFIT:
            divisor = self.effect["vulnerability_divisor"]
            floor = self.effect["vulnerability_floor"]
            buildings = tuple(
                dataclasses.replace(
                    b,
                    vulnerability_index=max(floor, b.vulnerability_index / divisor),
                )
                if b.building_id in targets
                else b
                for b in city.buildings
            )
            return dataclasses.replace(city, buildings=buildings)

        if self.intervention_type is InterventionType.ROAD_HARDENING:
            multiplier = self.effect["closure_multiplier"]
            roads = tuple(
                dataclasses.replace(
                    r,
                    closure_probability=r.closure_probability * multiplier,
                    susceptibility=r.susceptibility * multiplier,
                )
                if r.road_id in targets
                else r
                for r in city.roads
            )
            return dataclasses.replace(city, roads=roads)

        if self.intervention_type is InterventionType.HOSPITAL_SUPPORT:
            multiplier = self.effect["capacity_multiplier"]
            hospitals = tuple(
                dataclasses.replace(
                    h,
                    emergency_capacity=max(1, int(round(h.emergency_capacity * multiplier))),
                )
                if h.hospital_id in targets
                else h
                for h in city.hospitals
            )
            return dataclasses.replace(city, hospitals=hospitals)

        raise ValueError(f"unhandled intervention type {self.intervention_type!r}")


@dataclass(frozen=True, slots=True)
class Portfolio:
    """An ordered set of interventions evaluated together under one budget."""

    portfolio_id: str
    interventions: tuple[Intervention, ...] = ()

    @property
    def total_cost(self) -> float:
        return float(sum(i.cost for i in self.interventions))

    @property
    def intervention_types(self) -> tuple[str, ...]:
        return tuple(i.intervention_type.value for i in self.interventions)

    def within_budget(self, budget: float) -> bool:
        # Tiny tolerance so a portfolio that costs exactly the budget is feasible
        # despite float noise.
        return self.total_cost <= budget + 1e-6

    def target_ids(self) -> dict[str, tuple[str, ...]]:
        """Targeted ids grouped by intervention type value."""
        grouped: dict[str, list[str]] = {}
        for intervention in self.interventions:
            grouped.setdefault(intervention.intervention_type.value, []).extend(
                intervention.target_ids
            )
        return {key: tuple(values) for key, values in grouped.items()}

    def validate_targets(self, city: SyntheticCity) -> None:
        for intervention in self.interventions:
            intervention.validate_targets(city)

    def apply_to_city(self, city: SyntheticCity) -> SyntheticCity:
        """Apply every intervention in order. Different-type interventions touch
        disjoint entity collections, so order does not matter between types;
        same-type interventions compose deterministically."""
        for intervention in self.interventions:
            city = intervention.apply_to_city(city)
        return city


@dataclass(frozen=True, slots=True)
class InterventionOutcome:
    """Benefit of one portfolio, reported as a distribution, never a point.

    ``primary_*`` concern the primary objective (expected unreachable
    population). ``secondary`` carries interpretable side metrics reported
    separately — deliberately NOT folded into a single weighted score.
    """

    portfolio_id: str
    intervention_types: tuple[str, ...]
    portfolio_cost: float
    within_budget: bool
    n_simulations: int
    primary_metric: str
    # Mean over realisations of the primary objective, before and after.
    primary_objective_baseline: float
    primary_objective_after: float
    # Paired benefit = baseline[i] - after[i], summarised as a distribution.
    primary_benefit_mean: float
    primary_benefit_std: float
    primary_benefit_p05: float
    primary_benefit_p50: float
    primary_benefit_p95: float
    # Fraction of realisations in which the portfolio strictly improved the
    # primary objective relative to the paired baseline realisation.
    probability_of_improvement: float
    # Mean benefit per unit synthetic cost; nan when cost is zero.
    benefit_per_cost: float
    secondary: dict[str, float] = field(default_factory=dict)
    # Targeted entity ids grouped by intervention-type value, for provenance and
    # spatial output. Empty for the baseline (no interventions).
    target_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PortfolioComparison:
    """A full comparison run: baseline plus ranked portfolio outcomes.

    Reported metrics are computed on the EVALUATION realisations only; the
    disjoint SELECTION realisations are used exclusively to pick data-driven
    targets and never contribute to any reported performance number.
    """

    scenario_id: str
    seed: int
    n_simulations: int
    budget: float
    primary_metric: str
    baseline_outcome: InterventionOutcome
    # Ranked best-first by primary benefit (baseline excluded from the ranking).
    ranked_outcomes: tuple[InterventionOutcome, ...]
    # Prototype validation split (out-of-sample target selection).
    selection_fraction: float = SELECTION_FRACTION
    selection_indices: tuple[int, ...] = ()
    evaluation_indices: tuple[int, ...] = ()

    @property
    def n_selection(self) -> int:
        return len(self.selection_indices)

    @property
    def n_evaluation(self) -> int:
        return len(self.evaluation_indices)

    @property
    def best(self) -> InterventionOutcome | None:
        """The best feasible non-empty portfolio, or None if none was evaluated."""
        return self.ranked_outcomes[0] if self.ranked_outcomes else None

    @property
    def best_improves_on_baseline(self) -> bool:
        best = self.best
        return best is not None and best.primary_benefit_mean > 0.0


# --------------------------------------------------------------------------- #
# Effect / cost helpers and single-type intervention builders
# --------------------------------------------------------------------------- #


def _entity_ids(city: SyntheticCity, kind: InterventionType) -> set[str]:
    if kind is InterventionType.BUILDING_RETROFIT:
        return {b.building_id for b in city.buildings}
    if kind is InterventionType.ROAD_HARDENING:
        return {r.road_id for r in city.roads}
    if kind is InterventionType.HOSPITAL_SUPPORT:
        return {h.hospital_id for h in city.hospitals}
    raise ValueError(f"unhandled intervention type {kind!r}")


def _top_building_ids(city: SyntheticCity, n: int) -> tuple[str, ...]:
    """The ``n`` most vulnerable buildings (ties broken by id for determinism)."""
    ranked = sorted(
        city.buildings, key=lambda b: (-b.vulnerability_index, b.building_id)
    )
    return tuple(b.building_id for b in ranked[: max(0, n)])


def _top_link_ids_by_closure(
    city: SyntheticCity, closure_frequency: np.ndarray, n: int
) -> tuple[str, ...]:
    """The ``n`` links most frequently CLOSED in the baseline (ties by id).

    Data-driven targeting: the links whose failure the baseline Monte Carlo
    actually observes are the ones worth reinforcing. This finds the failure-
    prone links that drive isolation, which pure edge-betweenness centrality
    misses -- the central trunk links are not the ones whose closure strands a
    peripheral population unit.
    """
    order = sorted(
        range(len(city.roads)),
        key=lambda i: (-float(closure_frequency[i]), city.roads[i].road_id),
    )
    return tuple(city.roads[i].road_id for i in order[: max(0, n)])


def _top_hospital_ids(city: SyntheticCity, config: dict[str, Any], n: int) -> tuple[str, ...]:
    """The ``n`` most service-pressured hospitals on the undamaged network.

    Pressure is baseline assigned load / emergency capacity — the hospitals a
    support intervention would most plausibly target.
    """
    if n <= 0:
        return ()
    graph = build_graph(city)
    baseline = NearestHospitalAccessibility(config).evaluate(graph, city)
    pressure = baseline.hospital_utilisation
    ranked = sorted(
        city.hospitals, key=lambda h: (-pressure.get(h.hospital_id, 0.0), h.hospital_id)
    )
    return tuple(h.hospital_id for h in ranked[:n])


def building_retrofit(city: SyntheticCity, config: dict[str, Any]) -> Intervention:
    """Retrofit the most-vulnerable buildings (prototype)."""
    cfg = config["interventions"]
    retro = cfg["building_retrofit"]
    n = int(cfg["retrofit_n_buildings"])
    floor = float(config["synthetic_city"]["buildings"]["vulnerability"]["clip"][0])
    targets = _top_building_ids(city, n)
    return Intervention(
        intervention_id="RETROFIT",
        intervention_type=InterventionType.BUILDING_RETROFIT,
        target_ids=targets,
        cost=float(retro["cost_per_building"]) * len(targets),
        effect={
            "vulnerability_divisor": float(retro["fragility_median_uplift"]),
            "vulnerability_floor": floor,
        },
        description=(
            f"Retrofit the {len(targets)} most vulnerable buildings; uplift "
            f"fragility medians ×{retro['fragility_median_uplift']}."
        ),
    )


def road_hardening(
    city: SyntheticCity, config: dict[str, Any], closure_frequency: np.ndarray
) -> Intervention:
    """Harden the links most frequently closed in the baseline (prototype).

    ``closure_frequency`` is ``MonteCarloResult.link_closed_frequency`` from a
    baseline pre-pass, one value per link in ``city.roads`` order.
    """
    cfg = config["interventions"]
    harden = cfg["road_hardening"]
    n = int(cfg["harden_n_links"])
    targets = _top_link_ids_by_closure(city, closure_frequency, n)
    return Intervention(
        intervention_id="HARDEN",
        intervention_type=InterventionType.ROAD_HARDENING,
        target_ids=targets,
        cost=float(harden["cost_per_link"]) * len(targets),
        effect={"closure_multiplier": float(harden["closure_probability_multiplier"])},
        description=(
            f"Harden the {len(targets)} links most frequently closed in the "
            f"baseline; closure probability and debris susceptibility "
            f"×{harden['closure_probability_multiplier']}."
        ),
    )


def hospital_support(city: SyntheticCity, config: dict[str, Any]) -> Intervention:
    """Support the most service-pressured hospitals (prototype)."""
    cfg = config["interventions"]
    support = cfg["hospital_support"]
    n = int(cfg["support_n_hospitals"])
    targets = _top_hospital_ids(city, config, n)
    return Intervention(
        intervention_id="SUPPORT",
        intervention_type=InterventionType.HOSPITAL_SUPPORT,
        target_ids=targets,
        cost=float(support["cost_per_hospital"]) * len(targets),
        effect={"capacity_multiplier": float(support["emergency_capacity_multiplier"])},
        description=(
            f"Support the {len(targets)} most service-pressured hospitals; "
            f"emergency capacity ×{support['emergency_capacity_multiplier']}. "
            "Affects the service-pressure metric only."
        ),
    )


def build_candidate_interventions(
    city: SyntheticCity,
    config: dict[str, Any],
    selection_result: MonteCarloResult,
) -> tuple[Intervention, ...]:
    """The three single-type prototype interventions used as the candidate set.

    ``selection_result`` is a Monte Carlo run over the SELECTION realisations
    only; its ``link_closed_frequency`` drives data-driven road-hardening
    targeting, kept out-of-sample from the evaluation set. Interventions with no
    targets (e.g. a configured count of 0) are dropped, so an empty candidate set
    is possible and handled by the enumerator.
    """
    candidates = [
        building_retrofit(city, config),
        road_hardening(city, config, selection_result.link_closed_frequency),
        hospital_support(city, config),
    ]
    return tuple(c for c in candidates if c.target_ids)


# --------------------------------------------------------------------------- #
# Train / evaluation split (out-of-sample target selection)
# --------------------------------------------------------------------------- #


def split_realisations(
    seed: int, n_simulations: int, selection_fraction: float = SELECTION_FRACTION
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Deterministically partition ``0 .. n_simulations-1`` into (selection, evaluation).

    The selection set is used only to choose data-driven targets; the disjoint
    evaluation set carries every reported metric, so target selection cannot
    inflate the reported performance. The partition is a seed-derived
    permutation, so it is reproducible, seed-dependent, and non-overlapping.
    Both sides are returned sorted and are guaranteed non-empty.

    This is a PROTOTYPE validation split, not a statistical validation of the
    model: it removes in-sample target-selection bias within the synthetic slice
    and says nothing about real-world validity.
    """
    if n_simulations < 2:
        raise ValueError("need at least 2 realisations to split")
    if not 0.0 < selection_fraction < 1.0:
        raise ValueError("selection_fraction must be in (0, 1)")

    n_selection = int(round(n_simulations * selection_fraction))
    n_selection = min(max(n_selection, 1), n_simulations - 1)

    order = named_generator(seed, "intervention.split").permutation(n_simulations)
    selection = tuple(sorted(int(i) for i in order[:n_selection]))
    evaluation = tuple(sorted(int(i) for i in order[n_selection:]))
    return selection, evaluation


# --------------------------------------------------------------------------- #
# Enumeration (the strategy boundary a future optimizer replaces)
# --------------------------------------------------------------------------- #


def enumerate_feasible_portfolios(
    candidates: tuple[Intervention, ...], budget: float
) -> tuple[Portfolio, ...]:
    """Every non-empty subset of ``candidates`` within budget, as portfolios.

    Deterministic exhaustive enumeration over the power set. For the synthetic
    slice the candidate set is tiny (three single-type interventions), so this
    is exact and cheap. Replace this one function with an MILP / heuristic
    optimizer if the candidate set ever grows combinatorially — the evaluation
    and domain model do not change.
    """
    portfolios: list[Portfolio] = []
    for size in range(1, len(candidates) + 1):
        for combo in itertools.combinations(candidates, size):
            portfolio = Portfolio(
                portfolio_id="+".join(i.intervention_id for i in combo),
                interventions=combo,
            )
            if portfolio.within_budget(budget):
                portfolios.append(portfolio)
    return tuple(portfolios)


# --------------------------------------------------------------------------- #
# Evaluation under common random numbers
# --------------------------------------------------------------------------- #


def _objective(result: MonteCarloResult) -> np.ndarray:
    """Per-realisation primary objective values (unreachable population)."""
    return result.metric(PRIMARY_METRIC)


def _service_pressure_mean(result: MonteCarloResult) -> float:
    """Mean over realisations of assigned-population / total emergency capacity.

    A clearly-named PROTOTYPE service-pressure indicator, not a queueing result.
    It carries the same documented scale artefact as ``hospital_utilisation``.
    Hospital support raises total capacity, so it lowers this index; road and
    building interventions can only raise reachable population, so they can only
    raise it.
    """
    total_capacity = sum(result.hospital_emergency_capacity.values())
    if total_capacity <= 0:
        return math.nan
    reachable = result.metric("population_reachable")
    return float(np.mean(reachable / total_capacity))


def _secondary_metrics(
    result: MonteCarloResult, baseline: MonteCarloResult
) -> dict[str, float]:
    """Interpretable side metrics, each reported on its own (no composite)."""

    def mean(name: str) -> float:
        return float(np.mean(result.metric(name)))

    def base_mean(name: str) -> float:
        return float(np.mean(baseline.metric(name)))

    pressure_after = _service_pressure_mean(result)
    pressure_baseline = _service_pressure_mean(baseline)
    return {
        "mean_travel_time_after": mean("mean_travel_time_min"),
        "delta_mean_travel_time": mean("mean_travel_time_min")
        - base_mean("mean_travel_time_min"),
        "median_travel_time_after": mean("median_travel_time_min"),
        "n_closed_after": mean("n_closed"),
        "delta_n_closed": mean("n_closed") - base_mean("n_closed"),
        "n_collapse_after": mean("n_collapse"),
        "delta_n_collapse": mean("n_collapse") - base_mean("n_collapse"),
        "mean_expected_damage_index_after": mean("mean_expected_damage_index"),
        "service_pressure_baseline": pressure_baseline,
        "service_pressure_after": pressure_after,
        "delta_service_pressure": pressure_after - pressure_baseline,
    }


def evaluate_portfolio(
    city: SyntheticCity,
    config: dict[str, Any],
    portfolio: Portfolio,
    *,
    seed: int,
    realisation_indices: Sequence[int],
    baseline_result: MonteCarloResult,
) -> InterventionOutcome:
    """Evaluate one portfolio under common random numbers against ``baseline_result``.

    ``baseline_result`` must have been produced by ``run_monte_carlo`` on the
    unmodified ``city`` over the same ``realisation_indices`` (the evaluation
    set), in the same order — position *k* is then paired between baseline and
    portfolio because both used the same realisation index there.
    """
    portfolio.validate_targets(city)

    realisation_indices = tuple(int(i) for i in realisation_indices)
    budget = float(config["interventions"]["budget"])
    baseline_objective = _objective(baseline_result)

    if portfolio.interventions:
        modified = portfolio.apply_to_city(city)
        result = run_monte_carlo(
            modified, config, realisation_indices=realisation_indices, seed=seed
        )
    else:
        # Empty portfolio is the baseline itself; reuse it rather than re-running.
        result = baseline_result

    objective = _objective(result)
    benefit = baseline_objective - objective  # >0 means fewer unreachable

    cost = portfolio.total_cost
    benefit_mean = float(np.mean(benefit))
    return InterventionOutcome(
        portfolio_id=portfolio.portfolio_id,
        intervention_types=portfolio.intervention_types,
        portfolio_cost=cost,
        within_budget=portfolio.within_budget(budget),
        n_simulations=len(realisation_indices),
        primary_metric=PRIMARY_METRIC,
        primary_objective_baseline=float(np.mean(baseline_objective)),
        primary_objective_after=float(np.mean(objective)),
        primary_benefit_mean=benefit_mean,
        primary_benefit_std=float(np.std(benefit, ddof=0)),
        primary_benefit_p05=float(np.percentile(benefit, 5)),
        primary_benefit_p50=float(np.percentile(benefit, 50)),
        primary_benefit_p95=float(np.percentile(benefit, 95)),
        probability_of_improvement=float(np.mean(benefit > 0.0)),
        benefit_per_cost=(benefit_mean / cost) if cost > 0 else math.nan,
        secondary=_secondary_metrics(result, baseline_result),
        target_ids=portfolio.target_ids(),
    )


def _rank_key(outcome: InterventionOutcome) -> tuple[float, float, float]:
    """Best-first: most benefit, then most benefit-per-cost, then cheapest."""
    per_cost = outcome.benefit_per_cost
    per_cost = per_cost if math.isfinite(per_cost) else -math.inf
    return (-outcome.primary_benefit_mean, -per_cost, outcome.portfolio_cost)


def compare_portfolios(
    city: SyntheticCity,
    config: dict[str, Any],
    portfolios: tuple[Portfolio, ...] | None = None,
    *,
    seed: int | None = None,
    n_simulations: int | None = None,
) -> PortfolioComparison:
    """Evaluate the baseline and every feasible portfolio, ranked by benefit.

    The ``n_simulations`` realisations are split deterministically into a
    SELECTION set (``selection_fraction``) and a disjoint EVALUATION set. Only
    the selection set is used to pick data-driven targets (road hardening); every
    reported metric is computed on the evaluation set, so target selection cannot
    inflate the reported performance. Inside the evaluation set, the baseline and
    every portfolio share the same realisation indices, so comparisons remain
    paired (common random numbers). The baseline is evaluated once and reused.

    With ``portfolios=None`` the candidate set is built from config and every
    feasible non-empty subset is enumerated.
    """
    seed = int(config["random"]["seed"]) if seed is None else int(seed)
    if n_simulations is None:
        n_simulations = int(config["monte_carlo"]["n_simulations"])
    budget = float(config["interventions"]["budget"])
    selection_fraction = float(
        config["interventions"].get("selection_fraction", SELECTION_FRACTION)
    )

    selection_indices, evaluation_indices = split_realisations(
        seed, n_simulations, selection_fraction
    )

    # SELECTION set: used ONLY to choose data-driven targets (road hardening).
    selection_result = run_monte_carlo(
        city, config, realisation_indices=selection_indices, seed=seed
    )

    # EVALUATION set: every reported metric is computed here, out-of-sample.
    baseline_result = run_monte_carlo(
        city, config, realisation_indices=evaluation_indices, seed=seed
    )
    baseline_portfolio = Portfolio(portfolio_id="BASELINE", interventions=())
    baseline_outcome = evaluate_portfolio(
        city,
        config,
        baseline_portfolio,
        seed=seed,
        realisation_indices=evaluation_indices,
        baseline_result=baseline_result,
    )

    if portfolios is None:
        candidates = build_candidate_interventions(city, config, selection_result)
        portfolios = enumerate_feasible_portfolios(candidates, budget)
    else:
        portfolios = tuple(p for p in portfolios if p.within_budget(budget))

    outcomes = [
        evaluate_portfolio(
            city,
            config,
            portfolio,
            seed=seed,
            realisation_indices=evaluation_indices,
            baseline_result=baseline_result,
        )
        for portfolio in portfolios
    ]
    ranked = tuple(sorted(outcomes, key=_rank_key))

    return PortfolioComparison(
        scenario_id=config["hazard"]["scenario_id"],
        seed=seed,
        n_simulations=n_simulations,
        budget=budget,
        primary_metric=PRIMARY_METRIC,
        baseline_outcome=baseline_outcome,
        ranked_outcomes=ranked,
        selection_fraction=selection_fraction,
        selection_indices=selection_indices,
        evaluation_indices=evaluation_indices,
    )
