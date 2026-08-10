"""Prototype intervention optimization.

NOT a scientifically validated optimization model. It is a budget-constrained
comparison of a small number of candidate intervention portfolios, evaluated by
re-running the Monte Carlo engine with modified parameters. Report it only as
"Prototype intervention optimization".

Known limitations, stated up front:
  - greedy / enumerative search, no optimality guarantee
  - interaction effects between interventions are captured only insofar as the
    simulation captures them
  - costs are invented unit costs, not engineering estimates
  - the ranking is a ranking under the prototype model, not under reality

Implemented in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mre.models import InterventionType, SyntheticCity

__all__ = ["Intervention", "Portfolio", "InterventionOutcome", "compare_portfolios"]


@dataclass(frozen=True, slots=True)
class Intervention:
    """A single action, and the entities it applies to."""

    intervention_type: InterventionType
    # building_id / road_id / hospital_id values this action targets.
    target_ids: tuple[str, ...]
    cost: float

    def apply(self, config: dict[str, Any]) -> dict[str, Any]:
        """Return a modified config expressing this action's modelled effect.

        Effects act on model PARAMETERS (fragility medians, closure
        probabilities, emergency capacity), not on results directly.
        """
        raise NotImplementedError("Phase 5")


@dataclass(frozen=True, slots=True)
class Portfolio:
    label: str
    interventions: tuple[Intervention, ...]

    @property
    def total_cost(self) -> float:
        return sum(i.cost for i in self.interventions)

    def within_budget(self, budget: float) -> bool:
        return self.total_cost <= budget


@dataclass(frozen=True, slots=True)
class InterventionOutcome:
    """Benefit of a portfolio, reported as a distribution summary rather than a
    point estimate."""

    portfolio_label: str
    total_cost: float
    # Mean change vs. the no-intervention case; negative = faster.
    delta_mean_travel_time_min: float
    delta_population_unreachable: float
    # 5th-95th percentile band of the travel-time change across realisations.
    delta_travel_time_p05_p95: tuple[float, float]
    benefit_per_cost: float


def compare_portfolios(
    city: SyntheticCity,
    config: dict[str, Any],
    portfolios: tuple[Portfolio, ...],
) -> tuple[InterventionOutcome, ...]:
    """Evaluate each affordable portfolio against the no-intervention baseline.

    Every portfolio is evaluated with the SAME seed so that differences reflect
    the interventions rather than sampling noise.
    """
    raise NotImplementedError("Phase 5")
