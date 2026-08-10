"""Probabilistic building damage — PROTOTYPE, NOT A VALIDATED FRAGILITY MODEL.

Continuous lognormal fragility. For each damage state *ds* the probability of
reaching or exceeding it, given intensity proxy ``I``:

    P(D >= ds | I) = Phi( ln( I / (median_ds / v) ) / beta )

where ``v`` is the building's vulnerability index (higher = more fragile, hence
the median is DIVIDED by it) and ``Phi`` is the standard normal CDF. State
probabilities are consecutive differences of the exceedance curves:

    P(NONE)     = 1 - P(D >= SLIGHT)
    P(SLIGHT)   = P(D >= SLIGHT)   - P(D >= MODERATE)
    P(MODERATE) = P(D >= MODERATE) - P(D >= SEVERE)
    P(SEVERE)   = P(D >= SEVERE)   - P(D >= COLLAPSE)
    P(COLLAPSE) = P(D >= COLLAPSE)

This requires the medians to increase with severity; ``tests/test_config.py``
enforces that, since otherwise the differences go negative.

There is no threshold rule anywhere: no "intensity > X means collapse". Damage
is sampled from the full distribution, and the distribution itself is retained
alongside the sample.

**The medians and beta in config/default.toml are INVENTED.** They are not
derived from Turkish building stock, not from TBDY/TDY, not from HAZUS, not
from any empirical damage database. The lognormal *shape* is conventional and
defensible; uncalibrated *parameters* make the output non-quantitative. Do not
state that any real building will reach any damage state.

Replaceable: implement ``FragilityModel`` with validated curves; the Monte
Carlo driver is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy.stats import norm

from mre.models import Building, DamageState

__all__ = [
    "FragilityModel",
    "PrototypeLognormalFragility",
    "DamageResult",
    "EXCEEDANCE_STATES",
]

# Damage states that have an exceedance curve; NONE is the complement.
EXCEEDANCE_STATES = ("SLIGHT", "MODERATE", "SEVERE", "COLLAPSE")

_N_STATES = len(DamageState)
_STATE_VALUES = np.arange(_N_STATES, dtype=float)
_MAX_STATE_VALUE = float(_N_STATES - 1)


@dataclass(frozen=True, slots=True)
class DamageResult:
    """The full distribution AND the sampled outcome.

    Keeping the probability vector is the point: a sampled state alone throws
    away everything the model actually knows.
    """

    # (n_buildings, 5), rows sum to 1, columns ordered by DamageState.value
    probabilities: np.ndarray
    # (n_buildings,) integer DamageState.value
    states: np.ndarray
    # (n_buildings,) expected damage index in [0, 1]
    expected_damage_index: np.ndarray

    def state_enums(self) -> list[DamageState]:
        return [DamageState(int(v)) for v in self.states]

    def collapse_mask(self) -> np.ndarray:
        return self.states == DamageState.COLLAPSE.value


class FragilityModel(Protocol):
    def damage_probabilities(
        self, buildings: tuple[Building, ...], intensity: np.ndarray
    ) -> np.ndarray: ...

    def sample_damage(
        self,
        buildings: tuple[Building, ...],
        intensity: np.ndarray,
        rng: np.random.Generator,
    ) -> DamageResult: ...


class PrototypeLognormalFragility:
    """Lognormal fragility with per-building vulnerability scaling."""

    def __init__(self, config: dict[str, Any]) -> None:
        damage_cfg = config["damage"]
        self.medians = np.array(
            [damage_cfg["median_intensity"][s] for s in EXCEEDANCE_STATES], dtype=float
        )
        self.beta = float(damage_cfg["beta_ln"])
        if self.beta <= 0:
            raise ValueError("beta_ln must be positive")
        if np.any(np.diff(self.medians) <= 0):
            raise ValueError(
                "fragility medians must strictly increase with severity; "
                "otherwise state probabilities go negative"
            )

    def exceedance_probabilities(
        self, buildings: tuple[Building, ...], intensity: np.ndarray
    ) -> np.ndarray:
        """``P(D >= ds | I)`` for the four non-NONE states. Shape (n, 4)."""
        intensity = np.asarray(intensity, dtype=float)
        if intensity.shape != (len(buildings),):
            raise ValueError(
                f"intensity shape {intensity.shape} does not match "
                f"{len(buildings)} buildings"
            )

        vulnerability = np.array([b.vulnerability_index for b in buildings], dtype=float)
        # Higher vulnerability lowers the effective median => fragile sooner.
        effective_medians = self.medians[None, :] / vulnerability[:, None]

        # I == 0 means no shaking and therefore no exceedance. Evaluating
        # log(0) would emit a warning, so mask rather than compute it.
        positive = intensity > 0.0
        exceedance = np.zeros((len(buildings), len(EXCEEDANCE_STATES)), dtype=float)
        if np.any(positive):
            ratio = intensity[positive, None] / effective_medians[positive, :]
            exceedance[positive, :] = norm.cdf(np.log(ratio) / self.beta)
        return exceedance

    def damage_probabilities(
        self, buildings: tuple[Building, ...], intensity: np.ndarray
    ) -> np.ndarray:
        """Shape (n, 5); rows sum to 1, columns ordered by ``DamageState.value``."""
        exceedance = self.exceedance_probabilities(buildings, intensity)

        probabilities = np.empty((len(buildings), _N_STATES), dtype=float)
        probabilities[:, 0] = 1.0 - exceedance[:, 0]
        probabilities[:, 1:_N_STATES - 1] = exceedance[:, :-1] - exceedance[:, 1:]
        probabilities[:, -1] = exceedance[:, -1]

        # Guard against float noise producing a tiny negative, then renormalise
        # so rows sum to exactly 1 within tolerance.
        np.clip(probabilities, 0.0, 1.0, out=probabilities)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    @staticmethod
    def expected_damage_index(probabilities: np.ndarray) -> np.ndarray:
        """Probability-weighted mean damage state, normalised to [0, 1].

        ``EDI = sum_k P(k) * k / 4``. A continuous severity summary; it is NOT
        a loss ratio and does not convert to cost or casualties.
        """
        return (probabilities * _STATE_VALUES).sum(axis=1) / _MAX_STATE_VALUE

    def sample_damage(
        self,
        buildings: tuple[Building, ...],
        intensity: np.ndarray,
        rng: np.random.Generator,
    ) -> DamageResult:
        """Inverse-CDF sample per building, retaining the distribution."""
        probabilities = self.damage_probabilities(buildings, intensity)
        cumulative = np.cumsum(probabilities, axis=1)
        draws = rng.random(len(buildings))[:, None]
        states = np.clip((draws > cumulative).sum(axis=1), 0, _N_STATES - 1)

        return DamageResult(
            probabilities=probabilities,
            states=states.astype(int),
            expected_damage_index=self.expected_damage_index(probabilities),
        )
