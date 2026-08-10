"""Ground-shaking input — PROTOTYPE, NOT A VALIDATED HAZARD MODEL.

MRE-001 uses a distance-decay intensity field with lognormal variability. The
output is a **dimensionless intensity proxy**, deliberately NOT labelled PGA,
PGV, SA(T), or MMI, because it is not calibrated to any of them. Relabelling it
as one of those would manufacture false precision.

Formulation, for a site at distance ``d`` from the source:

    d_eff = sqrt(d_km^2 + h^2)                      near-source saturation
    I_med = I0 * exp(-decay_per_km * d_eff)         median field
    I     = clip(I_med * exp(N(0, sigma_ln)), 0, I_max)

``h`` (``near_source_saturation_km``) prevents a singularity at zero distance,
playing the role a finite-fault depth term plays in a real GMPE. Magnitude is
descriptive metadata only and does not enter the calculation.

Replaceable: implement ``HazardField`` with OpenQuake, a real GMPE, or an
imported ShakeMap raster, and nothing downstream changes.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from mre.models import Building, EarthquakeScenario

__all__ = [
    "HazardField",
    "PointSourceDecayField",
    "LineSourceDecayField",
    "hazard_field_from_config",
]


def _building_coordinates(buildings: tuple[Building, ...]) -> np.ndarray:
    return np.array([(b.easting, b.northing) for b in buildings], dtype=float)


class HazardField(Protocol):
    """Maps building locations to a shaking-intensity proxy."""

    def median_intensity(self, buildings: tuple[Building, ...]) -> np.ndarray:
        """The deterministic median field, with no random component."""
        ...

    def sample(
        self, buildings: tuple[Building, ...], rng: np.random.Generator
    ) -> np.ndarray:
        """One stochastic realisation of intensity, one value per building."""
        ...


class _DecayField:
    """Shared decay maths for the point- and line-source prototypes."""

    def __init__(self, scenario: EarthquakeScenario) -> None:
        self.scenario = scenario

    def _distance_km(self, coordinates: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def median_intensity(self, buildings: tuple[Building, ...]) -> np.ndarray:
        scenario = self.scenario
        distance_km = self._distance_km(_building_coordinates(buildings))
        effective_km = np.hypot(distance_km, scenario.near_source_saturation_km)
        median = scenario.intensity_at_source * np.exp(-scenario.decay_per_km * effective_km)
        return np.clip(median, 0.0, scenario.max_intensity)

    def sample(
        self, buildings: tuple[Building, ...], rng: np.random.Generator
    ) -> np.ndarray:
        """Median field times an independent lognormal residual per site.

        Residuals are INDEPENDENT, not spatially correlated. Real ground-motion
        residuals are correlated over hundreds of metres to kilometres, so this
        UNDERSTATES the probability of large clustered damage.
        """
        scenario = self.scenario
        median = self.median_intensity(buildings)
        residual = rng.lognormal(mean=0.0, sigma=scenario.intensity_sigma_ln, size=len(buildings))
        return np.clip(median * residual, 0.0, scenario.max_intensity)


class PointSourceDecayField(_DecayField):
    """Radial decay from a configurable epicenter. The MRE-001 default."""

    def _distance_km(self, coordinates: np.ndarray) -> np.ndarray:
        epicenter_x, epicenter_y = self.scenario.epicenter
        return (
            np.hypot(coordinates[:, 0] - epicenter_x, coordinates[:, 1] - epicenter_y) / 1000.0
        )


class LineSourceDecayField(_DecayField):
    """Decay from the nearest point on a finite line source.

    A crude stand-in for rupture extent: distance is measured to the segment,
    not to a single point.
    """

    def _distance_km(self, coordinates: np.ndarray) -> np.ndarray:
        (x1, y1), (x2, y2) = self.scenario.source_line
        start = np.array([x1, y1], dtype=float)
        end = np.array([x2, y2], dtype=float)

        segment = end - start
        length_squared = float(segment @ segment)
        if length_squared == 0.0:
            return np.hypot(coordinates[:, 0] - x1, coordinates[:, 1] - y1) / 1000.0

        offsets = coordinates - start
        t = np.clip((offsets @ segment) / length_squared, 0.0, 1.0)
        nearest = start + t[:, None] * segment
        deltas = coordinates - nearest
        return np.hypot(deltas[:, 0], deltas[:, 1]) / 1000.0


_MODELS = {
    "point_source_decay": PointSourceDecayField,
    "line_source_decay": LineSourceDecayField,
}


def hazard_field_from_config(config: dict) -> HazardField:
    """Build the hazard field named by ``config['hazard']['model']``."""
    name = config["hazard"]["model"]
    try:
        model = _MODELS[name]
    except KeyError:
        raise ValueError(
            f"unknown hazard model {name!r}; available: {sorted(_MODELS)}"
        ) from None
    return model(EarthquakeScenario.from_config(config))
