"""Real-data pilot layer — Phase 6.

**This package is the documented replacement of the synthetic ``CityBuilder``
seam with real, licensed open data.** It exists so the engine can run on a real
İstanbul pilot area (Zeytinburnu) without inventing anything.

Scientific-integrity contract for everything under ``mre.real``:

- **Never fabricate a value that is not in the source data.** Missing
  attributes are recorded as ``UNKNOWN`` (see ``UNKNOWN``), never filled with a
  plausible guess presented as real.
- **Real geometry, prototype physics.** OSM supplies real building footprints,
  the real road network and its topology, and real hospital locations. It does
  **not** supply structural attributes (construction year, structural system,
  and — for most buildings — storey count). The fragility/vulnerability
  parameters therefore remain the **PROTOTYPE** parameters of the synthetic
  engine, applied to real footprints. The resulting damage layer is
  *"prototype fragility on real geometry"* and is **not** a real damage
  estimate. This is stated in every output.
- **No real population data.** Open data does not give a licensed population
  distribution for the pilot, so demand is a **uniform proxy grid** clearly
  labelled ``population_source = "UNIFORM_PROXY"`` — accessibility results are
  about geographic/network access, not real population impact.
- **Intensity stays a proxy.** The scenario is sourced (Main Marmara Fault /
  Princes' Islands segment) but the intensity field is the same dimensionless
  ``scenario-derived intensity proxy`` as the synthetic engine — never
  relabelled PGA/PGV/SA(T)/MMI.

See ``docs/SCIENTIFIC_ASSUMPTIONS.md`` (§10 Real-data pilot) and the
``provenance.json`` written next to every real-pilot output.
"""

from __future__ import annotations

from enum import Enum

# Sentinel written into any attribute that is genuinely absent from the source.
# The rule: an attribute is either a real value from the source, or this exact
# string. There is no third, invented option.
UNKNOWN = "UNKNOWN"


class CityMode(str, Enum):
    """Which world the engine is running on.

    ``SYNTHETIC`` is the invented Phase 0-5 city. ``REAL_PILOT`` is the real
    Zeytinburnu open-data pilot. The distinction is carried into every output
    so a reader can never mistake one for the other.
    """

    SYNTHETIC = "SYNTHETIC"
    REAL_PILOT = "REAL_PILOT"


__all__ = ["UNKNOWN", "CityMode"]
