"""The real-pilot hazard scenario: real fault geometry, prototype proxy field.

The fault SYSTEM and segment are **real and cited** (Main Marmara Fault,
Princes' Islands segment — see ``mre.real.pilot.scenario_provenance``). The
source line below is placed at the segment's **real approximate location** in
EPSG:32635, ~18-19 km south of the Zeytinburnu pilot.

What is **not** real, and is labelled so everywhere: the intensity *field*. It
is the same dimensionless ``scenario-derived intensity proxy`` as the synthetic
engine. Its magnitude (``intensity_at_source``) and decay (``decay_per_km``) are
**PROTOTYPE tuning parameters**, exactly as ``docs/SCIENTIFIC_ASSUMPTIONS.md``
already documents them to be — chosen so the pilot spans a meaningful proxy
gradient, **NOT** a calibrated ground-motion prediction equation and **NOT** a
real PGA/PGV/SA(T)/MMI estimate at the real fault distance. Nothing here is a
prediction of shaking at any real address.
"""

from __future__ import annotations

import copy
from typing import Any

from mre.config import _deep_merge

__all__ = ["REAL_PILOT_HAZARD", "real_pilot_config"]

# Princes' Islands segment approximate endpoints in EPSG:32635 (converted from
# published ~40.86-40.88 N / 28.95-29.15 E). REAL fault location, ~18.5 km from
# the pilot centroid.
_PIS_SEGMENT_WEST = [664349.0, 4525046.0]
_PIS_SEGMENT_EAST = [681152.0, 4527661.0]

REAL_PILOT_HAZARD: dict[str, Any] = {
    "scenario_id": "MMF-PIS-M74",
    "magnitude": 7.4,
    # Line source along the real segment; distance is to the nearest point on it.
    "model": "line_source_decay",
    "epicenter": [672750.0, 4526353.0],  # segment midpoint, for the point model
    "source_line": [_PIS_SEGMENT_WEST, _PIS_SEGMENT_EAST],
    "near_source_saturation_km": 3.0,
    "max_intensity": 2.0,
    # PROXY TUNING, not calibrated attenuation. Lower decay + higher source
    # value than the synthetic default so that the proxy field is non-negligible
    # at the pilot's real ~18.5 km distance and spans a gradient. The absolute
    # values carry NO ground-motion meaning. See module docstring and
    # docs/SCIENTIFIC_ASSUMPTIONS.md §2 (decay is a documented tuning choice).
    "intensity_at_source": 0.70,
    "decay_per_km": 0.055,
    "intensity_sigma_ln": 0.25,
}


def real_pilot_config(base_config: dict[str, Any]) -> dict[str, Any]:
    """A copy of ``base_config`` with the real-pilot hazard scenario overlaid.

    Only the ``[hazard]`` block is replaced; every other prototype parameter
    (fragility medians, disruption, accessibility, interventions) is inherited
    unchanged from the synthetic engine and remains a PROTOTYPE parameter applied
    to real geometry.
    """
    return _deep_merge(copy.deepcopy(base_config), {"hazard": REAL_PILOT_HAZARD})
