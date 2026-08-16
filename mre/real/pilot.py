"""The Zeytinburnu real-data pilot: area definition, sourced scenario, provenance.

**Why Zeytinburnu.** It is the canonical published İstanbul earthquake-risk
pilot district: the Japan International Cooperation Agency (JICA) and the
İstanbul Metropolitan Municipality studied it as the pilot application of the
*Earthquake Master Plan for İstanbul* (2002), and Boğaziçi University/KOERI
published a seismic risk assessment of its building stock as an explicit pilot.
It sits on soft soil next to the Marmara Sea and directly faces the Main Marmara
Fault's Princes' Islands segment — the ~250-year seismic-gap segment for which a
M≈7.4 earthquake is considered overdue. That gives the scenario real, citable
provenance, which is the point of the pilot.

Nothing in the coordinates below is a claim about a specific building. The bbox
is a study-area extent; the scenario is a **published** rupture context, not a
prediction of when or exactly how a rupture will occur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "PilotArea",
    "ZEYTINBURNU",
    "scenario_provenance",
    "integrity_notes",
    "build_pilot_provenance",
]


@dataclass(frozen=True, slots=True)
class PilotArea:
    name: str
    study_area: str
    # south, west, north, east (WGS84)
    bbox: tuple[float, float, float, float]
    target_crs: str
    justification: str


# ~2.6 km2 core of Zeytinburnu. Measured OSM content at build time: several
# thousand real building footprints, several hundred road ways, and multiple
# real hospitals (needed for a meaningful nearest-hospital accessibility
# comparison). Dense real urban fabric — which itself demonstrates the density
# gap the synthetic slice documents. Within the 1-5 km2 pilot scope.
ZEYTINBURNU = PilotArea(
    name="zeytinburnu_pilot",
    study_area="Zeytinburnu district core, İstanbul, Türkiye",
    bbox=(40.9890, 28.8970, 41.0050, 28.9170),
    target_crs="EPSG:32635",
    justification=(
        "Canonical published İstanbul earthquake-risk pilot district "
        "(JICA–İMM Earthquake Master Plan for İstanbul, 2002; Boğaziçi/KOERI "
        "seismic risk assessment pilot). Soft soil beside the Marmara Sea, "
        "directly facing the Main Marmara Fault Princes' Islands segment. Dense, "
        "well-mapped OSM coverage at a size (~1.4 km²) that stays tractable."
    ),
)


def scenario_provenance() -> dict[str, Any]:
    """Published context for the pilot scenario. Citations, not a prediction.

    The intensity field driven from this scenario is a **dimensionless
    scenario-derived intensity proxy** — deliberately NOT PGA/PGV/SA(T)/MMI, and
    NOT calibrated to any of them. The magnitude and fault context are sourced;
    the intensity *values* remain a prototype proxy.
    """
    return {
        "scenario_id": "MMF-PIS-M74",
        "name": "Main Marmara Fault — Princes' Islands segment scenario",
        "magnitude_context": "Mw ~7.4 (published expectation for the segment)",
        "fault_system": "North Anatolian Fault / Main Marmara Fault, Princes' Islands segment",
        "intensity_field": "scenario-derived dimensionless intensity proxy (NOT PGA/PGV/SA(T)/MMI)",
        "why_this_segment": (
            "The Princes' Islands segment south of İstanbul is the seismic-gap "
            "segment that has not ruptured in ~250 years and is considered "
            "capable of a M~7.4 event."
        ),
        "sources": [
            {
                "citation": "JICA & İstanbul Metropolitan Municipality (2002), "
                "The Study on a Disaster Prevention/Mitigation Basic Plan in "
                "İstanbul including Seismic Microzonation (Earthquake Master "
                "Plan for İstanbul); Zeytinburnu pilot.",
                "type": "government/published study",
            },
            {
                "citation": "Boğaziçi University/KOERI, Seismic Risk Assessment "
                "of Existing Building Stock in İstanbul — A Pilot Application in "
                "Zeytinburnu District.",
                "type": "peer-reviewed / institutional",
            },
            {
                "citation": "Recent literature on eastward rupture progression "
                "of the Main Marmara Fault toward İstanbul and the Princes' "
                "Islands seismic gap (e.g. Science, 2025).",
                "type": "peer-reviewed",
            },
        ],
        "disclaimer": (
            "RESEARCH PROTOTYPE — NOT A REAL-WORLD PREDICTION. The fault context "
            "and magnitude are published; the intensity field is a prototype "
            "proxy and the fragility parameters are prototype values applied to "
            "real geometry. This is not an operational İstanbul earthquake "
            "prediction or a real damage/loss estimate."
        ),
    }


def build_pilot_provenance(
    n_buildings: int, n_roads: int, n_hospitals: int, acquired_at: str
) -> "Any":
    """Assemble the ProvenanceRecord for a Zeytinburnu build. Shared by the fetch
    and run scripts so both describe the sources identically."""
    from mre.real.provenance import DataSource, ProvenanceRecord

    def osm(name, geometry_type, count, attributes, missing):
        return DataSource(
            name=name,
            provider="OpenStreetMap contributors (via Overpass API)",
            url="https://www.openstreetmap.org / https://overpass-api.de",
            license="Open Database License (ODbL) 1.0",
            acquisition_date=acquired_at,
            crs="EPSG:4326 (reprojected to EPSG:32635)",
            geometry_type=geometry_type,
            spatial_coverage=ZEYTINBURNU.study_area,
            attributes=tuple(attributes),
            missing_attributes=tuple(missing),
            availability="available",
            feature_count=int(count),
            notes="Fetched live from Overpass; regenerate rather than commit.",
        )

    return ProvenanceRecord(
        pilot_name=ZEYTINBURNU.name,
        study_area=ZEYTINBURNU.study_area,
        bbox_wgs84=ZEYTINBURNU.bbox,
        target_crs=ZEYTINBURNU.target_crs,
        generated_at=acquired_at,
        sources=(
            osm("osm_buildings", "Polygon", n_buildings,
                ("osm_id", "building", "building_levels", "name"),
                ("construction_year", "structural_type", "occupancy", "occupants")),
            osm("osm_roads", "LineString", n_roads,
                ("osm_id", "highway", "road_class", "name", "maxspeed", "node_ids"),
                ("capacity", "lanes(not-ingested)")),
            osm("osm_hospitals", "Point", n_hospitals,
                ("osm_id", "name", "emergency"),
                ("capacity", "emergency_capacity", "beds")),
        ),
        scenario_provenance=scenario_provenance(),
        integrity_notes=integrity_notes(),
    )


def integrity_notes() -> tuple[str, ...]:
    """The honest-status notes embedded in every real-pilot provenance record."""
    return (
        "Real geometry, prototype physics: OSM supplies real building "
        "footprints, the real road network + topology, and real hospital "
        "locations. Structural attributes (construction year, structural "
        "system, and most storey counts) are NOT in OSM and are recorded "
        "UNKNOWN.",
        "The fragility/vulnerability parameters remain PROTOTYPE values from the "
        "synthetic engine, applied to real footprints. The damage layer is "
        "'prototype fragility on real geometry', NOT a real damage estimate.",
        "No licensed real population distribution is used; demand is a UNIFORM "
        "PROXY grid labelled population_source=UNIFORM_PROXY. Accessibility "
        "results describe geographic/network access, not real population impact.",
        "Hospital emergency capacities are almost never in OSM; where absent "
        "they are UNKNOWN and a PROTOTYPE emergency capacity is assigned for the "
        "accessibility model, labelled as such — never presented as the real "
        "capacity of a named hospital.",
        "The intensity field is a dimensionless scenario-derived proxy, not "
        "PGA/PGV/SA(T)/MMI.",
    )
