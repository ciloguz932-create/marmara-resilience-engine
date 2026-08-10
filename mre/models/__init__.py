"""Domain schemas shared by every stage of the pipeline.

These types are the contract between stages. The synthetic-data layer produces
them and the real-data layer (future) must produce the same types, so that the
simulation core never learns whether its input was invented or observed.

Nothing here performs science; these are structures and vocabularies only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from shapely.geometry import Polygon, box

__all__ = [
    "DamageState",
    "StructuralType",
    "Occupancy",
    "RoadClass",
    "LinkState",
    "InterventionType",
    "Building",
    "Hospital",
    "RoadLink",
    "PopulationUnit",
    "EarthquakeScenario",
    "SyntheticCity",
]


class DamageState(Enum):
    """Ordered damage states. Order matters: comparisons rely on ``value``."""

    NONE = 0
    SLIGHT = 1
    MODERATE = 2
    SEVERE = 3
    COLLAPSE = 4


class StructuralType(Enum):
    RC_FRAME = "RC_FRAME"
    RC_SHEAR_WALL = "RC_SHEAR_WALL"
    MASONRY = "MASONRY"
    STEEL = "STEEL"
    OTHER = "OTHER"


class Occupancy(Enum):
    RESIDENTIAL = "RESIDENTIAL"
    COMMERCIAL = "COMMERCIAL"
    PUBLIC = "PUBLIC"
    INDUSTRIAL = "INDUSTRIAL"


class RoadClass(Enum):
    ARTERIAL = "ARTERIAL"
    COLLECTOR = "COLLECTOR"
    LOCAL = "LOCAL"


class LinkState(Enum):
    """Post-earthquake condition of a road link."""

    OPEN = "OPEN"
    DEGRADED = "DEGRADED"
    CLOSED = "CLOSED"


class InterventionType(Enum):
    BUILDING_RETROFIT = "BUILDING_RETROFIT"
    ROAD_HARDENING = "ROAD_HARDENING"
    HOSPITAL_SUPPORT = "HOSPITAL_SUPPORT"


@dataclass(frozen=True, slots=True)
class Building:
    building_id: str
    easting: float
    northing: float
    floors: int
    construction_year: int
    structural_type: StructuralType
    occupancy: Occupancy
    occupants: int
    # Dimensionless prototype vulnerability index. Higher = MORE fragile:
    # fragility medians are DIVIDED by it. See mre.buildings.
    vulnerability_index: float
    # Side length of the square footprint, in metres.
    footprint_side_m: float = 12.0

    def footprint(self) -> Polygon:
        """Square footprint centred on the building point."""
        half = self.footprint_side_m / 2.0
        return box(
            self.easting - half,
            self.northing - half,
            self.easting + half,
            self.northing + half,
        )


@dataclass(frozen=True, slots=True)
class Hospital:
    hospital_id: str
    easting: float
    northing: float
    capacity: int
    emergency_capacity: int
    # Nearest road-network node, resolved when the city is built.
    node_id: int


@dataclass(frozen=True, slots=True)
class RoadLink:
    road_id: str
    from_node: int
    to_node: int
    length_m: float
    road_class: RoadClass
    # Free-flow travel time in minutes.
    travel_time_min: float
    # Intrinsic probability of closure before building-collapse effects.
    closure_probability: float
    # Betweenness-derived importance, filled in by mre.roads.
    criticality: float = 0.0
    # Multiplier on debris-driven disruption. Narrow streets block more easily.
    susceptibility: float = 1.0


@dataclass(frozen=True, slots=True)
class PopulationUnit:
    population_id: str
    easting: float
    northing: float
    population_count: int
    node_id: int


@dataclass(frozen=True, slots=True)
class EarthquakeScenario:
    """A synthetic scenario. ``magnitude`` is descriptive metadata only -- it
    does not enter the intensity calculation.

    Intensity is a dimensionless PROXY field. It is NOT PGA, PGV, SA(T), or
    MMI, and must never be relabelled as one.
    """

    scenario_id: str
    magnitude: float
    intensity_at_source: float
    decay_per_km: float
    intensity_sigma_ln: float
    near_source_saturation_km: float = 3.0
    max_intensity: float = 2.0
    # Point source. Used by PointSourceDecayField.
    epicenter: tuple[float, float] = (0.0, 0.0)
    # Line source. Used by LineSourceDecayField.
    source_line: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 0.0),
        (1.0, 0.0),
    )

    @classmethod
    def from_config(cls, config: dict) -> "EarthquakeScenario":
        hazard = config["hazard"]
        (x1, y1), (x2, y2) = hazard["source_line"]
        epicenter_x, epicenter_y = hazard["epicenter"]
        return cls(
            scenario_id=hazard["scenario_id"],
            magnitude=hazard["magnitude"],
            intensity_at_source=hazard["intensity_at_source"],
            decay_per_km=hazard["decay_per_km"],
            intensity_sigma_ln=hazard["intensity_sigma_ln"],
            near_source_saturation_km=hazard["near_source_saturation_km"],
            max_intensity=hazard["max_intensity"],
            epicenter=(epicenter_x, epicenter_y),
            source_line=((x1, y1), (x2, y2)),
        )


@dataclass(frozen=True, slots=True)
class SyntheticCity:
    """A complete, self-consistent synthetic world."""

    crs: str
    seed: int
    buildings: tuple[Building, ...] = field(default_factory=tuple)
    hospitals: tuple[Hospital, ...] = field(default_factory=tuple)
    roads: tuple[RoadLink, ...] = field(default_factory=tuple)
    population: tuple[PopulationUnit, ...] = field(default_factory=tuple)
    # node_id -> (easting, northing)
    nodes: dict[int, tuple[float, float]] = field(default_factory=dict)
