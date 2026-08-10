"""Phase 1: the inter-stage contract.

These types are what lets a validated model replace a prototype one later, so
their shape is worth pinning down before anything depends on it.
"""

from __future__ import annotations

import dataclasses

import pytest

from mre.models import (
    Building,
    DamageState,
    Hospital,
    InterventionType,
    LinkState,
    Occupancy,
    RoadClass,
    RoadLink,
    StructuralType,
    SyntheticCity,
)


def test_damage_states_are_ordered():
    """Severity comparisons and array indexing both rely on this order."""
    assert [s.value for s in DamageState] == [0, 1, 2, 3, 4]
    assert DamageState.COLLAPSE.value > DamageState.SEVERE.value
    assert [s.name for s in DamageState] == [
        "NONE",
        "SLIGHT",
        "MODERATE",
        "SEVERE",
        "COLLAPSE",
    ]


def test_intervention_types_are_the_three_prototypes():
    assert {t.name for t in InterventionType} == {
        "BUILDING_RETROFIT",
        "ROAD_HARDENING",
        "HOSPITAL_SUPPORT",
    }


def test_link_states():
    assert {s.name for s in LinkState} == {"OPEN", "DEGRADED", "CLOSED"}


def _building(**overrides) -> Building:
    defaults = dict(
        building_id="B0001",
        easting=660_000.0,
        northing=4_535_000.0,
        floors=5,
        construction_year=1985,
        structural_type=StructuralType.RC_FRAME,
        occupancy=Occupancy.RESIDENTIAL,
        occupants=40,
        vulnerability_index=1.2,
    )
    return Building(**{**defaults, **overrides})


def test_building_is_immutable():
    """Stages share these objects across realisations; mutation would leak
    state between Monte Carlo runs."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        _building().floors = 9


def test_road_link_criticality_defaults_to_zero():
    link = RoadLink(
        road_id="R0001",
        from_node=0,
        to_node=1,
        length_m=120.0,
        road_class=RoadClass.LOCAL,
        travel_time_min=0.36,
        closure_probability=0.01,
    )
    assert link.criticality == 0.0


def test_hospital_carries_emergency_capacity_separately():
    hospital = Hospital(
        hospital_id="H01",
        easting=661_000.0,
        northing=4_536_000.0,
        capacity=400,
        emergency_capacity=60,
        node_id=12,
    )
    assert hospital.emergency_capacity < hospital.capacity


def test_empty_city_is_constructible():
    city = SyntheticCity(crs="EPSG:32635", seed=1)
    assert city.buildings == ()
    assert city.nodes == {}
