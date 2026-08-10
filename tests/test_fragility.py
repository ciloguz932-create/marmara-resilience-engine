"""Phase 3 — fragility model, tested independently of the rest of the engine.

These tests check that the formulation behaves as a probability model should.
They do NOT validate the parameters, which are invented.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mre.buildings import EXCEEDANCE_STATES, PrototypeLognormalFragility
from mre.models import Building, DamageState, Occupancy, StructuralType

SEED = 20260810


def _building(vulnerability: float = 1.0, index: int = 0) -> Building:
    return Building(
        building_id=f"B{index:05d}",
        easting=660_000.0,
        northing=4_540_000.0,
        floors=5,
        construction_year=1985,
        structural_type=StructuralType.RC_FRAME,
        occupancy=Occupancy.RESIDENTIAL,
        occupants=40,
        vulnerability_index=vulnerability,
    )


@pytest.fixture
def model(config) -> PrototypeLognormalFragility:
    return PrototypeLognormalFragility(config)


# --- probability axioms ---------------------------------------------------


def test_probabilities_sum_to_one(model, city):
    intensity = np.linspace(0.0, 1.5, len(city.buildings))
    probabilities = model.damage_probabilities(city.buildings, intensity)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-12)


def test_probabilities_are_within_zero_and_one(model, city):
    intensity = np.linspace(0.0, 2.0, len(city.buildings))
    probabilities = model.damage_probabilities(city.buildings, intensity)
    assert (probabilities >= 0.0).all()
    assert (probabilities <= 1.0).all()
    assert np.isfinite(probabilities).all()


def test_probability_matrix_shape_and_column_order(model, city):
    probabilities = model.damage_probabilities(city.buildings, np.full(len(city.buildings), 0.4))
    assert probabilities.shape == (len(city.buildings), len(DamageState))
    assert len(EXCEEDANCE_STATES) == len(DamageState) - 1


def test_zero_intensity_means_no_damage(model):
    buildings = (_building(), _building(2.0, 1))
    probabilities = model.damage_probabilities(buildings, np.zeros(2))
    np.testing.assert_allclose(probabilities[:, DamageState.NONE.value], 1.0)
    np.testing.assert_allclose(probabilities[:, 1:], 0.0)


def test_mismatched_intensity_shape_is_rejected(model, city):
    with pytest.raises(ValueError, match="does not match"):
        model.damage_probabilities(city.buildings, np.zeros(3))


# --- monotonicity ---------------------------------------------------------


def test_collapse_probability_increases_with_intensity(model):
    intensity = np.linspace(0.01, 2.0, 60)
    buildings = tuple(_building(index=i) for i in range(len(intensity)))
    probabilities = model.damage_probabilities(buildings, intensity)

    collapse = probabilities[:, DamageState.COLLAPSE.value]
    none = probabilities[:, DamageState.NONE.value]
    assert np.all(np.diff(collapse) >= -1e-12)
    assert np.all(np.diff(none) <= 1e-12)


def test_exceedance_curves_are_ordered_by_severity(model):
    intensity = np.full(40, 0.45)
    buildings = tuple(_building(index=i) for i in range(40))
    exceedance = model.exceedance_probabilities(buildings, intensity)
    # P(>=SLIGHT) >= P(>=MODERATE) >= P(>=SEVERE) >= P(>=COLLAPSE)
    assert np.all(np.diff(exceedance, axis=1) <= 1e-12)


def test_expected_damage_index_increases_with_intensity(model):
    intensity = np.linspace(0.01, 2.0, 60)
    buildings = tuple(_building(index=i) for i in range(len(intensity)))
    edi = model.expected_damage_index(model.damage_probabilities(buildings, intensity))
    assert np.all(np.diff(edi) >= -1e-12)
    assert (edi >= 0.0).all() and (edi <= 1.0).all()


# --- vulnerability heterogeneity -----------------------------------------


def test_higher_vulnerability_yields_a_worse_distribution(model):
    """Same shaking, different buildings -- the distributions must differ."""
    fragile = _building(2.0, 0)
    sturdy = _building(0.5, 1)
    probabilities = model.damage_probabilities((fragile, sturdy), np.full(2, 0.4))

    assert (
        probabilities[0, DamageState.COLLAPSE.value]
        > probabilities[1, DamageState.COLLAPSE.value]
    )
    assert probabilities[0, DamageState.NONE.value] < probabilities[1, DamageState.NONE.value]
    assert not np.allclose(probabilities[0], probabilities[1])


def test_collapse_probability_is_monotonic_in_vulnerability(model):
    vulnerabilities = np.linspace(0.4, 2.5, 30)
    buildings = tuple(_building(float(v), i) for i, v in enumerate(vulnerabilities))
    probabilities = model.damage_probabilities(buildings, np.full(len(buildings), 0.4))
    collapse = probabilities[:, DamageState.COLLAPSE.value]
    assert np.all(np.diff(collapse) >= -1e-12)


def test_real_city_produces_heterogeneous_distributions(model, city):
    probabilities = model.damage_probabilities(city.buildings, np.full(len(city.buildings), 0.35))
    assert len(np.unique(probabilities[:, DamageState.COLLAPSE.value])) > 900


# --- no threshold rules ---------------------------------------------------


def test_no_deterministic_threshold_at_the_median(model, config):
    """At its median a building must be uncertain, not doomed.

    Guards against anyone replacing the continuous formulation with an
    "intensity > X means collapse" rule.
    """
    median = config["damage"]["median_intensity"]["COLLAPSE"]
    buildings = (_building(1.0),)
    probabilities = model.damage_probabilities(buildings, np.array([median]))
    collapse = probabilities[0, DamageState.COLLAPSE.value]
    # By construction P(D >= COLLAPSE) = Phi(0) = 0.5 at the median.
    assert collapse == pytest.approx(0.5, abs=1e-9)
    assert 0.0 < collapse < 1.0


def test_even_extreme_intensity_leaves_probabilities_continuous(model):
    buildings = (_building(1.0),)
    probabilities = model.damage_probabilities(buildings, np.array([5.0]))
    assert probabilities[0, DamageState.COLLAPSE.value] > 0.95
    assert probabilities[0, DamageState.COLLAPSE.value] <= 1.0


# --- configuration guards -------------------------------------------------


def test_non_monotonic_medians_are_rejected(mutable_config):
    mutable_config["damage"]["median_intensity"]["SEVERE"] = 0.05
    with pytest.raises(ValueError, match="strictly increase"):
        PrototypeLognormalFragility(mutable_config)


def test_non_positive_beta_is_rejected(mutable_config):
    mutable_config["damage"]["beta_ln"] = 0.0
    with pytest.raises(ValueError, match="beta_ln"):
        PrototypeLognormalFragility(mutable_config)


def test_smaller_beta_sharpens_the_transition(mutable_config, config):
    """Lower dispersion => closer to a step function, but never a step."""
    sharp = dict(mutable_config)
    sharp["damage"] = dict(mutable_config["damage"], beta_ln=0.15)
    buildings = (_building(1.0),)
    intensity = np.array([config["damage"]["median_intensity"]["COLLAPSE"] * 1.4])

    broad_p = PrototypeLognormalFragility(config).damage_probabilities(buildings, intensity)
    sharp_p = PrototypeLognormalFragility(sharp).damage_probabilities(buildings, intensity)
    assert (
        sharp_p[0, DamageState.COLLAPSE.value] > broad_p[0, DamageState.COLLAPSE.value]
    )
    assert sharp_p[0, DamageState.COLLAPSE.value] < 1.0


# --- reproducibility ------------------------------------------------------


def test_probabilities_are_deterministic(model, city):
    intensity = np.linspace(0.05, 1.2, len(city.buildings))
    np.testing.assert_array_equal(
        model.damage_probabilities(city.buildings, intensity),
        model.damage_probabilities(city.buildings, intensity),
    )


def test_model_is_independent_of_building_order(model):
    buildings = (_building(0.8, 0), _building(1.6, 1))
    intensity = np.array([0.3, 0.5])
    forward = model.damage_probabilities(buildings, intensity)
    reversed_ = model.damage_probabilities(buildings[::-1], intensity[::-1])
    np.testing.assert_allclose(forward, reversed_[::-1])


def test_vulnerability_enters_only_through_the_index(model):
    """Two buildings differing in every attribute EXCEPT vulnerability must get
    identical distributions -- the model must not read anything else."""
    a = _building(1.1, 0)
    b = dataclasses.replace(
        a,
        building_id="B99999",
        floors=11,
        construction_year=1960,
        structural_type=StructuralType.MASONRY,
        occupancy=Occupancy.INDUSTRIAL,
        occupants=999,
        easting=661_000.0,
    )
    probabilities = model.damage_probabilities((a, b), np.full(2, 0.4))
    np.testing.assert_allclose(probabilities[0], probabilities[1])
