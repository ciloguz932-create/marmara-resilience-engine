"""Phase 3 — damage sampling.

Distinct from test_fragility.py: that tests the distribution, this tests the
draw from it, and that the distribution survives sampling.
"""

from __future__ import annotations

import numpy as np
import pytest

from mre.buildings import PrototypeLognormalFragility
from mre.models import Building, DamageState, Occupancy, StructuralType
from mre.rng import named_generator

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


@pytest.fixture
def intensity(city) -> np.ndarray:
    rng = named_generator(SEED, "test.intensity")
    return rng.uniform(0.05, 1.0, len(city.buildings))


def test_sampled_states_are_valid(model, city, intensity):
    result = model.sample_damage(city.buildings, intensity, named_generator(SEED, "damage:0"))
    assert result.states.shape == (len(city.buildings),)
    assert result.states.min() >= 0
    assert result.states.max() <= DamageState.COLLAPSE.value
    assert set(result.states.tolist()) <= {s.value for s in DamageState}


def test_state_enums_round_trip(model, city, intensity):
    result = model.sample_damage(city.buildings, intensity, named_generator(SEED, "damage:0"))
    enums = result.state_enums()
    assert len(enums) == len(city.buildings)
    assert all(isinstance(e, DamageState) for e in enums)
    assert [e.value for e in enums] == result.states.tolist()


def test_probability_vector_is_retained_alongside_the_sample(model, city, intensity):
    """The whole point: a sampled state alone discards what the model knows."""
    result = model.sample_damage(city.buildings, intensity, named_generator(SEED, "damage:0"))
    assert result.probabilities.shape == (len(city.buildings), len(DamageState))
    np.testing.assert_allclose(
        result.probabilities, model.damage_probabilities(city.buildings, intensity)
    )
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, rtol=1e-12)


def test_expected_damage_index_is_valid(model, city, intensity):
    result = model.sample_damage(city.buildings, intensity, named_generator(SEED, "damage:0"))
    edi = result.expected_damage_index
    assert edi.shape == (len(city.buildings),)
    assert (edi >= 0.0).all() and (edi <= 1.0).all()
    assert np.isfinite(edi).all()
    np.testing.assert_allclose(
        edi, (result.probabilities * np.arange(5)).sum(axis=1) / 4.0
    )


def test_collapse_mask_matches_states(model, city, intensity):
    result = model.sample_damage(city.buildings, intensity, named_generator(SEED, "damage:0"))
    np.testing.assert_array_equal(
        result.collapse_mask(), result.states == DamageState.COLLAPSE.value
    )


# --- reproducibility ------------------------------------------------------


def test_sampling_is_reproducible(model, city, intensity):
    first = model.sample_damage(city.buildings, intensity, named_generator(SEED, "damage:0"))
    second = model.sample_damage(city.buildings, intensity, named_generator(SEED, "damage:0"))
    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_array_equal(first.probabilities, second.probabilities)


def test_different_seed_changes_the_sample(model, city, intensity):
    first = model.sample_damage(city.buildings, intensity, named_generator(SEED, "damage:0"))
    second = model.sample_damage(city.buildings, intensity, named_generator(SEED + 1, "damage:0"))
    assert not np.array_equal(first.states, second.states)
    # ...but the distribution is unchanged: only the draw moved.
    np.testing.assert_array_equal(first.probabilities, second.probabilities)


# --- the sampler actually follows the distribution ------------------------


def test_empirical_frequencies_match_the_distribution(model):
    """Law of large numbers over 40,000 identical buildings.

    This is the test that would catch an off-by-one in the inverse-CDF sampler,
    which would otherwise produce plausible-looking but systematically wrong
    damage.
    """
    n = 40_000
    buildings = tuple(_building(1.0, i) for i in range(n))
    intensity = np.full(n, 0.45)

    result = model.sample_damage(buildings, intensity, named_generator(SEED, "damage:mc"))
    expected = result.probabilities[0]
    observed = np.bincount(result.states, minlength=5) / n

    np.testing.assert_allclose(observed, expected, atol=0.01)


def test_sampling_spans_multiple_states_on_the_real_city(model, city, intensity):
    result = model.sample_damage(city.buildings, intensity, named_generator(SEED, "damage:0"))
    assert len(set(result.states.tolist())) >= 4


def test_more_shaking_produces_worse_sampled_damage(model, city):
    weak = model.sample_damage(
        city.buildings, np.full(len(city.buildings), 0.15), named_generator(SEED, "damage:0")
    )
    strong = model.sample_damage(
        city.buildings, np.full(len(city.buildings), 0.9), named_generator(SEED, "damage:0")
    )
    assert strong.states.mean() > weak.states.mean()
    assert strong.expected_damage_index.mean() > weak.expected_damage_index.mean()
