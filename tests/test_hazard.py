"""Phase 3 — synthetic hazard field.

Intensity is a dimensionless PROXY. These tests check internal consistency and
reproducibility; none of them validates the model against reality, because
there is nothing here to validate against.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest

from mre.hazard import (
    LineSourceDecayField,
    PointSourceDecayField,
    hazard_field_from_config,
)
from mre.models import EarthquakeScenario
from mre.rng import named_generator

SEED = 20260810


@pytest.fixture
def field(config) -> PointSourceDecayField:
    return hazard_field_from_config(config)


# --- reproducibility ------------------------------------------------------


def test_sampling_is_reproducible(field, city):
    first = field.sample(city.buildings, named_generator(SEED, "hazard:0"))
    second = field.sample(city.buildings, named_generator(SEED, "hazard:0"))
    np.testing.assert_array_equal(first, second)


def test_different_seed_changes_the_realisation(field, city):
    first = field.sample(city.buildings, named_generator(SEED, "hazard:0"))
    second = field.sample(city.buildings, named_generator(SEED + 1, "hazard:0"))
    assert not np.array_equal(first, second)


def test_different_realisation_changes_the_field(field, city):
    first = field.sample(city.buildings, named_generator(SEED, "hazard:0"))
    second = field.sample(city.buildings, named_generator(SEED, "hazard:1"))
    assert not np.array_equal(first, second)


def test_median_field_is_deterministic(field, city):
    np.testing.assert_array_equal(
        field.median_intensity(city.buildings), field.median_intensity(city.buildings)
    )


# --- bounds and variation -------------------------------------------------


def test_intensity_is_non_negative_and_bounded(field, city, config):
    intensity = field.sample(city.buildings, named_generator(SEED, "hazard:0"))
    assert intensity.shape == (len(city.buildings),)
    assert (intensity >= 0).all()
    assert (intensity <= config["hazard"]["max_intensity"]).all()
    assert np.isfinite(intensity).all()


def test_field_varies_spatially(field, city):
    intensity = field.sample(city.buildings, named_generator(SEED, "hazard:0"))
    assert intensity.std() > 0
    assert intensity.max() > intensity.min()


def test_median_field_alone_varies_spatially(field, city):
    """Variation must come from geometry, not only from the random residual."""
    assert field.median_intensity(city.buildings).std() > 0


def test_zero_sigma_removes_the_random_component(mutable_config, city):
    mutable_config["hazard"]["intensity_sigma_ln"] = 0.0
    field = hazard_field_from_config(mutable_config)
    sampled = field.sample(city.buildings, named_generator(SEED, "hazard:0"))
    np.testing.assert_allclose(sampled, field.median_intensity(city.buildings))


# --- epicenter influence --------------------------------------------------


def test_intensity_decreases_with_distance_from_the_epicenter(field, city, config):
    epicenter_x, epicenter_y = config["hazard"]["epicenter"]
    distance = np.array(
        [np.hypot(b.easting - epicenter_x, b.northing - epicenter_y) for b in city.buildings]
    )
    median = field.median_intensity(city.buildings)

    # Monotone by construction, so the rank correlation must be exactly -1.
    order = np.argsort(distance)
    assert np.all(np.diff(median[order]) <= 1e-12)
    assert median[np.argmin(distance)] == median.max()
    assert median[np.argmax(distance)] == median.min()


def test_moving_the_epicenter_moves_the_field(mutable_config, city):
    before = hazard_field_from_config(mutable_config).median_intensity(city.buildings)
    mutable_config["hazard"]["epicenter"] = [700000.0, 4600000.0]
    after = hazard_field_from_config(mutable_config).median_intensity(city.buildings)
    assert not np.array_equal(before, after)
    # Far away epicenter => weaker shaking everywhere.
    assert after.mean() < before.mean()


# --- configuration sensitivity -------------------------------------------


def test_larger_decay_lowers_intensity(mutable_config, city):
    weak = copy.deepcopy(mutable_config)
    weak["hazard"]["decay_per_km"] = mutable_config["hazard"]["decay_per_km"] * 3
    assert (
        hazard_field_from_config(weak).median_intensity(city.buildings).mean()
        < hazard_field_from_config(mutable_config).median_intensity(city.buildings).mean()
    )


def test_larger_source_intensity_raises_intensity(mutable_config, city):
    strong = copy.deepcopy(mutable_config)
    strong["hazard"]["intensity_at_source"] = mutable_config["hazard"]["intensity_at_source"] * 1.5
    assert (
        hazard_field_from_config(strong).median_intensity(city.buildings).mean()
        > hazard_field_from_config(mutable_config).median_intensity(city.buildings).mean()
    )


def test_max_intensity_caps_the_field(mutable_config, city):
    mutable_config["hazard"]["max_intensity"] = 0.05
    intensity = hazard_field_from_config(mutable_config).sample(
        city.buildings, named_generator(SEED, "hazard:0")
    )
    assert intensity.max() <= 0.05


def test_near_source_saturation_prevents_a_singularity(mutable_config, city):
    """A site exactly at the epicenter must not blow up."""
    building = city.buildings[0]
    mutable_config["hazard"]["epicenter"] = [building.easting, building.northing]
    field = hazard_field_from_config(mutable_config)
    median = field.median_intensity((building,))
    assert np.isfinite(median).all()
    assert median[0] < mutable_config["hazard"]["intensity_at_source"]


# --- model selection ------------------------------------------------------


def test_line_source_model_is_selectable(mutable_config, city):
    point_field = hazard_field_from_config(mutable_config)
    assert isinstance(point_field, PointSourceDecayField)

    mutable_config["hazard"]["model"] = "line_source_decay"
    line_field = hazard_field_from_config(mutable_config)
    assert isinstance(line_field, LineSourceDecayField)

    assert not np.array_equal(
        line_field.median_intensity(city.buildings),
        point_field.median_intensity(city.buildings),
    )


def test_line_source_distance_is_measured_to_the_segment(config, city):
    """Not to an endpoint: a site beside the middle of the rupture is close."""
    scenario = EarthquakeScenario.from_config(config)
    field = LineSourceDecayField(scenario)
    (x1, y1), (x2, y2) = scenario.source_line
    midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)

    coordinates = np.array([midpoint, (x1, y1)], dtype=float)
    distances = field._distance_km(coordinates)
    assert distances[0] == pytest.approx(0.0, abs=1e-9)


def test_unknown_model_is_rejected(mutable_config):
    mutable_config["hazard"]["model"] = "not_a_model"
    with pytest.raises(ValueError, match="unknown hazard model"):
        hazard_field_from_config(mutable_config)


def test_scenario_carries_magnitude_as_metadata_only(config, city):
    """Magnitude must not affect intensity, because the prototype does not
    model it. Silent dependence would be worse than absence."""
    scenario = EarthquakeScenario.from_config(config)
    baseline = PointSourceDecayField(scenario).median_intensity(city.buildings)
    bigger = PointSourceDecayField(
        dataclasses.replace(scenario, magnitude=9.0)
    ).median_intensity(city.buildings)
    np.testing.assert_array_equal(baseline, bigger)
