"""Phase 3 — machine-readable outputs and the Blender handoff boundary."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from mre.models import DamageState, LinkState
from mre.outputs import (
    buildings_geodataframe,
    population_geodataframe,
    roads_geodataframe,
    summarise,
    write_city_layers,
    write_scenario_outputs,
)
from mre.simulation import run_scenario

SEED = 20260810


@pytest.fixture(scope="module")
def result(request):
    city = request.getfixturevalue("city")
    config = request.getfixturevalue("config")
    return run_scenario(city, config, seed=SEED, realisation=0)


# --- geodataframes --------------------------------------------------------


def test_layers_keep_their_crs(city):
    """Silent CRS loss is the failure this project most needs to catch."""
    for frame in (
        buildings_geodataframe(city),
        roads_geodataframe(city),
        population_geodataframe(city),
    ):
        assert frame.crs is not None
        assert frame.crs.to_string() == city.crs
        assert frame.crs.is_projected


def test_building_geometries_are_squares_in_metres(city):
    frame = buildings_geodataframe(city)
    assert len(frame) == len(city.buildings)
    assert (frame.area > 0).all()
    assert frame.geometry.is_valid.all()
    np.testing.assert_allclose(
        frame.area.to_numpy(),
        np.array([b.footprint_side_m**2 for b in city.buildings]),
    )


def test_road_geometries_match_link_lengths(city):
    frame = roads_geodataframe(city)
    np.testing.assert_allclose(
        frame.length.to_numpy(), np.array([r.length_m for r in city.roads])
    )


# --- writing --------------------------------------------------------------


def test_write_city_layers_produces_all_four_layers(city, tmp_path):
    written = write_city_layers(city, tmp_path)
    path = written["city_gpkg"]
    assert path.is_file()

    for layer, expected in [
        ("buildings", len(city.buildings)),
        ("roads", len(city.roads)),
        ("hospitals", len(city.hospitals)),
        ("population_units", len(city.population)),
    ]:
        frame = gpd.read_file(path, layer=layer)
        assert len(frame) == expected
        assert frame.crs.to_string() == city.crs


def test_scenario_outputs_are_written(city, result, tmp_path):
    written = write_scenario_outputs(city, result, tmp_path)
    assert set(written) == {
        "scenario_gpkg",
        "damage_probabilities_csv",
        "summary_json",
    }
    for path in written.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_damage_layer_carries_hazard_and_damage(city, result, tmp_path):
    path = write_scenario_outputs(city, result, tmp_path)["scenario_gpkg"]
    frame = gpd.read_file(path, layer="buildings_damage")

    assert len(frame) == len(city.buildings)
    assert frame.crs.to_string() == city.crs
    np.testing.assert_allclose(frame["intensity_proxy"].to_numpy(), result.intensity)
    np.testing.assert_allclose(
        frame["expected_damage_index"].to_numpy(), result.damage.expected_damage_index
    )
    assert set(frame["damage_state"]) <= {s.name for s in DamageState}


def test_disrupted_roads_layer_carries_link_states(city, result, tmp_path):
    path = write_scenario_outputs(city, result, tmp_path)["scenario_gpkg"]
    frame = gpd.read_file(path, layer="roads_disrupted")

    assert len(frame) == len(city.roads)
    assert set(frame["link_state"]) <= {s.value for s in LinkState}
    assert (frame["closure_probability"] >= 0).all()
    assert (frame["closure_probability"] <= 1).all()


def test_probability_csv_retains_full_distributions(city, result, tmp_path):
    path = write_scenario_outputs(city, result, tmp_path)["damage_probabilities_csv"]
    frame = pd.read_csv(path)

    assert len(frame) == len(city.buildings)
    columns = [f"p_{s.name}" for s in DamageState]
    assert all(c in frame.columns for c in columns)
    np.testing.assert_allclose(frame[columns].to_numpy().sum(axis=1), 1.0, rtol=1e-9)
    np.testing.assert_allclose(frame[columns].to_numpy(), result.damage.probabilities, atol=1e-9)


def test_summary_json_is_valid_and_carries_the_disclaimer(city, result, tmp_path):
    path = write_scenario_outputs(city, result, tmp_path)["summary_json"]
    summary = json.loads(path.read_text(encoding="utf-8"))

    assert summary["seed"] == SEED
    assert summary["scenario_id"] == result.scenario_id
    assert summary["n_buildings"] == len(city.buildings)
    assert sum(summary["damage_state_counts"].values()) == len(city.buildings)
    assert "SYNTHETIC PROTOTYPE" in summary["disclaimer"]
    assert "not a validated" in summary["disclaimer"].lower()


def test_summary_is_json_serialisable(result):
    json.dumps(summarise(result))


def test_outputs_stay_small(city, result, tmp_path):
    """Guard against an accidental per-realisation data dump."""
    written = write_city_layers(city, tmp_path)
    written.update(write_scenario_outputs(city, result, tmp_path))
    for label, path in written.items():
        assert path.stat().st_size < 5_000_000, f"{label} is unexpectedly large"


def test_written_outputs_are_reproducible(city, config, tmp_path):
    """Same seed, same numbers on disk."""
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    for directory in (first_dir, second_dir):
        result = run_scenario(city, config, seed=SEED, realisation=0)
        write_scenario_outputs(city, result, directory)

    a = pd.read_csv(next(first_dir.glob("damage_probabilities_*.csv")))
    b = pd.read_csv(next(second_dir.glob("damage_probabilities_*.csv")))
    pd.testing.assert_frame_equal(a, b)
