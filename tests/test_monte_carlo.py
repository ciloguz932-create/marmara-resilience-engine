"""Phase 4 — Monte Carlo orchestration.

The guarantee that matters most: **realisation i run alone is identical to
realisation i extracted from a larger batch.** Everything else about
reproducibility follows from it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mre.data import build_synthetic_city
from mre.simulation import (
    ModelSuite,
    build_models,
    iter_realisations,
    run_monte_carlo,
    run_scenario,
)

SEED = 20260810
N = 12


@pytest.fixture(scope="module")
def run(request):
    city = request.getfixturevalue("city")
    config = request.getfixturevalue("config")
    return run_monte_carlo(city, config, n_simulations=N, seed=SEED)


# --- configuration --------------------------------------------------------


def test_default_n_simulations_is_1000(config):
    assert config["monte_carlo"]["n_simulations"] == 1000


def test_n_simulations_is_respected(run):
    assert run.n_simulations == N
    assert len(run.records) == N
    assert [r.realisation for r in run.records] == list(range(N))


def test_default_n_comes_from_config(small_city, small_config, monkeypatch):
    """Without an explicit count, the config value is used."""
    small_config = dict(small_config)
    small_config["monte_carlo"] = {**small_config["monte_carlo"], "n_simulations": 3}
    result = run_monte_carlo(small_city, small_config, seed=SEED)
    assert result.n_simulations == 3


def test_zero_realisations_is_rejected(small_city, small_config):
    with pytest.raises(ValueError, match="at least 1"):
        run_monte_carlo(small_city, small_config, n_simulations=0, seed=SEED)


def test_seed_defaults_to_the_city_seed(small_city, small_config):
    result = run_monte_carlo(small_city, small_config, n_simulations=2)
    assert result.seed == small_city.seed


def test_scenario_and_seed_provenance_is_recorded(run, config):
    assert run.seed == SEED
    assert run.scenario_id == config["hazard"]["scenario_id"]


# --- the realisation-index guarantee -------------------------------------


def test_realisation_alone_matches_the_same_realisation_in_a_batch(city, config):
    """The Phase 3 guarantee, preserved through Monte Carlo orchestration."""
    batch = list(iter_realisations(city, config, n_simulations=5, seed=SEED))

    for index in (0, 3, 4):
        alone = run_scenario(city, config, realisation=index, seed=SEED)
        np.testing.assert_array_equal(alone.intensity, batch[index].intensity)
        np.testing.assert_array_equal(alone.damage.states, batch[index].damage.states)
        np.testing.assert_array_equal(alone.link_states, batch[index].link_states)
        assert (
            alone.accessibility.post_event.population_unreachable
            == batch[index].accessibility.post_event.population_unreachable
        )
        assert alone.accessibility.n_disrupted_routes == (
            batch[index].accessibility.n_disrupted_routes
        )


def test_batch_size_does_not_change_early_realisations(city, config):
    """Realisation 2 must be the same whether the run is 3 long or 8 long."""
    short = run_monte_carlo(city, config, n_simulations=3, seed=SEED)
    longer = run_monte_carlo(city, config, n_simulations=8, seed=SEED)
    assert short.records[2] == longer.records[2]
    assert short.records[:3] == longer.records[:3]


def test_realisations_are_not_all_identical(run):
    """Independent streams must actually differ, or the run is 1 realisation
    repeated N times."""
    assert len({r.n_collapse for r in run.records}) > 1
    assert len({r.n_closed for r in run.records}) > 1


# --- reproducibility ------------------------------------------------------


def test_same_seed_reproduces_the_whole_run(city, config):
    first = run_monte_carlo(city, config, n_simulations=N, seed=SEED)
    second = run_monte_carlo(city, config, n_simulations=N, seed=SEED)

    assert first.records == second.records
    np.testing.assert_array_equal(
        first.building_collapse_count, second.building_collapse_count
    )
    np.testing.assert_array_equal(first.link_closed_count, second.link_closed_count)
    np.testing.assert_array_equal(
        first.unit_unreachable_count, second.unit_unreachable_count
    )


def test_different_seed_changes_the_run(config):
    city_a = build_synthetic_city(config, seed=SEED)
    city_b = build_synthetic_city(config, seed=SEED + 1)
    a = run_monte_carlo(city_a, config, n_simulations=N, seed=SEED)
    b = run_monte_carlo(city_b, config, n_simulations=N, seed=SEED + 1)

    assert a.records != b.records
    assert not np.array_equal(a.building_collapse_count, b.building_collapse_count)


def test_different_seed_same_city_still_changes_the_draws(city, config):
    a = run_monte_carlo(city, config, n_simulations=N, seed=SEED)
    b = run_monte_carlo(city, config, n_simulations=N, seed=SEED + 7)
    assert [r.n_collapse for r in a.records] != [r.n_collapse for r in b.records]
    # Same city => same deterministic baseline.
    assert a.baseline.hospital_load == b.baseline.hospital_load


# --- paired baseline comparison ------------------------------------------


def test_baseline_is_computed_once_and_is_deterministic(run, city, config):
    """The pre-earthquake network is a property of the city, not of a draw."""
    other = run_monte_carlo(city, config, n_simulations=3, seed=SEED + 99)
    assert run.baseline.population_reachable == other.baseline.population_reachable
    assert run.baseline.hospital_load == other.baseline.hospital_load
    np.testing.assert_array_equal(
        run.baseline.travel_time_min, other.baseline.travel_time_min
    )


def test_deltas_are_measured_against_that_baseline(run):
    """Each realisation is compared with the same pre-event state, never with
    another realisation."""
    baseline_mean = run.baseline.mean_travel_time_min
    for record in run.records:
        assert record.delta_mean_travel_time_min == pytest.approx(
            record.mean_travel_time_min - baseline_mean
        )


def test_post_event_is_never_better_than_baseline_on_average(run):
    """Disruption can only remove or slow links, so travel time cannot improve."""
    assert all(r.delta_mean_travel_time_min >= -1e-9 for r in run.records)
    assert all(r.population_unreachable >= run.baseline.population_unreachable for r in run.records)


# --- accumulators ---------------------------------------------------------


def test_spatial_counters_are_bounded_by_the_run_length(run, city):
    assert run.building_collapse_count.shape == (len(city.buildings),)
    assert run.link_closed_count.shape == (len(city.roads),)
    assert run.unit_unreachable_count.shape == (len(city.population),)

    for counter in (
        run.building_collapse_count,
        run.link_closed_count,
        run.link_degraded_count,
        run.unit_unreachable_count,
    ):
        assert counter.min() >= 0
        assert counter.max() <= run.n_simulations


def test_frequencies_are_probabilities(run):
    for frequency in (
        run.building_collapse_frequency,
        run.link_closed_frequency,
        run.link_degraded_frequency,
        run.unit_unreachable_frequency,
    ):
        assert (frequency >= 0.0).all()
        assert (frequency <= 1.0).all()
        assert np.isfinite(frequency).all()


def test_reachable_and_unreachable_counts_partition_the_run(run):
    np.testing.assert_array_equal(
        run.unit_reachable_count + run.unit_unreachable_count,
        np.full(run.unit_reachable_count.shape, run.n_simulations),
    )


def test_mean_edi_is_in_range(run):
    mean_edi = run.building_mean_edi
    assert (mean_edi >= 0.0).all()
    assert (mean_edi <= 1.0).all()
    assert np.isfinite(mean_edi).all()


def test_unit_mean_travel_time_excludes_unreachable_realisations(run):
    """Averaged over realisations where the unit WAS reachable; NaN if never."""
    mean_time = run.unit_mean_travel_time_min
    reachable_ever = run.unit_reachable_count > 0
    assert np.isfinite(mean_time[reachable_ever]).all()
    assert np.isnan(mean_time[~reachable_ever]).all() or reachable_ever.all()


def test_collapse_frequency_tracks_vulnerability(run, city):
    """Sanity: more vulnerable buildings collapse more often across the run."""
    vulnerability = np.array([b.vulnerability_index for b in city.buildings])
    frequency = run.building_collapse_frequency
    assert np.corrcoef(vulnerability, frequency)[0, 1] > 0.2


# --- records --------------------------------------------------------------


def test_damage_counts_sum_to_the_building_total(run, city):
    for record in run.records:
        total = (
            record.n_none
            + record.n_slight
            + record.n_moderate
            + record.n_severe
            + record.n_collapse
        )
        assert total == len(city.buildings)


def test_link_states_sum_to_the_link_total(run, city):
    for record in run.records:
        assert record.n_open + record.n_degraded + record.n_closed == len(city.roads)


def test_no_nan_or_inf_in_any_record(run):
    for record in run.records:
        for name, value in record.flat_row().items():
            if isinstance(value, float):
                assert math.isfinite(value), f"{name} is {value}"


def test_flat_row_expands_hospital_columns(run, city):
    row = run.records[0].flat_row()
    for hospital in city.hospitals:
        assert f"load_{hospital.hospital_id}" in row
        assert f"utilisation_{hospital.hospital_id}" in row
    assert not any(isinstance(v, dict) for v in row.values())


def test_hospital_loads_sum_to_reachable_population_every_realisation(run):
    """The documented model constraint, checked on every draw."""
    for record in run.records:
        assert sum(record.hospital_load.values()) == record.population_reachable


# --- orchestration is model-agnostic -------------------------------------


def test_models_can_be_injected(city, config):
    """The driver must not depend on which concrete models it is running."""
    models = build_models(config)
    assert isinstance(models, ModelSuite)

    injected = run_monte_carlo(city, config, n_simulations=4, seed=SEED, models=models)
    default = run_monte_carlo(city, config, n_simulations=4, seed=SEED)
    assert injected.records == default.records


def test_models_are_built_once_per_run(city, config, monkeypatch):
    """Rebuilding the disruption model per realisation would recompute the
    building/link adjacency index every time."""
    import mre.simulation as simulation

    calls = {"n": 0}
    original = simulation.build_models

    def counting_build_models(cfg):
        calls["n"] += 1
        return original(cfg)

    monkeypatch.setattr(simulation, "build_models", counting_build_models)
    run_monte_carlo(city, config, n_simulations=5, seed=SEED)
    assert calls["n"] == 1
