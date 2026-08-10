"""Phase 4 — uncertainty reporting and Monte Carlo outputs.

Two things are being protected here:

1. Summary statistics must never contain NaN or infinity. Non-finite
   realisations are excluded and *counted*, not propagated.
2. Percentiles must be described as empirical quantiles, never as confidence
   intervals, and the reports must not imply that more realisations make the
   model more valid.
"""

from __future__ import annotations

import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from mre.outputs import (
    DISTRIBUTION_METRICS,
    distribution_summary,
    realisations_dataframe,
    summarise_monte_carlo,
    write_monte_carlo_outputs,
)
from mre.simulation import run_monte_carlo

SEED = 20260810
N = 12


@pytest.fixture(scope="module")
def run(request):
    city = request.getfixturevalue("city")
    config = request.getfixturevalue("config")
    return run_monte_carlo(city, config, n_simulations=N, seed=SEED)


@pytest.fixture(scope="module")
def summary(run, request):
    return summarise_monte_carlo(run, request.getfixturevalue("config"))


# --- distribution_summary -------------------------------------------------


def test_summary_reports_all_required_statistics():
    stats = distribution_summary(np.arange(100.0))
    for key in ("mean", "std", "p05", "p25", "p50", "p75", "p95"):
        assert key in stats
        assert stats[key] is not None


def test_percentiles_are_ordered():
    stats = distribution_summary(np.random.default_rng(0).normal(size=500))
    assert stats["min"] <= stats["p05"] <= stats["p25"] <= stats["p50"]
    assert stats["p50"] <= stats["p75"] <= stats["p95"] <= stats["max"]


def test_percentiles_are_correct_for_a_known_distribution():
    stats = distribution_summary(np.arange(1.0, 101.0))
    assert stats["p50"] == pytest.approx(50.5)
    assert stats["mean"] == pytest.approx(50.5)
    assert stats["p05"] == pytest.approx(5.95)


def test_non_finite_values_are_excluded_and_counted():
    """The NaN-contamination guard, at its source."""
    values = np.array([1.0, 2.0, np.nan, np.inf, 3.0, -np.inf])
    stats = distribution_summary(values)

    assert stats["n"] == 6
    assert stats["n_finite"] == 3
    assert stats["n_excluded"] == 3
    assert stats["mean"] == pytest.approx(2.0)
    assert math.isfinite(stats["p50"])


def test_all_non_finite_yields_nulls_not_a_crash():
    stats = distribution_summary(np.array([np.nan, np.inf]))
    assert stats["n_finite"] == 0
    assert stats["mean"] is None
    assert stats["p50"] is None


def test_single_realisation_has_zero_spread():
    stats = distribution_summary(np.array([4.0]))
    assert stats["mean"] == 4.0
    assert stats["std"] == 0.0
    assert stats["p05"] == stats["p95"] == 4.0


def test_custom_percentiles_are_honoured():
    stats = distribution_summary(np.arange(100.0), percentiles=(10, 90))
    assert "p10" in stats and "p90" in stats


# --- the full summary -----------------------------------------------------


def test_every_declared_metric_is_summarised(summary):
    assert set(summary["metrics"]) == set(DISTRIBUTION_METRICS)
    for name, stats in summary["metrics"].items():
        assert stats["n"] == N, name


def test_required_metric_families_are_present(summary):
    metrics = summary["metrics"]
    # damage states, EDI, travel time, populations, roads, utilisation
    for name in (
        "n_collapse",
        "mean_expected_damage_index",
        "mean_travel_time_min",
        "median_travel_time_min",
        "population_reachable",
        "population_unreachable",
        "n_closed",
        "n_degraded",
        "n_open",
        "n_disrupted_routes",
    ):
        assert name in metrics
    assert summary["hospitals"], "hospital utilisation distributions missing"


def test_no_nan_or_inf_anywhere_in_the_summary(summary):
    """Requirement E: reported summary statistics must be clean."""

    def walk(node, path="root"):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        elif isinstance(node, float):
            assert math.isfinite(node), f"{path} is {node}"

    walk(summary)


def test_summary_is_json_serialisable(summary):
    text = json.dumps(summary)
    assert "NaN" not in text
    assert "Infinity" not in text


def test_summary_carries_run_provenance(summary, config):
    assert summary["seed"] == SEED
    assert summary["n_simulations"] == N
    assert summary["scenario_id"] == config["hazard"]["scenario_id"]
    assert summary["percentiles"] == config["monte_carlo"]["percentiles"]


# --- scientific framing ---------------------------------------------------


def test_percentiles_are_not_called_confidence_intervals(summary):
    note = summary["uncertainty_note"]
    assert "NOT confidence intervals" in note
    assert "empirical quantiles" in note.lower()


def test_summary_denies_that_more_runs_mean_validity(summary):
    assert "more realisations narrows" in summary["uncertainty_note"].lower()
    assert "SYNTHETIC PROTOTYPE" in summary["disclaimer"]


def test_summary_distinguishes_sampling_variability_from_model_uncertainty(summary):
    note = summary["uncertainty_note"].lower()
    assert "sampling variability" in note
    assert "epistemic" in note


def test_utilisation_scale_artefact_is_declared(summary):
    """The synthetic population/capacity mismatch must be stated wherever the
    numbers are reported, or utilisation will be misread as an overload ratio."""
    note = summary["baseline"]["utilisation_scale_note"]
    assert "artefact" in note.lower()
    assert "not a finding" in note.lower()
    assert summary["baseline"]["population_per_emergency_bed"] > 1.0


# --- baseline block -------------------------------------------------------


def test_baseline_block_describes_the_pairing(summary):
    assert "paired comparison" in summary["baseline"]["note"]
    assert summary["baseline"]["population_unreachable"] == 0


def test_hospital_block_reports_capacity_and_distributions(summary, city):
    assert set(summary["hospitals"]) == {h.hospital_id for h in city.hospitals}
    for hospital in city.hospitals:
        block = summary["hospitals"][hospital.hospital_id]
        assert block["emergency_capacity"] == hospital.emergency_capacity
        assert block["utilisation"]["p05"] <= block["utilisation"]["p95"]
        assert block["load"]["mean"] >= 0


# --- tabular output -------------------------------------------------------


def test_realisations_dataframe_has_one_row_per_realisation(run):
    frame = realisations_dataframe(run)
    assert len(frame) == run.n_simulations
    assert frame["realisation"].tolist() == list(range(run.n_simulations))


def test_realisations_dataframe_schema_is_stable(run, city):
    frame = realisations_dataframe(run)
    for column in (
        "realisation",
        "n_collapse",
        "mean_expected_damage_index",
        "n_open",
        "n_degraded",
        "n_closed",
        "mean_travel_time_min",
        "median_travel_time_min",
        "population_reachable",
        "population_unreachable",
        "n_disrupted_routes",
        "delta_mean_travel_time_min",
    ):
        assert column in frame.columns
    for hospital in city.hospitals:
        assert f"utilisation_{hospital.hospital_id}" in frame.columns


def test_realisations_dataframe_has_no_nan(run):
    frame = realisations_dataframe(run)
    assert not frame.select_dtypes(include=[float]).isna().any().any()
    assert np.isfinite(frame.select_dtypes(include=[float]).to_numpy()).all()


# --- files ----------------------------------------------------------------


def test_monte_carlo_outputs_are_written(city, run, config, tmp_path):
    written = write_monte_carlo_outputs(city, run, tmp_path, config)
    assert set(written) == {
        "realisations_csv",
        "monte_carlo_summary_json",
        "monte_carlo_gpkg",
    }
    for path in written.values():
        assert path.is_file() and path.stat().st_size > 0


def test_written_csv_round_trips(city, run, config, tmp_path):
    path = write_monte_carlo_outputs(city, run, tmp_path, config)["realisations_csv"]
    frame = pd.read_csv(path)
    assert len(frame) == run.n_simulations
    assert frame["n_collapse"].tolist() == [r.n_collapse for r in run.records]


def test_written_gpkg_layers_keep_their_crs(city, run, config, tmp_path):
    path = write_monte_carlo_outputs(city, run, tmp_path, config)["monte_carlo_gpkg"]
    for layer, expected in [
        ("buildings_frequency", len(city.buildings)),
        ("roads_frequency", len(city.roads)),
        ("population_frequency", len(city.population)),
        ("hospitals", len(city.hospitals)),
    ]:
        frame = gpd.read_file(path, layer=layer)
        assert len(frame) == expected
        assert frame.crs is not None
        assert frame.crs.to_string() == city.crs


def test_spatial_frequencies_are_written_as_probabilities(city, run, config, tmp_path):
    path = write_monte_carlo_outputs(city, run, tmp_path, config)["monte_carlo_gpkg"]
    buildings = gpd.read_file(path, layer="buildings_frequency")
    assert buildings["collapse_frequency"].between(0.0, 1.0).all()
    roads = gpd.read_file(path, layer="roads_frequency")
    assert roads["closed_frequency"].between(0.0, 1.0).all()


def test_outputs_stay_small(city, run, config, tmp_path):
    written = write_monte_carlo_outputs(city, run, tmp_path, config)
    for label, path in written.items():
        assert path.stat().st_size < 5_000_000, f"{label} unexpectedly large"
