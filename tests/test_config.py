"""Phase 1: config loading and the geo-env bootstrap."""

from __future__ import annotations

import os

from mre.config import DEFAULT_CONFIG, _deep_merge, bootstrap_geo_env, load_config


def test_default_config_exists_and_loads():
    assert DEFAULT_CONFIG.is_file()
    config = load_config()
    for section in (
        "project",
        "environment",
        "random",
        "synthetic_city",
        "hazard",
        "damage",
        "monte_carlo",
        "interventions",
    ):
        assert section in config, f"missing [{section}]"


def test_crs_is_projected_metre_based():
    """Distances, travel times, and adjacency radii are all in metres."""
    assert load_config()["project"]["crs"] == "EPSG:32635"


def test_seed_is_configured():
    assert isinstance(load_config()["random"]["seed"], int)


def test_monte_carlo_default_is_1000():
    assert load_config()["monte_carlo"]["n_simulations"] == 1000


def test_damage_states_are_the_five_prototype_states():
    assert load_config()["damage"]["states"] == [
        "NONE",
        "SLIGHT",
        "MODERATE",
        "SEVERE",
        "COLLAPSE",
    ]


def test_fragility_medians_are_monotonic():
    """Medians must increase with severity or the difference-of-exceedance
    formulation yields negative probabilities."""
    medians = load_config()["damage"]["median_intensity"]
    ordered = [medians[s] for s in ("SLIGHT", "MODERATE", "SEVERE", "COLLAPSE")]
    assert ordered == sorted(ordered)
    assert all(v > 0 for v in ordered)


def test_bootstrap_sets_gdal_and_proj_data():
    applied = bootstrap_geo_env(force=True)
    assert "GDAL_DATA" in applied
    assert "PROJ_DATA" in applied
    assert os.environ["PROJ_LIB"] == os.environ["PROJ_DATA"]


def test_bootstrap_is_idempotent():
    assert bootstrap_geo_env() == bootstrap_geo_env()


def test_bootstrap_makes_geopandas_crs_aware():
    """Regression guard for the pyarrow/PROJ DLL clash found in Phase 2.

    If PROJ's DLLs are not bound first, geopandas sets HAS_PYPROJ = False and
    silently drops every CRS instead of raising -- the worst possible failure
    mode for a project whose distances must be in metres.
    """
    import geopandas._compat

    assert geopandas._compat.HAS_PYPROJ, "PROJ DLLs lost to another library"


def test_deep_merge_preserves_unrelated_keys():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    assert _deep_merge(base, {"a": {"y": 99}}) == {"a": {"x": 1, "y": 99}, "b": 3}
