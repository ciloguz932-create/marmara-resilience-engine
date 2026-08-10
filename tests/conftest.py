"""Shared fixtures, and the geo-env bootstrap.

pytest imports ``conftest.py`` first, so importing ``mre`` here sets
``GDAL_DATA``/``PROJ_DATA`` and binds PROJ's DLLs before geopandas, pyogrio,
pyproj, pandas, or osgeo are touched by a test module.

This matters twice over:

  - ``pyogrio`` resolves GDAL's data directory once at import and caches it;
  - ``pandas`` 3.x pulls ``pyarrow``, whose bundled native libraries shadow
    PROJ's, after which geopandas silently drops every CRS.

Import order is load-bearing here, not cosmetic. See docs/ARCHITECTURE.md.
"""

from __future__ import annotations

import mre  # noqa: F401  -- import for side effect; must be first

import copy  # noqa: E402

import pytest  # noqa: E402

from mre.config import load_config  # noqa: E402
from mre.data import build_synthetic_city  # noqa: E402

SEED = 20260810


@pytest.fixture(scope="session")
def config() -> dict:
    return load_config()


@pytest.fixture
def mutable_config(config) -> dict:
    """A deep copy, so a test can vary parameters without leaking."""
    return copy.deepcopy(config)


@pytest.fixture(scope="session")
def small_config() -> dict:
    """A reduced city: same code paths, far cheaper to build repeatedly."""
    cfg = copy.deepcopy(load_config())
    cfg["synthetic_city"]["n_buildings"] = 120
    cfg["synthetic_city"]["road_grid"] = [6, 6]
    cfg["synthetic_city"]["population"]["grid"] = [4, 4]
    return cfg


@pytest.fixture(scope="session")
def city(config):
    """The full 1,000-building city. Session-scoped: betweenness is not free."""
    return build_synthetic_city(config, seed=SEED)


@pytest.fixture(scope="session")
def small_city(small_config):
    return build_synthetic_city(small_config, seed=SEED)
