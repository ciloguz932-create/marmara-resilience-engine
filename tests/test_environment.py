"""Phase 0/1: the environment is the thing most likely to break silently."""

from __future__ import annotations

import sys

import pytest

import mre


def test_python_is_312():
    assert sys.version_info[:2] == (3, 12)


@pytest.mark.parametrize(
    "name", ["numpy", "scipy", "shapely", "pyproj", "geopandas", "networkx", "pandas"]
)
def test_required_package_imports(name):
    __import__(name)


def test_gdal_available():
    from osgeo import gdal

    assert gdal.__version__.startswith("3.")


def test_proj_data_resolves():
    """Without GDAL_DATA/PROJ_DATA this raises; it is the classic QGIS trap."""
    import pyproj

    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32635", always_xy=True)
    easting, northing = transformer.transform(29.0, 41.0)
    # Sanity, not precision: UTM 35N easting near the central meridian.
    assert 600_000 < easting < 750_000
    assert 4_400_000 < northing < 4_700_000


def test_package_version_exposed():
    assert mre.__version__
