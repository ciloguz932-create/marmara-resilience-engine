"""Phase 2, smoke test 1 — Python + GDAL.

Exercises the raster path end to end: create, georeference, write, reopen,
read back. A hazard raster (ShakeMap, DEM) would travel exactly this route, so
a failure here is a failure of the whole future real-hazard pathway.
"""

from __future__ import annotations

import mre  # noqa: F401  -- MUST be imported before osgeo; see conftest

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from osgeo import gdal, osr  # noqa: E402

from mre.config import load_config  # noqa: E402

gdal.UseExceptions()


def test_gdal_version_is_the_qgis_bundled_one():
    assert gdal.__version__.startswith("3.13")


def test_gdal_geotiff_driver_available():
    assert gdal.GetDriverByName("GTiff") is not None


def test_gdal_data_files_resolve():
    """Without GDAL_DATA, spatial-reference lookups degrade silently."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32635)
    assert srs.IsProjected()
    assert "UTM" in srs.ExportToWkt() or "utm" in srs.ExportToProj4()


def test_write_and_read_georeferenced_raster(tmp_path):
    """Round-trip a small synthetic intensity-like grid."""
    path = tmp_path / "smoke.tif"
    width, height, pixel_m = 16, 12, 100.0
    origin_e, origin_n = 660_000.0, 4_540_000.0

    values = np.linspace(0.0, 1.0, width * height, dtype=np.float32).reshape(height, width)

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), width, height, 1, gdal.GDT_Float32)
    # North-up: negative north-south pixel size.
    dataset.SetGeoTransform((origin_e, pixel_m, 0.0, origin_n, 0.0, -pixel_m))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32635)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(values)
    dataset = None  # flush to disk

    assert path.is_file()

    reopened = gdal.Open(str(path))
    assert (reopened.RasterXSize, reopened.RasterYSize) == (width, height)

    read_back = reopened.GetRasterBand(1).ReadAsArray()
    np.testing.assert_allclose(read_back, values, rtol=1e-6)

    transform = reopened.GetGeoTransform()
    assert transform[0] == pytest.approx(origin_e)
    assert transform[5] == pytest.approx(-pixel_m)

    written_srs = osr.SpatialReference(wkt=reopened.GetProjection())
    assert written_srs.GetAuthorityCode(None) == "32635"
    reopened = None


def test_configured_crs_is_loadable_by_gdal():
    epsg = int(load_config()["project"]["crs"].split(":")[1])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    # Metres, because adjacency radii and travel times depend on it.
    assert srs.GetLinearUnitsName().lower().startswith("met")
