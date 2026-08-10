"""Phase 2, smoke test 2 — GeoPandas / Shapely.

The vector path: build a GeoDataFrame, assign a CRS, reproject, write a
GeoPackage, read it back with CRS and attributes intact. This is the exact
mechanism `mre.outputs.write_city_layers` will use to hand results to Blender
and QGIS, so it is worth proving before anything depends on it.
"""

from __future__ import annotations

import mre  # noqa: F401  -- MUST be imported before geopandas/pyogrio; see conftest

import geopandas as gpd  # noqa: E402
import pytest  # noqa: E402
from shapely.geometry import LineString, Point, Polygon  # noqa: E402

from mre.config import load_config  # noqa: E402

CRS = load_config()["project"]["crs"]


@pytest.fixture
def buildings() -> gpd.GeoDataFrame:
    """Three square footprints, the shape the synthetic city will produce."""
    squares = [
        Polygon([(e, n), (e + 20, n), (e + 20, n + 20), (e, n + 20)])
        for e, n in [(660_000, 4_540_000), (660_100, 4_540_050), (660_200, 4_540_100)]
    ]
    return gpd.GeoDataFrame(
        {"building_id": ["B0001", "B0002", "B0003"], "floors": [4, 8, 2]},
        geometry=squares,
        crs=CRS,
    )


def test_geodataframe_carries_crs(buildings):
    assert buildings.crs is not None
    assert buildings.crs.to_string() == CRS
    assert buildings.crs.is_projected


def test_areas_are_metres_squared(buildings):
    """20 m squares => 400 m^2. If this fails the CRS is not metric and every
    distance-based parameter is meaningless."""
    assert buildings.area.tolist() == pytest.approx([400.0] * 3)


def test_reprojection_round_trip(buildings):
    geographic = buildings.to_crs("EPSG:4326")
    # Marmara-plausible degree range; a sanity check, not a precision claim.
    minx, miny, maxx, maxy = geographic.total_bounds
    assert 26 < minx < 32 and 26 < maxx < 32
    assert 39 < miny < 42 and 39 < maxy < 42

    back = geographic.to_crs(CRS)
    assert back.geometry.iloc[0].centroid.x == pytest.approx(
        buildings.geometry.iloc[0].centroid.x, abs=0.01
    )


def test_geopackage_write_and_read(tmp_path, buildings):
    path = tmp_path / "smoke_city.gpkg"
    buildings.to_file(path, layer="buildings", driver="GPKG")
    assert path.is_file()

    read_back = gpd.read_file(path, layer="buildings")
    assert len(read_back) == 3
    assert read_back.crs.to_string() == CRS
    assert read_back["building_id"].tolist() == ["B0001", "B0002", "B0003"]
    assert read_back["floors"].tolist() == [4, 8, 2]


def test_multi_layer_geopackage(tmp_path, buildings):
    """One file, several layers — how city layers will be shipped."""
    path = tmp_path / "multi.gpkg"
    hospitals = gpd.GeoDataFrame(
        {"hospital_id": ["H01"]}, geometry=[Point(660_150, 4_540_075)], crs=CRS
    )
    roads = gpd.GeoDataFrame(
        {"road_id": ["R0001"]},
        geometry=[LineString([(660_000, 4_540_000), (660_200, 4_540_100)])],
        crs=CRS,
    )
    buildings.to_file(path, layer="buildings", driver="GPKG")
    hospitals.to_file(path, layer="hospitals", driver="GPKG")
    roads.to_file(path, layer="roads", driver="GPKG")

    for layer, expected in [("buildings", 3), ("hospitals", 1), ("roads", 1)]:
        assert len(gpd.read_file(path, layer=layer)) == expected


def test_geojson_write_and_read(tmp_path, buildings):
    """The Blender handoff format. GeoJSON is defined in WGS84."""
    path = tmp_path / "smoke.geojson"
    buildings.to_crs("EPSG:4326").to_file(path, driver="GeoJSON")
    read_back = gpd.read_file(path)
    assert len(read_back) == 3
    assert read_back.crs.to_epsg() == 4326
