"""Phase 2, smoke test 3 — synthetic geometry creation.

Proves the geometric primitives the synthetic city will be built from, without
implementing the city itself (that is Phase 3). Specifically:

  - seeded, reproducible random point placement in a bounded extent
  - footprint construction from points
  - a regular road grid with metrically correct edge lengths
  - point-to-line distance, the mechanism behind hazard decay from a line
    source and behind road/building adjacency

No science here — just the geometry those stages will stand on.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, Point, box
from shapely.strtree import STRtree

import mre  # noqa: F401
from mre.config import load_config

CONFIG = load_config()
EXTENT_X, EXTENT_Y = CONFIG["synthetic_city"]["extent_m"]
ORIGIN_E = CONFIG["synthetic_city"]["origin_easting"]
ORIGIN_N = CONFIG["synthetic_city"]["origin_northing"]


def _random_points(seed: int, n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eastings = ORIGIN_E + rng.uniform(0, EXTENT_X, n)
    northings = ORIGIN_N + rng.uniform(0, EXTENT_Y, n)
    return np.column_stack([eastings, northings])


def test_point_placement_is_reproducible():
    """Same seed, same city. This is the reproducibility guarantee in miniature."""
    np.testing.assert_array_equal(_random_points(20260810), _random_points(20260810))
    assert not np.array_equal(_random_points(1), _random_points(2))


def test_points_fall_inside_the_study_extent():
    coords = _random_points(20260810)
    extent = box(ORIGIN_E, ORIGIN_N, ORIGIN_E + EXTENT_X, ORIGIN_N + EXTENT_Y)
    assert all(extent.contains(Point(e, n)) for e, n in coords)


def test_footprints_from_points_have_expected_area():
    """Square footprints centred on building points, sized by a half-width."""
    half_width = 7.5
    coords = _random_points(20260810, n=20)
    footprints = [
        box(e - half_width, n - half_width, e + half_width, n + half_width) for e, n in coords
    ]
    assert all(f.area == pytest.approx((2 * half_width) ** 2) for f in footprints)
    assert all(f.is_valid for f in footprints)


def test_regular_road_grid_geometry():
    """Grid nodes and edges, with lengths in metres."""
    nx_cells, ny_cells = CONFIG["synthetic_city"]["road_grid"]
    dx, dy = EXTENT_X / nx_cells, EXTENT_Y / ny_cells

    nodes = {
        (i, j): (ORIGIN_E + i * dx, ORIGIN_N + j * dy)
        for i in range(nx_cells + 1)
        for j in range(ny_cells + 1)
    }
    assert len(nodes) == (nx_cells + 1) * (ny_cells + 1)

    horizontal = [
        LineString([nodes[(i, j)], nodes[(i + 1, j)]])
        for i in range(nx_cells)
        for j in range(ny_cells + 1)
    ]
    vertical = [
        LineString([nodes[(i, j)], nodes[(i, j + 1)]])
        for i in range(nx_cells + 1)
        for j in range(ny_cells)
    ]
    assert len(horizontal) + len(vertical) == 2 * nx_cells * (ny_cells + 1)
    assert all(link.length == pytest.approx(dx) for link in horizontal)
    assert all(link.length == pytest.approx(dy) for link in vertical)


def test_travel_time_from_length_and_speed():
    """length_m / speed -> minutes. The RoadLink.travel_time_min convention."""
    speed_kmh = CONFIG["synthetic_city"]["roads"]["speed_kmh"]["ARTERIAL"]
    link = LineString([(ORIGIN_E, ORIGIN_N), (ORIGIN_E + 1000.0, ORIGIN_N)])
    minutes = (link.length / 1000.0) / speed_kmh * 60.0
    # 1 km at 50 km/h = 1.2 min
    assert minutes == pytest.approx(1.2)


def test_distance_to_line_source():
    """Hazard decay is driven by distance to the source line."""
    (x1, y1), (x2, y2) = CONFIG["hazard"]["source_line"]
    source = LineString([(x1, y1), (x2, y2)])

    on_line = Point((x1 + x2) / 2, (y1 + y2) / 2)
    assert source.distance(on_line) == pytest.approx(0.0, abs=1.0)

    offset = Point(on_line.x, on_line.y + 5000.0)
    assert source.distance(offset) == pytest.approx(5000.0, rel=0.02)
    assert source.distance(offset) > source.distance(on_line)


def test_spatial_index_finds_buildings_near_a_link():
    """Road disruption needs 'collapsed buildings within R of this link'.
    STRtree is how that stays tractable for 1,000+ buildings."""
    radius = CONFIG["roads"]["disruption"]["adjacency_radius_m"]
    link = LineString([(ORIGIN_E, ORIGIN_N), (ORIGIN_E + 500.0, ORIGIN_N)])

    near = Point(ORIGIN_E + 250.0, ORIGIN_N + radius / 2)
    far = Point(ORIGIN_E + 250.0, ORIGIN_N + radius * 10)
    tree = STRtree([near, far])

    hits = tree.query(link.buffer(radius))
    matched = {id(tree.geometries.take(i)) for i in hits}
    assert any(tree.geometries.take(i).equals(near) for i in hits)
    assert not any(tree.geometries.take(i).equals(far) for i in hits)
    assert len(matched) == 1
