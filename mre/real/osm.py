"""OpenStreetMap acquisition for the real pilot, via the Overpass API.

Fetches real building footprints, the real road network (with OSM node-id
topology, so connectivity is the real OSM topology rather than a guess), and
real hospital locations for a bounding box. Returns GeoDataFrames in EPSG:4326
with the raw OSM attributes preserved and every absent attribute marked
``UNKNOWN`` — nothing is invented here.

Network access lives **only** in this module and is driven by scripts, never by
``import mre``. The tests never call Overpass; they exercise the parsing and
adapter logic on small in-memory fixtures.

Licence: OpenStreetMap data is © OpenStreetMap contributors, available under the
Open Database License (ODbL). Any use of the pilot outputs inherits that licence.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.polygon import orient

from mre.real import UNKNOWN

__all__ = ["OSMExtract", "OVERPASS_ENDPOINTS", "fetch_osm_extract", "overpass_query"]

OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

_USER_AGENT = "MRE-research-prototype/0.1 (academic earthquake-resilience pilot)"

# OSM highway= value -> prototype road class. A documented, conventional
# mapping; the class names are the engine's, the source values are real OSM.
HIGHWAY_TO_CLASS = {
    "motorway": "ARTERIAL", "motorway_link": "ARTERIAL",
    "trunk": "ARTERIAL", "trunk_link": "ARTERIAL",
    "primary": "ARTERIAL", "primary_link": "ARTERIAL",
    "secondary": "COLLECTOR", "secondary_link": "COLLECTOR",
    "tertiary": "COLLECTOR", "tertiary_link": "COLLECTOR",
    "residential": "LOCAL", "living_street": "LOCAL", "unclassified": "LOCAL",
    "service": "LOCAL", "road": "LOCAL",
}
# Only these highway values are ingested as vehicular road links. Footways,
# steps, cycleways, etc. are excluded — they are not part of the emergency road
# network the model reasons about.
ROUTABLE_HIGHWAYS = frozenset(HIGHWAY_TO_CLASS)


@dataclass(frozen=True, slots=True)
class OSMExtract:
    """Raw OSM geometry for one pilot bbox, in EPSG:4326.

    ``roads`` carries a ``node_ids`` column (JSON-encoded list of OSM node ids)
    so the adapter can rebuild the real OSM topology. ``bbox`` is
    (south, west, north, east).
    """

    bbox: tuple[float, float, float, float]
    buildings: gpd.GeoDataFrame
    roads: gpd.GeoDataFrame
    hospitals: gpd.GeoDataFrame
    acquired_at: str


def overpass_query(query: str, *, timeout: int = 180, retries: int = 3) -> dict[str, Any]:
    """POST an Overpass QL query, trying each endpoint with patient retries.

    Raises the last error if every endpoint fails. Overpass is a shared free
    service and is often busy (504/429); the exponential backoff handles
    transient load, not a bug. Being a polite client matters — this is public
    infrastructure.
    """
    body = ("data=" + urllib.parse.quote(query)).encode()
    last_error: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(retries):
            try:
                request = urllib.request.Request(
                    endpoint, data=body, headers={"User-Agent": _USER_AGENT}
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return json.load(response)
            except Exception as exc:  # noqa: BLE001 - report and try the next endpoint
                last_error = exc
                time.sleep(5 * (attempt + 1) ** 2)  # 5s, 20s, 45s
    raise RuntimeError(f"Overpass request failed on all endpoints: {last_error}")


def _tag(tags: dict[str, str], key: str) -> str:
    """A tag value, or ``UNKNOWN`` when the source does not carry it."""
    value = tags.get(key)
    return value if value not in (None, "") else UNKNOWN


def _levels(tags: dict[str, str]) -> Any:
    """OSM ``building:levels`` as an int if present and numeric, else ``UNKNOWN``.

    Never guessed. A building with no ``building:levels`` tag has an UNKNOWN
    storey count, full stop.
    """
    raw = tags.get("building:levels")
    if raw is None:
        return UNKNOWN
    try:
        return int(float(str(raw).split(";")[0].split(",")[0].strip()))
    except (ValueError, TypeError):
        return UNKNOWN


def _ways_and_nodes(elements: list[dict]) -> tuple[dict[int, tuple[float, float]], list[dict]]:
    """Split an Overpass element list into a node-id->(lon,lat) map and ways."""
    nodes: dict[int, tuple[float, float]] = {}
    ways: list[dict] = []
    for element in elements:
        if element["type"] == "node":
            nodes[element["id"]] = (element["lon"], element["lat"])
        elif element["type"] == "way":
            ways.append(element)
    return nodes, ways


def _way_polygon(way: dict, nodes: dict[int, tuple[float, float]]) -> Polygon | None:
    coords = [nodes[n] for n in way.get("nodes", []) if n in nodes]
    if len(coords) < 3:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    try:
        polygon = orient(Polygon(coords), sign=1.0)
        return polygon if polygon.is_valid or polygon.buffer(0).is_valid else None
    except Exception:  # noqa: BLE001
        return None


def _fetch_buildings(bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    s, w, n, e = bbox
    query = (
        f"[out:json][timeout:180];"
        f"(way[building]({s},{w},{n},{e}););(._;>;);out body;"
    )
    data = overpass_query(query)
    nodes, ways = _ways_and_nodes(data.get("elements", []))
    records = []
    for way in ways:
        polygon = _way_polygon(way, nodes)
        if polygon is None:
            continue
        tags = way.get("tags", {})
        records.append(
            {
                "osm_id": way["id"],
                "osm_type": "way",
                "building": _tag(tags, "building"),
                "building_levels": _levels(tags),
                "construction_year": UNKNOWN,   # OSM does not carry this
                "structural_type": UNKNOWN,     # OSM does not carry this
                "name": _tag(tags, "name"),
                "geometry": polygon,
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def _fetch_roads(bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    s, w, n, e = bbox
    query = (
        f"[out:json][timeout:180];"
        f"(way[highway]({s},{w},{n},{e}););(._;>;);out body;"
    )
    data = overpass_query(query)
    nodes, ways = _ways_and_nodes(data.get("elements", []))
    records = []
    for way in ways:
        tags = way.get("tags", {})
        highway = tags.get("highway")
        if highway not in ROUTABLE_HIGHWAYS:
            continue
        node_ids = [n for n in way.get("nodes", []) if n in nodes]
        if len(node_ids) < 2:
            continue
        coords = [nodes[i] for i in node_ids]
        records.append(
            {
                "osm_id": way["id"],
                "highway": highway,
                "road_class": HIGHWAY_TO_CLASS[highway],
                "name": _tag(tags, "name"),
                "maxspeed": _tag(tags, "maxspeed"),
                "oneway": _tag(tags, "oneway"),
                "node_ids": json.dumps(node_ids),
                "geometry": LineString(coords),
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def _fetch_hospitals(bbox: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    s, w, n, e = bbox
    query = (
        f"[out:json][timeout:120];"
        f"(node[amenity=hospital]({s},{w},{n},{e});"
        f" way[amenity=hospital]({s},{w},{n},{e});"
        f" relation[amenity=hospital]({s},{w},{n},{e}););"
        f"out center tags;"
    )
    data = overpass_query(query)
    records = []
    for element in data.get("elements", []):
        if element["type"] == "node":
            lon, lat = element["lon"], element["lat"]
        else:  # way / relation with 'center'
            center = element.get("center")
            if not center:
                continue
            lon, lat = center["lon"], center["lat"]
        tags = element.get("tags", {})
        records.append(
            {
                "osm_id": element["id"],
                "osm_type": element["type"],
                "name": _tag(tags, "name"),
                "emergency": _tag(tags, "emergency"),
                "capacity": _tag(tags, "capacity"),        # almost always UNKNOWN
                "beds": _tag(tags, "beds"),                # almost always UNKNOWN
                "geometry": Point(lon, lat),
            }
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def fetch_osm_extract(bbox: tuple[float, float, float, float]) -> OSMExtract:
    """Fetch buildings, roads, and hospitals for ``bbox`` (south, west, north, east)."""
    from mre.real.provenance import utc_now_iso

    buildings = _fetch_buildings(bbox)
    time.sleep(2)  # space the sub-queries — polite to shared Overpass servers
    roads = _fetch_roads(bbox)
    time.sleep(2)
    hospitals = _fetch_hospitals(bbox)
    return OSMExtract(
        bbox=bbox,
        buildings=buildings,
        roads=roads,
        hospitals=hospitals,
        acquired_at=utc_now_iso(),
    )
