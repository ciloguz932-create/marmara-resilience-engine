"""Fetch real OSM data for the Zeytinburnu pilot and write raw layers + provenance.

    .venv\\Scripts\\python.exe scripts\\fetch_real_pilot.py

REAL DATA. Buildings, roads, and hospitals come from OpenStreetMap
(© OpenStreetMap contributors, ODbL) via the Overpass API. Nothing is invented:
attributes the source does not carry are written as ``UNKNOWN``. Writes to
``outputs/real_pilot/raw/`` and a ``provenance.json`` recording every source,
its licence, acquisition date, CRS, and which attributes were missing.

This is the only step that touches the network. It is deliberately separate from
simulation so a build can be re-run offline from the cached raw layers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mre  # noqa: E402  -- MUST precede pandas/geopandas; see docs/ARCHITECTURE.md
from mre.real import UNKNOWN  # noqa: E402
from mre.real.osm import fetch_osm_extract  # noqa: E402
from mre.real.pilot import ZEYTINBURNU, build_pilot_provenance  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "real_pilot")
    args = parser.parse_args()

    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"mre {mre.__version__} | REAL-DATA PILOT fetch (OSM / Overpass)")
    print(f"pilot: {ZEYTINBURNU.study_area}")
    print(f"bbox (S,W,N,E): {ZEYTINBURNU.bbox}")
    print("fetching buildings / roads / hospitals from Overpass ...")

    extract = fetch_osm_extract(ZEYTINBURNU.bbox)
    acquired_at = extract.acquired_at

    print(
        f"  buildings: {len(extract.buildings):>6}   "
        f"roads: {len(extract.roads):>5}   hospitals: {len(extract.hospitals):>3}"
    )
    if len(extract.buildings) == 0 or len(extract.roads) == 0:
        print("ERROR: empty building or road layer — refusing to write an empty pilot.", file=sys.stderr)
        return 1

    buildings_path = raw_dir / "buildings_osm_4326.geojson"
    roads_path = raw_dir / "roads_osm_4326.geojson"
    hospitals_path = raw_dir / "hospitals_osm_4326.geojson"

    extract.buildings.to_file(buildings_path, driver="GeoJSON")
    extract.roads.to_file(roads_path, driver="GeoJSON")
    extract.hospitals.to_file(hospitals_path, driver="GeoJSON")

    provenance = build_pilot_provenance(
        n_buildings=len(extract.buildings),
        n_roads=len(extract.roads),
        n_hospitals=len(extract.hospitals),
        acquired_at=acquired_at,
    )
    provenance_path = provenance.write(args.out_dir / "provenance.json")

    ok, problems = provenance.is_complete()
    print("\nwritten:")
    for label, path in [
        ("buildings_raw", buildings_path),
        ("roads_raw", roads_path),
        ("hospitals_raw", hospitals_path),
        ("provenance", provenance_path),
    ]:
        print(f"  {label:<16} {path}  ({path.stat().st_size:,} bytes)")
    print(f"\nprovenance complete: {ok}" + ("" if ok else f"  problems={problems}"))
    print(
        f"UNKNOWN marking active — buildings with no OSM structural attributes "
        f"carry {UNKNOWN!r}, never a guessed value."
    )
    print("\nOSM data © OpenStreetMap contributors, ODbL. Next: scripts/run_real_pilot.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
