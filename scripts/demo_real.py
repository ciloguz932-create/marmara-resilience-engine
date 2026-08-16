"""ONE-COMMAND REAL-DATA PILOT DEMO: validate → simulate → intervene → Blender.

    .venv\\Scripts\\python.exe scripts\\demo_real.py

Runs the whole real Zeytinburnu pilot and produces the 3D scene:

    fetch (cached OSM) → normalise → adapt → VALIDATE → scenario + Monte Carlo
    + prototype intervention → real-pilot GIS/JSON → real_scene.json
    → blender/build_real_scene.py → outputs/real_pilot/blender/marmara_real_pilot.blend (+ preview.png)

REAL-DATA PILOT — RESEARCH PROTOTYPE, NOT A REAL-WORLD PREDICTION. Real geometry
(OSM, ODbL); prototype fragility/intensity on real geometry; UNKNOWN structural
attributes and hospital capacities; UNIFORM demand proxy. See
docs/SCIENTIFIC_ASSUMPTIONS.md §10. If the raw OSM layers are missing, run
scripts/fetch_real_pilot.py first (needs network).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mre  # noqa: E402  -- MUST precede pandas/geopandas; see docs/ARCHITECTURE.md
from mre.config import blender_executable, load_config  # noqa: E402
from mre.optimization import compare_portfolios  # noqa: E402
from mre.outputs import summarise_monte_carlo, write_intervention_outputs  # noqa: E402
from mre.real.outputs import (  # noqa: E402
    write_real_city_layers,
    write_real_hazard_layer,
    write_real_results_layer,
    write_real_scenario_json,
)
from mre.real.pilot import ZEYTINBURNU, scenario_provenance  # noqa: E402
from mre.real.scenario import real_pilot_config  # noqa: E402
from mre.simulation import run_monte_carlo, run_scenario  # noqa: E402

# Import the shared build+validate path from the run script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_real_pilot import build_and_validate  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BLENDER_SCRIPT = REPO_ROOT / "blender" / "build_real_scene.py"

SCALE = 0.03           # metres -> Blender units
HEIGHT_SCALE = 0.03    # metres of building height -> Blender units
SIMPLIFY_TOL_M = 0.6   # footprint simplification tolerance (metres)
DISPLAY_FALLBACK_FLOORS = 3  # visual-only height where storeys are UNKNOWN

VISUAL = {
    "scale": SCALE,
    "block_gap": 18.0,
    "road_width": 0.45,
    "hospital_size": 1.1,
    "legend_step": 3.4,
    "title_size": 3.0,
    "damage_color": {
        "NONE": (0.80, 0.80, 0.78), "SLIGHT": (0.74, 0.80, 0.28),
        "MODERATE": (0.93, 0.65, 0.12), "SEVERE": (0.85, 0.28, 0.10),
        "COLLAPSE": (0.30, 0.05, 0.05),
    },
    "road_color": {"OPEN": (0.55, 0.55, 0.55), "DEGRADED": (0.95, 0.60, 0.05), "CLOSED": (0.85, 0.12, 0.12)},
    "hospital_color": (0.10, 0.45, 0.95),
    "targeted_highlight_color": (0.97, 0.90, 0.12),
}


def build_scene_json(real_city, scenario_result, comparison, out_path: Path) -> Path:
    city = real_city.city
    polys = real_city.building_polygons
    battrs = real_city.building_attributes

    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)

    def lx(x):
        return (x - minx) * SCALE

    def ly(y):
        return (y - miny) * SCALE

    best = comparison.best
    targeted_buildings = set(best.target_ids.get("BUILDING_RETROFIT", ())) if best else set()
    targeted_roads = set(best.target_ids.get("ROAD_HARDENING", ())) if best else set()

    damage_after = [s.name for s in scenario_result.damage.state_enums()]

    buildings_json = []
    for i, poly in enumerate(polys):
        simplified = poly.simplify(SIMPLIFY_TOL_M, preserve_topology=True)
        geom = simplified if simplified.geom_type == "Polygon" and not simplified.is_empty else poly
        try:
            ring = [(round(lx(x), 4), round(ly(y), 4)) for x, y in geom.exterior.coords]
        except AttributeError:
            continue
        floors_attr = battrs[i]["floors"]
        floors = floors_attr if isinstance(floors_attr, (int, float)) else DISPLAY_FALLBACK_FLOORS
        height = max(1, int(floors)) * 3.0 * HEIGHT_SCALE
        buildings_json.append(
            {
                "ring": ring,
                "height": round(height, 4),
                "damage_after": damage_after[i],
                "targeted": battrs[i]["building_id"] in targeted_buildings,
            }
        )

    link_state_after = [s.value for s in scenario_result.link_states]
    roads_json = []
    for i, link in enumerate(city.roads):
        a = city.nodes[link.from_node]
        b = city.nodes[link.to_node]
        roads_json.append(
            {
                "line": [(round(lx(a[0]), 4), round(ly(a[1]), 4)), (round(lx(b[0]), 4), round(ly(b[1]), 4))],
                "state_after": link_state_after[i],
                "targeted": real_city.road_attributes[i]["road_id"] in targeted_roads,
            }
        )

    hospitals_json = [{"x": round(lx(h.easting), 4), "y": round(ly(h.northing), 4)} for h in city.hospitals]

    intervention = {
        "has_best_portfolio": best is not None,
        "best_portfolio_id": best.portfolio_id if best else None,
        "improves_on_baseline": comparison.best_improves_on_baseline,
        "primary_benefit_mean": best.primary_benefit_mean if best else None,
        "probability_of_improvement": best.probability_of_improvement if best else None,
    }

    scene = {
        "meta": {
            "title": "MARMARA RESILIENCE ENGINE",
            "subtitle": "REAL-DATA PILOT — Zeytinburnu, İstanbul (OpenStreetMap, ODbL)",
            "disclaimer": "RESEARCH PROTOTYPE — NOT A REAL-WORLD PREDICTION. Prototype fragility/intensity on real geometry.",
            "scenario": scenario_provenance()["name"],
        },
        "visual": VISUAL,
        "extent": {"min_x": 0.0, "min_y": 0.0, "max_x": lx(maxx), "max_y": ly(maxy)},
        "buildings": buildings_json,
        "roads": roads_json,
        "hospitals": hospitals_json,
        "intervention": intervention,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scene), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-n", "--n-simulations", type=int, default=150)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "real_pilot")
    parser.add_argument("--no-blender", action="store_true")
    args = parser.parse_args()

    config = real_pilot_config(load_config())
    seed = args.seed if args.seed is not None else config["random"]["seed"]
    out_dir = args.out_dir

    print(f"mre {mre.__version__} | REAL-DATA PILOT DEMO — NOT A REAL-WORLD PREDICTION")
    print(f"pilot: {ZEYTINBURNU.study_area}")

    print("\n[1/6] normalise + adapt + VALIDATE (from cached OSM) ...")
    real_city, validation, validation_path = build_and_validate(out_dir, config, seed)
    if not validation.passed:
        print(f"VALIDATION FAILED — see {validation_path}. Refusing to build a scene.", file=sys.stderr)
        return 1
    city = real_city.city
    print(f"      {len(city.buildings)} buildings / {len(city.roads)} links / {len(city.hospitals)} hospitals — validation passed ({validation.n_warnings} warnings)")

    print("[2/6] scenario realisation 0 ...")
    scenario_result = run_scenario(city, config, realisation=0, seed=seed)
    print(f"      damage: {scenario_result.summary()['damage_state_counts']}")

    print("[3/6] Monte Carlo + intervention comparison ...")
    started = time.perf_counter()
    mc_result = run_monte_carlo(city, config, n_simulations=args.n_simulations, seed=seed)
    comparison = compare_portfolios(city, config, seed=seed, n_simulations=args.n_simulations)
    best = comparison.best
    if best is not None:
        verb = "improves" if comparison.best_improves_on_baseline else "no improvement"
        print(f"      best: {best.portfolio_id} ({verb}, {best.primary_benefit_mean:+.1f}) [{time.perf_counter()-started:.1f}s]")

    print("[4/6] writing real-pilot outputs ...")
    write_real_city_layers(real_city, out_dir)
    write_real_hazard_layer(real_city, scenario_result, out_dir)
    write_real_results_layer(real_city, mc_result, out_dir)
    write_real_scenario_json(real_city, scenario_result, config, scenario_provenance(), out_dir)
    (out_dir / "monte_carlo_summary.json").write_text(
        json.dumps(summarise_monte_carlo(mc_result, config), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_intervention_outputs(city, comparison, out_dir / "interventions")

    print("[5/6] building real scene description ...")
    blender_dir = out_dir / "blender"
    scene_json = build_scene_json(real_city, scenario_result, comparison, blender_dir / "real_scene.json")
    print(f"      {scene_json}")

    if args.no_blender:
        print("\n--no-blender: stopping after scene JSON + GIS outputs.")
        return 0

    blender = blender_executable(config)
    if blender is None:
        print(f"\nBlender not configured; scene JSON written to {scene_json}. Skipping render.")
        return 0

    blend_path = blender_dir / "marmara_real_pilot.blend"
    print(f"[6/6] building Blender scene with {blender} ...")
    result = subprocess.run(
        [str(blender), "--background", "--python", str(BLENDER_SCRIPT), "--",
         str(scene_json.resolve()), str(blend_path.resolve())],
        capture_output=True, text=True,
    )
    print(result.stdout[-3000:])
    if result.returncode != 0:
        print(result.stderr[-3000:], file=sys.stderr)
        return 1

    preview = blend_path.with_suffix(".png")
    print("\n" + "=" * 72)
    print("REAL-DATA PILOT DEMO READY  —  RESEARCH PROTOTYPE, NOT A PREDICTION")
    print("=" * 72)
    print(f"  Blender scene : {blend_path.resolve()}")
    if preview.is_file():
        print(f"  Preview image : {preview.resolve()}")
    print(f"  GIS/JSON      : {out_dir.resolve()}")
    print(f"  Validation    : {validation_path.resolve()}")
    print(f"  Provenance    : {(out_dir / 'provenance.json').resolve()}")
    print("\nBASELINE → EARTHQUAKE → +INTERVENTION on the real Zeytinburnu fabric. "
          "Real geometry (OSM, ODbL); prototype physics. See docs/SCIENTIFIC_ASSUMPTIONS.md §10.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
