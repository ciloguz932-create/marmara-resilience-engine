"""ONE-COMMAND DEMO: run the engine, write GIS outputs, build a Blender scene.

    .venv\\Scripts\\python.exe scripts\\demo.py

Pipeline:

    build_synthetic_city
      -> run_scenario(realisation=0)                 buildings/roads damage+state
      -> run_monte_carlo(n_simulations=1, seed=SAME)  population reachability for
                                                       that SAME realisation 0
                                                       (identical named streams)
      -> compare_portfolios(...)                      Phase 5 intervention ranking
      -> mre.outputs.write_*                          city.gpkg / scenario / MC /
                                                        intervention GIS+CSV+JSON
      -> blender_scene.json                           compact scene description
      -> blender/build_scene.py (headless, embedded Python, no bpy in this venv)
                                                       -> outputs/demo/mre_demo.blend
                                                          outputs/demo/mre_demo.png

SYNTHETIC PROTOTYPE. Every number here is invented; see
docs/SCIENTIFIC_ASSUMPTIONS.md. This script computes nothing new -- it only
runs the existing Phase 3-5 engine once and hands the result to a rendering
layer. If the Blender executable is not configured, the JSON and every
GIS/CSV/JSON artifact are still written; only the .blend/.png step is skipped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mre  # noqa: E402  -- MUST precede pandas/geopandas; see docs/ARCHITECTURE.md
from mre.config import blender_executable, load_config  # noqa: E402
from mre.data import build_synthetic_city  # noqa: E402
from mre.models import DamageState, LinkState  # noqa: E402
from mre.optimization import compare_portfolios  # noqa: E402
from mre.outputs import (  # noqa: E402
    write_city_layers,
    write_intervention_outputs,
    write_monte_carlo_outputs,
    write_scenario_outputs,
)
from mre.simulation import run_monte_carlo, run_scenario  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BLENDER_BUILD_SCRIPT = REPO_ROOT / "blender" / "build_scene.py"

# --- purely visual constants; NOT scientific parameters, so they stay here
# rather than in config/default.toml. Colours by DamageState/LinkState below
# are a readability convention (docs/SCIENTIFIC_ASSUMPTIONS.md is the source
# of truth for what the underlying categories mean).
SCALE = 0.02  # metres -> Blender units (8,000 m city -> 160 units)
BLOCK_GAP = 30.0  # Blender units between the three side-by-side city blocks

DAMAGE_COLOR = {
    "NONE": (0.82, 0.82, 0.80),
    "SLIGHT": (0.74, 0.80, 0.28),
    "MODERATE": (0.93, 0.65, 0.12),
    "SEVERE": (0.85, 0.28, 0.10),
    "COLLAPSE": (0.32, 0.05, 0.05),
}
DAMAGE_HEIGHT_MULTIPLIER = {
    "NONE": 1.0,
    "SLIGHT": 0.95,
    "MODERATE": 0.80,
    "SEVERE": 0.55,
    "COLLAPSE": 0.22,
}
ROAD_COLOR = {
    "OPEN": (0.55, 0.55, 0.55),
    "DEGRADED": (0.95, 0.60, 0.05),
    "CLOSED": (0.85, 0.12, 0.12),
}
HOSPITAL_COLOR = (0.10, 0.45, 0.95)
POPULATION_REACHABLE_COLOR = (0.65, 0.65, 0.65)
POPULATION_UNREACHABLE_COLOR = (0.85, 0.15, 0.15)
TARGETED_HIGHLIGHT_COLOR = (0.95, 0.90, 0.10)


def _building_json(city, damage_states: list[str] | None) -> list[dict[str, Any]]:
    out = []
    for index, building in enumerate(city.buildings):
        state = damage_states[index] if damage_states is not None else "NONE"
        out.append(
            {
                "id": building.building_id,
                "x": building.easting,
                "y": building.northing,
                "side_m": building.footprint_side_m,
                "height_m": building.floors * 3.0,
                "damage_state": state,
            }
        )
    return out


def _road_json(city, link_states: list[str] | None) -> list[dict[str, Any]]:
    out = []
    for index, link in enumerate(city.roads):
        x1, y1 = city.nodes[link.from_node]
        x2, y2 = city.nodes[link.to_node]
        state = link_states[index] if link_states is not None else "OPEN"
        out.append(
            {
                "id": link.road_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "road_class": link.road_class.value,
                "link_state": state,
            }
        )
    return out


def _hospital_json(city) -> list[dict[str, Any]]:
    return [
        {
            "id": h.hospital_id,
            "x": h.easting,
            "y": h.northing,
            "capacity": h.capacity,
            "emergency_capacity": h.emergency_capacity,
        }
        for h in city.hospitals
    ]


def _population_json(city, unreachable_frequency) -> list[dict[str, Any]]:
    return [
        {
            "id": p.population_id,
            "x": p.easting,
            "y": p.northing,
            "count": p.population_count,
            "unreachable": bool(unreachable_frequency[i] > 0.5),
        }
        for i, p in enumerate(city.population)
    ]


def build_scene_json(
    city, scenario_result, mc_single, comparison, out_path: Path
) -> Path:
    """Compact, JSON-safe scene description for the Blender-side builder.

    No geopandas / bpy mixing: this runs in the project .venv and reads only
    the in-memory dataclasses already produced by the engine for realisation 0
    -- ``run_scenario(realisation=0)`` and ``run_monte_carlo(n_simulations=1)``
    draw from the SAME named streams (``hazard:0``, ``damage:0``,
    ``disruption:0``), so the single-realisation damage/road state and the
    Monte-Carlo-derived population reachability describe the identical draw.
    """
    xs = [n[0] for n in city.nodes.values()]
    ys = [n[1] for n in city.nodes.values()]
    extent = {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)}

    damage_states = [s.name for s in scenario_result.damage.state_enums()]
    link_states = [s.value for s in scenario_result.link_states]
    unreachable_frequency = mc_single.unit_unreachable_frequency

    best = comparison.best
    intervention = {
        "has_best_portfolio": best is not None,
        "best_portfolio_id": best.portfolio_id if best is not None else None,
        "intervention_types": list(best.intervention_types) if best is not None else [],
        "primary_objective_baseline": (
            best.primary_objective_baseline if best is not None else None
        ),
        "primary_objective_after": best.primary_objective_after if best is not None else None,
        "primary_benefit_mean": best.primary_benefit_mean if best is not None else None,
        "probability_of_improvement": (
            best.probability_of_improvement if best is not None else None
        ),
        "improves_on_baseline": comparison.best_improves_on_baseline,
        "targeted_building_ids": list(best.target_ids.get("BUILDING_RETROFIT", ())) if best else [],
        "targeted_road_ids": list(best.target_ids.get("ROAD_HARDENING", ())) if best else [],
        "targeted_hospital_ids": list(best.target_ids.get("HOSPITAL_SUPPORT", ())) if best else [],
    }

    scene = {
        "meta": {
            "scenario_id": scenario_result.scenario_id,
            "seed": scenario_result.seed,
            "realisation": scenario_result.realisation,
            "disclaimer": (
                "SYNTHETIC PROTOTYPE -- invented city, invented parameters, "
                "dimensionless intensity proxy. Not a validated model of any "
                "real place. See docs/SCIENTIFIC_ASSUMPTIONS.md."
            ),
        },
        "visual": {
            "scale": SCALE,
            "block_gap": BLOCK_GAP,
            "damage_color": DAMAGE_COLOR,
            "damage_height_multiplier": DAMAGE_HEIGHT_MULTIPLIER,
            "road_color": ROAD_COLOR,
            "hospital_color": HOSPITAL_COLOR,
            "population_reachable_color": POPULATION_REACHABLE_COLOR,
            "population_unreachable_color": POPULATION_UNREACHABLE_COLOR,
            "targeted_highlight_color": TARGETED_HIGHLIGHT_COLOR,
        },
        "extent": extent,
        "buildings_before": _building_json(city, None),
        "buildings_after": _building_json(city, damage_states),
        "roads_before": _road_json(city, None),
        "roads_after": _road_json(city, link_states),
        "hospitals": _hospital_json(city),
        "population": _population_json(city, unreachable_frequency),
        "intervention": intervention,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scene), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None, help="master seed")
    parser.add_argument(
        "-n",
        "--n-simulations",
        type=int,
        default=300,
        help="Monte Carlo / intervention realisation count for this demo run "
        "(smaller than the config default of 1000, for a fast demo; override "
        "for a slower, tighter-banded run).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "demo")
    parser.add_argument("--no-blender", action="store_true", help="skip the Blender step")
    args = parser.parse_args()

    config = load_config()
    seed = args.seed if args.seed is not None else config["random"]["seed"]
    out_dir = args.out_dir

    print(f"mre {mre.__version__} | SYNTHETIC PROTOTYPE -- not a validated model")
    print(f"seed={seed} n_simulations={args.n_simulations} out_dir={out_dir}")

    print("\n[1/5] building synthetic city ...")
    city = build_synthetic_city(config, seed=seed)
    print(
        f"      {len(city.buildings)} buildings, {len(city.roads)} links, "
        f"{len(city.hospitals)} hospitals, {len(city.population)} population units"
    )

    print("[2/5] running scenario realisation 0 ...")
    scenario_result = run_scenario(city, config, realisation=0, seed=seed)
    print(f"      damage states: {scenario_result.summary()['damage_state_counts']}")
    print(f"      links: {scenario_result.summary()['links']}")

    print("[3/5] running 1-realisation Monte Carlo (population reachability for realisation 0) ...")
    mc_single = run_monte_carlo(city, config, n_simulations=1, seed=seed)

    print(f"[4/5] running Phase 5 intervention comparison (n={args.n_simulations}) ...")
    started = time.perf_counter()
    comparison = compare_portfolios(city, config, seed=seed, n_simulations=args.n_simulations)
    elapsed = time.perf_counter() - started
    best = comparison.best
    if best is not None:
        verb = "improves" if comparison.best_improves_on_baseline else "does NOT improve"
        print(
            f"      best portfolio: {best.portfolio_id} ({verb} on baseline; "
            f"mean benefit {best.primary_benefit_mean:+.2f} unreachable population) "
            f"[{elapsed:.1f}s]"
        )
    else:
        print(f"      no feasible portfolio found [{elapsed:.1f}s]")

    print("[5/5] writing outputs ...")
    written: dict[str, Path] = {}
    written.update(write_city_layers(city, out_dir))
    written.update(write_scenario_outputs(city, scenario_result, out_dir))
    written.update(write_monte_carlo_outputs(city, mc_single, out_dir, config))
    written.update(write_intervention_outputs(city, comparison, out_dir / "interventions"))
    for label, path in written.items():
        print(f"      {label:<28} {path}")

    scene_json = build_scene_json(
        city, scenario_result, mc_single, comparison, out_dir / "blender_scene.json"
    )
    print(f"      {'blender_scene_json':<28} {scene_json}")

    if args.no_blender:
        print("\n--no-blender: skipping scene generation.")
        return 0

    blender = blender_executable(config)
    if blender is None:
        print(
            "\nBlender executable not found (see [environment].blender_exe in "
            "config/default.toml). Scene data was written to "
            f"{scene_json}; run blender/build_scene.py manually once Blender "
            "is available. Skipping the visualization step."
        )
        return 0

    blend_path = out_dir / "mre_demo.blend"
    print(f"\nbuilding Blender scene with {blender} ...")
    result = subprocess.run(
        [
            str(blender),
            "--background",
            "--python",
            str(BLENDER_BUILD_SCRIPT),
            "--",
            str(scene_json.resolve()),
            str(blend_path.resolve()),
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout[-4000:])
    if result.returncode != 0:
        print(result.stderr[-4000:], file=sys.stderr)
        print(f"\nBlender scene build FAILED (exit {result.returncode}).", file=sys.stderr)
        return 1

    png_path = blend_path.with_suffix(".png")
    print("\n" + "=" * 72)
    print("DEMO READY")
    print("=" * 72)
    print(f"  Blender scene : {blend_path.resolve()}")
    if png_path.is_file():
        print(f"  Preview image : {png_path.resolve()}")
    print(f"  GIS/CSV/JSON  : {out_dir.resolve()}")
    print(
        "\nOpen the .blend file in Blender 5.2 to present: BEFORE / AFTER "
        "EARTHQUAKE / AFTER INTERVENTION blocks, left to right, with a "
        "damage/road-state legend. SYNTHETIC PROTOTYPE -- see "
        "docs/SCIENTIFIC_ASSUMPTIONS.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
