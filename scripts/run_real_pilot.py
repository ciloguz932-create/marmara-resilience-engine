"""Run the MRE engine on the real Zeytinburnu pilot (offline from cached OSM).

    .venv\\Scripts\\python.exe scripts\\run_real_pilot.py -n 200

REAL-DATA PILOT — RESEARCH PROTOTYPE, NOT A REAL-WORLD PREDICTION. Reads the
cached raw OSM layers (run scripts/fetch_real_pilot.py first), normalises them to
EPSG:32635, adapts them into the engine's models, validates the build, runs the
scenario + Monte Carlo + prototype intervention comparison, and writes the
real-pilot outputs. Geometry is real (OSM, ODbL); fragility and intensity are
prototype values applied to real geometry; structural attributes and hospital
capacities are UNKNOWN; demand is a uniform proxy. See docs/SCIENTIFIC_ASSUMPTIONS.md §10.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mre  # noqa: E402  -- MUST precede pandas/geopandas; see docs/ARCHITECTURE.md
import geopandas as gpd  # noqa: E402
from mre.config import load_config  # noqa: E402
from mre.optimization import compare_portfolios  # noqa: E402
from mre.outputs import summarise_monte_carlo, write_intervention_outputs  # noqa: E402
from mre.real.adapter import build_real_city  # noqa: E402
from mre.real.normalize import normalize_layers  # noqa: E402
from mre.real.outputs import (  # noqa: E402
    write_real_city_layers,
    write_real_hazard_layer,
    write_real_results_layer,
    write_real_scenario_json,
)
from mre.real.pilot import ZEYTINBURNU, build_pilot_provenance, scenario_provenance  # noqa: E402
from mre.real.scenario import real_pilot_config  # noqa: E402
from mre.real.validation import validate_real_pilot  # noqa: E402
from mre.simulation import run_monte_carlo, run_scenario  # noqa: E402


def load_raw(raw_dir: Path):
    paths = {
        "buildings": raw_dir / "buildings_osm_4326.geojson",
        "roads": raw_dir / "roads_osm_4326.geojson",
        "hospitals": raw_dir / "hospitals_osm_4326.geojson",
    }
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise SystemExit(
            "Raw OSM layers not found:\n  " + "\n  ".join(missing)
            + "\nRun scripts/fetch_real_pilot.py first (requires network)."
        )
    return (
        gpd.read_file(paths["buildings"]),
        gpd.read_file(paths["roads"]),
        gpd.read_file(paths["hospitals"]),
    )


def build_and_validate(out_dir: Path, config, seed: int):
    """Shared build+validate path used by this script and demo_real.py."""
    raw_b, raw_r, raw_h = load_raw(out_dir / "raw")
    bn, rn, hn, cleaning = normalize_layers(raw_b, raw_r, raw_h)
    real_city = build_real_city(bn, rn, hn, config, seed=seed)

    acquired_at = "unknown"
    prov_path = out_dir / "provenance.json"
    if prov_path.is_file():
        acquired_at = json.loads(prov_path.read_text(encoding="utf-8")).get("generated_at", "unknown")
    provenance = build_pilot_provenance(len(bn), len(rn), len(hn), acquired_at)
    provenance.write(prov_path)

    validation = validate_real_pilot(real_city, (bn, rn, hn), provenance, seed=seed)
    validation_path = out_dir / "validation.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        json.dumps({**validation.to_dict(), "cleaning": cleaning.to_dict()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return real_city, validation, validation_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-n", "--n-simulations", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "real_pilot")
    args = parser.parse_args()

    config = real_pilot_config(load_config())
    seed = args.seed if args.seed is not None else config["random"]["seed"]

    print(f"mre {mre.__version__} | REAL-DATA PILOT — RESEARCH PROTOTYPE, NOT A PREDICTION")
    print(f"pilot: {ZEYTINBURNU.study_area}")
    print(f"scenario: {scenario_provenance()['name']} (proxy intensity field)")

    print("\n[1/6] normalise + adapt + validate ...")
    real_city, validation, validation_path = build_and_validate(args.out_dir, config, seed)
    city = real_city.city
    print(
        f"      {len(city.buildings)} real buildings, {len(city.roads)} road links, "
        f"{len(city.hospitals)} hospitals, {len(city.population)} demand-proxy units"
    )
    print(f"      validation: passed={validation.passed}  warnings={validation.n_warnings}  -> {validation_path}")
    for c in validation.checks:
        if c["status"] != "pass":
            print(f"        [{c['status'].upper()}] {c['check']}: {c['detail']}")
    if not validation.passed:
        print("\nVALIDATION FAILED — refusing to present results. See validation.json.", file=sys.stderr)
        return 1

    print("[2/6] scenario realisation 0 ...")
    scenario_result = run_scenario(city, config, realisation=0, seed=seed)
    dmg = scenario_result.summary()["damage_state_counts"]
    print(f"      damage (prototype fragility on real geometry): {dmg}")

    print(f"[3/6] Monte Carlo (n={args.n_simulations}) ...")
    started = time.perf_counter()
    mc_result = run_monte_carlo(city, config, n_simulations=args.n_simulations, seed=seed)
    print(f"      {mc_result.n_simulations} realisations in {time.perf_counter()-started:.1f}s")

    print(f"[4/6] prototype intervention comparison (n={args.n_simulations}) ...")
    comparison = compare_portfolios(city, config, seed=seed, n_simulations=args.n_simulations)
    best = comparison.best
    if best is not None:
        verb = "improves" if comparison.best_improves_on_baseline else "does NOT improve"
        print(f"      best: {best.portfolio_id} ({verb}; mean benefit {best.primary_benefit_mean:+.2f} unreachable demand-proxy)")

    print("[5/6] writing real-pilot outputs ...")
    written: dict[str, Path] = {}
    written.update(write_real_city_layers(real_city, args.out_dir))
    written.update(write_real_hazard_layer(real_city, scenario_result, args.out_dir))
    written.update(write_real_results_layer(real_city, mc_result, args.out_dir))
    written.update(write_real_scenario_json(real_city, scenario_result, config, scenario_provenance(), args.out_dir))
    mc_summary_path = args.out_dir / "monte_carlo_summary.json"
    mc_summary_path.write_text(
        json.dumps(summarise_monte_carlo(mc_result, config), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    written["monte_carlo_summary_json"] = mc_summary_path
    written.update(write_intervention_outputs(city, comparison, args.out_dir / "interventions"))
    for label, path in written.items():
        print(f"      {label:<28} {path}")

    print("[6/6] done.")
    print(
        "\nREAL geometry (OSM, ODbL) + PROTOTYPE physics. UNKNOWN structural "
        "attributes/capacities; UNIFORM demand proxy; scenario-derived intensity "
        "proxy. NOT a real damage/loss prediction. Next: scripts/demo_real.py for the Blender scene."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
