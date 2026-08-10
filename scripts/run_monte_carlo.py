"""Run a Monte Carlo experiment and write distribution outputs.

    .venv\\Scripts\\python.exe scripts\\run_monte_carlo.py --seed 20260810 -n 1000 --out-dir outputs

SYNTHETIC PROTOTYPE. The city is invented; the fragility parameters are
invented; intensity is a dimensionless proxy. Percentiles are empirical
quantiles of the simulated distribution, NOT confidence intervals. Running more
realisations narrows the bands without making the model more valid. See
docs/SCIENTIFIC_ASSUMPTIONS.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mre  # noqa: E402  -- MUST precede pandas/geopandas; see docs/ARCHITECTURE.md
from mre.config import load_config  # noqa: E402
from mre.data import build_synthetic_city  # noqa: E402
from mre.outputs import summarise_monte_carlo, write_monte_carlo_outputs  # noqa: E402
from mre.simulation import run_monte_carlo  # noqa: E402


def _band(label: str, stats: dict, unit: str = "") -> str:
    if stats["mean"] is None:
        return f"  {label:<32} (no finite values)"
    return (
        f"  {label:<32} mean={stats['mean']:>10.3f}{unit}  "
        f"sd={stats['std']:>8.3f}  "
        f"p05={stats['p05']:>9.3f}  p50={stats['p50']:>9.3f}  p95={stats['p95']:>9.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="master seed")
    parser.add_argument("-n", "--n-simulations", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config = load_config()
    seed = args.seed if args.seed is not None else config["random"]["seed"]
    n_simulations = (
        args.n_simulations
        if args.n_simulations is not None
        else config["monte_carlo"]["n_simulations"]
    )

    print(f"mre {mre.__version__} | SYNTHETIC PROTOTYPE -- not a validated model")
    print(f"seed={seed} n_simulations={n_simulations}")

    city = build_synthetic_city(config, seed=seed)
    print(
        f"city: {len(city.buildings)} buildings, {len(city.roads)} links, "
        f"{len(city.hospitals)} hospitals, {len(city.population)} population units"
    )

    started = time.perf_counter()
    result = run_monte_carlo(city, config, n_simulations=n_simulations, seed=seed)
    elapsed = time.perf_counter() - started
    print(f"ran {result.n_simulations} realisations in {elapsed:.1f}s")

    summary = summarise_monte_carlo(result, config)
    baseline = summary["baseline"]

    print("\n-- pre-earthquake baseline (deterministic) --")
    print(f"  mean travel time            {baseline['mean_travel_time_min']:.3f} min")
    print(f"  median travel time          {baseline['median_travel_time_min']:.3f} min")
    print(f"  population reachable        {baseline['population_reachable']:,}")
    print(f"  population unreachable      {baseline['population_unreachable']:,}")
    print(f"  hospital utilisation        {baseline['hospital_utilisation']}")

    print("\n-- post-earthquake distributions --")
    for name, unit in [
        ("n_collapse", ""),
        ("mean_expected_damage_index", ""),
        ("n_closed", ""),
        ("n_degraded", ""),
        ("mean_travel_time_min", ""),
        ("median_travel_time_min", ""),
        ("population_unreachable", ""),
        ("population_reachable", ""),
        ("n_disrupted_routes", ""),
        ("delta_mean_travel_time_min", ""),
        ("n_components_after", ""),
    ]:
        print(_band(name, summary["metrics"][name], unit))

    print("\n-- hospital utilisation (post-event) --")
    for hospital_id, stats in summary["hospitals"].items():
        utilisation = stats["utilisation"]
        print(
            f"  {hospital_id}  baseline={stats['baseline_utilisation']:.3f}  "
            f"mean={utilisation['mean']:.3f}  "
            f"p05={utilisation['p05']:.3f}  p50={utilisation['p50']:.3f}  "
            f"p95={utilisation['p95']:.3f}"
        )

    if not args.no_write:
        written = write_monte_carlo_outputs(city, result, args.out_dir, config)
        print("\nwritten:")
        for label, path in written.items():
            print(f"  {label:<26} {path}  ({path.stat().st_size:,} bytes)")

    print(f"\n{summary['uncertainty_note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
