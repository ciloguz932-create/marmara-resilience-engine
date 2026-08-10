"""Run one synthetic scenario end to end and write outputs.

    .venv\\Scripts\\python.exe scripts\\run_scenario.py --seed 20260810 --out-dir outputs

SYNTHETIC PROTOTYPE. The city is invented; the fragility parameters are
invented; intensity is a dimensionless proxy. Nothing here describes a real
place. See docs/SCIENTIFIC_ASSUMPTIONS.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mre  # noqa: E402  -- MUST precede pandas/geopandas; see docs/ARCHITECTURE.md
from mre.config import load_config  # noqa: E402
from mre.data import build_synthetic_city  # noqa: E402
from mre.outputs import write_city_layers, write_scenario_outputs  # noqa: E402
from mre.simulation import run_scenario  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="master seed")
    parser.add_argument("--realisation", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--no-write", action="store_true", help="compute only")
    args = parser.parse_args()

    config = load_config()
    seed = args.seed if args.seed is not None else config["random"]["seed"]

    print(f"mre {mre.__version__} | SYNTHETIC PROTOTYPE -- not a validated model")
    print(f"seed={seed} realisation={args.realisation}")

    city = build_synthetic_city(config, seed=seed)
    print(
        f"city: {len(city.buildings)} buildings, {len(city.roads)} links, "
        f"{len(city.hospitals)} hospitals, {len(city.population)} population units, "
        f"crs={city.crs}"
    )

    result = run_scenario(city, config, realisation=args.realisation, seed=seed)
    summary = result.summary()

    intensity = summary["intensity"]
    print(
        f"intensity proxy: min={intensity['min']:.4f} "
        f"mean={intensity['mean']:.4f} max={intensity['max']:.4f}"
    )
    print(f"damage states: {summary['damage_state_counts']}")
    print(f"mean expected damage index: {summary['mean_expected_damage_index']:.4f}")
    print(f"links: {summary['links']}")
    print(
        f"network after: connected={summary['network_connected_after']} "
        f"components={summary['n_components_after']}"
    )

    if not args.no_write:
        written = write_city_layers(city, args.out_dir)
        written.update(write_scenario_outputs(city, result, args.out_dir))
        print("\nwritten:")
        for label, path in written.items():
            print(f"  {label:<26} {path}  ({path.stat().st_size:,} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
