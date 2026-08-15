"""Run the Phase 5 prototype intervention comparison and write outputs.

    .venv\\Scripts\\python.exe scripts\\run_interventions.py --seed 20260810 -n 1000 --out-dir outputs\\interventions

PROTOTYPE INTERVENTION OPTIMIZATION. The city is invented; the fragility, road,
and intervention parameters are invented; intensity is a dimensionless proxy.
The primary objective is expected unreachable population; portfolios are compared
under common random numbers (same seed, same realisation streams). Percentiles
are empirical quantiles, NOT confidence intervals. Nothing here describes a real
place, and the ranking has no optimality guarantee. See
docs/SCIENTIFIC_ASSUMPTIONS.md section 7.
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
from mre.optimization import compare_portfolios  # noqa: E402
from mre.outputs import write_intervention_outputs  # noqa: E402


def _fmt(outcome) -> str:
    sec = outcome.secondary
    return (
        f"  {outcome.portfolio_id:<18} cost={outcome.portfolio_cost:>10,.0f}  "
        f"benefit(mean)={outcome.primary_benefit_mean:>+7.2f}  "
        f"p05={outcome.primary_benefit_p05:>+5.0f} p95={outcome.primary_benefit_p95:>+5.0f}  "
        f"P(improve)={outcome.probability_of_improvement:>4.2f}  "
        f"d_n_closed={sec['delta_n_closed']:>+6.2f}  "
        f"d_n_collapse={sec['delta_n_collapse']:>+7.2f}  "
        f"d_pressure={sec['delta_service_pressure']:>+8.2f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None, help="master seed")
    parser.add_argument("-n", "--n-simulations", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/interventions"))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    config = load_config()
    seed = args.seed if args.seed is not None else config["random"]["seed"]
    n_simulations = (
        args.n_simulations
        if args.n_simulations is not None
        else config["monte_carlo"]["n_simulations"]
    )
    budget = config["interventions"]["budget"]

    print(f"mre {mre.__version__} | PROTOTYPE INTERVENTION OPTIMIZATION -- not validated")
    print(f"seed={seed} n_simulations={n_simulations} budget={budget:,.0f}")

    city = build_synthetic_city(config, seed=seed)
    print(
        f"city: {len(city.buildings)} buildings, {len(city.roads)} links, "
        f"{len(city.hospitals)} hospitals, {len(city.population)} population units"
    )

    started = time.perf_counter()
    comparison = compare_portfolios(
        city, config, seed=seed, n_simulations=n_simulations
    )
    elapsed = time.perf_counter() - started

    baseline = comparison.baseline_outcome
    print(
        f"\nevaluated baseline + {len(comparison.ranked_outcomes)} feasible "
        f"portfolios in {elapsed:.1f}s"
    )
    print(
        f"validation split: {comparison.n_selection} selection / "
        f"{comparison.n_evaluation} evaluation realisations "
        f"(selection_fraction={comparison.selection_fraction:.2f}); "
        "targets selected out-of-sample, metrics reported on the evaluation set"
    )
    print(
        "primary objective: EXPECTED UNREACHABLE POPULATION "
        f"(baseline mean = {baseline.primary_objective_baseline:.2f})"
    )

    print("\n-- portfolios, ranked best-first by primary benefit --")
    for outcome in comparison.ranked_outcomes:
        print(_fmt(outcome))

    best = comparison.best
    if best is not None:
        verb = "improves on" if comparison.best_improves_on_baseline else "does NOT improve"
        print(
            f"\nbest feasible portfolio: {best.portfolio_id}  "
            f"(cost {best.portfolio_cost:,.0f}; {verb} the baseline primary objective)"
        )
        print(
            "  reduces expected unreachable population from "
            f"{best.primary_objective_baseline:.2f} to "
            f"{best.primary_objective_after:.2f} "
            f"(mean benefit {best.primary_benefit_mean:+.2f}, "
            f"in {best.probability_of_improvement:.0%} of realisations)"
        )

    if not args.no_write:
        written = write_intervention_outputs(city, comparison, args.out_dir)
        print("\nwritten:")
        for label, path in written.items():
            print(f"  {label:<32} {path}  ({path.stat().st_size:,} bytes)")

    print(
        "\nPROTOTYPE: invented costs and effect multipliers; ranking under the "
        "prototype model only, no optimality guarantee. HOSPITAL_SUPPORT moves the "
        "service-pressure metric only, not the primary objective."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
