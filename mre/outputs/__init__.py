"""Phase 3/4/5 — machine-readable outputs and the Blender handoff boundary.

Writes GeoPackage (spatial layers), CSV (tabular / probability matrices), and
JSON (distribution summaries with seed provenance). Nothing here computes
anything scientific; this module only serialises what earlier stages already
produced. See docs/OUTPUT_SCHEMA.md for the stable column/field contract this
module must not silently break, and docs/ARCHITECTURE.md section 6a.

Every writer calls ``_require_crs`` and raises rather than emitting an
unprojected layer -- silent CRS loss is the failure mode this project can
least afford (see the pyarrow/PROJ DLL clash in docs/ARCHITECTURE.md).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from mre.models import DamageState, SyntheticCity

__all__ = [
    "DISTRIBUTION_METRICS",
    "SYNTHETIC_DISCLAIMER",
    "distribution_summary",
    "buildings_geodataframe",
    "roads_geodataframe",
    "population_geodataframe",
    "hospitals_geodataframe",
    "summarise",
    "write_city_layers",
    "write_scenario_outputs",
    "realisations_dataframe",
    "summarise_monte_carlo",
    "write_monte_carlo_outputs",
    "write_intervention_outputs",
]

SYNTHETIC_DISCLAIMER = (
    "SYNTHETIC PROTOTYPE. The city is invented, the fragility and disruption "
    "parameters are invented, and intensity is a dimensionless proxy. This is "
    "not a validated model of any real place. See docs/SCIENTIFIC_ASSUMPTIONS.md."
)

UNCERTAINTY_NOTE = (
    "Percentiles are empirical quantiles of the simulated distribution, "
    "describing Monte Carlo sampling variability within the model's own "
    "assumptions -- they are NOT confidence intervals or predictive intervals. "
    "Model (epistemic) uncertainty in the model form and the invented parameter "
    "values is NOT represented and does not shrink with more realisations. "
    "Running more realisations narrows the bands without making the model more "
    "valid; precision is not accuracy."
)

# Every RealisationRecord field that is a per-realisation scalar metric.
# Excludes the row index ("realisation") and the per-hospital dicts, which are
# summarised separately, per hospital, in `summarise_monte_carlo`.
DISTRIBUTION_METRICS = (
    "intensity_mean",
    "intensity_max",
    "n_none",
    "n_slight",
    "n_moderate",
    "n_severe",
    "n_collapse",
    "mean_expected_damage_index",
    "n_open",
    "n_degraded",
    "n_closed",
    "n_components_after",
    "network_connected_after",
    "mean_travel_time_min",
    "median_travel_time_min",
    "population_reachable",
    "population_unreachable",
    "n_disrupted_routes",
    "n_newly_unreachable_units",
    "population_newly_unreachable",
    "delta_mean_travel_time_min",
    "delta_median_travel_time_min",
)


def _require_crs(frame: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    """Raise rather than write a layer with no CRS. See module docstring."""
    if frame.crs is None:
        raise ValueError(
            f"{label}: refusing to write a layer with no CRS -- this is the "
            "pyarrow/PROJ DLL clash failure mode; see docs/ARCHITECTURE.md."
        )
    return frame


# --------------------------------------------------------------------------- #
# distribution_summary -- the single NaN/inf-safe statistics primitive
# --------------------------------------------------------------------------- #


def distribution_summary(
    values: np.ndarray, percentiles: tuple[int, ...] = (5, 25, 50, 75, 95)
) -> dict[str, Any]:
    """Mean/std/min/max/percentiles over the FINITE values only.

    Non-finite values (NaN, +-inf) are excluded and counted in ``n_excluded``,
    never propagated into a statistic. If every value is non-finite, every
    statistic is ``None`` rather than NaN, so the result stays JSON-safe.
    """
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    n = int(array.size)
    n_finite = int(finite.size)

    stats: dict[str, Any] = {
        "n": n,
        "n_finite": n_finite,
        "n_excluded": n - n_finite,
    }

    if n_finite == 0:
        stats["mean"] = None
        stats["std"] = None
        stats["min"] = None
        stats["max"] = None
        for p in percentiles:
            stats[f"p{p:02d}"] = None
        return stats

    stats["mean"] = float(np.mean(finite))
    stats["std"] = float(np.std(finite, ddof=0))
    stats["min"] = float(np.min(finite))
    stats["max"] = float(np.max(finite))
    for p in percentiles:
        stats[f"p{p:02d}"] = float(np.percentile(finite, p))
    return stats


# --------------------------------------------------------------------------- #
# GeoDataFrames -- the synthetic world, as geometry
# --------------------------------------------------------------------------- #


def buildings_geodataframe(city: SyntheticCity) -> gpd.GeoDataFrame:
    frame = gpd.GeoDataFrame(
        {
            "building_id": [b.building_id for b in city.buildings],
            "floors": [b.floors for b in city.buildings],
            "construction_year": [b.construction_year for b in city.buildings],
            "structural_type": [b.structural_type.value for b in city.buildings],
            "occupancy": [b.occupancy.value for b in city.buildings],
            "occupants": [b.occupants for b in city.buildings],
            "vulnerability_index": [b.vulnerability_index for b in city.buildings],
            "footprint_side_m": [b.footprint_side_m for b in city.buildings],
        },
        geometry=[b.footprint() for b in city.buildings],
        crs=city.crs,
    )
    return _require_crs(frame, "buildings")


def roads_geodataframe(city: SyntheticCity) -> gpd.GeoDataFrame:
    geometries = [
        LineString([city.nodes[link.from_node], city.nodes[link.to_node]])
        for link in city.roads
    ]
    frame = gpd.GeoDataFrame(
        {
            "road_id": [r.road_id for r in city.roads],
            "from_node": [r.from_node for r in city.roads],
            "to_node": [r.to_node for r in city.roads],
            "road_class": [r.road_class.value for r in city.roads],
            "length_m": [r.length_m for r in city.roads],
            "travel_time_min": [r.travel_time_min for r in city.roads],
            "criticality": [r.criticality for r in city.roads],
            "susceptibility": [r.susceptibility for r in city.roads],
        },
        geometry=geometries,
        crs=city.crs,
    )
    return _require_crs(frame, "roads")


def population_geodataframe(city: SyntheticCity) -> gpd.GeoDataFrame:
    frame = gpd.GeoDataFrame(
        {
            "population_id": [p.population_id for p in city.population],
            "population_count": [p.population_count for p in city.population],
            "node_id": [p.node_id for p in city.population],
        },
        geometry=[Point(p.easting, p.northing) for p in city.population],
        crs=city.crs,
    )
    return _require_crs(frame, "population_units")


def hospitals_geodataframe(city: SyntheticCity) -> gpd.GeoDataFrame:
    frame = gpd.GeoDataFrame(
        {
            "hospital_id": [h.hospital_id for h in city.hospitals],
            "capacity": [h.capacity for h in city.hospitals],
            "emergency_capacity": [h.emergency_capacity for h in city.hospitals],
            "node_id": [h.node_id for h in city.hospitals],
        },
        geometry=[Point(h.easting, h.northing) for h in city.hospitals],
        crs=city.crs,
    )
    return _require_crs(frame, "hospitals")


def write_city_layers(city: SyntheticCity, out_dir: Path) -> dict[str, Path]:
    """Write ``city.gpkg`` with layers buildings/roads/hospitals/population_units."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "city.gpkg"

    buildings_geodataframe(city).to_file(path, layer="buildings", driver="GPKG")
    roads_geodataframe(city).to_file(path, layer="roads", driver="GPKG")
    hospitals_geodataframe(city).to_file(path, layer="hospitals", driver="GPKG")
    population_geodataframe(city).to_file(path, layer="population_units", driver="GPKG")

    return {"city_gpkg": path}


# --------------------------------------------------------------------------- #
# Single-realisation outputs (Phase 3)
# --------------------------------------------------------------------------- #


def summarise(result: Any) -> dict[str, Any]:
    """JSON-safe summary of one ``ScenarioResult``, with the disclaimer attached."""
    data = result.summary()
    data["disclaimer"] = SYNTHETIC_DISCLAIMER
    return data


def _scenario_tag(result: Any) -> str:
    return f"{result.scenario_id}_seed{result.seed}_r{result.realisation}"


def write_scenario_outputs(
    city: SyntheticCity, result: Any, out_dir: Path
) -> dict[str, Path]:
    """Write one realisation: damage/disruption layers, probability CSV, summary JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _scenario_tag(result)

    scenario_gpkg = out_dir / f"scenario_{tag}.gpkg"

    buildings_damage = buildings_geodataframe(city)
    buildings_damage["intensity_proxy"] = result.intensity
    buildings_damage["damage_state"] = [s.name for s in result.damage.state_enums()]
    buildings_damage["expected_damage_index"] = result.damage.expected_damage_index
    _require_crs(buildings_damage, "buildings_damage").to_file(
        scenario_gpkg, layer="buildings_damage", driver="GPKG"
    )

    roads_disrupted = roads_geodataframe(city)
    roads_disrupted["link_state"] = [s.value for s in result.link_states]
    roads_disrupted["closure_probability"] = result.link_closure_probabilities
    _require_crs(roads_disrupted, "roads_disrupted").to_file(
        scenario_gpkg, layer="roads_disrupted", driver="GPKG"
    )

    probability_columns = {
        f"p_{state.name}": result.damage.probabilities[:, state.value]
        for state in DamageState
    }
    damage_probabilities_csv = out_dir / f"damage_probabilities_{tag}.csv"
    pd.DataFrame(
        {
            "building_id": [b.building_id for b in city.buildings],
            **probability_columns,
            "intensity_proxy": result.intensity,
            "expected_damage_index": result.damage.expected_damage_index,
        }
    ).to_csv(damage_probabilities_csv, index=False)

    summary_json = out_dir / f"summary_{tag}.json"
    summary_json.write_text(json.dumps(summarise(result), indent=2), encoding="utf-8")

    return {
        "scenario_gpkg": scenario_gpkg,
        "damage_probabilities_csv": damage_probabilities_csv,
        "summary_json": summary_json,
    }


# --------------------------------------------------------------------------- #
# Monte Carlo outputs (Phase 4)
# --------------------------------------------------------------------------- #


def realisations_dataframe(run: Any) -> pd.DataFrame:
    """One row per realisation, hospital dicts expanded to ``load_<id>`` /
    ``utilisation_<id>`` columns."""
    return pd.DataFrame([record.flat_row() for record in run.records])


def _mc_tag(run: Any) -> str:
    return f"{run.scenario_id}_seed{run.seed}_n{run.n_simulations}"


def summarise_monte_carlo(run: Any, config: dict[str, Any]) -> dict[str, Any]:
    """The full uncertainty-aware summary: baseline + distributions + per-hospital.

    Never propagates NaN/inf: every leaf is either a finite JSON-safe scalar or
    ``None`` where the underlying statistic is undefined (see
    ``distribution_summary``).
    """
    percentiles = tuple(config["monte_carlo"]["percentiles"])

    baseline = run.baseline
    total_emergency_capacity = int(sum(run.hospital_emergency_capacity.values()))
    population_total = baseline.population_total
    population_per_emergency_bed = (
        population_total / total_emergency_capacity if total_emergency_capacity else math.nan
    )

    baseline_block = {
        "note": (
            "Deterministic pre-earthquake baseline for this city. Every "
            "realisation is a paired comparison against this same baseline, "
            "never against another realisation."
        ),
        "mean_travel_time_min": float(baseline.mean_travel_time_min),
        "median_travel_time_min": float(baseline.median_travel_time_min),
        "population_reachable": int(baseline.population_reachable),
        "population_unreachable": int(baseline.population_unreachable),
        "hospital_load": {k: int(v) for k, v in baseline.hospital_load.items()},
        "hospital_utilisation": {
            k: float(v) for k, v in baseline.hospital_utilisation.items()
        },
        "total_emergency_capacity": total_emergency_capacity,
        "population_per_emergency_bed": float(population_per_emergency_bed),
        "utilisation_scale_note": (
            "Population and hospital emergency capacity were chosen "
            "independently in the synthetic city, giving a large "
            "population-per-bed ratio and utilisation values that can exceed "
            "1.0 by a wide margin. This is a parameter artefact, not a finding "
            "about emergency capacity -- utilisation is interpretable here only "
            "as a relative comparison between hospitals and between scenarios."
        ),
    }

    metrics = {
        name: distribution_summary(run.metric(name), percentiles)
        for name in DISTRIBUTION_METRICS
    }

    hospitals: dict[str, Any] = {}
    for hospital_id, emergency_capacity in run.hospital_emergency_capacity.items():
        utilisation_values = np.array(
            [r.hospital_utilisation.get(hospital_id, math.nan) for r in run.records],
            dtype=float,
        )
        load_values = np.array(
            [r.hospital_load.get(hospital_id, 0) for r in run.records], dtype=float
        )
        hospitals[hospital_id] = {
            "emergency_capacity": int(emergency_capacity),
            "utilisation": distribution_summary(utilisation_values, percentiles),
            "load": distribution_summary(load_values, percentiles),
            "baseline_load": int(baseline.hospital_load.get(hospital_id, 0)),
            "baseline_utilisation": float(
                baseline.hospital_utilisation.get(hospital_id, math.nan)
            ),
        }

    return {
        "scenario_id": run.scenario_id,
        "seed": int(run.seed),
        "n_simulations": int(run.n_simulations),
        "percentiles": list(config["monte_carlo"]["percentiles"]),
        "baseline": baseline_block,
        "metrics": metrics,
        "hospitals": hospitals,
        "uncertainty_note": UNCERTAINTY_NOTE,
        "disclaimer": SYNTHETIC_DISCLAIMER,
    }


def write_monte_carlo_outputs(
    city: SyntheticCity, run: Any, out_dir: Path, config: dict[str, Any]
) -> dict[str, Path]:
    """Write per-realisation CSV, the distribution-summary JSON, and spatial
    frequency layers (collapse/closure/unreachability rates)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _mc_tag(run)

    realisations_csv = out_dir / f"realisations_{tag}.csv"
    realisations_dataframe(run).to_csv(realisations_csv, index=False)

    monte_carlo_summary_json = out_dir / f"monte_carlo_summary_{tag}.json"
    monte_carlo_summary_json.write_text(
        json.dumps(summarise_monte_carlo(run, config), indent=2), encoding="utf-8"
    )

    monte_carlo_gpkg = out_dir / f"monte_carlo_{tag}.gpkg"

    buildings_frequency = buildings_geodataframe(city)
    buildings_frequency["collapse_frequency"] = run.building_collapse_frequency
    buildings_frequency["mean_expected_damage_index"] = run.building_mean_edi
    _require_crs(buildings_frequency, "buildings_frequency").to_file(
        monte_carlo_gpkg, layer="buildings_frequency", driver="GPKG"
    )

    roads_frequency = roads_geodataframe(city)
    roads_frequency["closed_frequency"] = run.link_closed_frequency
    roads_frequency["degraded_frequency"] = run.link_degraded_frequency
    _require_crs(roads_frequency, "roads_frequency").to_file(
        monte_carlo_gpkg, layer="roads_frequency", driver="GPKG"
    )

    baseline_times = np.where(
        np.isfinite(run.baseline.travel_time_min), run.baseline.travel_time_min, np.nan
    )
    population_frequency = population_geodataframe(city)
    population_frequency["unreachable_frequency"] = run.unit_unreachable_frequency
    population_frequency["mean_travel_time_min"] = run.unit_mean_travel_time_min
    population_frequency["baseline_travel_time_min"] = baseline_times
    _require_crs(population_frequency, "population_frequency").to_file(
        monte_carlo_gpkg, layer="population_frequency", driver="GPKG"
    )

    hospitals_geodataframe(city).to_file(monte_carlo_gpkg, layer="hospitals", driver="GPKG")

    return {
        "realisations_csv": realisations_csv,
        "monte_carlo_summary_json": monte_carlo_summary_json,
        "monte_carlo_gpkg": monte_carlo_gpkg,
    }


# --------------------------------------------------------------------------- #
# Intervention comparison outputs (Phase 5)
# --------------------------------------------------------------------------- #

INTERVENTION_NOTE = (
    "PROTOTYPE INTERVENTION OPTIMIZATION. Costs and effect multipliers are "
    "invented, not engineering estimates. The search is an exhaustive "
    "enumeration over a tiny candidate set with no optimality guarantee. "
    "HOSPITAL_SUPPORT moves only the service-pressure metric, never the "
    "primary objective, under the current accessibility model."
)


def _outcome_row(outcome: Any, rank: int | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "rank": rank,
        "portfolio_id": outcome.portfolio_id,
        "intervention_types": (
            "+".join(outcome.intervention_types) if outcome.intervention_types else "NONE"
        ),
        "portfolio_cost": float(outcome.portfolio_cost),
        "within_budget": bool(outcome.within_budget),
        "n_simulations": int(outcome.n_simulations),
        "primary_metric": outcome.primary_metric,
        "primary_objective_baseline": float(outcome.primary_objective_baseline),
        "primary_objective_after": float(outcome.primary_objective_after),
        "primary_benefit": float(outcome.primary_benefit_mean),
        "primary_benefit_std": float(outcome.primary_benefit_std),
        "p05_benefit": float(outcome.primary_benefit_p05),
        "p50_benefit": float(outcome.primary_benefit_p50),
        "p95_benefit": float(outcome.primary_benefit_p95),
        "probability_of_improvement": float(outcome.probability_of_improvement),
        "benefit_per_cost": (
            float(outcome.benefit_per_cost) if math.isfinite(outcome.benefit_per_cost) else None
        ),
    }
    for key, value in outcome.secondary.items():
        row[f"secondary_{key}"] = float(value) if math.isfinite(value) else None
    return row


def _outcome_dict(outcome: Any) -> dict[str, Any]:
    row = _outcome_row(outcome, None)
    row.pop("rank")
    row["target_ids"] = {k: list(v) for k, v in outcome.target_ids.items()}
    return row


def _intervention_tag(comparison: Any) -> str:
    return f"{comparison.scenario_id}_seed{comparison.seed}_n{comparison.n_simulations}"


def write_intervention_outputs(
    city: SyntheticCity, comparison: Any, out_dir: Path
) -> dict[str, Path]:
    """Write the ranked portfolio comparison CSV, summary JSON, and targeted-
    entity spatial layers for the best feasible portfolio."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _intervention_tag(comparison)

    rows = [_outcome_row(comparison.baseline_outcome, None)]
    rows.extend(
        _outcome_row(outcome, rank)
        for rank, outcome in enumerate(comparison.ranked_outcomes, start=1)
    )
    comparison_csv = out_dir / f"intervention_comparison_{tag}.csv"
    pd.DataFrame(rows).to_csv(comparison_csv, index=False)

    best = comparison.best
    summary = {
        "scenario_id": comparison.scenario_id,
        "seed": int(comparison.seed),
        "n_simulations": int(comparison.n_simulations),
        "budget": float(comparison.budget),
        "primary_metric": comparison.primary_metric,
        "primary_objective_note": (
            "Expected unreachable population -- lower is better. Deliberately "
            "the single primary objective; secondary metrics are reported "
            "separately, never folded into a weighted score."
        ),
        "validation_split": {
            "selection_fraction": float(comparison.selection_fraction),
            "n_selection": int(comparison.n_selection),
            "n_evaluation": int(comparison.n_evaluation),
            "note": (
                "Data-driven road-hardening targets are chosen on the selection "
                "subset only; every reported metric is computed on the disjoint "
                "evaluation subset (out-of-sample). Common random numbers are "
                "preserved within the evaluation set."
            ),
        },
        "baseline": _outcome_dict(comparison.baseline_outcome),
        "ranked_portfolios": [_outcome_dict(o) for o in comparison.ranked_outcomes],
        "best_portfolio": {
            "portfolio_id": best.portfolio_id if best is not None else None,
            "improves_on_baseline": comparison.best_improves_on_baseline,
            "note": (
                "Chosen by the primary objective through enumeration, not "
                "hard-coded. No optimality guarantee."
            ),
        },
        "uncertainty_note": UNCERTAINTY_NOTE,
        "intervention_note": INTERVENTION_NOTE,
        "disclaimer": SYNTHETIC_DISCLAIMER,
    }
    summary_json = out_dir / f"intervention_summary_{tag}.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    targets = best.target_ids if best is not None else {}
    targeted_buildings = set(targets.get("BUILDING_RETROFIT", ()))
    targeted_roads = set(targets.get("ROAD_HARDENING", ()))
    targeted_hospitals = set(targets.get("HOSPITAL_SUPPORT", ()))

    targets_gpkg = out_dir / f"intervention_targets_{tag}.gpkg"

    buildings_targeted = buildings_geodataframe(city)
    buildings_targeted["targeted"] = buildings_targeted["building_id"].isin(targeted_buildings)
    _require_crs(buildings_targeted, "buildings_targeted").to_file(
        targets_gpkg, layer="buildings_targeted", driver="GPKG"
    )

    roads_targeted = roads_geodataframe(city)
    roads_targeted["targeted"] = roads_targeted["road_id"].isin(targeted_roads)
    _require_crs(roads_targeted, "roads_targeted").to_file(
        targets_gpkg, layer="roads_targeted", driver="GPKG"
    )

    hospitals_targeted = hospitals_geodataframe(city)
    hospitals_targeted["targeted"] = hospitals_targeted["hospital_id"].isin(targeted_hospitals)
    _require_crs(hospitals_targeted, "hospitals_targeted").to_file(
        targets_gpkg, layer="hospitals_targeted", driver="GPKG"
    )

    return {
        "intervention_comparison_csv": comparison_csv,
        "intervention_summary_json": summary_json,
        "intervention_targets_gpkg": targets_gpkg,
    }
