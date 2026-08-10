# Output schema — MRE-001

**Stable contract.** Downstream analysis and the future Blender importer depend
on these names. Columns may be *added*; existing columns must not be renamed,
removed, or have their meaning changed without a note in the changelog below.

All outputs are synthetic. See
[SCIENTIFIC_ASSUMPTIONS.md](SCIENTIFIC_ASSUMPTIONS.md).

---

## Conventions

| Convention | Meaning |
|---|---|
| CRS | `EPSG:32635` for every spatial layer. Coordinates are projected metres. |
| `intensity_proxy` | Dimensionless. **Not** PGA, PGV, SA(T), or MMI. |
| Travel time | Minutes. |
| `inf` travel time | Unreachable. Never included in any mean or median. |
| `nan` | Undefined, not zero. Appears only where documented below. |
| `*_frequency` | Fraction of realisations, in [0, 1]. |
| Percentile keys | `p05`, `p25`, `p50`, `p75`, `p95` — **empirical quantiles, not confidence intervals**. |

Filename tags: `<scenario_id>_seed<seed>_r<realisation>` for single runs,
`<scenario_id>_seed<seed>_n<n_simulations>` for Monte Carlo runs.

---

## 1. `city.gpkg` — the synthetic world

Written by `write_city_layers`. Deterministic given the seed.

**Layer `buildings`** (polygon, square footprints)

| Column | Type | Notes |
|---|---|---|
| `building_id` | str | `B00000`… unique |
| `floors` | int | |
| `construction_year` | int | |
| `structural_type` | str | `RC_FRAME`, `RC_SHEAR_WALL`, `MASONRY`, `STEEL`, `OTHER` |
| `occupancy` | str | `RESIDENTIAL`, `COMMERCIAL`, `PUBLIC`, `INDUSTRIAL` |
| `occupants` | int | Not used as accessibility demand |
| `vulnerability_index` | float | Higher = more fragile |
| `footprint_side_m` | float | |

**Layer `roads`** (linestring)

| Column | Type | Notes |
|---|---|---|
| `road_id` | str | `R00000`… unique |
| `from_node`, `to_node` | int | |
| `road_class` | str | `ARTERIAL`, `COLLECTOR`, `LOCAL` |
| `length_m`, `travel_time_min` | float | Free-flow |
| `criticality` | float | Edge betweenness on the undamaged network, [0, 1] |
| `susceptibility` | float | Debris-blockage multiplier |

**Layer `hospitals`** (point): `hospital_id`, `capacity`, `emergency_capacity`, `node_id`.

**Layer `population_units`** (point): `population_id`, `population_count`, `node_id`.
Decoupled from buildings by design.

---

## 2. Single-realisation outputs

Written by `write_scenario_outputs`.

**`scenario_<tag>.gpkg`, layer `buildings_damage`** — the `buildings` columns plus:

| Column | Type | Notes |
|---|---|---|
| `intensity_proxy` | float | Dimensionless |
| `damage_state` | str | `NONE`, `SLIGHT`, `MODERATE`, `SEVERE`, `COLLAPSE` |
| `expected_damage_index` | float | [0, 1]; not a loss ratio |

**layer `roads_disrupted`** — the `roads` columns plus:

| Column | Type | Notes |
|---|---|---|
| `link_state` | str | `OPEN`, `DEGRADED`, `CLOSED` |
| `closure_probability` | float | [0, 1], before sampling |

**`damage_probabilities_<tag>.csv`** — full distributions, one row per building:
`building_id`, `p_NONE`, `p_SLIGHT`, `p_MODERATE`, `p_SEVERE`, `p_COLLAPSE`,
`intensity_proxy`, `expected_damage_index`. The five `p_*` columns sum to 1.

**`summary_<tag>.json`** — counts and provenance for one realisation, plus
`disclaimer`. One realisation is one draw, not a result.

---

## 3. Monte Carlo outputs

Written by `write_monte_carlo_outputs`.

### 3.1 `realisations_<tag>.csv` — one row per realisation

| Column | Type | Notes |
|---|---|---|
| `realisation` | int | 0 … n−1 |
| `intensity_mean`, `intensity_max` | float | |
| `n_none`, `n_slight`, `n_moderate`, `n_severe`, `n_collapse` | int | Sum = building count |
| `mean_expected_damage_index` | float | [0, 1] |
| `n_open`, `n_degraded`, `n_closed` | int | Sum = link count |
| `n_components_after` | int | 1 = still connected |
| `network_connected_after` | bool | |
| `mean_travel_time_min` | float | Population-weighted, **reachable only** |
| `median_travel_time_min` | float | Population-weighted, **reachable only** |
| `population_reachable` | int | |
| `population_unreachable` | int | Reachable + unreachable = total |
| `n_disrupted_routes` | int | Units slower than baseline by > tolerance, or newly unreachable |
| `n_newly_unreachable_units` | int | |
| `population_newly_unreachable` | int | |
| `delta_mean_travel_time_min` | float | Post − baseline (paired) |
| `delta_median_travel_time_min` | float | Post − baseline (paired) |
| `load_<hospital_id>` | int | Assigned population; sums to `population_reachable` |
| `utilisation_<hospital_id>` | float | `load / emergency_capacity`; **unbounded** |

### 3.2 `monte_carlo_summary_<tag>.json`

```
scenario_id, seed, n_simulations, percentiles[]
baseline: {
  note, mean_travel_time_min, median_travel_time_min,
  population_reachable, population_unreachable,
  hospital_load{}, hospital_utilisation{},
  total_emergency_capacity, population_per_emergency_bed,
  utilisation_scale_note
}
metrics: { <metric>: {n, n_finite, n_excluded, mean, std, min, max, p05..p95} }
hospitals: { <id>: {utilisation{...}, load{...},
                    baseline_load, baseline_utilisation, emergency_capacity} }
uncertainty_note, disclaimer
```

`n_excluded` counts realisations whose value was non-finite — for example a
realisation leaving nobody reachable, where mean travel time is undefined.
**Non-finite values are excluded from the statistics and counted, never
propagated.** If `n_finite` is 0, every statistic is `null`.

Metrics summarised are listed in `mre.outputs.DISTRIBUTION_METRICS`.

### 3.3 `monte_carlo_<tag>.gpkg` — spatial frequencies

**`buildings_frequency`** — `buildings` columns plus `collapse_frequency` [0,1]
and `mean_expected_damage_index` [0,1].

**`roads_frequency`** — `roads` columns plus `closed_frequency` and
`degraded_frequency`, both [0,1].

**`population_frequency`** — `population_units` columns plus:

| Column | Notes |
|---|---|
| `unreachable_frequency` | [0, 1] |
| `mean_travel_time_min` | Mean over realisations where the unit **was reachable**; `nan` if never reachable |
| `baseline_travel_time_min` | Pre-earthquake; `nan` if unreachable at baseline |

**`hospitals`** — as in `city.gpkg`.

---

## Reading these numbers

- Percentiles describe **Monte Carlo sampling variability within the model's
  assumptions**. They are not confidence intervals, not predictive intervals,
  and exclude epistemic uncertainty in the model form and the invented
  parameters — which dominates.
- `utilisation_*` is a demand-pressure indicator, not a queueing result. In the
  default synthetic city, population and hospital capacity were chosen
  independently, giving ~585 people per emergency bed and utilisation values
  in the hundreds. That is a **parameter artefact**, meaningful only as a
  relative comparison.
- More realisations narrow the bands. They do not make the model more valid.

---

## Changelog

| Date | Change |
|---|---|
| 2026-08-10 | Initial schema: city, single-realisation, and Monte Carlo outputs (Phase 3 + Phase 4). |
