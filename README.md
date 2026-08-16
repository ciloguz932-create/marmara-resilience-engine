# Marmara Resilience Engine

Synthetic earthquake resilience simulation and intervention optimization prototype.

> **This is a synthetic research prototype, not an operational Istanbul
> earthquake prediction system.**

It runs in two modes, kept strictly separate everywhere:

- **`SYNTHETIC`** — an entirely invented city (Phases 0–5.1). The default.
- **`REAL_PILOT`** — a real İstanbul study area (Zeytinburnu) built from real
  OpenStreetMap geometry (Phase 6). **Real geometry, prototype physics:** the
  building footprints, road network + topology, and hospital locations are real;
  the fragility and intensity parameters are prototype values applied to that
  real geometry, structural attributes and hospital capacities are `UNKNOWN`,
  and demand is a uniform proxy. It is **not** a real damage or loss prediction
  for any real building. See [Real-data pilot](#real-data-pilot-phase-6) and
  [docs/SCIENTIFIC_ASSUMPTIONS.md §10](docs/SCIENTIFIC_ASSUMPTIONS.md).

**MRE-001 — first vertical-slice prototype**, now with a Blender
visualization layer on top of the unchanged Phase 0-5 simulation pipeline:

```
earthquake scenario → hazard input → probabilistic building damage
→ road disruption → changed road network → hospital accessibility
→ intervention comparison → uncertainty-aware results → Blender scene
```

Long-term research objective:

> Given an earthquake scenario and limited intervention resources, determine
> which combination of interventions can reduce risk and improve emergency
> accessibility most effectively.

MRE-001 does not attempt to answer that. It builds the pipeline, end to end,
on a **fully synthetic city**, and now renders that pipeline's output as a
readable 3D scene.

---

## ⚠️ Status and scope

MRE-001 is a **research / software-architecture prototype**, not a validated
engineering earthquake-risk model.

- The city is **entirely synthetic**. It is not Istanbul, not simplified
  Istanbul, not anonymised Istanbul.
- Fragility parameters are **invented**. They are not a Turkish fragility model.
- The engine makes **no claim about any real building, road, or hospital**, and
  does not predict that any individual structure will collapse.
- It is **not** a deterministic earthquake prediction model.
- The intervention comparison is a **"Prototype intervention optimization"**
  with no optimality guarantee.

Read [docs/SCIENTIFIC_ASSUMPTIONS.md](docs/SCIENTIFIC_ASSUMPTIONS.md) before
interpreting any output.

---

## WHAT IT DOES

Builds a fully synthetic city (buildings, road network, hospitals, population
units), applies a synthetic earthquake scenario, samples probabilistic
building damage and road disruption, re-evaluates hospital accessibility on
the disrupted network, repeats that thousands of times for uncertainty bands,
and ranks a small set of prototype interventions (retrofit / road hardening /
hospital support) under a budget — all reproducible from one seed. A Blender
scene then renders one such run as a readable BEFORE / AFTER EARTHQUAKE /
AFTER INTERVENTION comparison.

## ARCHITECTURE

```
mre/            simulation engine
  models/       shared domain schemas (inter-stage contract)
  rng.py        named seeded random streams
  data/         world construction; synthetic ↔ real swap point
  hazard/       ground-shaking intensity field
  buildings/    fragility and damage sampling
  roads/        network construction and disruption
  hospitals/    accessibility metrics
  simulation/   Monte Carlo driver
  optimization/ prototype intervention comparison
  outputs/      serialisation and GIS/JSON export — the Blender handoff boundary
blender/        visualization scripts (headless, outside the engine)
config/         default.toml, plus gitignored local.toml
data/           raw / interim / processed / synthetic (all gitignored)
docs/           architecture and scientific assumptions
scripts/        environment checks, CLI entrypoints, and the one-command demo
tests/          pytest suite (298 tests)
notebooks/      exploration
```

Each scientific stage sits behind a `Protocol` seam (`HazardField`,
`FragilityModel`, `DisruptionModel`, `AccessibilityModel`, `CityBuilder`) so a
validated model can replace a prototype one without touching the Monte Carlo
driver. Every stochastic draw comes from a **named** seeded stream
(`mre.rng.named_generator`), never a bare `np.random.*` call — this is what
makes Monte Carlo runs reproducible and makes Phase 5's paired
common-random-numbers comparison possible. Full rationale, the QGIS/GDAL/PROJ
environment story, and the pyarrow/PROJ DLL clash: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Blender is **not** part of the calculation engine. It never appears in an
`import` inside `mre/`; the boundary is a file on disk (`mre.outputs` writes,
`blender/build_scene.py` reads a small JSON export), and every
Blender-dependent step skips cleanly when the executable is absent.

## PHASES COMPLETED

| Phase | Content | Status |
|---|---|---|
| 0 | Python 3.12 environment, stack verification | ✅ done |
| 1 | Repository skeleton, config, docs, module seams | ✅ done |
| 2 | Smoke tests: GDAL, GeoPandas/Shapely, geometry, NetworkX, Blender headless | ✅ done |
| 3 | Synthetic city, hazard, fragility, road disruption | ✅ done |
| 4 | Hospital accessibility, Monte Carlo, uncertainty reporting | ✅ done |
| 5 | Prototype intervention optimization (budget, portfolios, CRN, ranking) | ✅ done |
| 5.1 | Validation hardening — out-of-sample target selection split | ✅ done |
| 6a | Blender visualization (BEFORE / AFTER / INTERVENTION scene, one-command demo) | ✅ done |
| 6b | Real-data pilot — real OSM geometry (Zeytinburnu), provenance, validation, real-city Blender scene | ✅ done |
| 7+ | Ingested AFAD/MTA/İBB layers; validated fragility; real population | not scoped |

## END-TO-END DEMO

One command runs the full pipeline — scenario, Monte Carlo population
reachability, Phase 5 intervention ranking, every GIS/CSV/JSON artifact, and
the Blender scene:

```bash
.venv\Scripts\python.exe scripts\demo.py
```

Writes to `outputs\demo\`, prints every file it wrote, and — if Blender is
configured (`[environment].blender_exe` in `config/default.toml`) — builds
`outputs\demo\mre_demo.blend` and a `.png` preview, then prints exactly which
file to open. If Blender is not available, everything else is still written
and the step is skipped rather than failing (`--no-blender` to skip on
purpose). `-n` sets the intervention-comparison realisation count (default
300, for a demo that finishes in well under a minute; the config default of
1,000 is used by the individual `run_*` scripts below for a tighter-banded
run).

## BLENDER VISUALIZATION

`blender/build_scene.py` (run headlessly by `scripts/demo.py`, or by hand —
see [blender/README.md](blender/README.md)) builds three side-by-side city
blocks in one scene:

```
BEFORE — NORMAL CITY  →  AFTER — EARTHQUAKE SCENARIO  →  AFTER INTERVENTION
```

- Buildings are extruded footprints, coloured by damage state (`NONE` /
  `SLIGHT` / `MODERATE` / `SEVERE` / `COLLAPSE`) and rendered shorter as
  damage increases — a deliberate readability cue, not a geometric claim.
- Roads are coloured by link state (`OPEN` / `DEGRADED` / `CLOSED`).
- Hospitals are marked distinctly (a cross marker); population units are
  small spheres, sized by population count and coloured red where the Monte
  Carlo run found that unit unreachable **in that same realisation** (the
  scene generator runs a 1-realisation Monte Carlo at the same seed as the
  single scenario, so the two are drawn from identical named streams and
  describe the same event, not two different draws).
- The `AFTER INTERVENTION` block highlights the entities the best-ranked
  Phase 5 portfolio actually targets, plus its benefit numbers as on-scene
  text.
- A legend and the synthetic-prototype disclaimer are placed in-scene.

The boundary between the engine and Blender is a single JSON file
(`outputs/demo/blender_scene.json`) plus the standard `mre.outputs`
GeoPackage/CSV/JSON artifacts; `blender/build_scene.py` never imports
`geopandas` or any `mre` module, and `scripts/demo.py` never imports `bpy`.
Colours and height multipliers are a rendering convention, not a scientific
parameter — see [blender/README.md](blender/README.md).

## MONTE CARLO

```bash
.venv\Scripts\python.exe scripts\run_monte_carlo.py --seed 20260810 -n 1000 --out-dir outputs
```

Hazard field → probabilistic building damage → probabilistic road disruption →
disrupted network → hospital accessibility, over N Monte Carlo realisations,
each compared against the same deterministic pre-earthquake baseline.

Results are **distributions, not single numbers**: mean, standard deviation, and
P05/P25/P50/P75/P95 for every metric. Those percentiles are empirical quantiles
of the simulated distribution — **not confidence intervals**, and running more
realisations narrows them without making the model more valid.

Outputs are GeoPackage (spatial layers and per-element frequencies), CSV
(per-realisation records, full damage probability vectors), and JSON
(distribution summaries with seed provenance). Schema:
[docs/OUTPUT_SCHEMA.md](docs/OUTPUT_SCHEMA.md).

## INTERVENTION OPTIMIZATION

```bash
.venv\Scripts\python.exe scripts\run_interventions.py --seed 20260810 -n 1000 --out-dir outputs\interventions
```

Given a synthetic budget, the engine enumerates every feasible portfolio of
three prototype interventions — `BUILDING_RETROFIT`, `ROAD_HARDENING`,
`HOSPITAL_SUPPORT` — and ranks them by a single interpretable objective:
**expected unreachable population**. Each portfolio is evaluated by re-running
the Monte Carlo engine on a modified copy of the city under **common random
numbers** (same seed, same realisation streams), so realisation *i* of every
portfolio sees the identical hazard field and draws and the comparison is
paired, not a difference between unrelated samples.

Benefit is reported as a distribution (mean, P05/P50/P95, and probability of
improvement), never as a single "best". Secondary metrics (travel time, closures,
collapses, service pressure) are reported separately and never folded into a
weighted score. **This is a "Prototype intervention optimization"**: costs and
effect multipliers are invented, the search has no optimality guarantee, and
`HOSPITAL_SUPPORT` — under the current accessibility model, where capacity does
not gate assignment — moves only the service-pressure metric, never the primary
objective.

## REAL-DATA PILOT (Phase 6)

A real İstanbul study area — **Zeytinburnu district core (~2.6 km²)** — built
from real open data, run through the same engine, and rendered as a real city.

```bash
.venv\Scripts\python.exe scripts\fetch_real_pilot.py     # once, needs network (OSM/Overpass)
.venv\Scripts\python.exe scripts\demo_real.py            # validate → simulate → intervene → Blender
```

`demo_real.py` writes to `outputs\real_pilot\` and builds
`outputs\real_pilot\blender\marmara_real_pilot.blend` (+ `preview.png`):
`BASELINE — REAL CITY → EARTHQUAKE SCENARIO → + INTERVENTION`, with real
building footprints extruded, roads coloured by link state, hospitals marked,
the best-portfolio targets highlighted, a legend, and the
`RESEARCH PROTOTYPE — NOT A REAL-WORLD PREDICTION` banner in-scene.

**Real geometry, prototype physics — the honest split, enforced in code:**

| | Source / status |
|---|---|
| Building footprints, centroids | **REAL** — OpenStreetMap (ODbL) |
| Storey counts (~92%), occupancy | **REAL** — OSM `building:levels` / `building` tag |
| Road network + topology, classes | **REAL** — OSM node topology, `highway` tag |
| Hospital locations + names | **REAL** — OSM `amenity=hospital` |
| Construction year, structural system | **`UNKNOWN`** — not in open data, never invented |
| Hospital capacity / beds | **`UNKNOWN`** — recorded, not guessed |
| Per-building vulnerability | **PROTOTYPE** — uniform, labelled `PROTOTYPE_UNIFORM` |
| Fragility medians / β | **PROTOTYPE** — invented (§3), applied to real geometry |
| Intensity field | **PROTOTYPE proxy** — real fault *location* (Princes' Islands segment), proxy magnitude; not PGA/PGV/MMI |
| Hospital emergency capacity | **PROTOTYPE** — assigned; does not affect who can reach a hospital |
| Population / demand | **UNIFORM PROXY** — no real census; `population_source=UNIFORM_PROXY` |

**Data sources & licence.** Buildings, roads, and hospitals: © OpenStreetMap
contributors, **Open Database License (ODbL)**, fetched via the Overpass API.
Every source, its licence, acquisition date, CRS (`EPSG:32635`), attributes, and
**missing** attributes are recorded in `outputs\real_pilot\provenance.json`. The
scenario cites the JICA–İMM Zeytinburnu pilot, the Boğaziçi/KOERI risk
assessment, and Main Marmara Fault literature. AFAD/MTA/İBB datasets are recorded
as **context/citations**, not ingested as layers (they require portal/licensing
steps this pilot does not perform) — marked as such, never silently substituted.

**Validation.** Every build runs `mre.real.validation` (valid geometry, single
CRS, no NaN in required fields, no non-positive lengths, road connectivity,
entities in the study area, non-empty demand proxy, reproducibility, provenance
completeness) and writes `outputs\real_pilot\validation.json`; the demo refuses
to present results unless it passes.

**Outputs.** `outputs\real_pilot\`: `buildings.gpkg`, `roads.gpkg`,
`hospitals.gpkg`, `hazard.gpkg`, `results.gpkg`, `scenario.json`,
`provenance.json`, `validation.json`, `monte_carlo_summary.json`, and
`blender\marmara_real_pilot.blend` + `preview.png`. Every GPKG carries `*_source`
columns and literal `UNKNOWN` values so each attribute's origin is auditable.

Full assumptions and limitations: [docs/SCIENTIFIC_ASSUMPTIONS.md §10](docs/SCIENTIFIC_ASSUMPTIONS.md).

## VALIDATION

- **Out-of-sample target selection (Phase 5.1).** Data-driven road-hardening
  targets are chosen on a **selection** subset of realisations
  (`selection_fraction`, default 30%); every reported metric is computed on
  the disjoint **evaluation** subset, so target selection cannot inflate the
  reported benefit. This is a *prototype validation split* that removes an
  in-sample selection artefact — **not** a statistical validation of the
  model against reality.
- **Real-pilot build validation (Phase 6).** `mre.real.validation` gates every
  real-pilot build (geometry validity, single CRS, no NaN in required fields, no
  non-positive road lengths, road connectivity, entities in the study area,
  non-empty demand proxy, reproducibility, provenance completeness) and writes
  `validation.json`; results are not presented unless it passes.
- **Test suite.** 315 tests (`.venv\Scripts\python.exe -m pytest -q -W error`)
  cover reproducibility (named-stream isolation, no bare `np.random.*` calls),
  the scientific invariants (fragility medians strictly increasing, no
  threshold rule, unreachable population excluded from travel-time means, CRS
  never silently dropped), Monte Carlo NaN/inf hygiene, every output schema, and
  the real-pilot integrity contract (real values kept, absent values marked
  `UNKNOWN` and never invented, prototype/proxy assignments labelled).
- **What is *not* validated:** no comparison against observed damage from a
  real event, no validated fragility, no real ground-motion, no real population,
  no external peer review, no epistemic-uncertainty logic tree. See
  [docs/SCIENTIFIC_ASSUMPTIONS.md §9–§10](docs/SCIENTIFIC_ASSUMPTIONS.md) for the
  full gap list.

## LIMITATIONS

- The city, hazard, fragility, disruption, and intervention parameters are
  **entirely invented** — not fitted to Turkish building stock, TBDY/TDY,
  HAZUS, or any empirical damage database.
- `intensity_proxy` is a **dimensionless proxy**, not PGA, PGV, SA(T), or MMI.
- No casualty model, no liquefaction/landslide/fire-following/tsunami, no
  time-dependent response (debris clearance, surge capacity), no traffic
  assignment, hospitals assumed fully functional post-event.
- Percentile bands describe **Monte Carlo sampling variability only** — they
  exclude model (epistemic) uncertainty and the synthetic-assumption gap
  entirely, and are therefore far narrower than genuine uncertainty about a
  real earthquake.
- `hospital_utilisation` carries a ~2-3 order-of-magnitude population/capacity
  scale artefact (declared wherever it is reported) — read it only as a
  relative comparison, never as an overload ratio.
- The full, itemised limitation list — one section per pipeline stage — lives
  in [docs/SCIENTIFIC_ASSUMPTIONS.md](docs/SCIENTIFIC_ASSUMPTIONS.md); read it
  before quoting any number from this engine.

## HOW TO RUN

Python 3.12 from the QGIS 3.44.13 bundled interpreter, so that the
GDAL/PROJ/GEOS stack is consistent and no second GDAL is installed.

```bash
"C:\Program Files\QGIS 3.44.13\apps\Python312\python.exe" -m venv --system-site-packages .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Verify the environment:

```bash
.venv\Scripts\python.exe scripts\verify_env.py
```

Run the test suite:

```bash
.venv\Scripts\python.exe -m pytest -q -W error
```

Run the one-command **synthetic** demo (scenario + Monte Carlo + interventions + Blender scene):

```bash
.venv\Scripts\python.exe scripts\demo.py
```

Run the **real-data pilot** (fetch once with network, then simulate + Blender offline):

```bash
.venv\Scripts\python.exe scripts\fetch_real_pilot.py
.venv\Scripts\python.exe scripts\demo_real.py
```

Or run each stage independently:

```bash
.venv\Scripts\python.exe scripts\run_scenario.py --seed 20260810 --out-dir outputs
.venv\Scripts\python.exe scripts\run_monte_carlo.py --seed 20260810 -n 1000 --out-dir outputs
.venv\Scripts\python.exe scripts\run_interventions.py --seed 20260810 -n 1000 --out-dir outputs\interventions
```

Details and rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Reproducibility

One master seed in `config/default.toml` drives everything, through **named**
random streams — so changing the building count leaves the road network
bit-identical, and realisation *i* is the same whether run alone or in a batch.
Synthetic data is regenerated from the seed rather than committed: the seed is
the artifact.

## Layout

See [ARCHITECTURE](#architecture) above.
