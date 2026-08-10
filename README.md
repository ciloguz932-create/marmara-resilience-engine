# Marmara Resilience Engine

**MRE-001 — first vertical-slice prototype.**

Long-term research objective:

> Given an earthquake scenario and limited intervention resources, determine
> which combination of interventions can reduce risk and improve emergency
> accessibility most effectively.

MRE-001 does not attempt to answer that. It builds the pipeline, end to end, on
a **fully synthetic city**.

```
earthquake scenario → hazard input → probabilistic building damage
→ road disruption → changed road network → hospital accessibility
→ intervention comparison → uncertainty-aware results
```

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

## Environment

Python 3.12 from the QGIS 3.44.12 bundled interpreter, so that the
GDAL/PROJ/GEOS stack is consistent and no second GDAL is installed.

```bash
"C:\Program Files\QGIS 3.44.12\apps\Python312\python.exe" -m venv --system-site-packages .venv
```

Verify the environment:

```bash
.venv\Scripts\python.exe scripts\verify_env.py
```

Run a single synthetic scenario:

```bash
.venv\Scripts\python.exe scripts\run_scenario.py --seed 20260810 --out-dir outputs
```

Run a 1,000-realisation Monte Carlo experiment:

```bash
.venv\Scripts\python.exe scripts\run_monte_carlo.py --seed 20260810 -n 1000 --out-dir outputs
```

Details and rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Blender is **not** part of the calculation engine — it is the future
visualization layer, driven headlessly from files on disk.

---

## Layout

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
  outputs/      serialisation and GIS export
blender/        visualization scripts (headless, outside the engine)
config/         default.toml, plus gitignored local.toml
data/           raw / interim / processed / synthetic (all gitignored)
docs/           architecture and scientific assumptions
scripts/        environment checks and CLI entrypoints
tests/          pytest suite
notebooks/      exploration
```

---

## What the engine produces

Hazard field → probabilistic building damage → probabilistic road disruption →
disrupted network → hospital accessibility, over 1,000 Monte Carlo realisations,
each compared against the same deterministic pre-earthquake baseline.

Results are **distributions, not single numbers**: mean, standard deviation, and
P05/P25/P50/P75/P95 for every metric. Those percentiles are empirical quantiles
of the simulated distribution — **not confidence intervals**, and running more
realisations narrows them without making the model more valid.

Outputs are GeoPackage (spatial layers and per-element frequencies), CSV
(per-realisation records, full damage probability vectors), and JSON
(distribution summaries with seed provenance). Schema:
[docs/OUTPUT_SCHEMA.md](docs/OUTPUT_SCHEMA.md). The future Blender importer
reads the GeoPackage.

Intervention comparison is a later phase.

## Reproducibility

One master seed in `config/default.toml` drives everything, through **named**
random streams — so changing the building count leaves the road network
bit-identical, and realisation *i* is the same whether run alone or in a batch.
Synthetic data is regenerated from the seed rather than committed: the seed is
the artifact.

---

## Roadmap

| Phase | Content | Status |
|---|---|---|
| 0 | Python 3.12 environment, stack verification | ✅ done |
| 1 | Repository skeleton, config, docs, module seams | ✅ done |
| 2 | Smoke tests: GDAL, GeoPandas/Shapely, geometry, NetworkX, Blender headless | ✅ done |
| 3 | Synthetic city, hazard, fragility, road disruption | ✅ done |
| 4 | Hospital accessibility, Monte Carlo, uncertainty reporting | ✅ done |
| 5 | Prototype intervention optimization | planned |
| 6+ | Real-data layer, Blender visualization | not scoped |
