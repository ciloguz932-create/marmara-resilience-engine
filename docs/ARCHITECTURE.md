# Architecture — MRE-001

## 1. Runtime

MRE-001 runs on **Python 3.12.13**, taken from the QGIS 3.44.12 bundled
interpreter:

```
C:\Program Files\QGIS 3.44.12\apps\Python312\python.exe
```

The project venv is created from it with `--system-site-packages`:

```bash
"C:\Program Files\QGIS 3.44.12\apps\Python312\python.exe" -m venv --system-site-packages .venv
```

**Why this interpreter.** `osgeo`/GDAL has no reliable installable wheel on
Windows, and building it from source requires a standalone GDAL that this
project deliberately does not install. QGIS already ships a mutually consistent
GDAL 3.13.1 / PROJ 9.8.1 / GEOS stack with matching Python bindings. Inheriting
it is the only path that avoids a second, conflicting GDAL on the machine.

**Constraints held by this decision:**

- the global Python 3.14 installation is never modified
- no standalone GDAL is installed
- no conda
- Blender keeps its own embedded Python; `bpy` is never imported into this venv

**Verified stack** (see `scripts/verify_env.py`):

| Package | Version |
|---|---|
| Python | 3.12.13 |
| GDAL (`osgeo`) | 3.13.1 |
| PROJ (via pyproj 3.7.2) | 9.8.1 |
| geopandas | 1.1.3 |
| shapely | 2.1.2 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| pandas | 3.0.3 |
| networkx | 3.6.1 |
| pyogrio | present (geopandas I/O backend; `fiona` absent and not required) |

### The GDAL_DATA / PROJ_DATA trap

The QGIS installer does **not** set `GDAL_DATA` or `PROJ_DATA` machine-wide.
Without them `pyproj` cannot find `proj.db` and every reprojection fails, with
an error that surfaces far from its cause. `mre.config.bootstrap_geo_env()` sets
them in `os.environ` **for the current process only**, and is called from
`mre/__init__.py` before any geospatial import. The machine environment is never
touched.

Paths live in `config/default.toml` under `[environment]`, overridable via a
gitignored `config/local.toml`.

### The pyarrow / PROJ DLL clash

Found by the Phase 2 smoke tests, and more dangerous than the trap above.

`pandas` 3.x pulls in `pyarrow`, which ships its own copies of common native
libraries. On Windows the first loader wins for the life of the process, so if
pyarrow loads first, `pyproj._context` binds against pyarrow's copies and fails
with *"entry point not found"*.

The failure mode is the problem: `geopandas` catches that ImportError and sets
`HAS_PYPROJ = False`, then **silently drops every CRS** — `.crs` assignments
become `None` with only a `UserWarning`, and layers get written with no
projection. For a project whose adjacency radii and travel times are only
meaningful in metres, silent CRS loss is the worst available outcome.

`mre.config._preload_proj()` therefore imports `pyproj.network` during
bootstrap, before anything else can claim those DLLs. Importing `pyproj` alone
is **not** sufficient — `pyproj.network` is what forces the full native chain to
load. It also adds the QGIS `bin` directory via `os.add_dll_directory()`.

This is why `import mre` must come first, and why `tests/conftest.py` exists.
`tests/test_config.py::test_bootstrap_makes_geopandas_crs_aware` is the
regression guard.

---

## 2. Pipeline

```
EarthquakeScenario
        │
        ▼
  mre.hazard          intensity proxy per building        (stochastic)
        │
        ▼
  mre.buildings       damage state per building           (stochastic)
        │
        ▼
  mre.roads           link states: OPEN/DEGRADED/CLOSED   (stochastic)
        │
        ▼
  mre.roads           post-event graph
        │
        ▼
  mre.hospitals       AccessibilityResult
        │
        ▼
  mre.simulation      × N realisations → distributions
        │
        ▼
  mre.optimization    portfolio comparison under budget
        │
        ▼
  mre.outputs         GeoPackage / GeoJSON / JSON summary
        │
        ▼
  blender/            3D visualization (out of engine)
```

---

## 3. Module map

| Module | Responsibility | Phase |
|---|---|---|
| `mre.config` | config load, GDAL/PROJ bootstrap, Blender discovery | 1 ✅ |
| `mre.models` | shared domain schemas — the inter-stage contract | 1 ✅ |
| `mre.rng` | named seeded streams | 3 ✅ |
| `mre.data` | world construction; the synthetic↔real swap point | 3 ✅ |
| `mre.hazard` | intensity field | 3 ✅ |
| `mre.buildings` | fragility and damage sampling | 3 ✅ |
| `mre.roads` | graph construction and disruption | 3 ✅ |
| `mre.hospitals` | accessibility metrics, paired comparison | 4 ✅ |
| `mre.simulation` | scenario driver + Monte Carlo orchestration | 3–4 ✅ |
| `mre.outputs` | serialisation, uncertainty summaries, GIS export | 3–4 ✅ |
| `mre.optimization` | prototype intervention comparison | 5 |

`mre.rng` is not in the original Phase 1 tree. It was added so `mre.data` and
`mre.simulation` can both seed themselves without importing each other.

### Seeding

Every stochastic step draws from a **named** stream:

```
SeedSequence(entropy=seed, spawn_key=(crc32(name),))
```

Names are hierarchical by convention — `city.roads`, `city.buildings`,
`hazard:0`, `damage:0`, `disruption:0`. Because streams are named rather than
spawned in sequence, a stage that consumes a different number of draws never
shifts another stage's numbers, and adding a stage later cannot perturb
existing results. Changing `n_buildings` leaves the road network bit-identical;
`tests/test_synthetic_city.py::test_named_streams_isolate_stages` enforces it.

CRC32 rather than `hash()`: `hash()` is salted per process, so it would produce
different streams on every run.

There is no module-level global RNG and no bare `np.random.*` call in `mre` —
`tests/test_reproducibility.py` greps the source and fails if one appears.

---

## 4. Replaceability seams

The point of the architecture is that each scientific stage can be replaced
without rewriting the engine. Each seam is a `Protocol`:

| Seam | Protocol | Replaced by |
|---|---|---|
| World | `mre.data.CityBuilder` | OSM / real building inventory / census |
| Hazard | `mre.hazard.HazardField` | **OpenQuake**, a real GMPE, or a ShakeMap raster |
| Damage | `mre.buildings.FragilityModel` | validated regional fragility curves |
| Disruption | `mre.roads.DisruptionModel` | debris/bridge-aware model |
| Access | `mre.hospitals.AccessibilityModel` | capacity-constrained assignment |

`run_scenario()` accepts `hazard=`, `fragility=`, and `disruption=` arguments
typed as those protocols and defaults to the prototypes. Substituting a
validated model is an argument, not an edit:

```python
run_scenario(city, config, hazard=OpenQuakeField(...))
```

Concrete implementations are selected by name from config
(`hazard.model = "point_source_decay"`), so a new model becomes available by
registering it — the driver never imports one directly.

The simulation driver depends only on these protocols, never on a concrete
implementation.

**Real-data rule:** the real-data layer must produce the same `mre.models`
types. The simulation core must never be able to tell whether its input was
invented or observed.

---

## 5. Reproducibility

- One master seed in `config/default.toml`.
- `numpy.random.SeedSequence(seed).spawn(n)` yields independent child streams.
- Realisation *i* always uses child stream *i* — results are independent of
  execution order and survive future parallelisation unchanged.
- Run summaries record seed, scenario id, and config provenance.
- Synthetic data is **regenerated, never committed**: the seed is the artifact.

---

## 6. Boundaries

**NetworkX** is the graph backend. **OSMnx is not a dependency** — it is an
optional future integration. The synthetic slice must run with no network
access and no downloads.

**Blender is not part of the calculation engine.** It never appears in an
`import` inside `mre/`. The boundary is a file on disk: the engine writes
GeoPackage/GeoJSON, and a separate headless Blender process reads it. Blender is
located via `config/default.toml` rather than PATH, and every Blender-dependent
test skips cleanly when the executable is absent.

**QGIS** is used for its libraries only; the `qgis` Python module itself is not
imported and the desktop application is not automated.

---

## 5a. Monte Carlo orchestration

`mre.simulation` owns looping, seeding, and accumulation — and nothing else. It
knows the hazard, fragility, disruption, and accessibility models only through
their protocols, bundled in a `ModelSuite` built **once per run**. Rebuilding
them per realisation would recompute the building/link adjacency index every
time, and would couple the orchestrator to concrete classes.

```
build_models(config)  ->  ModelSuite(hazard, fragility, disruption, accessibility)
run_monte_carlo       ->  iter_realisations -> run_scenario -> RealisationRecord
```

**Memory discipline.** A 1,000-realisation run must not retain 1,000 graphs or
1,000 per-building arrays. Each realisation is reduced to a flat
`RealisationRecord` plus running spatial counters (collapse count per building,
closed/degraded count per link, unreachable count per unit); the graphs and
arrays are then discarded.

**Baseline pairing.** The pre-earthquake network is a property of the city, not
of a realisation, so baseline accessibility is deterministic and computed once.
Every realisation is compared against that same baseline. Accessibility is never
compared across two unrelated realisations.

**Reporting.** `distribution_summary` excludes non-finite values and counts them
in `n_excluded` rather than propagating NaN. Percentiles are empirical
quantiles, never described as confidence intervals.

## 6a. Outputs and the Blender handoff

`mre.outputs` writes, per run:

| File | Contents |
|---|---|
| `city.gpkg` | layers `buildings`, `roads`, `hospitals`, `population_units` |
| `scenario_<id>_seed<n>_r<i>.gpkg` | `buildings_damage` (intensity proxy, sampled state, EDI), `roads_disrupted` (link state, closure probability) |
| `damage_probabilities_<tag>.csv` | full 5-state probability vector per building |
| `summary_<tag>.json` | counts, distributions, seed provenance, disclaimer |

GeoPackage for geometry (one file, many layers, CRS preserved); CSV for the
probability matrix, which is tabular and would bloat a geometry layer; JSON for
the summary.

**The future Blender importer should read the GeoPackage**, whose coordinates
are projected metres. Every writer calls `_require_crs()` and raises rather
than emitting an unprojected layer — silent CRS loss is the failure mode this
project can least afford.

## 7. Data handling

`data/raw`, `interim`, `processed`, `synthetic` are gitignored except for
`.gitkeep`. Large real datasets are out of scope for MRE-001 and, when they
arrive, will be fetched by a documented download script rather than committed
or stored in Git LFS.
