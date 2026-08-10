# CLAUDE.md — Marmara Resilience Engine

Working notes for agents and contributors. Read
[docs/SCIENTIFIC_ASSUMPTIONS.md](docs/SCIENTIFIC_ASSUMPTIONS.md) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before changing anything.

## Environment — hard rules

- Python is **3.12 from the QGIS bundle**, via `.venv` created with
  `--system-site-packages`. Always invoke `.venv\Scripts\python.exe` explicitly.
- **Do not modify the global Python 3.14 installation.**
- **Do not install a standalone GDAL.** `osgeo` comes from QGIS.
- **Do not install conda.**
- **Do not add the geo stack to `pyproject.toml` dependencies.** Listing
  numpy/geopandas/pyproj/GDAL there would invite pip to rebuild them and break
  the environment. `dependencies` is intentionally empty.
- `GDAL_DATA` / `PROJ_DATA` are set per-process by `mre.config.bootstrap_geo_env()`,
  called from `mre/__init__.py`. Never set them machine-wide.
- **`import mre` must come before `pyproj`, `osgeo`, `pandas`, or `geopandas`
  in any entry point.** Not style — load-bearing. pandas 3.x pulls pyarrow,
  whose bundled DLLs shadow PROJ's; if pyarrow wins the race, geopandas sets
  `HAS_PYPROJ = False` and **silently drops every CRS** instead of raising.
  `mre.config._preload_proj()` claims those DLLs first. See
  docs/ARCHITECTURE.md, "The pyarrow / PROJ DLL clash". `tests/conftest.py`
  enforces the ordering for the test suite; new scripts must do it themselves.
- Blender uses its **own embedded Python**. Never import `bpy` into this venv;
  drive Blender headlessly via `blender --background --python <script>`.
- Blender is located through `config/default.toml`, **not** PATH. Anything
  Blender-dependent must skip cleanly when the executable is absent.

## Scientific-integrity rules

These are not style preferences.

- The city is **synthetic**. Never present synthetic output as Istanbul or
  Marmara data, and never plot it on a real basemap.
- Never state that a specific real building will collapse.
- Never call the fragility parameters a Turkish or validated fragility model.
- Never describe the engine as earthquake prediction.
- Always label the intervention module output **"Prototype intervention
  optimization"**.
- Report distributions with percentile bands, not bare means. A single number
  from a Monte Carlo run is a lie of omission.
- **Never call percentile bands confidence intervals or predictive intervals.**
  They are empirical quantiles of the simulated distribution.
- Never imply that more realisations make the model more valid. More
  realisations narrow the bands; precision is not accuracy.
- Keep sampling variability, model uncertainty, and synthetic assumptions
  distinct — only the first is represented.
- `hospital_utilisation` is a demand-pressure indicator, not a queueing result,
  and in the default city it is inflated ~2-3 orders of magnitude by a
  population/capacity scale artefact. Report the artefact wherever the number
  appears.
- Any new simplification must be added to `docs/SCIENTIFIC_ASSUMPTIONS.md` in
  the same change that introduces it.
- The `I` field is a **dimensionless intensity proxy**. Do not relabel it PGA,
  PGV, SA(T), or MMI.

## Design rules

- Stages talk to each other only through `mre.models` types and the `Protocol`
  seams listed in `docs/ARCHITECTURE.md`. The simulation driver must never
  import a concrete hazard/fragility/disruption implementation directly.
- Real data must produce the **same** `mre.models` types as synthetic data. The
  engine must not be able to tell the difference.
- NetworkX is the graph backend. **OSMnx is optional and future** — never make
  it a required import. The synthetic slice must run with no network access.
- Randomness flows through explicitly passed `numpy.random.Generator` objects
  obtained from `mre.rng.named_generator(seed, name)`. No module-level global
  RNG, no bare `np.random.*` calls — `tests/test_reproducibility.py` greps the
  source and fails if one appears.
- New stochastic stages get a **new stream name**, never a reused one. Reusing
  a name couples two stages' draws and breaks controlled comparison.
- Model parameters belong in `config/default.toml`, never hard-coded. Anything
  invented must say so in a comment and appear in
  `docs/SCIENTIFIC_ASSUMPTIONS.md`.
- Fragility medians are **divided** by `vulnerability_index` — higher index
  means more fragile. Getting this backwards inverts the entire damage model
  while still producing plausible-looking output.
- Keep the full probability vector alongside any sampled outcome. A sampled
  state alone discards what the model knows.
- Never mutate the baseline network. `apply_link_states` returns a copy so
  before/after comparison stays possible.
- Unreachability is a **result**: return `inf`, never raise.
- **Never let unreachable population into a travel-time mean or median.**
  Exclude it and count it separately. Averaging in `inf` gives NaN; averaging
  in the threshold silently understates the loss. Both hide the effect being
  measured.
- Where a statistic is undefined (nobody reachable), report `nan`, never `0.0`
  — zero reads as instant access.
- `distribution_summary` excludes non-finite values and reports `n_excluded`.
  Never propagate NaN into a summary.
- Compare post-event against the **deterministic baseline for the same city**.
  Never compare two realisations against each other.
- Monte Carlo must not retain graphs or per-building arrays per realisation —
  reduce to a `RealisationRecord` plus running counters.
- Build the `ModelSuite` once per run, not per realisation.
- Blender never appears in an `import` inside `mre/`.

## Data rules

- `data/**` is gitignored. Synthetic worlds are regenerated from the seed, not
  committed.
- Do not download large real datasets, and do not create API keys.

## Workflow

- Proceed phase by phase; run tests and report before advancing.
- Show and explain any package-install or environment-modifying command
  **before** running it.
- Do not commit or push automatically.
- Do not modify files outside the repository.

## Commands

```bash
.venv\Scripts\python.exe scripts\verify_env.py
```

```bash
.venv\Scripts\python.exe -m pytest
```
