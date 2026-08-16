# Blender layer

**Not part of the calculation engine.** Nothing in `mre/` imports `bpy`, and
nothing here is required for a simulation run.

Blender uses its **own embedded Python**, not the project `.venv`. Scripts here
run headlessly:

```bash
"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --python <script>
```

The executable is resolved from `config/default.toml` via
`mre.config.blender_executable()`, never from PATH. It returns `None` when
Blender is absent, and callers must skip rather than fail.

## Pipeline

```
scripts/demo.py (.venv, no bpy)
  build_synthetic_city
  → run_scenario(realisation=0)                 damage/link states
  → run_monte_carlo(n_simulations=1, same seed)  population reachability for
                                                  that SAME realisation
                                                  (identical named streams)
  → compare_portfolios(...)                      Phase 5 intervention ranking
  → mre.outputs.write_*                          city.gpkg / scenario / MC /
                                                    intervention GIS+CSV+JSON
  → outputs/demo/blender_scene.json              compact scene description
                                                  (positions, damage/link
                                                  states, targeted ids — no
                                                  geometry libraries needed)
        │
        ▼  file on disk — the only boundary
blender/build_scene.py (Blender's embedded Python, no geopandas)
  reads blender_scene.json → builds three side-by-side city blocks
  (BEFORE / AFTER EARTHQUAKE / AFTER INTERVENTION), a damage/road-state
  legend, camera and lights → outputs/demo/mre_demo.blend (+ .png preview)
```

`scripts/demo.py` never imports `bpy`; `blender/build_scene.py` never imports
`geopandas`, `mre`, or any project module. The only channel between them is
`blender_scene.json` on disk, and the GeoPackage/CSV/JSON files written by
`mre.outputs` alongside it (for downstream GIS use, independent of Blender).

## Running it

```bash
.venv\Scripts\python.exe scripts\demo.py
```

Writes everything under `outputs/demo/` and prints the `.blend` path to open.
`--no-blender` stops after the JSON/GIS outputs (e.g. if Blender is not
installed on this machine); `-n` controls the intervention-comparison
realisation count (default 300, smaller than the config default of 1,000, so
the one-command demo finishes in well under a minute).

## Visual conventions (not scientific parameters)

Colours, building-height multipliers by damage state, marker sizes, and block
layout live in `scripts/demo.py` (the `DAMAGE_COLOR` / `ROAD_COLOR` /
`DAMAGE_HEIGHT_MULTIPLIER` constants), not in `config/default.toml` — they are
a rendering convention for readability, carrying no scientific claim. The
underlying categories (`DamageState`, `LinkState`) and every number behind
them come unmodified from the Phase 3-5 engine. Building footprints and
positions are drawn to true relative scale; nothing is exaggerated for effect
except the damage-state height multiplier (a deliberate readability cue — a
COLLAPSE building renders shorter than a NONE building of the same floor
count) and the intervention-target highlight overlay.

## Status

`blender/smoke_mesh.py` remains the Phase 2 headless smoke test (a tiny
synthetic mesh, no external assets). `blender/build_scene.py` is the demo
visualization described above.
