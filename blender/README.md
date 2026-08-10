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

## Intended pipeline

```
simulation results → GeoJSON / GeoPackage → Blender → 3D visualization
```

The boundary is a file on disk. The engine writes; Blender reads. Neither
imports the other.

## Status

Phase 2 adds a headless smoke test only — a tiny synthetic mesh written to a
temporary artifact, no external assets downloaded. The full visualization is
not scoped yet.
