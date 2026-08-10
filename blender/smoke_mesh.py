"""Blender headless smoke test — runs INSIDE Blender's embedded Python.

    blender.exe --background --python blender/smoke_mesh.py -- <output_path>

Builds a tiny synthetic mesh from explicit vertex/face data (the same way
extruded building footprints will be built later) and writes it to
``<output_path>``. Extension decides the format: ``.glb`` exports via glTF,
``.blend`` saves the scene. If glTF export is unavailable, it falls back to
``.blend`` and says so rather than failing silently.

Downloads nothing and imports no external asset. Never imported by ``mre`` —
``bpy`` exists only in Blender's own interpreter.
"""

import sys
from pathlib import Path

import bpy  # noqa: F401 -- provided by Blender, not by the project venv

# A 2x2 m footprint extruded to 3 m: 8 vertices, 6 quads.
VERTICES = [
    (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0),
    (0.0, 0.0, 3.0), (2.0, 0.0, 3.0), (2.0, 2.0, 3.0), (0.0, 2.0, 3.0),
]
FACES = [
    (0, 1, 2, 3),  # base
    (4, 5, 6, 7),  # roof
    (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),  # walls
]


def parse_output_path() -> Path:
    """Read the path after the ``--`` separator Blender passes through."""
    if "--" not in sys.argv:
        raise SystemExit("SMOKE FAIL: expected '-- <output_path>'")
    args = sys.argv[sys.argv.index("--") + 1:]
    if not args:
        raise SystemExit("SMOKE FAIL: no output path given after '--'")
    return Path(args[0])


def build_mesh(name="MRE_SmokeBuilding"):
    """Create the mesh from raw data — the extruded-footprint pattern."""
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(VERTICES, [], FACES)
    mesh.update()
    if mesh.validate():
        raise SystemExit("SMOKE FAIL: mesh failed validation")

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def write(path: Path) -> str:
    """Write the scene. Returns the format actually used."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".glb":
        try:
            bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB")
            return "GLB"
        except (AttributeError, RuntimeError) as exc:
            print(f"SMOKE INFO: glTF export unavailable ({exc}); falling back to .blend")
            path = path.with_suffix(".blend")

    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    return "BLEND"


def main() -> None:
    output = parse_output_path()

    # Empty scene: no default cube, so the vertex count below is unambiguous.
    bpy.ops.wm.read_factory_settings(use_empty=True)

    obj = build_mesh()
    print(f"SMOKE INFO: blender {bpy.app.version_string}")
    print(f"SMOKE INFO: vertices={len(obj.data.vertices)} polygons={len(obj.data.polygons)}")

    if len(obj.data.vertices) != len(VERTICES) or len(obj.data.polygons) != len(FACES):
        raise SystemExit("SMOKE FAIL: unexpected mesh topology")

    written_format = write(output)
    final = output if output.is_file() else output.with_suffix(".blend")

    if not final.is_file() or final.stat().st_size == 0:
        raise SystemExit(f"SMOKE FAIL: no artifact written at {final}")

    print(f"SMOKE OK: format={written_format} path={final} bytes={final.stat().st_size}")


if __name__ == "__main__":
    main()
