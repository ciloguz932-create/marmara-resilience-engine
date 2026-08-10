"""Phase 2, smoke test 5 — Blender headless execution.

Blender is NOT part of the calculation engine. This test only proves that the
visualization layer can be driven headlessly from a file-on-disk boundary.

`bpy` is never imported here — it exists only inside Blender's embedded Python.
The engine's side of the boundary is a subprocess call plus an output file.

Skips cleanly when Blender is absent: no test in this repository may require it.
"""

from __future__ import annotations

import subprocess

import pytest

from mre.config import REPO_ROOT, blender_executable

BLENDER = blender_executable()
SCRIPT = REPO_ROOT / "blender" / "smoke_mesh.py"

pytestmark = pytest.mark.blender

requires_blender = pytest.mark.skipif(
    BLENDER is None, reason="Blender not installed; visualization layer is optional"
)


def test_blender_smoke_script_exists():
    assert SCRIPT.is_file()


def test_engine_never_imports_bpy():
    """The boundary rule, enforced rather than documented."""
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / "mre").rglob("*.py")
        if "import bpy" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"bpy imported inside the engine: {offenders}"


@requires_blender
def test_blender_is_discovered_from_config_not_path():
    assert BLENDER is not None
    assert BLENDER.is_file()


@requires_blender
def test_blender_headless_builds_and_writes_mesh(tmp_path):
    """Run Blender in background mode; it must write a non-empty artifact."""
    output = tmp_path / "smoke.glb"

    result = subprocess.run(
        [str(BLENDER), "--background", "--python", str(SCRIPT), "--", str(output)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    combined = result.stdout + result.stderr

    assert "SMOKE FAIL" not in combined, combined
    assert result.returncode == 0, combined
    assert "SMOKE OK" in combined, combined
    assert "vertices=8 polygons=6" in combined, combined

    written = output if output.is_file() else output.with_suffix(".blend")
    assert written.is_file(), f"no artifact at {output} or {written}\n{combined}"
    assert written.stat().st_size > 0
