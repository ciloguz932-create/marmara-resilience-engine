"""Phase 0 environment verification. Runs without pytest.

    .venv\\Scripts\\python.exe scripts\\verify_env.py

Checks the interpreter, the required scientific stack, that GDAL/PROJ data
files resolve, and reports whether optional tooling is present. Exits non-zero
if any required check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mre  # noqa: E402  -- must precede pyproj/osgeo; sets GDAL_DATA/PROJ_DATA
from mre.config import blender_executable, load_config  # noqa: E402

REQUIRED = ["numpy", "scipy", "shapely", "pyproj", "geopandas", "networkx", "pandas"]
OPTIONAL = ["pytest", "pyogrio", "matplotlib", "fiona"]

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "", *, required: bool = True) -> None:
    if ok:
        status = "OK"
    elif required:
        status = "FAIL"
        failures.append(label)
    else:
        status = "absent"
    print(f"  {label:<24} {status:<8} {detail}")


def main() -> int:
    print(f"MRE-001 environment check\n\ninterpreter: {sys.executable}")

    print("\n[1] Python 3.12")
    check("python", sys.version_info[:2] == (3, 12), ".".join(map(str, sys.version_info[:3])))

    print("\n[2] pip")
    try:
        import pip

        check("pip", True, pip.__version__)
    except ImportError:
        check("pip", False)

    print("\n[3] Required packages")
    for name in REQUIRED:
        try:
            mod = __import__(name)
            check(name, True, getattr(mod, "__version__", ""))
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            check(name, False, f"{type(exc).__name__}: {exc}")

    print("\n[4] GDAL")
    try:
        from osgeo import gdal

        check("osgeo.gdal", True, gdal.__version__)
    except Exception as exc:  # noqa: BLE001
        check("osgeo.gdal", False, f"{type(exc).__name__}: {exc}")

    print("\n[5] PROJ data resolves (the QGIS GDAL_DATA/PROJ_DATA trap)")
    try:
        import pyproj

        transformer = pyproj.Transformer.from_crs(
            "EPSG:4326", load_config()["project"]["crs"], always_xy=True
        )
        easting, northing = transformer.transform(29.0, 41.0)
        check("reprojection", True, f"PROJ {pyproj.proj_version_str} -> {easting:.1f}, {northing:.1f}")
    except Exception as exc:  # noqa: BLE001
        check("reprojection", False, f"{type(exc).__name__}: {exc}")

    print("\n[6] Optional tooling")
    for name in OPTIONAL:
        try:
            mod = __import__(name)
            check(name, True, getattr(mod, "__version__", ""), required=False)
        except Exception:  # noqa: BLE001
            check(name, False, "not installed", required=False)

    blender = blender_executable()
    check("blender", blender is not None, str(blender or "not found"), required=False)

    print(f"\nmre {mre.__version__}")
    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    print("\nAll required checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
