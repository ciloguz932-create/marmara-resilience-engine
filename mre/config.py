"""Configuration loading and geospatial environment bootstrap.

Two responsibilities:

1. ``bootstrap_geo_env()`` -- point GDAL and PROJ at the data directories that
   ship with QGIS. The QGIS installer does not set ``GDAL_DATA`` / ``PROJ_DATA``
   globally, so without this ``pyproj`` cannot find ``proj.db`` and every
   reprojection fails. We set them in ``os.environ`` for this process only; the
   machine-wide environment is never modified.

2. ``load_config()`` -- read ``config/default.toml``, overlaid by an optional
   gitignored ``config/local.toml`` for machine-specific paths.

The bootstrap also pre-binds PROJ's native DLLs; see ``_preload_proj``.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_CONFIG = CONFIG_DIR / "default.toml"
LOCAL_CONFIG = CONFIG_DIR / "local.toml"

_bootstrapped = False


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load the project configuration as a plain nested dict.

    ``config/local.toml`` overlays ``config/default.toml`` when present.
    """
    source = path or DEFAULT_CONFIG
    with source.open("rb") as fh:
        config = tomllib.load(fh)
    if path is None and LOCAL_CONFIG.exists():
        with LOCAL_CONFIG.open("rb") as fh:
            config = _deep_merge(config, tomllib.load(fh))
    return config


def _preload_proj(qgis_root: str | None) -> None:
    """Bind PROJ's native DLLs before any other library can shadow them.

    On Windows, ``pandas`` 3.x pulls in ``pyarrow``, which ships its own copies
    of common native libraries. If pyarrow loads first, ``pyproj._context``
    resolves against pyarrow's copies and dies with "entry point not found" --
    after which ``geopandas`` silently sets ``HAS_PYPROJ = False`` and every
    ``.crs`` assignment and ``.to_crs()`` call fails or degrades to ``None``.

    Whichever library loads first wins for the life of the process, so we make
    that PROJ. Importing ``pyproj.network`` is what forces the full native
    chain to load; importing ``pyproj`` alone is not sufficient.

    Discovered by Phase 2 smoke tests: ``tests/test_smoke_geopandas.py`` passed
    only when a GDAL test happened to run first and load the DLLs by accident.
    """
    if qgis_root:
        bin_dir = Path(qgis_root) / "bin"
        if bin_dir.is_dir() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(bin_dir))

    # Import for side effect, after GDAL_DATA/PROJ_DATA are already in place.
    import pyproj.network  # noqa: F401


def bootstrap_geo_env(config: dict[str, Any] | None = None, *, force: bool = False) -> dict[str, str]:
    """Set ``GDAL_DATA`` / ``PROJ_DATA`` / ``PROJ_LIB`` for this process.

    Idempotent, and never overwrites values the user has already exported.
    Returns the variables that ended up set, for diagnostics. Missing
    directories are reported rather than silently ignored -- a wrong path here
    produces confusing CRS errors much later.
    """
    global _bootstrapped
    if _bootstrapped and not force:
        return {k: os.environ[k] for k in ("GDAL_DATA", "PROJ_DATA", "PROJ_LIB") if k in os.environ}

    cfg = config if config is not None else load_config()
    env_cfg = cfg.get("environment", {})

    wanted = {
        "GDAL_DATA": env_cfg.get("gdal_data"),
        "PROJ_DATA": env_cfg.get("proj_data"),
        "PROJ_LIB": env_cfg.get("proj_data"),
    }

    applied: dict[str, str] = {}
    for var, value in wanted.items():
        if not value:
            continue
        if os.environ.get(var) and not force:
            applied[var] = os.environ[var]
            continue
        if not Path(value).is_dir():
            raise FileNotFoundError(
                f"{var} target does not exist: {value}. "
                f"Fix [environment] in {DEFAULT_CONFIG} or override in {LOCAL_CONFIG}."
            )
        os.environ[var] = value
        applied[var] = value

    _preload_proj(env_cfg.get("qgis_root"))

    _bootstrapped = True
    return applied


def blender_executable(config: dict[str, Any] | None = None) -> Path | None:
    """Path to the Blender executable, or ``None`` if it is not installed.

    Blender is never on PATH in this setup and is never required by the
    simulation core -- callers must handle ``None`` by skipping.
    """
    cfg = config if config is not None else load_config()
    raw = cfg.get("environment", {}).get("blender_exe")
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None
