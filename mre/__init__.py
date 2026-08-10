"""MARMARA RESILIENCE ENGINE -- MRE-001 prototype.

MRE-001 is a research/software-architecture vertical slice running on a fully
SYNTHETIC city. It is not a validated earthquake-risk model and makes no claim
about any real building, road, or hospital. See docs/SCIENTIFIC_ASSUMPTIONS.md.

Importing this package sets GDAL/PROJ data paths for the current process. That
must happen before ``pyproj`` or ``osgeo`` are first imported, so it is done
here rather than left to individual modules.
"""

from __future__ import annotations

from mre.config import bootstrap_geo_env

__version__ = "0.1.0.dev0"

bootstrap_geo_env()

__all__ = ["__version__", "bootstrap_geo_env"]
