"""Data provenance records — the proof that the real pilot is not fabricated.

Every real dataset the pilot ingests is described by a ``DataSource``: where it
came from, under what licence, when it was acquired, its CRS and geometry, which
attributes it actually carries, and — explicitly — which attributes it does
**not** carry (so a downstream reader knows what was marked ``UNKNOWN`` versus
invented). ``ProvenanceRecord`` bundles the sources for one pilot build and is
serialised to ``provenance.json`` next to the outputs.

If a value is not attributable to a listed source, it does not belong in a
real-pilot output. This module is how that rule is made auditable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["DataSource", "ProvenanceRecord", "utc_now_iso"]


def utc_now_iso() -> str:
    """Acquisition timestamp, UTC, ISO-8601 — recorded, never guessed."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True, slots=True)
class DataSource:
    """One real dataset, fully attributed.

    ``availability`` is one of ``available`` / ``partial`` / ``unavailable`` /
    ``restricted`` — the honest status vocabulary. ``missing_attributes`` names
    the fields the source does **not** provide, which are therefore ``UNKNOWN``
    downstream rather than filled in.
    """

    name: str
    provider: str
    url: str
    license: str
    acquisition_date: str
    crs: str
    geometry_type: str
    spatial_coverage: str
    attributes: tuple[str, ...] = ()
    missing_attributes: tuple[str, ...] = ()
    availability: str = "available"
    feature_count: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Provenance for one real-pilot build."""

    pilot_name: str
    study_area: str
    bbox_wgs84: tuple[float, float, float, float]  # south, west, north, east
    target_crs: str
    generated_at: str = field(default_factory=utc_now_iso)
    sources: tuple[DataSource, ...] = ()
    scenario_provenance: dict[str, Any] = field(default_factory=dict)
    integrity_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "pilot_name": self.pilot_name,
            "study_area": self.study_area,
            "bbox_wgs84": {
                "south": self.bbox_wgs84[0],
                "west": self.bbox_wgs84[1],
                "north": self.bbox_wgs84[2],
                "east": self.bbox_wgs84[3],
            },
            "target_crs": self.target_crs,
            "generated_at": self.generated_at,
            "sources": [s.to_dict() for s in self.sources],
            "scenario_provenance": self.scenario_provenance,
            "integrity_notes": list(self.integrity_notes),
        }

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def is_complete(self) -> tuple[bool, list[str]]:
        """A provenance record is complete when every source names a licence,
        an acquisition date, a CRS, and coverage — no blanks. Returns
        ``(ok, problems)``."""
        problems: list[str] = []
        if not self.sources:
            problems.append("no data sources recorded")
        for source in self.sources:
            for required in ("provider", "url", "license", "acquisition_date", "crs", "spatial_coverage"):
                if not getattr(source, required):
                    problems.append(f"source {source.name!r} missing {required}")
        if not self.scenario_provenance:
            problems.append("no scenario provenance recorded")
        return (not problems), problems
