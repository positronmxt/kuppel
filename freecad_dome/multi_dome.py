"""Multi-dome project planning — multiple domes with connecting corridors.

This module implements ROADMAP item III3: Mitu kupli ühe projekti raames.

Features
--------
- **DomeInstance** — one dome in the project with positional offset, rotation
  and optional parameter overrides (e.g. a smaller annex).
- **Corridor** — rectangular connecting passage between two domes at belt
  level.  Geometry includes wall thickness, material, and floor/roof planes.
- **MultiDomePlan** — the complete project layout with merged foundation plan,
  merged BOM, and inter-dome corridor list.
- ``plan_multi_dome()`` — entry-point that resolves dome placement, generates
  corridors, merges foundation anchors, and merges BOM rows.
- ``write_multi_dome_report()`` — JSON report with per-dome summaries,
  corridor geometry, merged foundation, and merged BOM.

Usage::

    from freecad_dome.multi_dome import plan_multi_dome, write_multi_dome_report

    plan = plan_multi_dome(dome, params)
    write_multi_dome_report(plan, Path("exports"))
"""

from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .parameters import DomeParameters

__all__ = [
    "DomeInstance",
    "Corridor",
    "MultiDomePlan",
    "parse_dome_instances",
    "parse_corridor_definitions",
    "plan_multi_dome",
    "write_multi_dome_report",
]

log = logging.getLogger(__name__)

Vector2 = Tuple[float, float]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DomeInstance:
    """One dome within a multi-dome project.

    ``index == 0`` is always the primary (main) dome whose parameters come
    from the project DomeParameters.  Secondary domes (index >= 1) carry
    positional offsets and optional parameter overrides.
    """

    index: int
    label: str                      # human-readable label ("Peakuppel", "Anneks")
    offset_x_m: float = 0.0        # X offset from project origin
    offset_y_m: float = 0.0        # Y offset from project origin
    rotation_deg: float = 0.0      # rotation around Z
    radius_m: float = 5.0          # effective radius (from overrides or primary)
    hemisphere_ratio: float = 0.5
    belt_radius_m: float = 0.0     # computed from radius + hemisphere_ratio
    belt_height_m: float = 0.0     # computed belt Z
    overrides: Dict[str, Any] = field(default_factory=dict)

    @property
    def centre(self) -> Vector2:
        """2-D position of dome centre."""
        return (self.offset_x_m, self.offset_y_m)

    def distance_to(self, other: "DomeInstance") -> float:
        """Horizontal distance between dome centres."""
        dx = self.offset_x_m - other.offset_x_m
        dy = self.offset_y_m - other.offset_y_m
        return math.sqrt(dx * dx + dy * dy)

    def edge_distance_to(self, other: "DomeInstance") -> float:
        """Approximate clearance between dome shells."""
        return max(self.distance_to(other) - self.belt_radius_m - other.belt_radius_m, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "offset_x_m": round(self.offset_x_m, 4),
            "offset_y_m": round(self.offset_y_m, 4),
            "rotation_deg": round(self.rotation_deg, 2),
            "radius_m": round(self.radius_m, 4),
            "hemisphere_ratio": round(self.hemisphere_ratio, 4),
            "belt_radius_m": round(self.belt_radius_m, 4),
            "belt_height_m": round(self.belt_height_m, 4),
            "overrides": self.overrides,
        }


@dataclass(slots=True)
class Corridor:
    """Connecting passage between two domes.

    The corridor is modelled as a rectangular tunnel whose long axis runs
    from the belt circle of *from_dome* to the belt circle of *to_dome*.
    """

    from_dome: int          # index of source dome
    to_dome: int            # index of destination dome
    width_m: float = 1.2
    height_m: float = 2.1
    wall_thickness_m: float = 0.15
    material: str = "wood"  # "wood" | "steel" | "glass"
    length_m: float = 0.0   # computed: edge-to-edge distance
    azimuth_deg: float = 0.0  # direction from source to destination
    # End-point coordinates (belt-circle intersection, computed)
    start_x_m: float = 0.0
    start_y_m: float = 0.0
    end_x_m: float = 0.0
    end_y_m: float = 0.0

    @property
    def floor_area_m2(self) -> float:
        return self.length_m * self.width_m

    @property
    def wall_area_m2(self) -> float:
        """Both side walls + ceiling (no floor)."""
        return (2.0 * self.height_m + self.width_m) * self.length_m

    @property
    def volume_m3(self) -> float:
        return self.length_m * self.width_m * self.height_m

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_dome": self.from_dome,
            "to_dome": self.to_dome,
            "width_m": round(self.width_m, 4),
            "height_m": round(self.height_m, 4),
            "wall_thickness_m": round(self.wall_thickness_m, 4),
            "material": self.material,
            "length_m": round(self.length_m, 4),
            "azimuth_deg": round(self.azimuth_deg, 2),
            "start_x_m": round(self.start_x_m, 4),
            "start_y_m": round(self.start_y_m, 4),
            "end_x_m": round(self.end_x_m, 4),
            "end_y_m": round(self.end_y_m, 4),
            "floor_area_m2": round(self.floor_area_m2, 3),
            "wall_area_m2": round(self.wall_area_m2, 3),
            "volume_m3": round(self.volume_m3, 3),
        }


@dataclass
class MultiDomePlan:
    """Complete multi-dome project layout."""

    domes: List[DomeInstance] = field(default_factory=list)
    corridors: List[Corridor] = field(default_factory=list)

    # Merged foundation anchors (offset-adjusted coordinates)
    merged_foundation_anchors: List[Dict[str, Any]] = field(default_factory=list)

    # Merged BOM across all domes + corridors
    merged_bom: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def dome_count(self) -> int:
        return len(self.domes)

    @property
    def corridor_count(self) -> int:
        return len(self.corridors)

    @property
    def total_floor_area_m2(self) -> float:
        """Sum of dome floor areas + corridor floor areas."""
        dome_area = 0.0
        for d in self.domes:
            dome_area += math.pi * d.belt_radius_m ** 2
        corr_area = sum(c.floor_area_m2 for c in self.corridors)
        return dome_area + corr_area

    def summary(self) -> str:
        return (
            f"{self.dome_count} domes, {self.corridor_count} corridors, "
            f"total floor {self.total_floor_area_m2:.1f} m², "
            f"{len(self.merged_bom)} BOM items"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dome_count": self.dome_count,
            "corridor_count": self.corridor_count,
            "total_floor_area_m2": round(self.total_floor_area_m2, 3),
            "domes": [d.to_dict() for d in self.domes],
            "corridors": [c.to_dict() for c in self.corridors],
            "merged_foundation_anchors": self.merged_foundation_anchors,
            "merged_bom": self.merged_bom,
        }


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_dome_instances(
    params: DomeParameters,
) -> List[DomeInstance]:
    """Parse dome instance definitions from parameters.

    The primary dome (index 0) is always created from the base params.
    Secondary domes are parsed from ``dome_instances_json``.
    """
    R = float(params.radius_m)
    hemi = float(params.hemisphere_ratio)
    belt_z = R * (1.0 - 2.0 * hemi) if hemi < 1.0 else -R
    disc = R * R - belt_z * belt_z
    belt_r = math.sqrt(max(disc, 0.0))

    primary = DomeInstance(
        index=0,
        label="Peakuppel",
        radius_m=R,
        hemisphere_ratio=hemi,
        belt_radius_m=belt_r,
        belt_height_m=belt_z,
    )
    domes = [primary]

    raw = getattr(params, "dome_instances_json", "[]")
    try:
        entries = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        entries = []

    if not isinstance(entries, list):
        entries = []

    for i, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            continue
        overrides = entry.get("overrides", {})
        r = float(overrides.get("radius_m", R))
        h = float(overrides.get("hemisphere_ratio", hemi))
        bz = r * (1.0 - 2.0 * h) if h < 1.0 else -r
        d2 = r * r - bz * bz
        br = math.sqrt(max(d2, 0.0))

        domes.append(DomeInstance(
            index=i,
            label=entry.get("label", f"Kuppel {i + 1}"),
            offset_x_m=float(entry.get("offset_x_m", 0.0)),
            offset_y_m=float(entry.get("offset_y_m", 0.0)),
            rotation_deg=float(entry.get("rotation_deg", 0.0)),
            radius_m=r,
            hemisphere_ratio=h,
            belt_radius_m=br,
            belt_height_m=bz,
            overrides=overrides,
        ))

    return domes


def parse_corridor_definitions(
    params: DomeParameters,
    domes: List[DomeInstance],
) -> List[Corridor]:
    """Parse corridor definitions from parameters and compute geometry.

    Falls back to the project-level corridor dimensions when individual
    corridor dicts don't specify width/height.
    """
    raw = getattr(params, "corridor_definitions_json", "[]")
    try:
        entries = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        entries = []

    if not isinstance(entries, list):
        entries = []

    default_w = float(params.corridor_width_m)
    default_h = float(params.corridor_height_m)
    default_t = float(params.corridor_wall_thickness_m)
    default_mat = str(params.corridor_material)
    dome_map = {d.index: d for d in domes}

    corridors: List[Corridor] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        fi = int(entry.get("from_dome", 0))
        ti = int(entry.get("to_dome", 1))
        if fi not in dome_map or ti not in dome_map or fi == ti:
            log.warning("Skipping invalid corridor definition: %s", entry)
            continue

        src = dome_map[fi]
        dst = dome_map[ti]

        width = float(entry.get("width_m", default_w))
        height = float(entry.get("height_m", default_h))
        thick = float(entry.get("wall_thickness_m", default_t))
        mat = str(entry.get("material", default_mat))

        corr = _compute_corridor(src, dst, width, height, thick, mat)
        corridors.append(corr)

    return corridors


# ---------------------------------------------------------------------------
# Corridor geometry
# ---------------------------------------------------------------------------

def _compute_corridor(
    src: DomeInstance,
    dst: DomeInstance,
    width: float,
    height: float,
    thickness: float,
    material: str,
) -> Corridor:
    """Compute corridor endpoints and length between two domes."""
    dx = dst.offset_x_m - src.offset_x_m
    dy = dst.offset_y_m - src.offset_y_m
    dist = math.sqrt(dx * dx + dy * dy)

    if dist < 1e-9:
        azimuth = 0.0
        ux, uy = 1.0, 0.0
    else:
        azimuth = math.degrees(math.atan2(dy, dx)) % 360.0
        ux, uy = dx / dist, dy / dist

    # Start at belt circle of source dome
    sx = src.offset_x_m + ux * src.belt_radius_m
    sy = src.offset_y_m + uy * src.belt_radius_m
    # End at belt circle of destination dome
    ex = dst.offset_x_m - ux * dst.belt_radius_m
    ey = dst.offset_y_m - uy * dst.belt_radius_m

    length = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)

    return Corridor(
        from_dome=src.index,
        to_dome=dst.index,
        width_m=width,
        height_m=height,
        wall_thickness_m=thickness,
        material=material,
        length_m=max(length, 0.0),
        azimuth_deg=azimuth,
        start_x_m=sx,
        start_y_m=sy,
        end_x_m=ex,
        end_y_m=ey,
    )


# ---------------------------------------------------------------------------
# Foundation merging
# ---------------------------------------------------------------------------

def _merge_foundations(
    domes: List[DomeInstance],
    params: DomeParameters,
) -> List[Dict[str, Any]]:
    """Merge anchor bolt positions across all domes, offset-adjusted.

    Each anchor gets its coordinates shifted by the dome's positional offset
    so all anchors are in one unified coordinate system.
    """
    ftype = getattr(params, "foundation_type", "strip")
    bolt_dia = getattr(params, "foundation_bolt_diameter_m", 0.016)

    all_anchors: List[Dict[str, Any]] = []

    for dome_inst in domes:
        R = dome_inst.radius_m
        hemi = dome_inst.hemisphere_ratio
        belt_z = dome_inst.belt_height_m
        belt_r = dome_inst.belt_radius_m

        # Generate evenly-spaced anchors around the belt circle
        # Use the dome's own values; the actual count comes from tessellation
        # but for planning we estimate based on typical spacing (~0.4 m)
        spacing_target = 0.4
        circumference = 2.0 * math.pi * belt_r
        n_anchors = max(int(round(circumference / spacing_target)), 6)

        cos_rot = math.cos(math.radians(dome_inst.rotation_deg))
        sin_rot = math.sin(math.radians(dome_inst.rotation_deg))

        for i in range(n_anchors):
            angle = 2.0 * math.pi * i / n_anchors
            # Local coordinates
            lx = belt_r * math.cos(angle)
            ly = belt_r * math.sin(angle)
            # Rotate
            rx = lx * cos_rot - ly * sin_rot
            ry = lx * sin_rot + ly * cos_rot
            # Offset
            gx = rx + dome_inst.offset_x_m
            gy = ry + dome_inst.offset_y_m
            azimuth = math.degrees(math.atan2(gy - dome_inst.offset_y_m,
                                               gx - dome_inst.offset_x_m)) % 360.0

            all_anchors.append({
                "dome_index": dome_inst.index,
                "dome_label": dome_inst.label,
                "anchor_index": i,
                "x_m": round(gx, 4),
                "y_m": round(gy, 4),
                "z_m": round(belt_z, 4),
                "azimuth_deg": round(azimuth, 2),
                "bolt_diameter_m": bolt_dia,
            })

    return all_anchors


# ---------------------------------------------------------------------------
# BOM merging
# ---------------------------------------------------------------------------

def _merge_bom(
    domes: List[DomeInstance],
    corridors: List[Corridor],
    params: DomeParameters,
) -> List[Dict[str, Any]]:
    """Build a consolidated BOM for the entire multi-dome project."""
    bom: List[Dict[str, Any]] = []

    for dome_inst in domes:
        # Approximate timber for struts
        R = dome_inst.radius_m
        hemi = dome_inst.hemisphere_ratio
        belt_r = dome_inst.belt_radius_m
        # Surface area of dome cap
        cap_area = 2.0 * math.pi * R * R * hemi
        # Rough strut length total ≈ 3 × surface_area / R
        est_strut_length = 3.0 * cap_area / R if R > 0 else 0.0

        bom.append({
            "dome": dome_inst.label,
            "item": f"Timber struts (dome {dome_inst.index})",
            "type": "timber",
            "estimated_total_m": round(est_strut_length, 1),
        })

        # Panel covering
        bom.append({
            "dome": dome_inst.label,
            "item": f"Panel covering (dome {dome_inst.index})",
            "type": "covering",
            "estimated_area_m2": round(cap_area, 1),
        })

        # Foundation anchors
        circumference = 2.0 * math.pi * belt_r
        n_anchors = max(int(round(circumference / 0.4)), 6)
        bom.append({
            "dome": dome_inst.label,
            "item": f"Anchor bolts (dome {dome_inst.index})",
            "type": "hardware",
            "quantity": n_anchors,
        })

    # Corridor materials
    for corr in corridors:
        if corr.length_m <= 0:
            continue

        bom.append({
            "corridor": f"Dome {corr.from_dome} → Dome {corr.to_dome}",
            "item": f"Corridor frame ({corr.material})",
            "type": "corridor_frame",
            "length_m": round(corr.length_m, 2),
            "wall_area_m2": round(corr.wall_area_m2, 2),
        })

        bom.append({
            "corridor": f"Dome {corr.from_dome} → Dome {corr.to_dome}",
            "item": "Corridor floor",
            "type": "corridor_floor",
            "area_m2": round(corr.floor_area_m2, 2),
        })

    return bom


# ---------------------------------------------------------------------------
# Main planning function
# ---------------------------------------------------------------------------

def plan_multi_dome(
    dome: Any,
    params: DomeParameters,
) -> MultiDomePlan:
    """Create a MultiDomePlan from the primary dome and project parameters.

    Parameters
    ----------
    dome : TessellatedDome
        Primary dome (index 0) — only used for consistency; per-dome
        tessellation would be done by separate pipeline runs.
    params : DomeParameters
        Current configuration including multi-dome JSON fields.
    """
    domes = parse_dome_instances(params)
    corridors = parse_corridor_definitions(params, domes)

    merged_anchors: List[Dict[str, Any]] = []
    if getattr(params, "merge_foundation", True):
        merged_anchors = _merge_foundations(domes, params)

    merged_bom: List[Dict[str, Any]] = []
    if getattr(params, "merge_bom", True):
        merged_bom = _merge_bom(domes, corridors, params)

    plan = MultiDomePlan(
        domes=domes,
        corridors=corridors,
        merged_foundation_anchors=merged_anchors,
        merged_bom=merged_bom,
    )

    log.info("Multi-dome plan: %s", plan.summary())
    return plan


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_multi_dome_report(
    plan: MultiDomePlan,
    out_dir: Path,
) -> Path:
    """Write a JSON report with the full multi-dome project layout."""
    report = {
        "multi_dome_plan": plan.to_dict(),
        "summary": plan.summary(),
        "notes": [
            f"Primary dome: {plan.domes[0].label}" if plan.domes else "No domes",
            f"Total floor area: {plan.total_floor_area_m2:.1f} m²",
            f"Corridors: {plan.corridor_count}",
            f"Merged BOM items: {len(plan.merged_bom)}",
            f"Merged foundation anchors: {len(plan.merged_foundation_anchors)}",
        ],
    }

    out = Path(out_dir) / "multi_dome_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Wrote multi-dome report to %s", out)
    return out
