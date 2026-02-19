"""Ventilation system for the geodesic dome.

Identifies panels for ventilation openings, computes ventilation area,
and generates hinge-point geometry and metadata for the BOM.

Greenhouse domes require 15–25 % of floor area as operable ventilation
to prevent overheating.  This module supports three placement strategies:

- **apex**  — Top-most panel(s) for stack-effect exhaust.
- **ring**  — A ring of panels at a configurable height band for intake/cross-flow.
- **manual** — Explicit panel indices chosen by the user.

Vent panels keep their structural frame (struts) intact; only the covering
(glass/polycarbonate) is marked as openable.  Hinge points are placed along
one edge of each vent panel so the covering can swing outward.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .parameters import DomeParameters
from .tessellation import TessellatedDome, Vector3

__all__ = [
    "VentPanel",
    "VentilationPlan",
    "plan_ventilation",
    "select_apex_vents",
    "select_ring_vents",
    "floor_area_m2",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VentPanel:
    """A single panel designated as a ventilation opening."""

    panel_index: int
    area_m2: float
    centroid: Vector3
    hinge_edge: Tuple[int, int]  # Node indices of the hinge edge.
    hinge_points_m: Tuple[Vector3, Vector3]  # 3D coords of hinge endpoints.
    open_direction: Vector3  # Unit vector: outward swing direction.
    vent_type: str  # "apex" | "ring" | "manual"


@dataclass(slots=True)
class VentilationPlan:
    """Complete ventilation layout for a dome."""

    vents: List[VentPanel]
    floor_area_m2: float
    total_vent_area_m2: float
    vent_ratio: float  # total_vent_area / floor_area (target: 0.15–0.25)
    target_ratio_min: float = 0.15
    target_ratio_max: float = 0.25

    @property
    def meets_target(self) -> bool:
        return self.target_ratio_min <= self.vent_ratio <= self.target_ratio_max

    def summary(self) -> str:
        status = "OK" if self.meets_target else "WARNING"
        return (
            f"{len(self.vents)} vent panels, "
            f"{self.total_vent_area_m2:.2f} m² / {self.floor_area_m2:.2f} m² "
            f"= {self.vent_ratio * 100:.1f}% [{status}]"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vent_count": len(self.vents),
            "floor_area_m2": round(self.floor_area_m2, 3),
            "total_vent_area_m2": round(self.total_vent_area_m2, 3),
            "vent_ratio": round(self.vent_ratio, 4),
            "meets_target": self.meets_target,
            "target_ratio": [self.target_ratio_min, self.target_ratio_max],
            "panels": [
                {
                    "panel_index": v.panel_index,
                    "area_m2": round(v.area_m2, 4),
                    "centroid": [round(c, 4) for c in v.centroid],
                    "hinge_edge": list(v.hinge_edge),
                    "vent_type": v.vent_type,
                }
                for v in self.vents
            ],
        }


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan_ventilation(
    dome: TessellatedDome,
    params: DomeParameters,
) -> VentilationPlan:
    """Create a ventilation plan based on dome parameters.

    Combines apex and ring vent selection to meet the configured target.
    """
    floor = floor_area_m2(params)
    target_area = floor * params.ventilation_target_ratio

    vents: List[VentPanel] = []

    mode = params.ventilation_mode

    if mode == "manual":
        # User-specified panel indices.
        vents = _select_manual_vents(dome, params.ventilation_panel_indices)
    elif mode == "apex":
        vents = select_apex_vents(dome, params, max_panels=params.ventilation_apex_count)
    elif mode == "ring":
        vents = select_ring_vents(dome, params, max_panels=params.ventilation_ring_count)
    else:
        # Default: combine apex + ring to meet target.
        apex = select_apex_vents(dome, params, max_panels=params.ventilation_apex_count)
        vents.extend(apex)
        apex_area = sum(v.area_m2 for v in apex)
        remaining = max(0.0, target_area - apex_area)
        if remaining > 0:
            ring = select_ring_vents(
                dome, params,
                max_panels=params.ventilation_ring_count,
                exclude_indices={v.panel_index for v in apex},
            )
            vents.extend(ring)

    total_area = sum(v.area_m2 for v in vents)
    ratio = total_area / floor if floor > 0 else 0.0

    return VentilationPlan(
        vents=vents,
        floor_area_m2=floor,
        total_vent_area_m2=total_area,
        vent_ratio=ratio,
    )


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------


def select_apex_vents(
    dome: TessellatedDome,
    params: DomeParameters,
    max_panels: int = 1,
) -> List[VentPanel]:
    """Select the top-most panels as apex ventilation.

    The apex pentagon and the ring of hexagons around it are the best
    candidates for stack-effect exhaust.
    """
    scored = _score_panels_by_height(dome, descending=True)
    result: List[VentPanel] = []
    for panel_index, _height in scored[:max_panels]:
        vp = _make_vent_panel(dome, panel_index, vent_type="apex")
        if vp is not None:
            result.append(vp)
    return result


def select_ring_vents(
    dome: TessellatedDome,
    params: DomeParameters,
    max_panels: int = 6,
    height_ratio: float | None = None,
    exclude_indices: set[int] | None = None,
) -> List[VentPanel]:
    """Select panels in a horizontal ring for cross-ventilation intake.

    By default targets panels roughly at 40–60 % of the dome height to
    create effective stack-effect airflow with apex exhaust above.
    """
    if exclude_indices is None:
        exclude_indices = set()

    belt_z = params.radius_m * (1.0 - 2.0 * params.hemisphere_ratio)
    dome_top_z = params.radius_m  # approximate apex
    dome_height = dome_top_z - belt_z
    if dome_height <= 0:
        return []

    if height_ratio is None:
        height_ratio = params.ventilation_ring_height_ratio

    target_z = belt_z + dome_height * height_ratio

    # Score panels by proximity to the target height.
    panels_with_height = _score_panels_by_height(dome, descending=False)
    candidates = []
    for panel_index, centroid_z in panels_with_height:
        if panel_index in exclude_indices:
            continue
        dist = abs(centroid_z - target_z)
        candidates.append((dist, panel_index))
    candidates.sort()

    # Select evenly-spaced panels (by azimuth) from the closest group.
    close_group = candidates[: max(max_panels * 3, 20)]
    if not close_group:
        return []

    # Sort by azimuth for even spacing.
    with_azimuth = []
    for _dist, pidx in close_group:
        panel = dome.panels[pidx]
        pts = [dome.nodes[i] for i in panel.node_indices]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        az = math.atan2(cy, cx) % (2.0 * math.pi)
        with_azimuth.append((az, pidx))
    with_azimuth.sort()

    # Pick evenly spaced by azimuth.
    selected = _pick_evenly_spaced(with_azimuth, max_panels)

    result: List[VentPanel] = []
    for panel_index in selected:
        vp = _make_vent_panel(dome, panel_index, vent_type="ring")
        if vp is not None:
            result.append(vp)
    return result


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def floor_area_m2(params: DomeParameters) -> float:
    """Compute floor area (circular footprint at belt height)."""
    r = params.radius_m
    belt_z = r * (1.0 - 2.0 * params.hemisphere_ratio)
    disc = r * r - belt_z * belt_z
    if disc <= 0:
        return 0.0
    return math.pi * disc


def _score_panels_by_height(
    dome: TessellatedDome,
    descending: bool = True,
) -> List[Tuple[int, float]]:
    """Return (panel_index, centroid_z) sorted by height."""
    result = []
    for panel in dome.panels:
        pts = [dome.nodes[i] for i in panel.node_indices]
        cz = sum(p[2] for p in pts) / len(pts)
        result.append((panel.index, cz))
    result.sort(key=lambda x: x[1], reverse=descending)
    return result


def _polygon_area_3d(points: Sequence[Vector3]) -> float:
    """Compute area of a 3D polygon using the cross-product method."""
    n = len(points)
    if n < 3:
        return 0.0
    # Newell's method for polygon normal magnitude = 2 * area.
    sx = sy = sz = 0.0
    for i in range(n):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[(i + 1) % n]
        sx += (y1 - y2) * (z1 + z2)
        sy += (z1 - z2) * (x1 + x2)
        sz += (x1 - x2) * (y1 + y2)
    return 0.5 * math.sqrt(sx * sx + sy * sy + sz * sz)


def _make_vent_panel(
    dome: TessellatedDome,
    panel_index: int,
    vent_type: str,
) -> VentPanel | None:
    """Build a VentPanel from a dome panel index."""
    if panel_index < 0 or panel_index >= len(dome.panels):
        return None
    panel = dome.panels[panel_index]
    pts = [dome.nodes[i] for i in panel.node_indices]
    if len(pts) < 3:
        return None

    area = _polygon_area_3d(pts)
    centroid = (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
        sum(p[2] for p in pts) / len(pts),
    )

    # Choose hinge edge: the lowest (or most horizontal) edge is best for
    # a top-hinged vent.  For apex panels the "lowest" edge swings outward
    # from above.  For ring panels we pick the top edge so the panel opens
    # upward like a hopper window.
    node_ids = list(panel.node_indices)
    best_edge = (node_ids[0], node_ids[1])
    if vent_type == "apex":
        # Pick the lowest edge (lowest average z).
        best_z = float("inf")
        for i in range(len(node_ids)):
            a = node_ids[i]
            b = node_ids[(i + 1) % len(node_ids)]
            avg_z = (dome.nodes[a][2] + dome.nodes[b][2]) * 0.5
            if avg_z < best_z:
                best_z = avg_z
                best_edge = (a, b)
    else:
        # Pick the highest edge for hopper-style opening.
        best_z = float("-inf")
        for i in range(len(node_ids)):
            a = node_ids[i]
            b = node_ids[(i + 1) % len(node_ids)]
            avg_z = (dome.nodes[a][2] + dome.nodes[b][2]) * 0.5
            if avg_z > best_z:
                best_z = avg_z
                best_edge = (a, b)

    hinge_a = dome.nodes[best_edge[0]]
    hinge_b = dome.nodes[best_edge[1]]

    # Open direction: panel normal (outward from dome center).
    nx, ny, nz = panel.normal
    # Ensure outward (away from origin, since normals stored inward).
    dot = centroid[0] * nx + centroid[1] * ny + centroid[2] * nz
    if dot < 0:
        nx, ny, nz = -nx, -ny, -nz
    norm_len = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm_len > 1e-12:
        nx /= norm_len
        ny /= norm_len
        nz /= norm_len

    return VentPanel(
        panel_index=panel_index,
        area_m2=area,
        centroid=centroid,
        hinge_edge=best_edge,
        hinge_points_m=(hinge_a, hinge_b),
        open_direction=(nx, ny, nz),
        vent_type=vent_type,
    )


def _select_manual_vents(
    dome: TessellatedDome,
    indices: Sequence[int],
) -> List[VentPanel]:
    """Build vent panels from explicit panel indices."""
    result: List[VentPanel] = []
    for idx in indices:
        vp = _make_vent_panel(dome, idx, vent_type="manual")
        if vp is not None:
            result.append(vp)
    return result


def _pick_evenly_spaced(
    items_with_azimuth: List[Tuple[float, int]],
    count: int,
) -> List[int]:
    """Pick *count* items spaced as evenly as possible around the circle."""
    if not items_with_azimuth:
        return []
    if len(items_with_azimuth) <= count:
        return [idx for _, idx in items_with_azimuth]

    step = 2.0 * math.pi / count
    selected: List[int] = []
    used: set[int] = set()

    for slot in range(count):
        target_az = slot * step
        best_idx = -1
        best_dist = float("inf")
        for az, pidx in items_with_azimuth:
            if pidx in used:
                continue
            d = abs((az - target_az + math.pi) % (2.0 * math.pi) - math.pi)
            if d < best_dist:
                best_dist = d
                best_idx = pidx
        if best_idx >= 0:
            selected.append(best_idx)
            used.add(best_idx)

    return selected
