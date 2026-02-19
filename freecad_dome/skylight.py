"""Skylight / window module for the geodesic dome.

Replaces selected dome panels with openable skylights or fixed windows.
Each skylight has its own glazing thickness (which may differ from the
standard covering), a dedicated frame, and hinge geometry.

Placement strategies mirror the ventilation module:

- **apex**  — Top-most panel(s), ideal for stack-effect ventilation skylights.
- **ring**  — A ring of panels at configurable height for daylight intake.
- **manual** — Explicit panel indices chosen by the user.

The module produces a :class:`SkylightPlan` containing all selected panels
with their hinge edges, frame dimensions, and material specification.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .parameters import DomeParameters
from .tessellation import TessellatedDome, Vector3

__all__ = [
    "SkylightPanel",
    "SkylightPlan",
    "plan_skylights",
    "select_apex_skylights",
    "select_ring_skylights",
    "write_skylight_report",
    "create_skylight_solids",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SkylightPanel:
    """A single panel designated as a skylight / window."""

    panel_index: int
    area_m2: float
    centroid: Vector3
    hinge_edge: Tuple[int, int]          # Node indices of the hinge edge.
    hinge_points_m: Tuple[Vector3, Vector3]  # 3-D coords of hinge endpoints.
    open_direction: Vector3              # Unit vector: outward swing direction.
    glass_thickness_m: float
    frame_width_m: float
    material: str                        # "glass" | "polycarbonate"
    hinge_side: str                      # "top" | "bottom" | "left" | "right"
    placement: str                       # "apex" | "ring" | "manual"


@dataclass(slots=True)
class SkylightPlan:
    """Complete skylight layout for a dome."""

    skylights: List[SkylightPanel]
    total_skylight_area_m2: float
    dome_surface_area_m2: float
    skylight_ratio: float  # total_skylight_area / dome_surface_area

    def summary(self) -> str:
        return (
            f"{len(self.skylights)} skylights, "
            f"{self.total_skylight_area_m2:.2f} m² / "
            f"{self.dome_surface_area_m2:.2f} m² surface "
            f"= {self.skylight_ratio * 100:.1f}%"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skylight_count": len(self.skylights),
            "total_skylight_area_m2": round(self.total_skylight_area_m2, 4),
            "dome_surface_area_m2": round(self.dome_surface_area_m2, 4),
            "skylight_ratio": round(self.skylight_ratio, 4),
            "panels": [
                {
                    "panel_index": s.panel_index,
                    "area_m2": round(s.area_m2, 4),
                    "centroid": [round(c, 4) for c in s.centroid],
                    "hinge_edge": list(s.hinge_edge),
                    "glass_thickness_m": s.glass_thickness_m,
                    "frame_width_m": s.frame_width_m,
                    "material": s.material,
                    "hinge_side": s.hinge_side,
                    "placement": s.placement,
                }
                for s in self.skylights
            ],
        }


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def plan_skylights(
    dome: TessellatedDome,
    params: DomeParameters,
) -> SkylightPlan:
    """Create a skylight plan based on dome parameters.

    Delegates to the appropriate selection strategy and wraps results
    in a :class:`SkylightPlan`.
    """
    mode = params.skylight_position
    count = params.skylight_count

    skylights: List[SkylightPanel] = []

    if mode == "manual":
        skylights = _select_manual_skylights(dome, params)
    elif mode == "ring":
        skylights = select_ring_skylights(dome, params, max_panels=count)
    else:
        # Default: apex
        skylights = select_apex_skylights(dome, params, max_panels=count)

    surface = _dome_surface_area(dome)
    total = sum(s.area_m2 for s in skylights)
    ratio = total / surface if surface > 0 else 0.0

    return SkylightPlan(
        skylights=skylights,
        total_skylight_area_m2=total,
        dome_surface_area_m2=surface,
        skylight_ratio=ratio,
    )


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------


def select_apex_skylights(
    dome: TessellatedDome,
    params: DomeParameters,
    max_panels: int = 1,
) -> List[SkylightPanel]:
    """Select the top-most panels as skylights.

    The apex pentagon and its surrounding hexagons provide the best
    overhead daylight.
    """
    scored = _score_panels_by_height(dome, descending=True)
    result: List[SkylightPanel] = []
    for panel_index, _height in scored[:max_panels]:
        sp = _make_skylight_panel(dome, params, panel_index, placement="apex")
        if sp is not None:
            result.append(sp)
    return result


def select_ring_skylights(
    dome: TessellatedDome,
    params: DomeParameters,
    max_panels: int = 3,
    height_ratio: float = 0.6,
) -> List[SkylightPanel]:
    """Select panels in a horizontal ring for daylighting.

    Targets panels at roughly 60 % of dome height by default, providing
    good side-light distribution.
    """
    belt_z = params.radius_m * (1.0 - 2.0 * params.hemisphere_ratio)
    dome_top_z = params.radius_m
    dome_height = dome_top_z - belt_z
    if dome_height <= 0:
        return []

    target_z = belt_z + dome_height * height_ratio

    panels_with_height = _score_panels_by_height(dome, descending=False)
    candidates = []
    for panel_index, centroid_z in panels_with_height:
        dist = abs(centroid_z - target_z)
        candidates.append((dist, panel_index))
    candidates.sort()

    close_group = candidates[: max(max_panels * 3, 15)]
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

    selected = _pick_evenly_spaced(with_azimuth, max_panels)

    result: List[SkylightPanel] = []
    for panel_index in selected:
        sp = _make_skylight_panel(dome, params, panel_index, placement="ring")
        if sp is not None:
            result.append(sp)
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_skylight_report(plan: SkylightPlan, out_dir: Path) -> Path:
    """Write skylight plan as a JSON report file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "skylight_plan.json"
    path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    logging.info("Wrote skylight plan %s", path)
    return path


# ---------------------------------------------------------------------------
# FreeCAD 3-D geometry
# ---------------------------------------------------------------------------

_FC_SCALE: float = 1000.0  # metres → mm


def create_skylight_solids(
    plan: SkylightPlan,
    dome: TessellatedDome,
    document: Any = None,
) -> Optional[Any]:
    """Create FreeCAD Part::Feature solids for each skylight panel.

    For every :class:`SkylightPanel` in *plan*, creates:

    - A **glass pane** — the panel face extruded inward by
      ``glass_thickness_m``.
    - A **frame** — the panel outline extruded by the same thickness,
      offset inward by ``frame_width_m``, producing a border ring.

    All objects are placed under an ``App::DocumentObjectGroup`` named
    ``"Skylights"``.

    Parameters
    ----------
    plan : SkylightPlan
        Output of :func:`plan_skylights`.
    dome : TessellatedDome
        The tessellated dome (needed for panel vertex coordinates).
    document : optional
        Existing FreeCAD document.

    Returns
    -------
    The FreeCAD group object, or *None* on failure.
    """
    try:
        import FreeCAD  # type: ignore
        import Part  # type: ignore
        from FreeCAD import Vector  # type: ignore
    except ImportError:
        return None

    doc = document
    if doc is None:
        doc = FreeCAD.ActiveDocument
    if doc is None:
        doc = FreeCAD.newDocument("GeodesicDome")

    def _mm(m: float) -> float:
        return float(m) * _FC_SCALE

    created: list[Any] = []

    _MATERIAL_COLORS: Dict[str, tuple] = {
        "glass":         (0.40, 0.70, 0.90, 0.0),
        "polycarbonate": (0.85, 0.88, 0.75, 0.0),
    }
    _FRAME_COLOR = (0.25, 0.25, 0.25, 0.0)

    for sk in plan.skylights:
        pidx = sk.panel_index
        if pidx < 0 or pidx >= len(dome.panels):
            continue
        panel = dome.panels[pidx]
        pts = [dome.nodes[i] for i in panel.node_indices]
        if len(pts) < 3:
            continue

        # Convert to FreeCAD vectors (mm) ------------------------------------
        fc_pts = [Vector(_mm(p[0]), _mm(p[1]), _mm(p[2])) for p in pts]

        # Panel normal (outward)
        nx, ny, nz = sk.open_direction
        norm_vec = Vector(nx, ny, nz)
        norm_len = norm_vec.Length
        if norm_len < 1e-12:
            continue
        norm_vec = norm_vec * (1.0 / norm_len)

        thickness = _mm(sk.glass_thickness_m) if sk.glass_thickness_m > 0 else 6.0

        # --- Glass pane (full panel face extruded inward) -------------------
        try:
            wire_pts = list(fc_pts) + [fc_pts[0]]
            wire = Part.makePolygon(wire_pts)
            face = Part.Face(wire)
            glass_solid = face.extrude(norm_vec * (-thickness))

            g_obj = doc.addObject("Part::Feature", f"SkylightGlass_{pidx:03d}")
            g_obj.Label = f"SkylightGlass_{pidx:03d}"
            g_obj.Shape = glass_solid
            color = _MATERIAL_COLORS.get(sk.material, (0.40, 0.70, 0.90, 0.0))
            try:
                g_obj.ViewObject.ShapeColor = color
                g_obj.ViewObject.Transparency = 50
            except Exception:
                pass
            created.append(g_obj)
        except Exception as exc:
            logging.warning("Skylight glass %d failed: %s", pidx, exc)

        # --- Frame ring (panel outline minus inner offset) ------------------
        frame_w = _mm(sk.frame_width_m) if sk.frame_width_m > 0 else _mm(0.05)
        try:
            # Compute centroid
            cx = sum(v.x for v in fc_pts) / len(fc_pts)
            cy = sum(v.y for v in fc_pts) / len(fc_pts)
            cz = sum(v.z for v in fc_pts) / len(fc_pts)
            c = Vector(cx, cy, cz)

            # Scale inward for inner wire
            inner_pts: list[Any] = []
            for v in fc_pts:
                diff = v - c
                edge_len = diff.Length
                if edge_len < 1e-6:
                    inner_pts.append(v)
                else:
                    shrink = max(0.0, 1.0 - frame_w / edge_len)
                    inner_pts.append(c + diff * shrink)

            outer_wire = Part.makePolygon(list(fc_pts) + [fc_pts[0]])
            inner_wire = Part.makePolygon(list(inner_pts) + [inner_pts[0]])

            outer_face = Part.Face(outer_wire)
            inner_face = Part.Face(inner_wire)
            frame_face = outer_face.cut(inner_face)

            frame_solid = frame_face.extrude(norm_vec * (-thickness))

            f_obj = doc.addObject("Part::Feature", f"SkylightFrame_{pidx:03d}")
            f_obj.Label = f"SkylightFrame_{pidx:03d}"
            f_obj.Shape = frame_solid
            try:
                f_obj.ViewObject.ShapeColor = _FRAME_COLOR
            except Exception:
                pass
            created.append(f_obj)
        except Exception as exc:
            logging.warning("Skylight frame %d failed: %s", pidx, exc)

    if not created:
        return None

    # Group all under "Skylights" -----------------------------------------------
    grp = None
    try:
        for o in list(getattr(doc, "Objects", []) or []):
            if (
                str(getattr(o, "Name", "")) == "Skylights_Group"
                and str(getattr(o, "TypeId", "")) == "App::DocumentObjectGroup"
            ):
                grp = o
                break
        if grp is None:
            grp = doc.addObject("App::DocumentObjectGroup", "Skylights_Group")
            grp.Label = "Skylights"
        for co in created:
            grp.addObject(co)
    except Exception:
        pass

    try:
        doc.recompute()
    except Exception:
        pass

    logging.info("Created skylight solids: %d objects", len(created))
    return grp


# ---------------------------------------------------------------------------
# Geometry / selection helpers
# ---------------------------------------------------------------------------


def _dome_surface_area(dome: TessellatedDome) -> float:
    """Compute total dome surface area from panels."""
    total = 0.0
    for panel in dome.panels:
        pts = [dome.nodes[i] for i in panel.node_indices]
        total += _polygon_area_3d(pts)
    return total


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
    """Compute area of a 3-D polygon (Newell's method)."""
    n = len(points)
    if n < 3:
        return 0.0
    sx = sy = sz = 0.0
    for i in range(n):
        x1, y1, z1 = points[i]
        x2, y2, z2 = points[(i + 1) % n]
        sx += (y1 - y2) * (z1 + z2)
        sy += (z1 - z2) * (x1 + x2)
        sz += (x1 - x2) * (y1 + y2)
    return 0.5 * math.sqrt(sx * sx + sy * sy + sz * sz)


def _make_skylight_panel(
    dome: TessellatedDome,
    params: DomeParameters,
    panel_index: int,
    placement: str,
) -> Optional[SkylightPanel]:
    """Build a SkylightPanel from a dome panel index."""
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

    hinge_side = params.skylight_hinge_side
    best_edge = _choose_hinge_edge(dome, panel, hinge_side)

    hinge_a = dome.nodes[best_edge[0]]
    hinge_b = dome.nodes[best_edge[1]]

    # Open direction: panel normal (outward from dome center).
    nx, ny, nz = panel.normal
    dot = centroid[0] * nx + centroid[1] * ny + centroid[2] * nz
    if dot < 0:
        nx, ny, nz = -nx, -ny, -nz
    norm_len = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm_len > 1e-12:
        nx /= norm_len
        ny /= norm_len
        nz /= norm_len

    return SkylightPanel(
        panel_index=panel_index,
        area_m2=area,
        centroid=centroid,
        hinge_edge=best_edge,
        hinge_points_m=(hinge_a, hinge_b),
        open_direction=(nx, ny, nz),
        glass_thickness_m=params.skylight_glass_thickness_m,
        frame_width_m=params.skylight_frame_width_m,
        material=params.skylight_material,
        hinge_side=hinge_side,
        placement=placement,
    )


def _choose_hinge_edge(
    dome: TessellatedDome,
    panel: Any,
    hinge_side: str,
) -> Tuple[int, int]:
    """Choose the hinge edge based on the requested side.

    - ``"top"``    — highest average-z edge (hopper style, swings outward).
    - ``"bottom"`` — lowest average-z edge (awning style).
    - ``"left"``   — most-negative-azimuth edge (casement left).
    - ``"right"``  — most-positive-azimuth edge (casement right).
    """
    node_ids = list(panel.node_indices)
    edges = [
        (node_ids[i], node_ids[(i + 1) % len(node_ids)])
        for i in range(len(node_ids))
    ]
    if not edges:
        return (node_ids[0], node_ids[1]) if len(node_ids) >= 2 else (0, 0)

    if hinge_side in ("top", "bottom"):
        # Score by average z of edge endpoints.
        best = edges[0]
        best_score = float("-inf") if hinge_side == "top" else float("inf")
        for a, b in edges:
            avg_z = (dome.nodes[a][2] + dome.nodes[b][2]) * 0.5
            if hinge_side == "top" and avg_z > best_score:
                best_score = avg_z
                best = (a, b)
            elif hinge_side == "bottom" and avg_z < best_score:
                best_score = avg_z
                best = (a, b)
        return best

    # left / right — score by azimuth of edge midpoint.
    centroid_x = sum(dome.nodes[n][0] for n in node_ids) / len(node_ids)
    centroid_y = sum(dome.nodes[n][1] for n in node_ids) / len(node_ids)
    panel_az = math.atan2(centroid_y, centroid_x)

    best = edges[0]
    best_score = float("inf")  # angular distance from panel azimuth
    target_offset = -math.pi / 2 if hinge_side == "left" else math.pi / 2

    for a, b in edges:
        mx = (dome.nodes[a][0] + dome.nodes[b][0]) * 0.5
        my = (dome.nodes[a][1] + dome.nodes[b][1]) * 0.5
        edge_az = math.atan2(my, mx)
        # Angular distance from the target direction.
        diff = abs(((edge_az - panel_az - target_offset) + math.pi) % (2 * math.pi) - math.pi)
        if diff < best_score:
            best_score = diff
            best = (a, b)
    return best


def _select_manual_skylights(
    dome: TessellatedDome,
    params: DomeParameters,
) -> List[SkylightPanel]:
    """Build skylight panels from explicit panel indices."""
    result: List[SkylightPanel] = []
    for idx in params.skylight_panel_indices:
        sp = _make_skylight_panel(dome, params, idx, placement="manual")
        if sp is not None:
            result.append(sp)
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
